"""
Backend subprocess lifecycle manager for local orchestrator.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

logger = logging.getLogger("dpolaris.orchestrator.backend")


class BackendProcessError(RuntimeError):
    pass


class PortInUseByUnknownProcessError(BackendProcessError):
    def __init__(self, pid: int, port: int):
        super().__init__(f"Port {port} is already in use by PID {pid} (not managed by orchestrator).")
        self.pid = pid
        self.port = port


@dataclass
class BackendProcessConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    python_exe: Path = Path(sys.executable).resolve()
    workdir: Path = Path(__file__).resolve().parent.parent
    llm_provider: str = "none"
    health_timeout_seconds: float = 2.0
    start_timeout_seconds: float = 20.0
    stop_timeout_seconds: float = 8.0
    tail_lines: int = 400
    data_dir: Path = Path("~/dpolaris_data").expanduser()


class BackendProcessManager:
    def __init__(self, config: Optional[BackendProcessConfig] = None):
        self.config = config or BackendProcessConfig()
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._tail = deque(maxlen=max(50, int(self.config.tail_lines)))
        self._started_at: Optional[dt.datetime] = None
        self._last_restart_at: Optional[dt.datetime] = None
        self._restart_events: deque[dt.datetime] = deque(maxlen=2048)
        self._log_lock = threading.RLock()
        self._current_log_date: Optional[str] = None
        self._current_log_file = None

        self._run_dir = self.config.data_dir / "run"
        self._logs_dir = self.config.data_dir / "logs"
        self._pid_file = self._run_dir / "backend.pid"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def pid_file(self) -> Path:
        return self._pid_file

    @property
    def health_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/health"

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def get_state(self) -> dict:
        with self._lock:
            now = dt.datetime.utcnow()
            restart_24h = sum(1 for x in self._restart_events if (now - x).total_seconds() <= 86400)
            pid = self._process.pid if self._process is not None and self._process.poll() is None else None
            managed_pid = self._read_pid_file()
            port_owner_pid = self._find_pid_on_port(self.config.port)
            port_owner_unknown = bool(
                port_owner_pid
                and port_owner_pid != pid
                and port_owner_pid != managed_pid
            )
            return {
                "running": self.is_running(),
                "pid": pid,
                "health_url": self.health_url,
                "started_at": self._iso(self._started_at),
                "last_restart": self._iso(self._last_restart_at),
                "restart_count_24h": restart_24h,
                "pid_file": str(self._pid_file),
                "managed_pid": managed_pid,
                "port_owner_pid": port_owner_pid,
                "port_owner_unknown": port_owner_unknown,
                "python_executable": str(self.config.python_exe),
                "workdir": str(self.config.workdir),
            }

    def get_tail_lines(self, n: int = 20) -> list[str]:
        n = max(1, int(n))
        with self._lock:
            items = list(self._tail)
        return items[-n:]

    def check_health(self, timeout_seconds: Optional[float] = None) -> bool:
        timeout = timeout_seconds if timeout_seconds is not None else self.config.health_timeout_seconds
        try:
            req = urlrequest.Request(self.health_url, method="GET")
            with urlrequest.urlopen(req, timeout=max(0.2, float(timeout))) as resp:
                return int(resp.status) == 200
        except (urlerror.URLError, TimeoutError, OSError):
            return False

    def wait_until_healthy(self, timeout_seconds: Optional[float] = None) -> bool:
        timeout = timeout_seconds if timeout_seconds is not None else self.config.start_timeout_seconds
        deadline = time.monotonic() + max(1.0, float(timeout))
        while time.monotonic() < deadline:
            if self.check_health():
                return True
            if not self.is_running():
                return False
            time.sleep(0.4)
        return self.check_health(timeout_seconds=1.0)

    def restart_backend(self, reason: str = "unspecified") -> int:
        logger.warning("Restarting backend (reason=%s)", reason)
        self.stop_backend()
        return self.start_backend(reason=reason)

    def start_backend(self, reason: str = "manual") -> int:
        with self._lock:
            if self.is_running():
                return int(self._process.pid)  # type: ignore[arg-type]

            self._terminate_stale_managed_pid()

            owner_pid = self._find_pid_on_port(self.config.port)
            managed_pid = self._read_pid_file()
            if owner_pid is not None and owner_pid != managed_pid:
                raise PortInUseByUnknownProcessError(pid=owner_pid, port=self.config.port)

            cmd = [
                str(self.config.python_exe),
                "-m",
                "cli.main",
                "server",
                "--host",
                self.config.host,
                "--port",
                str(self.config.port),
            ]

            env = os.environ.copy()
            env["LLM_PROVIDER"] = self.config.llm_provider
            env["PYTHONUNBUFFERED"] = "1"

            creation_flags = 0
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                creation_flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

            process = subprocess.Popen(
                cmd,
                cwd=str(self.config.workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
            self._process = process
            self._started_at = dt.datetime.utcnow()
            self._last_restart_at = self._started_at
            self._restart_events.append(self._started_at)
            self._write_pid_file(process.pid)
            self._append_line("SYSTEM", f"backend started pid={process.pid} reason={reason}")

            self._start_stream_thread(process.stdout, "STDOUT")
            self._start_stream_thread(process.stderr, "STDERR")
            return int(process.pid)

    def stop_backend(self) -> None:
        with self._lock:
            process = self._process
            self._process = None

        if process is not None:
            self._append_line("SYSTEM", f"stopping backend pid={process.pid}")
            try:
                process.terminate()
            except Exception:
                pass

            try:
                process.wait(timeout=max(1.0, self.config.stop_timeout_seconds))
            except subprocess.TimeoutExpired:
                self._append_line("SYSTEM", f"terminate timeout pid={process.pid}; forcing kill")
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=3)
                except Exception:
                    pass
        else:
            self._terminate_stale_managed_pid()

        self._delete_pid_file()

    def _terminate_stale_managed_pid(self) -> None:
        pid = self._read_pid_file()
        if not pid:
            return
        if self._is_pid_alive(pid):
            self._append_line("SYSTEM", f"terminating stale managed pid={pid}")
            self._terminate_pid(pid)
            time.sleep(0.6)
            if self._is_pid_alive(pid):
                self._append_line("SYSTEM", f"forcing stale managed pid={pid}")
                self._kill_pid(pid)
        self._delete_pid_file()

    def _start_stream_thread(self, stream, channel: str) -> None:
        if stream is None:
            return

        def _reader():
            try:
                for line in iter(stream.readline, ""):
                    cleaned = line.rstrip("\r\n")
                    if cleaned:
                        self._append_line(channel, cleaned)
            except Exception as exc:
                self._append_line("SYSTEM", f"log stream error ({channel}): {exc}")

        thread = threading.Thread(target=_reader, name=f"backend-{channel.lower()}-reader", daemon=True)
        thread.start()

    def _append_line(self, channel: str, message: str) -> None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} [{channel}] {message}"
        with self._lock:
            self._tail.append(line)
        self._write_log_line(line)

    def _write_log_line(self, line: str) -> None:
        today = dt.datetime.now().strftime("%Y%m%d")
        with self._log_lock:
            if self._current_log_date != today or self._current_log_file is None:
                if self._current_log_file is not None:
                    try:
                        self._current_log_file.close()
                    except Exception:
                        pass
                path = self._logs_dir / f"backend-{today}.log"
                self._current_log_file = open(path, "a", encoding="utf-8")
                self._current_log_date = today
            try:
                self._current_log_file.write(line + "\n")
                self._current_log_file.flush()
            except Exception:
                pass

    def _read_pid_file(self) -> Optional[int]:
        try:
            if self._pid_file.exists():
                return int(self._pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None
        return None

    def _write_pid_file(self, pid: int) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(pid), encoding="utf-8")

    def _delete_pid_file(self) -> None:
        try:
            if self._pid_file.exists():
                self._pid_file.unlink()
        except Exception:
            pass

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    @staticmethod
    def _terminate_pid(pid: int) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)

    @staticmethod
    def _kill_pid(pid: int) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGKILL)

    @staticmethod
    def _find_pid_on_port(port: int) -> Optional[int]:
        try:
            if os.name == "nt":
                proc = subprocess.run(
                    ["netstat", "-ano", "-p", "tcp"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                needle = f":{int(port)}"
                for line in proc.stdout.splitlines():
                    if needle not in line or "LISTENING" not in line.upper():
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        return int(parts[-1])
                    except Exception:
                        continue
            else:
                proc = subprocess.run(
                    ["lsof", "-ti", f"tcp:{int(port)}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                first = (proc.stdout or "").strip().splitlines()
                if first:
                    return int(first[0].strip())
        except Exception:
            return None
        return None

    @staticmethod
    def _iso(value: Optional[dt.datetime]) -> Optional[str]:
        return value.isoformat() if value else None
