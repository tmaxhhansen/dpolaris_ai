"""
Backend control primitives for API-driven process lifecycle management.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

logger = logging.getLogger("dpolaris.backend.control")


class BackendControlError(RuntimeError):
    code = "backend_control_error"

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": str(self),
            "details": self.details,
        }


class PortInUseByUnknownProcessError(BackendControlError):
    code = "port_in_use_by_unmanaged_process"

    def __init__(self, *, port: int, owner_pid: int, owner_cmdline: Optional[str], managed_pid: Optional[int]):
        super().__init__(
            f"Port {port} is owned by PID {owner_pid}, which is not the managed backend PID.",
            details={
                "port": int(port),
                "owner_pid": int(owner_pid),
                "owner_cmdline": owner_cmdline,
                "managed_pid": managed_pid,
            },
        )
        self.port = int(port)
        self.owner_pid = int(owner_pid)
        self.owner_cmdline = owner_cmdline
        self.managed_pid = managed_pid


class UnsafeForceKillError(BackendControlError):
    code = "unsafe_force_kill_candidate"

    def __init__(self, *, pid: int, cmdline: Optional[str], cwd: Optional[str]):
        super().__init__(
            f"Refusing to force-kill PID {pid}; process does not match allowlist.",
            details={
                "pid": int(pid),
                "cmdline": cmdline,
                "cwd": cwd,
                "required_cmd_fragment": "-m cli.main server",
                "required_repo_fragment": "dpolaris_ai",
            },
        )
        self.pid = int(pid)
        self.cmdline = cmdline
        self.cwd = cwd


@dataclass
class BackendControlConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    data_dir: Path = Path("~/dpolaris_data").expanduser()
    python_executable: Path = Path(sys.executable).resolve()
    workdir: Path = Path(__file__).resolve().parent.parent
    health_timeout_seconds: float = 2.0
    startup_timeout_seconds: float = 25.0
    stop_timeout_seconds: float = 8.0


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso_utc(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    else:
        value = value.astimezone(dt.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _parse_iso(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _format_uptime(seconds: int) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class BackendControlManager:
    """Manage backend lifecycle using filesystem runtime metadata."""

    def __init__(self, config: Optional[BackendControlConfig] = None):
        self.config = config or BackendControlConfig()
        self._lock = RLock()
        self._run_dir = self.config.data_dir / "run"
        self._logs_dir = self.config.data_dir / "logs"
        self._pid_path = self._run_dir / "backend.pid"
        self._heartbeat_path = self._run_dir / "backend.heartbeat.json"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def pid_path(self) -> Path:
        return self._pid_path

    @property
    def heartbeat_path(self) -> Path:
        return self._heartbeat_path

    @property
    def health_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/health"

    def read_pid(self) -> Optional[int]:
        try:
            if self._pid_path.exists():
                return int(self._pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            return None
        return None

    def _write_pid(self, pid: int) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._pid_path.write_text(str(int(pid)), encoding="utf-8")

    def _delete_pid(self) -> None:
        try:
            if self._pid_path.exists():
                self._pid_path.unlink()
        except Exception:
            pass

    def read_heartbeat(self) -> dict[str, Any]:
        try:
            if not self._heartbeat_path.exists():
                return {}
            with open(self._heartbeat_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return {}
        return {}

    def _write_heartbeat(self, payload: dict[str, Any]) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        with open(self._heartbeat_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _delete_heartbeat(self) -> None:
        try:
            if self._heartbeat_path.exists():
                self._heartbeat_path.unlink()
        except Exception:
            pass

    @staticmethod
    def is_pid_alive(pid: Optional[int]) -> bool:
        if not pid:
            return False
        if psutil is not None:
            try:
                return bool(psutil.pid_exists(int(pid)))
            except Exception:
                return False
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    @staticmethod
    def process_cmdline(pid: Optional[int]) -> Optional[str]:
        if not pid:
            return None
        if psutil is not None:
            try:
                parts = psutil.Process(int(pid)).cmdline()
                return " ".join(parts) if parts else None
            except Exception:
                pass
        try:
            proc = subprocess.run(
                ["ps", "-p", str(int(pid)), "-o", "command="],
                capture_output=True,
                text=True,
                check=False,
            )
            text = (proc.stdout or "").strip()
            return text or None
        except Exception:
            return None

    @staticmethod
    def process_cwd(pid: Optional[int]) -> Optional[str]:
        if not pid or psutil is None:
            return None
        try:
            return str(psutil.Process(int(pid)).cwd())
        except Exception:
            return None

    @staticmethod
    def find_port_owner_pid(port: int) -> Optional[int]:
        target = int(port)
        try:
            if psutil is not None:
                for conn in psutil.net_connections(kind="tcp"):
                    laddr = getattr(conn, "laddr", None)
                    status = str(getattr(conn, "status", "")).upper()
                    if not laddr or status != "LISTEN":
                        continue
                    if int(getattr(laddr, "port", -1)) == target:
                        pid = getattr(conn, "pid", None)
                        if pid:
                            return int(pid)
        except Exception:
            pass

        try:
            if os.name == "nt":
                proc = subprocess.run(
                    ["netstat", "-ano", "-p", "tcp"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                needle = f":{target}"
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
                    ["lsof", "-nP", "-iTCP:%d" % target, "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for line in (proc.stdout or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        return int(line)
                    except Exception:
                        continue
        except Exception:
            return None
        return None

    def _is_force_kill_allowed(self, *, pid: int, cmdline: Optional[str], cwd: Optional[str]) -> bool:
        cmd = (cmdline or "").lower()
        if "-m cli.main server" not in cmd:
            return False

        repo_hint = "dpolaris_ai"
        if repo_hint in cmd:
            return True
        if cwd and repo_hint in cwd.lower():
            return True
        if repo_hint in str(self.config.workdir).lower():
            return True
        return False

    @staticmethod
    def _kill_pid(pid: int, *, sig: int) -> None:
        if os.name == "nt":
            force = "/F" if int(sig) == int(signal.SIGKILL) else ""
            cmd = ["taskkill", "/PID", str(int(pid)), "/T"]
            if force:
                cmd.append(force)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return
        os.kill(int(pid), sig)

    def _terminate_pid(self, pid: int, *, timeout_seconds: float) -> bool:
        if not self.is_pid_alive(pid):
            return True

        try:
            self._kill_pid(int(pid), sig=signal.SIGTERM)
        except Exception:
            pass

        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        while time.monotonic() < deadline:
            if not self.is_pid_alive(pid):
                return True
            time.sleep(0.2)

        try:
            self._kill_pid(int(pid), sig=signal.SIGKILL)
        except Exception:
            pass
        time.sleep(0.2)
        return not self.is_pid_alive(pid)

    def check_health(self, *, host: Optional[str] = None, port: Optional[int] = None, timeout_seconds: Optional[float] = None) -> bool:
        target_host = str(host or self.config.host)
        target_port = int(port or self.config.port)
        timeout = float(timeout_seconds or self.config.health_timeout_seconds)
        url = f"http://{target_host}:{target_port}/health"
        try:
            req = urlrequest.Request(url, method="GET")
            with urlrequest.urlopen(req, timeout=max(0.2, timeout)) as resp:
                return int(resp.status) == 200
        except (urlerror.URLError, TimeoutError, OSError):
            return False

    def wait_for_health(self, *, timeout_seconds: Optional[float] = None, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        timeout = float(timeout_seconds or self.config.startup_timeout_seconds)
        deadline = time.monotonic() + max(1.0, timeout)
        while time.monotonic() < deadline:
            if self.check_health(host=host, port=port):
                return True
            time.sleep(0.4)
        return self.check_health(host=host, port=port, timeout_seconds=1.0)

    def get_status(self, *, include_health_check: bool = True) -> dict[str, Any]:
        with self._lock:
            managed_pid = self.read_pid()
            heartbeat = self.read_heartbeat()

        host = str(heartbeat.get("host") or self.config.host)
        port = int(heartbeat.get("port") or self.config.port)
        pid_alive = self.is_pid_alive(managed_pid)
        port_owner_pid = self.find_port_owner_pid(port)
        port_owner_cmdline = self.process_cmdline(port_owner_pid) if port_owner_pid else None
        port_owner_cwd = self.process_cwd(port_owner_pid) if port_owner_pid else None
        port_conflict = bool(port_owner_pid and managed_pid and int(port_owner_pid) != int(managed_pid))
        running = bool(managed_pid and pid_alive and not port_conflict)

        started_at_raw = heartbeat.get("started_at")
        started_at_dt = _parse_iso(started_at_raw) if isinstance(started_at_raw, str) else None
        uptime_seconds = None
        if started_at_dt is not None:
            uptime_seconds = max(0, int((_utc_now() - started_at_dt).total_seconds()))

        last_heartbeat = heartbeat.get("last_heartbeat")
        last_health = heartbeat.get("last_health") if isinstance(heartbeat.get("last_health"), dict) else None
        health_snapshot = None
        if include_health_check:
            now_iso = _iso_utc(_utc_now())
            health_snapshot = {
                "ok": bool(self.check_health(host=host, port=port)),
                "timestamp": now_iso,
            }

        python_executable = str(heartbeat.get("python_executable") or self.config.python_executable)
        workdir = str(heartbeat.get("workdir") or self.config.workdir)
        managed = bool(managed_pid and (heartbeat.get("managed", True)))

        return {
            "managed": managed,
            "running": running,
            "pid": managed_pid,
            "pid_alive": pid_alive,
            "pid_file": str(self._pid_path),
            "heartbeat_file": str(self._heartbeat_path),
            "host": host,
            "port": port,
            "health_url": f"http://{host}:{port}/health",
            "python_executable": python_executable,
            "workdir": workdir,
            "started_at": started_at_raw,
            "uptime_seconds": uptime_seconds,
            "uptime": _format_uptime(uptime_seconds) if uptime_seconds is not None else None,
            "last_heartbeat": last_heartbeat,
            "last_health": last_health,
            "current_health": health_snapshot,
            "port_owner_pid": port_owner_pid,
            "port_owner_cmdline": port_owner_cmdline,
            "port_owner_cwd": port_owner_cwd,
            "port_conflict": port_conflict,
        }

    def touch_current_process_heartbeat(self, *, started_at: Optional[dt.datetime] = None, healthy: bool = True) -> dict[str, Any]:
        now = _utc_now()
        now_iso = _iso_utc(now)
        current_pid = int(os.getpid())
        with self._lock:
            existing = self.read_heartbeat()
            existing_started_at = existing.get("started_at")
            started_at_iso: str
            if isinstance(existing_started_at, str) and int(existing.get("pid", -1)) == current_pid:
                started_at_iso = existing_started_at
            elif started_at is not None:
                started_at_iso = _iso_utc(started_at)
            else:
                started_at_iso = now_iso

            payload = {
                "managed": True,
                "manager": "api_control_surface",
                "pid": current_pid,
                "host": self.config.host,
                "port": int(self.config.port),
                "health_url": self.health_url,
                "python_executable": str(Path(sys.executable).resolve()),
                "workdir": str(self.config.workdir),
                "started_at": started_at_iso,
                "last_heartbeat": now_iso,
                "last_health": {
                    "ok": bool(healthy),
                    "timestamp": now_iso,
                },
            }
            self._write_pid(current_pid)
            self._write_heartbeat(payload)
        return payload

    def clear_current_process_runtime_files(self) -> None:
        current_pid = int(os.getpid())
        with self._lock:
            managed_pid = self.read_pid()
            if managed_pid and int(managed_pid) == current_pid:
                self._delete_pid()
            heartbeat = self.read_heartbeat()
            hb_pid = heartbeat.get("pid") if isinstance(heartbeat, dict) else None
            try:
                hb_pid_int = int(hb_pid) if hb_pid is not None else None
            except Exception:
                hb_pid_int = None
            if hb_pid_int == current_pid:
                self._delete_heartbeat()

    def _build_start_command(self) -> list[str]:
        return [
            str(self.config.python_executable),
            "-m",
            "cli.main",
            "server",
            "--host",
            str(self.config.host),
            "--port",
            str(int(self.config.port)),
        ]

    def _spawn_backend_process(self) -> int:
        cmd = self._build_start_command()
        env = os.environ.copy()
        env["LLM_PROVIDER"] = "none"
        env["DPOLARIS_BACKEND_HOST"] = str(self.config.host)
        env["DPOLARIS_BACKEND_PORT"] = str(int(self.config.port))

        stamp = dt.datetime.now().strftime("%Y%m%d")
        log_path = self._logs_dir / f"backend-control-{stamp}.log"
        log_handle = open(log_path, "a", encoding="utf-8")
        creationflags = 0
        kwargs: dict[str, Any] = {
            "cwd": str(self.config.workdir),
            "env": env,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(cmd, **kwargs)
            return int(proc.pid)
        finally:
            try:
                log_handle.close()
            except Exception:
                pass

    def _ensure_port_available_or_kill(self, *, force: bool, managed_pid: Optional[int]) -> None:
        owner_pid = self.find_port_owner_pid(self.config.port)
        if owner_pid is None:
            return
        if managed_pid and int(owner_pid) == int(managed_pid):
            return

        owner_cmdline = self.process_cmdline(owner_pid)
        owner_cwd = self.process_cwd(owner_pid)
        if not force:
            raise PortInUseByUnknownProcessError(
                port=self.config.port,
                owner_pid=owner_pid,
                owner_cmdline=owner_cmdline,
                managed_pid=managed_pid,
            )

        if not self._is_force_kill_allowed(pid=owner_pid, cmdline=owner_cmdline, cwd=owner_cwd):
            raise UnsafeForceKillError(pid=owner_pid, cmdline=owner_cmdline, cwd=owner_cwd)

        terminated = self._terminate_pid(owner_pid, timeout_seconds=self.config.stop_timeout_seconds)
        if not terminated:
            raise BackendControlError(
                f"Failed to terminate forced port owner PID {owner_pid}.",
                details={"pid": owner_pid, "port": int(self.config.port)},
            )

        time.sleep(0.3)

    def start_backend(self, *, force: bool = False, wait_for_health_seconds: Optional[float] = None) -> dict[str, Any]:
        with self._lock:
            managed_pid = self.read_pid()
            if managed_pid and self.is_pid_alive(managed_pid):
                status = self.get_status(include_health_check=True)
                return {"status": "already_running", "healthy": bool((status.get("current_health") or {}).get("ok")), "backend": status}

            if managed_pid and not self.is_pid_alive(managed_pid):
                self._delete_pid()

            self._ensure_port_available_or_kill(force=force, managed_pid=managed_pid)

            pid = self._spawn_backend_process()
            now_iso = _iso_utc(_utc_now())
            self._write_pid(pid)
            self._write_heartbeat(
                {
                    "managed": True,
                    "manager": "api_control_surface",
                    "pid": pid,
                    "host": self.config.host,
                    "port": int(self.config.port),
                    "health_url": self.health_url,
                    "python_executable": str(self.config.python_executable),
                    "workdir": str(self.config.workdir),
                    "started_at": now_iso,
                    "last_heartbeat": now_iso,
                    "last_health": {"ok": False, "timestamp": now_iso},
                }
            )

        timeout = wait_for_health_seconds if wait_for_health_seconds is not None else self.config.startup_timeout_seconds
        healthy = self.wait_for_health(timeout_seconds=timeout)
        status = self.get_status(include_health_check=True)
        return {
            "status": "started" if healthy else "starting",
            "healthy": healthy,
            "backend": status,
        }

    def stop_backend(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            managed_pid = self.read_pid()
            if not managed_pid:
                status = self.get_status(include_health_check=True)
                return {"status": "not_running", "backend": status}

            if int(managed_pid) == int(os.getpid()):
                status = self.get_status(include_health_check=True)
                return {"status": "self_stop_required", "backend": status}

            if not self.is_pid_alive(managed_pid):
                self._delete_pid()
                self._delete_heartbeat()
                status = self.get_status(include_health_check=True)
                return {"status": "not_running", "backend": status}

            timeout = self.config.stop_timeout_seconds if not force else max(2.0, self.config.stop_timeout_seconds / 2)
            stopped = self._terminate_pid(managed_pid, timeout_seconds=timeout)
            if stopped:
                self._delete_pid()
                self._delete_heartbeat()
                status = self.get_status(include_health_check=True)
                return {"status": "stopped", "backend": status}

            status = self.get_status(include_health_check=True)
            return {
                "status": "error",
                "backend": status,
                "error": {
                    "error": "stop_timeout",
                    "message": f"Failed to stop managed backend PID {managed_pid}.",
                    "details": {"pid": managed_pid},
                },
            }

    def restart_backend(self, *, force: bool = False, wait_for_health_seconds: Optional[float] = None) -> dict[str, Any]:
        stop_result = self.stop_backend(force=force)
        if stop_result.get("status") == "self_stop_required":
            return stop_result
        if stop_result.get("status") == "error":
            return stop_result
        return self.start_backend(force=force, wait_for_health_seconds=wait_for_health_seconds)

    def spawn_restart_helper(self, *, old_pid: int, force: bool = False) -> int:
        cmd = [
            str(self.config.python_executable),
            "-m",
            "daemon.backend_control",
            "restart-helper",
            "--old-pid",
            str(int(old_pid)),
            "--host",
            str(self.config.host),
            "--port",
            str(int(self.config.port)),
            "--python-executable",
            str(self.config.python_executable),
            "--workdir",
            str(self.config.workdir),
            "--data-dir",
            str(self.config.data_dir),
        ]
        if force:
            cmd.append("--force")

        env = os.environ.copy()
        env["LLM_PROVIDER"] = "none"
        stamp = dt.datetime.now().strftime("%Y%m%d")
        log_path = self._logs_dir / f"backend-restart-helper-{stamp}.log"
        log_handle = open(log_path, "a", encoding="utf-8")
        kwargs: dict[str, Any] = {
            "cwd": str(self.config.workdir),
            "env": env,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(cmd, **kwargs)
            return int(proc.pid)
        finally:
            try:
                log_handle.close()
            except Exception:
                pass


def _run_restart_helper(args: argparse.Namespace) -> int:
    cfg = BackendControlConfig(
        host=str(args.host),
        port=int(args.port),
        data_dir=Path(args.data_dir).expanduser(),
        python_executable=Path(args.python_executable).expanduser().resolve(),
        workdir=Path(args.workdir).expanduser().resolve(),
    )
    manager = BackendControlManager(cfg)
    old_pid = int(args.old_pid)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not manager.is_pid_alive(old_pid):
            break
        time.sleep(0.2)
    time.sleep(0.4)

    try:
        result = manager.start_backend(force=bool(args.force), wait_for_health_seconds=cfg.startup_timeout_seconds)
        status = str(result.get("status") or "")
        if status in {"started", "already_running"}:
            return 0
        if status == "starting":
            return 0
        return 1
    except Exception:
        logger.exception("restart-helper failed")
        return 1


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dPolaris backend control helpers")
    sub = parser.add_subparsers(dest="command")

    helper = sub.add_parser("restart-helper", help="Wait for old pid to exit, then start backend")
    helper.add_argument("--old-pid", type=int, required=True)
    helper.add_argument("--host", default="127.0.0.1")
    helper.add_argument("--port", type=int, default=8420)
    helper.add_argument("--python-executable", required=True)
    helper.add_argument("--workdir", required=True)
    helper.add_argument("--data-dir", default=str(Path("~/dpolaris_data").expanduser()))
    helper.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.command == "restart-helper":
        return _run_restart_helper(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
