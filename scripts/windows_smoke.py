#!/usr/bin/env python3
"""
Windows smoke test for orchestrator + backend ownership.

Run with:
  .\\.venv\\Scripts\\python.exe scripts\\windows_smoke.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

HOST = "127.0.0.1"
PORT = 8420
TIMEOUT_SECONDS = 30

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = (REPO_ROOT / ".venv" / "Scripts" / "python.exe").resolve()


def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_repo_root() -> None:
    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        _print(f"[INFO] switching working directory to repo root: {REPO_ROOT}")
        os.chdir(REPO_ROOT)


def http_json(path: str, timeout: float = 2.0) -> tuple[int, Any]:
    url = f"http://{HOST}:{PORT}{path}"
    req = urlrequest.Request(url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw.strip() else {}
            return status, data
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        parsed: Any = {"raw": body}
        if body.strip():
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"raw": body}
        return int(exc.code), parsed


def get_port_owner_pid(port: int) -> int | None:
    proc = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pat = re.compile(rf"^\s*TCP\s+\S+:{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.IGNORECASE)
    for line in proc.stdout.splitlines():
        m = pat.match(line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def get_process_cmdline(pid: int) -> str:
    query = f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", query],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip()


def is_non_venv_python(cmdline: str) -> bool:
    if not cmdline:
        return False
    normalized = cmdline.lower().replace("/", "\\")
    venv = str(VENV_PY).lower().replace("/", "\\")
    return ("python" in normalized) and (venv not in normalized)


def start_orchestrator() -> subprocess.Popen[str]:
    if not VENV_PY.exists():
        raise RuntimeError(f"missing venv python: {VENV_PY}")

    args = [
        str(VENV_PY),
        "-m",
        "cli.main",
        "orchestrator",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--interval-health",
        "10",
        "--interval-scan",
        "1h",
        "--dry-run",
    ]
    flags = 0
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=flags,
    )


def main() -> int:
    ensure_repo_root()
    _print("=== windows_smoke.py ===")
    _print(f"repo: {REPO_ROOT}")

    owner = get_port_owner_pid(PORT)
    if owner:
        cmd = get_process_cmdline(owner)
        _print(f"[INFO] port {PORT} owner pid={owner}")
        _print(f"[INFO] owner cmdline: {cmd or '<unavailable>'}")
        if is_non_venv_python(cmd):
            _print("[FAIL] port is owned by non-venv python process.")
            _print("Suggested fix:")
            _print("  1) Stop that process")
            _print("  2) Start orchestrator with: .\\.venv\\Scripts\\python.exe -m cli.main orchestrator --host 127.0.0.1 --port 8420 --interval-health 10 --interval-scan 1h --dry-run")
            return 2

    orch_proc: subprocess.Popen[str] | None = None
    try:
        _print("[STEP] starting orchestrator via venv python...")
        orch_proc = start_orchestrator()
        _print(f"[INFO] orchestrator launch pid={orch_proc.pid}")

        health_ok = False
        status_ok = False
        backend_pid: int | None = None
        last_status: Any = {}

        deadline = time.time() + TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                code, _ = http_json("/health", timeout=2.0)
                health_ok = (code == 200)
            except Exception:
                health_ok = False

            try:
                scode, payload = http_json("/api/orchestrator/status", timeout=2.0)
                last_status = payload
                if scode == 200 and isinstance(payload, dict):
                    running = bool(payload.get("running", False))
                    backend = payload.get("backend_state") if isinstance(payload.get("backend_state"), dict) else {}
                    candidate = backend.get("pid")
                    if isinstance(candidate, int) and candidate > 0:
                        backend_pid = candidate
                    elif isinstance(candidate, str) and candidate.strip().isdigit():
                        backend_pid = int(candidate.strip())
                    status_ok = running and bool(backend_pid)
                else:
                    status_ok = False
            except Exception:
                status_ok = False

            if health_ok and status_ok:
                break
            time.sleep(0.8)

        if not health_ok:
            _print("[FAIL] /health did not return HTTP 200 within 30s")
            return 3
        if not status_ok:
            _print("[FAIL] /api/orchestrator/status did not report running=true with backend_state.pid")
            _print(f"[INFO] last status payload: {last_status}")
            if isinstance(last_status, dict):
                backend = last_status.get("backend_state")
                if isinstance(backend, dict) and backend.get("port_owner_unknown"):
                    _print("[INFO] backend reports port_owner_unknown=true")
                    _print(f"[INFO] conflicting pid={backend.get('port_owner_pid')}")
            return 4

        _print("[PASS] orchestrator started and backend is healthy")
        _print(f"[PASS] backend pid: {backend_pid}")
        return 0
    except Exception as exc:
        _print(f"[FAIL] unexpected error: {exc}")
        return 5
    finally:
        # Keep successful orchestrator runs alive; clean up failed launches.
        if orch_proc is not None and orch_proc.poll() is None:
            try:
                code, payload = http_json("/api/orchestrator/status", timeout=1.5)
                running = bool(payload.get("running", False)) if code == 200 and isinstance(payload, dict) else False
                backend = payload.get("backend_state") if isinstance(payload, dict) else {}
                has_pid = isinstance(backend, dict) and bool(backend.get("pid"))
                if not (running and has_pid):
                    orch_proc.terminate()
                    orch_proc.wait(timeout=5)
            except Exception:
                try:
                    orch_proc.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
