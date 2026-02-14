#!/usr/bin/env python3
"""
Self-healing regression test:
- require orchestrator running
- kill backend pid
- verify backend comes back healthy with new pid

Run with:
  .\\.venv\\Scripts\\python.exe scripts\\selfheal_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

HOST = "127.0.0.1"
PORT = 8420

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = (REPO_ROOT / ".venv" / "Scripts" / "python.exe").resolve()


def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_repo_root() -> None:
    if Path.cwd().resolve() != REPO_ROOT:
        _print(f"[INFO] switching working directory to {REPO_ROOT}")
        os.chdir(REPO_ROOT)


def http_json(path: str, timeout: float = 2.0) -> tuple[int, Any]:
    url = f"http://{HOST}:{PORT}{path}"
    req = urlrequest.Request(url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8", errors="replace")
            return status, (json.loads(raw) if raw.strip() else {})
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(body) if body.strip() else {}
        except Exception:
            parsed = {"raw": body}
        return int(exc.code), parsed


def start_orchestrator() -> subprocess.Popen[str]:
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
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=flags,
    )


def wait_for_orchestrator(timeout_seconds: int) -> tuple[bool, dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    last: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            hcode, _ = http_json("/health", timeout=2.0)
            scode, payload = http_json("/api/orchestrator/status", timeout=2.0)
            if scode == 200 and isinstance(payload, dict):
                last = payload
                running = bool(payload.get("running", False))
                backend = payload.get("backend_state") if isinstance(payload.get("backend_state"), dict) else {}
                pid = backend.get("pid")
                pid_ok = (isinstance(pid, int) and pid > 0) or (isinstance(pid, str) and pid.strip().isdigit())
                if running and pid_ok and hcode == 200:
                    return True, payload
        except Exception:
            pass
        time.sleep(0.8)
    return False, last


def kill_pid(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)


def parse_pid(payload: dict[str, Any]) -> int | None:
    backend = payload.get("backend_state") if isinstance(payload.get("backend_state"), dict) else {}
    pid = backend.get("pid")
    if isinstance(pid, int):
        return pid if pid > 0 else None
    if isinstance(pid, str) and pid.strip().isdigit():
        return int(pid.strip())
    return None


def parse_restart_count(payload: dict[str, Any]) -> int:
    val = payload.get("restart_count_24h")
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.strip().isdigit():
        return int(val.strip())
    backend = payload.get("backend_state") if isinstance(payload.get("backend_state"), dict) else {}
    bval = backend.get("restart_count_24h")
    if isinstance(bval, int):
        return bval
    if isinstance(bval, str) and bval.strip().isdigit():
        return int(bval.strip())
    return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=60, help="seconds to wait for self-heal")
    args = parser.parse_args()

    ensure_repo_root()
    if not VENV_PY.exists():
        _print(f"[FAIL] missing venv python: {VENV_PY}")
        return 2

    _print("=== selfheal_test.py ===")

    started_proc: subprocess.Popen[str] | None = None
    try:
        ok, status = wait_for_orchestrator(timeout_seconds=8)
        if not ok:
            _print("[INFO] orchestrator not ready, starting it...")
            started_proc = start_orchestrator()
            _print(f"[INFO] orchestrator launch pid={started_proc.pid}")
            ok, status = wait_for_orchestrator(timeout_seconds=30)
            if not ok:
                _print("[FAIL] orchestrator failed to become ready")
                _print(f"[INFO] last status: {status}")
                if isinstance(status, dict):
                    backend = status.get("backend_state")
                    if isinstance(backend, dict) and backend.get("port_owner_unknown"):
                        _print("[INFO] backend reports port_owner_unknown=true")
                        _print(f"[INFO] conflicting pid={backend.get('port_owner_pid')}")
                return 3

        old_pid = parse_pid(status)
        if not old_pid:
            _print("[FAIL] backend pid missing from /api/orchestrator/status")
            _print(f"[INFO] status payload: {status}")
            return 4

        old_restart_count = parse_restart_count(status)
        _print(f"[STEP] old backend pid={old_pid}, restart_count_24h={old_restart_count}")

        t0 = time.time()
        kill_pid(old_pid)
        _print("[STEP] sent backend kill signal")

        healed = False
        new_pid: int | None = None
        new_restart_count = old_restart_count
        deadline = time.time() + max(5, args.timeout)
        while time.time() < deadline:
            try:
                hcode, _ = http_json("/health", timeout=2.0)
                scode, payload = http_json("/api/orchestrator/status", timeout=2.0)
                if scode == 200 and isinstance(payload, dict):
                    candidate_pid = parse_pid(payload)
                    candidate_restart = parse_restart_count(payload)
                    running = bool(payload.get("running", False))
                    pid_changed = bool(candidate_pid and candidate_pid != old_pid)
                    restart_bumped = candidate_restart > old_restart_count >= 0
                    if running and hcode == 200 and (pid_changed or restart_bumped):
                        healed = True
                        new_pid = candidate_pid
                        new_restart_count = candidate_restart
                        break
            except Exception:
                pass
            time.sleep(0.8)

        elapsed = time.time() - t0
        if not healed:
            _print(f"[FAIL] backend did not self-heal within {args.timeout}s")
            return 5

        _print(f"[PASS] self-heal complete in {elapsed:.1f}s")
        _print(f"[PASS] old pid={old_pid}, new pid={new_pid}, restart_count_24h={new_restart_count}")
        return 0
    finally:
        if started_proc is not None and started_proc.poll() is None:
            try:
                code, payload = http_json("/api/orchestrator/status", timeout=1.5)
                running = bool(payload.get("running", False)) if code == 200 and isinstance(payload, dict) else False
                if not running:
                    started_proc.terminate()
            except Exception:
                try:
                    started_proc.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
