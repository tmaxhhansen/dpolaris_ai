#!/usr/bin/env python3
"""
Port ownership guard test:
- run dummy server on 127.0.0.1:8420
- start orchestrator
- verify dummy owner is NOT killed by orchestrator

Run with:
  .\\.venv\\Scripts\\python.exe scripts\\port_owner_guard_test.py
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8420

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = (REPO_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
RUNTIME_DIR = REPO_ROOT / ".runtime"
ORCH_LOG = RUNTIME_DIR / "port_owner_guard_orchestrator.log"


def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_repo_root() -> None:
    if Path.cwd().resolve() != REPO_ROOT:
        _print(f"[INFO] switching working directory to {REPO_ROOT}")
        os.chdir(REPO_ROOT)


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


def start_dummy_server() -> subprocess.Popen[str]:
    args = [str(VENV_PY), "-m", "http.server", str(PORT), "--bind", HOST]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=flags,
    )


def start_orchestrator() -> subprocess.Popen[str]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(ORCH_LOG, "w", encoding="utf-8")
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
    proc = subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=flags,
    )
    proc._guard_log_handle = logf  # type: ignore[attr-defined]
    return proc


def terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        try:
            handle = getattr(proc, "_guard_log_handle", None)
            if handle:
                handle.close()
        except Exception:
            pass
        return
    proc.terminate()
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=4)
    try:
        handle = getattr(proc, "_guard_log_handle", None)
        if handle:
            handle.close()
    except Exception:
        pass


def main() -> int:
    ensure_repo_root()
    if not VENV_PY.exists():
        _print(f"[FAIL] missing venv python: {VENV_PY}")
        return 2

    existing_owner = get_port_owner_pid(PORT)
    if existing_owner:
        _print(f"[FAIL] precondition failed: port {PORT} already in use by pid {existing_owner}")
        _print("Stop existing owner and rerun this test.")
        return 3

    _print("=== port_owner_guard_test.py ===")

    dummy: subprocess.Popen[str] | None = None
    orch: subprocess.Popen[str] | None = None
    try:
        dummy = start_dummy_server()
        _print(f"[STEP] started dummy server pid={dummy.pid}")

        deadline = time.time() + 8
        owner = None
        while time.time() < deadline:
            owner = get_port_owner_pid(PORT)
            if owner == dummy.pid:
                break
            time.sleep(0.3)
        if owner != dummy.pid:
            _print(f"[FAIL] dummy server failed to own port {PORT}; owner={owner}")
            return 4

        orch = start_orchestrator()
        _print(f"[STEP] started orchestrator pid={orch.pid}")

        time.sleep(12)

        owner_after = get_port_owner_pid(PORT)
        dummy_alive = dummy.poll() is None
        if not dummy_alive:
            _print("[FAIL] dummy process exited unexpectedly")
            return 5
        if owner_after != dummy.pid:
            _print(f"[FAIL] dummy no longer owns port {PORT}; owner={owner_after}, dummy={dummy.pid}")
            return 6

        log_text = ""
        if ORCH_LOG.exists():
            log_text = ORCH_LOG.read_text(encoding="utf-8", errors="replace")
        log_signal = (
            "already in use" in log_text.lower()
            or "port 8420" in log_text.lower()
            or "portinusebyunknownprocesserror" in log_text.lower()
        )
        if not log_signal:
            _print("[FAIL] orchestrator log did not show clear port-ownership warning")
            if log_text:
                tail = "\n".join(log_text.splitlines()[-20:])
                _print("[INFO] log tail:\n" + tail)
            return 7

        _print("[PASS] orchestrator respected unknown port owner and did not kill dummy server")
        return 0
    finally:
        terminate_process(orch)
        terminate_process(dummy)


if __name__ == "__main__":
    raise SystemExit(main())
