"""
Manual smoke script for orchestrator self-healing behavior.

Usage:
  python scripts/orchestrator_smoke.py --host 127.0.0.1 --port 8420 --wait-seconds 45
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib import request as urlrequest


def _health(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urlrequest.Request(url, method="GET")
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        os.kill(pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--wait-seconds", type=int, default=45)
    args = parser.parse_args()

    health_url = f"http://{args.host}:{args.port}/health"
    pid_file = Path("~/dpolaris_data/run/backend.pid").expanduser()

    print(f"[1/4] waiting for healthy backend at {health_url}")
    deadline = time.time() + 60
    while time.time() < deadline:
        if _health(health_url):
            break
        time.sleep(1)
    if not _health(health_url):
        print("Backend never became healthy. Start orchestrator first.")
        return 1

    if not pid_file.exists():
        print(f"Missing pid file: {pid_file}")
        return 1
    old_pid = int(pid_file.read_text(encoding="utf-8").strip())
    print(f"[2/4] killing managed backend pid={old_pid}")
    _kill_pid(old_pid)

    print("[3/4] waiting for orchestrator to restart backend")
    deadline = time.time() + max(5, args.wait_seconds)
    new_pid = old_pid
    while time.time() < deadline:
        if pid_file.exists():
            try:
                new_pid = int(pid_file.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        if new_pid != old_pid and _health(health_url):
            break
        time.sleep(1)

    print("[4/4] result")
    if new_pid != old_pid and _health(health_url):
        print(f"PASS: backend restarted from {old_pid} -> {new_pid} and is healthy")
        return 0

    print("FAIL: backend did not restart to a healthy state in time")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

