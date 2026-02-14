#!/usr/bin/env python3
"""Start local server, then smoke-test /health and /api/universe/list."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HOST = "127.0.0.1"
PORT = int(os.getenv("DPOLARIS_SMOKE_PORT", "8450"))
BASE_URL = f"http://{HOST}:{PORT}"


def get_json(path: str) -> tuple[int, object]:
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw
        return int(exc.code), payload
    except Exception as exc:
        return 0, {"error": str(exc)}


def wait_until_ready(timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status, payload = get_json("/health")
        if status == 200 and isinstance(payload, dict) and payload.get("status") == "healthy":
            return True
        time.sleep(0.8)
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = repo_root / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        print(f"FAIL: missing venv python at {python_exe}")
        return 1

    proc = subprocess.Popen(
        [str(python_exe), "-m", "cli.main", "server", "--host", HOST, "--port", str(PORT)],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        if not wait_until_ready():
            print("FAIL: server did not become healthy in time")
            return 1
        print("PASS: /health")

        status, payload = get_json("/api/universe/list")
        if status != 200 or not isinstance(payload, dict) or "universes" not in payload:
            print(f"FAIL: /api/universe/list status={status} payload={payload}")
            return 1
        if not isinstance(payload.get("universes"), list):
            print(f"FAIL: /api/universe/list universes is not a list: payload={payload}")
            return 1
        print("PASS: /api/universe/list")
        print(json.dumps(payload))
        print("PASS: smoke_universe_start_and_list")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
