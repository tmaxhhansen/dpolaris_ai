#!/usr/bin/env python3
"""
Smoke test for Java Control Center backend integration.

Checks:
1) Backend is up (starts it if needed)
2) GET /health
3) GET /api/status
4) GET /api/control/backend/status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HttpResult:
    status: int
    body_text: str
    data: Any


class SmokeFailure(RuntimeError):
    pass


def _request_json(url: str, *, timeout: int = 8) -> HttpResult:
    req = urllib.request.Request(url=url, method="GET", headers={"Accept": "application/json"})
    body = b""
    status = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp is not None else b""
        status = int(exc.code)
    except urllib.error.URLError as exc:
        raise SmokeFailure(f"Connection failed for {url}: {exc}") from exc

    text = body.decode("utf-8", errors="replace")
    parsed: Any = None
    if text.strip():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    return HttpResult(status=status, body_text=text, data=parsed)


def _wait_for_health(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + max(5, timeout_seconds)
    last_error = "no response"
    while time.time() < deadline:
        try:
            result = _request_json(f"{base_url}/health", timeout=4)
            if result.status == 200 and isinstance(result.data, dict) and result.data.get("status") == "healthy":
                return
            last_error = f"HTTP {result.status}"
        except SmokeFailure as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise SmokeFailure(f"/health did not become ready in {timeout_seconds}s (last error: {last_error})")


def _start_backend(repo_root: Path, host: str, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["LLM_PROVIDER"] = "none"
    env["DPOLARIS_BACKEND_HOST"] = host
    env["DPOLARIS_BACKEND_PORT"] = str(int(port))

    venv_python = repo_root / ".venv" / "bin" / "python"
    python_exe = venv_python if venv_python.exists() else Path(sys.executable).resolve()

    cmd = [
        str(python_exe),
        "-m",
        "cli.main",
        "server",
        "--host",
        host,
        "--port",
        str(int(port)),
    ]
    log_path = repo_root / "scripts" / "smoke_control_center.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")

    kwargs: dict[str, Any] = {
        "cwd": str(repo_root),
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

    print(f"[INFO] Using python executable for backend start: {python_exe}")

    try:
        proc = subprocess.Popen(cmd, **kwargs)
    finally:
        log_handle.close()
    return proc


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for backend control-center endpoints")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--startup-timeout", type=int, default=45)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_url = f"http://{args.host}:{args.port}"

    started_proc: subprocess.Popen[str] | None = None
    try:
        try:
            _wait_for_health(base_url, timeout_seconds=3)
            print("[INFO] Backend already healthy")
        except SmokeFailure:
            print("[INFO] Backend not healthy; starting backend")
            started_proc = _start_backend(repo_root, host=args.host, port=args.port)
            _wait_for_health(base_url, timeout_seconds=args.startup_timeout)
            print(f"[INFO] Started backend pid={started_proc.pid}")

        health = _request_json(f"{base_url}/health")
        _assert(health.status == 200, f"GET /health expected 200, got {health.status}")
        _assert(isinstance(health.data, dict), "GET /health expected JSON object")
        _assert("status" in health.data and "timestamp" in health.data, "GET /health missing required keys")
        print("[PASS] GET /health")

        api_status = _request_json(f"{base_url}/api/status")
        _assert(api_status.status == 200, f"GET /api/status expected 200, got {api_status.status}")
        _assert(isinstance(api_status.data, dict), "GET /api/status expected JSON object")
        print("[PASS] GET /api/status")

        backend_status = _request_json(f"{base_url}/api/control/backend/status")
        _assert(
            backend_status.status == 200,
            f"GET /api/control/backend/status expected 200, got {backend_status.status}",
        )
        _assert(isinstance(backend_status.data, dict), "GET /api/control/backend/status expected JSON object")
        for key in ("status", "pid", "running", "python_executable", "uptime"):
            _assert(key in backend_status.data, f"GET /api/control/backend/status missing key: {key}")
        print("[PASS] GET /api/control/backend/status")

        print("[PASS] Control Center smoke succeeded")
        return 0
    except SmokeFailure as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        if started_proc is not None and started_proc.poll() is not None:
            print(f"[WARN] Started backend exited early with code {started_proc.returncode}")


if __name__ == "__main__":
    raise SystemExit(main())
