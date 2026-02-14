#!/usr/bin/env python3
"""
Windows-friendly smoke test for dPolaris API.

What this script does:
1) Starts server with: python -m cli.main server --host 127.0.0.1 --port 8420
2) Waits for /health to return HTTP 200
3) Validates required endpoints and response contracts
4) Validates LLM disabled behavior for /api/chat
5) Verifies training endpoints return JSON even on failure
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
class Response:
    method: str
    path: str
    status: int
    content_type: str
    body_text: str
    json_data: Any


class SmokeTestFailure(Exception):
    pass


def _is_json_content_type(content_type: str) -> bool:
    return "application/json" in (content_type or "").lower()


def _request(base_url: str, method: str, path: str, body: dict[str, Any] | None = None, timeout: int = 20) -> Response:
    url = f"{base_url}{path}"
    payload = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=payload, headers=headers, method=method)

    raw = b""
    status = 0
    content_type = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = int(resp.getcode())
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp is not None else b""
        status = int(exc.code)
        content_type = exc.headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        raise SmokeTestFailure(f"{method} {path} connection error: {exc}") from exc

    body_text = raw.decode("utf-8", errors="replace")
    parsed: Any = None
    if body_text.strip():
        if _is_json_content_type(content_type):
            try:
                parsed = json.loads(body_text)
            except json.JSONDecodeError:
                parsed = None
        else:
            try:
                parsed = json.loads(body_text)
            except json.JSONDecodeError:
                parsed = None

    return Response(
        method=method,
        path=path,
        status=status,
        content_type=content_type,
        body_text=body_text,
        json_data=parsed,
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def _assert_json(resp: Response, label: str) -> None:
    _assert(resp.json_data is not None, f"{label}: expected JSON response, got content-type='{resp.content_type}'")


def _assert_keys(obj: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [k for k in keys if k not in obj]
    _assert(not missing, f"{label}: missing keys {missing}")


def _tail_file(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _wait_for_health(
    base_url: str,
    timeout_seconds: int,
    proc: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
) -> Response:
    deadline = time.time() + timeout_seconds
    last_error = "no response yet"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            extra = ""
            if log_path is not None:
                tail = _tail_file(log_path).strip()
                if tail:
                    extra = f"\nServer output tail:\n{tail}"
            raise SmokeTestFailure(
                f"Server process exited before /health was ready (exit code {proc.returncode}).{extra}"
            )
        try:
            resp = _request(base_url, "GET", "/health", timeout=5)
            if resp.status == 200 and isinstance(resp.json_data, dict):
                return resp
            last_error = f"HTTP {resp.status}"
        except SmokeTestFailure as exc:
            last_error = str(exc)
        time.sleep(1)
    raise SmokeTestFailure(f"/health did not become ready in {timeout_seconds}s (last error: {last_error})")


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run() -> int:
    parser = argparse.ArgumentParser(description="dPolaris Windows smoke test")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8420, help="Server port")
    parser.add_argument("--startup-timeout", type=int, default=90, help="Seconds to wait for /health")
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Do not launch server; assume it is already running",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_url = f"http://{args.host}:{args.port}"
    python_exe = sys.executable
    server_cmd = [python_exe, "-m", "cli.main", "server", "--host", args.host, "--port", str(args.port)]

    print("== dPolaris Smoke Test ==")
    print(f"Base URL: {base_url}")
    print("Server start command:")
    print(f"  LLM_PROVIDER=none {' '.join(server_cmd)}")

    proc: subprocess.Popen[str] | None = None
    log_handle = None
    log_path = repo_root / "scripts" / "smoke_server.log"
    try:
        if not args.no_start:
            env = os.environ.copy()
            env["LLM_PROVIDER"] = "none"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                server_cmd,
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        health_resp = _wait_for_health(
            base_url,
            args.startup_timeout,
            proc=proc,
            log_path=log_path if proc is not None else None,
        )
        _assert_json(health_resp, "GET /health")
        _assert(isinstance(health_resp.json_data, dict), "GET /health: expected JSON object")
        _assert_keys(health_resp.json_data, ["status", "timestamp"], "GET /health")
        print("[PASS] GET /health")

        status_resp = _request(base_url, "GET", "/api/status")
        _assert(status_resp.status == 200, f"GET /api/status: expected 200, got {status_resp.status}")
        _assert_json(status_resp, "GET /api/status")
        _assert(isinstance(status_resp.json_data, dict), "GET /api/status: expected JSON object")
        _assert_keys(
            status_resp.json_data,
            [
                "daemon_running",
                "total_memories",
                "total_trades",
                "models_available",
                "llm_provider",
                "llm_enabled",
            ],
            "GET /api/status",
        )
        print("[PASS] GET /api/status")

        models_resp = _request(base_url, "GET", "/api/models")
        _assert(models_resp.status == 200, f"GET /api/models: expected 200, got {models_resp.status}")
        _assert_json(models_resp, "GET /api/models")
        if isinstance(models_resp.json_data, dict):
            _assert_keys(models_resp.json_data, ["models"], "GET /api/models")
        else:
            _assert(isinstance(models_resp.json_data, list), "GET /api/models: expected JSON list or object")
        print("[PASS] GET /api/models")

        memories_resp = _request(base_url, "GET", "/api/memories?limit=5")
        _assert(memories_resp.status == 200, f"GET /api/memories?limit=5: expected 200, got {memories_resp.status}")
        _assert_json(memories_resp, "GET /api/memories?limit=5")
        _assert(isinstance(memories_resp.json_data, list), "GET /api/memories?limit=5: expected JSON array")
        if memories_resp.json_data:
            first = memories_resp.json_data[0]
            _assert(isinstance(first, dict), "GET /api/memories?limit=5: expected memory objects")
            _assert_keys(first, ["category", "content"], "GET /api/memories?limit=5 first item")
        print("[PASS] GET /api/memories?limit=5")

        chat_resp = _request(base_url, "POST", "/api/chat", body={"message": "smoke test"})
        _assert(
            chat_resp.status in (400, 503),
            f"POST /api/chat: expected 400 or 503 with LLM_PROVIDER=none, got {chat_resp.status}",
        )
        _assert_json(chat_resp, "POST /api/chat")
        _assert(isinstance(chat_resp.json_data, dict), "POST /api/chat: expected JSON object")
        _assert_keys(chat_resp.json_data, ["detail"], "POST /api/chat")
        print("[PASS] POST /api/chat (LLM disabled behavior)")

        train_resp = _request(base_url, "POST", "/api/train/AAPL")
        _assert_json(train_resp, "POST /api/train/AAPL")
        _assert(isinstance(train_resp.json_data, dict), "POST /api/train/AAPL: expected JSON object")
        _assert_keys(train_resp.json_data, ["symbol", "result"], "POST /api/train/AAPL")
        print(f"[PASS] POST /api/train/AAPL (HTTP {train_resp.status})")

        dl_job_resp = _request(
            base_url,
            "POST",
            "/api/jobs/deep-learning/train",
            body={"symbol": "AAPL", "model_type": "lstm", "epochs": 1},
        )
        _assert_json(dl_job_resp, "POST /api/jobs/deep-learning/train")
        _assert(isinstance(dl_job_resp.json_data, dict), "POST /api/jobs/deep-learning/train: expected JSON object")
        job_id: str | None = None
        if dl_job_resp.status >= 400:
            _assert_keys(dl_job_resp.json_data, ["detail"], "POST /api/jobs/deep-learning/train error")
        else:
            _assert_keys(dl_job_resp.json_data, ["id", "status"], "POST /api/jobs/deep-learning/train success")
            raw_id = dl_job_resp.json_data.get("id")
            if isinstance(raw_id, str) and raw_id.strip():
                job_id = raw_id.strip()
        print(f"[PASS] POST /api/jobs/deep-learning/train (HTTP {dl_job_resp.status})")

        job_path = f"/api/jobs/{job_id}" if job_id else "/api/jobs/smoke-test-missing-id"
        job_resp = _request(base_url, "GET", job_path)
        _assert_json(job_resp, f"GET {job_path}")
        _assert(isinstance(job_resp.json_data, dict), f"GET {job_path}: expected JSON object")
        if job_resp.status >= 400:
            _assert_keys(job_resp.json_data, ["detail"], f"GET {job_path} error")
        else:
            _assert_keys(job_resp.json_data, ["id", "status"], f"GET {job_path} success")
        print(f"[PASS] GET {job_path} (HTTP {job_resp.status})")

        print("\nSmoke test passed.")
        return 0
    except SmokeTestFailure as exc:
        print(f"\n[FAIL] {exc}")
        return 1
    finally:
        if log_handle is not None:
            log_handle.close()
        if proc is not None:
            _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(_run())
