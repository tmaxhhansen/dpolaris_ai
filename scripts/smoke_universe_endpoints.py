#!/usr/bin/env python3
"""Minimal smoke checks for universe endpoints."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8420")


def get_json(path: str) -> tuple[int, object]:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = body
        return int(exc.code), payload


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    status, payload = get_json("/health")
    if status != 200 or not isinstance(payload, dict) or payload.get("status") != "healthy":
        return fail(f"/health unexpected response: status={status}, payload={payload}")
    print("PASS: /health")

    status, payload = get_json("/api/universe/list")
    if status != 200 or not isinstance(payload, dict) or "universes" not in payload:
        return fail(f"/api/universe/list unexpected response: status={status}, payload={payload}")
    print("PASS: /api/universe/list")

    status, payload = get_json("/api/universe/merged")
    if status != 200 or not isinstance(payload, dict) or "tickers" not in payload:
        return fail(f"/api/universe/merged unexpected response: status={status}, payload={payload}")
    print("PASS: /api/universe/merged")

    status, payload = get_json("/api/universe/all")
    if status != 200 or not isinstance(payload, dict):
        return fail(f"/api/universe/all unexpected response: status={status}, payload={payload}")
    print("PASS: /api/universe/all")

    print("PASS: smoke_universe_endpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
