#!/usr/bin/env python3
"""Tiny smoke test for /health and /api/universe/list."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8420"


def fetch(path: str) -> tuple[int, object]:
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


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    status, payload = fetch("/health")
    if status != 200 or not isinstance(payload, dict) or payload.get("status") != "healthy":
        return fail(f"/health status={status} payload={payload}")
    print("PASS: /health")

    status, payload = fetch("/api/universe/list")
    if status != 200 or not isinstance(payload, dict) or "universes" not in payload:
        return fail(f"/api/universe/list status={status} payload={payload}")
    print(f"PASS: /api/universe/list total={payload.get('total')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
