#!/usr/bin/env python3
"""Tiny local smoke for health + universe list endpoints."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8420"


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


def main() -> int:
    status, payload = get_json("/health")
    if status != 200 or not isinstance(payload, dict) or payload.get("status") != "healthy":
        print(f"FAIL /health status={status} payload={payload}")
        return 1
    print("PASS /health")

    status, payload = get_json("/api/universe/list")
    if status != 200 or not isinstance(payload, dict) or "universes" not in payload:
        print(f"FAIL /api/universe/list status={status} payload={payload}")
        return 1
    print(f"PASS /api/universe/list payload={json.dumps(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
