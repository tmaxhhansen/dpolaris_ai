from __future__ import annotations

import asyncio
from pathlib import Path

from api import server


def test_scan_universe_endpoint_generates_fallback_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_repo_root", lambda: tmp_path)
    monkeypatch.setenv("DPOLARIS_FALLBACK_SYMBOLS", "AAPL,MSFT,NVDA")

    response = asyncio.run(server.get_scan_universe("nasdaq_top_500"))
    assert response["name"] == "nasdaq_top_500"
    assert response["count"] >= 3
    assert response["schema_version"] == "1.0.0"
    assert response["universe"]["universe_hash"]

    generated_path = Path(response["path"])
    assert generated_path.exists()
    assert generated_path.name == "nasdaq_top_500.json"


def test_scan_runs_endpoint_lists_runs_from_disk_state(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(server, "_runs_root", lambda: runs_root)
    monkeypatch.setattr(server, "scan_jobs", {})
    monkeypatch.setattr(server, "scan_job_order", [])

    run_id = "scan_demo_1234"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    server._json_dump(
        run_dir / "run_summary.json",
        {
            "run_id": run_id,
            "status": "completed",
            "created_at": "2026-02-10T10:00:00",
            "updated_at": "2026-02-10T10:05:00",
            "started_at": "2026-02-10T10:00:05",
            "completed_at": "2026-02-10T10:04:55",
            "universe": "combined_1000",
            "universe_hash": "abc123",
            "run_mode": "scan",
            "horizon_days": 5,
            "options_mode": True,
            "concurrency": 8,
            "total_tickers": 2,
            "progress_percent": 100.0,
        },
    )
    server._json_dump(
        run_dir / server.SCAN_INDEX_FILE,
        {
            "run_id": run_id,
            "count": 2,
            "items": [
                {"ticker": "AAPL", "status": "completed", "primary_score": 0.71},
                {"ticker": "MSFT", "status": "completed", "primary_score": 0.69},
            ],
        },
    )

    response = asyncio.run(server.list_scan_runs_endpoint(limit=50, status=None))
    assert response["count"] == 1
    item = response["runs"][0]
    assert item["runId"] == run_id
    assert item["status"] == "completed"
    assert item["universe_hash"] == "abc123"
    assert item["config_summary"].startswith("h=5d")
    assert float(item["primary_score"]) >= 0.69

