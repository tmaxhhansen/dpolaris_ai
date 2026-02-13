from __future__ import annotations

import asyncio
from pathlib import Path

from api import server


def _completed_payload(run_id: str, ticker: str) -> dict:
    return {
        "run_id": run_id,
        "ticker": ticker,
        "status": "completed",
        "generated_at": server.utc_now_iso(),
        "market_context": {"regime_label": "bullish_low_positive"},
        "price_volume_pattern_analysis": {},
        "options_decision_support": {"enabled": False, "warnings": [], "candidates": []},
        "risk_summary": {"position_sizing_guidance": "standard", "hard_warnings": []},
        "traceability": {},
        "signal": {"bias": "LONG", "confidence": 0.71},
        "prediction": {"model_type": "lstm"},
        "summary": {
            "ticker": ticker,
            "status": "completed",
            "primary_score": 0.71,
            "bias": "LONG",
            "confidence": 0.71,
            "model_type": "lstm",
            "target_horizon": 5,
            "dataset_range": {"start": "2024-01-01T00:00:00", "end": "2025-01-01T00:00:00", "bars": 300},
            "regime_label": "bullish_low_positive",
            "updated_at": server.utc_now_iso(),
        },
    }


def test_scan_job_persists_results_and_continues_on_failure(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("DPOLARIS_RUNS_DIR", str(runs_root))

    universe_payload = {
        "schema_version": "1.0.0",
        "universe_hash": "hash123",
        "merged": [{"symbol": "AAA"}, {"symbol": "BBB"}],
    }
    universe_rows = [{"symbol": "AAA"}, {"symbol": "BBB"}]
    monkeypatch.setattr(
        server,
        "_load_scan_universe",
        lambda _: (universe_payload, universe_rows, Path("/tmp/combined_1000.json")),
    )

    async def _fake_build(**kwargs):
        symbol = kwargs["symbol"]
        run_id = kwargs["run_id"]
        if symbol == "BBB":
            raise RuntimeError("ticker failed intentionally")
        return _completed_payload(run_id, symbol)

    monkeypatch.setattr(server, "_build_scan_ticker_payload", _fake_build)
    monkeypatch.setattr(server, "scan_jobs", {})
    monkeypatch.setattr(server, "scan_job_order", [])
    monkeypatch.setattr(server, "scan_job_queue", asyncio.Queue())

    req = server.ScanStartRequest(universe="combined_1000", options_mode=False)
    started = asyncio.run(server.start_scan(req))
    run_id = started["runId"]

    asyncio.run(server._run_scan_job(run_id))
    status = asyncio.run(server.get_scan_status(run_id))

    assert status["status"] == "completed"
    assert status["completedTickers"] == 1
    assert status["failedTickers"] == 1
    assert status["totalTickers"] == 2

    ok_file = runs_root / run_id / "scan_results" / "AAA.json"
    failed_file = runs_root / run_id / "scan_results" / "BBB.json"
    assert ok_file.exists()
    assert failed_file.exists()

    failed_detail = asyncio.run(server.get_scan_result_detail(run_id, "BBB"))
    assert failed_detail["status"] == "failed"
    assert "ticker failed intentionally" in failed_detail["error"]

    listed = asyncio.run(server.get_scan_results(run_id, page=1, page_size=50, status=None))
    assert listed["total"] == 2
    assert {row["ticker"] for row in listed["items"]} == {"AAA", "BBB"}

