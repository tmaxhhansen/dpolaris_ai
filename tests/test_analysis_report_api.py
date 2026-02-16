from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd


class _FakeMarket:
    async def get_historical(self, symbol: str, days: int = 365, use_cache: bool = True):
        _ = symbol, days, use_cache
        rng = np.random.default_rng(77)
        rows = 420
        dates = pd.date_range("2020-01-01", periods=rows, freq="B")
        close = 120.0 + np.cumsum(rng.normal(0.05, 1.2, rows))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, rows))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, rows)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, rows)))
        volume = rng.integers(1_000_000, 10_000_000, rows)
        return pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )


def test_analyze_report_endpoint_returns_required_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("DPOLARIS_ANALYSIS_DIR", str(tmp_path / "analysis"))

    from api import server

    monkeypatch.setattr(server, "market_service", _FakeMarket())

    payload = asyncio.run(server.generate_analysis_report_endpoint("AAPL"))

    assert payload["ticker"] == "AAPL"
    report_text = str(payload.get("report_text") or "")
    for heading in (
        "## Overview",
        "## Price/Volume Snapshot",
        "## Technical Indicators",
        "## Chart Patterns",
        "## Model Signals",
        "## News",
        "## Risk Notes",
        "## Next Steps",
    ):
        assert heading in report_text

    rows = asyncio.run(server.list_analysis_endpoint(limit=5))
    assert isinstance(rows, list)
    assert len(rows) >= 1

    detail = asyncio.run(server.get_analysis_artifact_endpoint(rows[0]["id"]))
    assert detail.get("report_text")
