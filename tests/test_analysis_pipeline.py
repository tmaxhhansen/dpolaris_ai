from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pipeline import compute_indicator_snapshot, detect_chart_patterns


def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    close = close.astype(float)
    rows = len(close)
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    open_ = close * 0.998
    high = close * 1.01
    low = close * 0.99
    volume = np.full(rows, 5_000_000.0)
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


def test_indicator_snapshot_contains_required_sections():
    rng = np.random.default_rng(123)
    close = 100.0 + np.cumsum(rng.normal(0.15, 0.8, 260))
    df = _make_ohlcv(close)

    indicators = compute_indicator_snapshot(df)

    for key in ("sma", "ema", "rsi", "macd", "atr", "bollinger", "volume_trend", "support_resistance"):
        assert key in indicators

    rsi_value = float(indicators["rsi"]["value"])
    assert 0.0 <= rsi_value <= 100.0


def test_detect_chart_patterns_finds_uptrend_channel():
    close = np.linspace(80.0, 140.0, 220)
    df = _make_ohlcv(close)

    patterns = detect_chart_patterns(df)
    trend = next(p for p in patterns if p["name"] == "trend_channel")

    assert trend["signal"] == "uptrend"
    assert float(trend["confidence"]) >= 0.5


def test_detect_chart_patterns_finds_double_bottom():
    close = np.linspace(100.0, 120.0, 220)
    close[110] = 89.5
    close[111] = 90.0
    close[112] = 91.0
    close[150] = 89.2
    close[151] = 89.8
    close[152] = 90.8
    close[-1] = 125.0
    df = _make_ohlcv(close)

    patterns = detect_chart_patterns(df)
    names = {p["name"] for p in patterns}
    assert "double_bottom" in names
