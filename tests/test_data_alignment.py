from __future__ import annotations

import pandas as pd

from data.alignment import assert_no_peek, causal_asof_join, resample_ohlcv


def _base_price_df() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-03 14:30:00+00:00", periods=6, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [1000, 1200, 900, 1100, 1300, 1400],
            "adj_close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "dividend": [0.0] * 6,
            "split_factor": [1.0] * 6,
            "session": ["regular"] * 6,
            "is_market_holiday": [False] * 6,
            "is_synthetic": [False] * 6,
        }
    )


def test_causal_asof_join_uses_latest_available_without_peek():
    price_df = _base_price_df()
    macro_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-03 14:20:00+00:00"),
                pd.Timestamp("2024-01-03 14:45:00+00:00"),
            ],
            "cpi_surprise": [0.1, -0.2],
        }
    )

    aligned = causal_asof_join(price_df, macro_df, right_prefix="macro")
    assert assert_no_peek(aligned)

    pre_switch = aligned[aligned["timestamp"] < pd.Timestamp("2024-01-03 14:45:00+00:00")]
    post_switch = aligned[aligned["timestamp"] >= pd.Timestamp("2024-01-03 14:45:00+00:00")]

    assert (pre_switch["macro__cpi_surprise"] == 0.1).all()
    assert (post_switch["macro__cpi_surprise"] == -0.2).all()


def test_resample_ohlcv_to_15m():
    price_df = _base_price_df()
    out = resample_ohlcv(price_df, timeframe="15m")

    assert len(out) >= 2
    first = out.iloc[0]

    assert first["open"] == 100
    assert first["high"] == 103
    assert first["low"] == 99
    assert first["volume"] == 1000 + 1200 + 900
