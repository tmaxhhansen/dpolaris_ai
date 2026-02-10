from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=rows, freq="B")

    drift = np.linspace(0.0001, 0.0005, rows)
    shock = rng.normal(0.0, 0.01, rows)
    returns = drift + shock

    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, rows))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, rows)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, rows)))
    volume = rng.integers(2_000_000, 20_000_000, rows)

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


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_ohlcv()
