from __future__ import annotations

import pandas as pd

from data.schema import canonicalize_price_frame


def _sample_raw_frame(include_split: bool = False) -> pd.DataFrame:
    data = {
        "date": [
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ],
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1_000_000, 1_100_000, 1_200_000],
    }
    if include_split:
        data["split_factor"] = [0.0, 2.0, None]
    return pd.DataFrame(data)


def test_canonicalize_without_split_column_defaults_to_one() -> None:
    raw = _sample_raw_frame(include_split=False)
    out = canonicalize_price_frame(raw)

    assert "split_factor" in out.columns
    assert out["split_factor"].tolist() == [1.0, 1.0, 1.0]


def test_canonicalize_normalizes_zero_or_missing_split_factor() -> None:
    raw = _sample_raw_frame(include_split=True)
    out = canonicalize_price_frame(raw)

    # 0 and missing values are treated as "no split" => 1.0.
    assert out["split_factor"].tolist() == [1.0, 2.0, 1.0]
