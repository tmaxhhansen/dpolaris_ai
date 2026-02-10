from __future__ import annotations

import numpy as np
import pandas as pd

from features import (
    FeatureSpec,
    TechnicalFeatureLibrary,
    build_default_registry,
    generate_fundamentals_features,
)


def _price_intraday() -> pd.DataFrame:
    ts = pd.date_range("2024-03-01 13:00:00+00:00", periods=8, freq="1h")
    close = np.array([100.0, 100.2, 100.1, 100.4, 100.8, 101.1, 101.0, 101.3], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.array([1000, 900, 1100, 1200, 950, 980, 1020, 1080], dtype=float),
        }
    )


def test_fundamentals_dont_appear_before_release():
    price = _price_intraday()
    fundamentals = pd.DataFrame(
        {
            "filing_timestamp": [
                pd.Timestamp("2024-03-01 15:00:00+00:00"),
                pd.Timestamp("2024-03-01 17:00:00+00:00"),
            ],
            "revenue": [100.0, 120.0],
            "free_cash_flow": [10.0, 12.0],
            "total_debt": [60.0, 66.0],
            "total_equity": [40.0, 44.0],
            "ebit": [20.0, 22.0],
            "interest_expense": [5.0, 5.5],
        }
    )

    out = generate_fundamentals_features(price, fundamentals_df=fundamentals)

    pre_first = out.loc[price["timestamp"] < pd.Timestamp("2024-03-01 15:00:00+00:00"), "fund_debt_to_equity"]
    between = out.loc[
        (price["timestamp"] >= pd.Timestamp("2024-03-01 15:00:00+00:00"))
        & (price["timestamp"] < pd.Timestamp("2024-03-01 17:00:00+00:00")),
        "fund_debt_to_equity",
    ]
    post_second = out.loc[price["timestamp"] >= pd.Timestamp("2024-03-01 17:00:00+00:00"), "fund_debt_to_equity"]

    assert pre_first.isna().all()
    assert np.allclose(between.values, np.array([1.5] * len(between)))
    assert np.allclose(post_second.values, np.array([1.5] * len(post_second)))

    # QoQ growth should only become available after the second released filing.
    pre_second_growth = out.loc[price["timestamp"] < pd.Timestamp("2024-03-01 17:00:00+00:00"), "fund_revenue_growth_qoq"]
    post_second_growth = out.loc[price["timestamp"] >= pd.Timestamp("2024-03-01 17:00:00+00:00"), "fund_revenue_growth_qoq"]
    assert pre_second_growth.isna().all()
    assert np.allclose(post_second_growth.dropna().values, np.array([0.2] * len(post_second_growth.dropna())))


def test_earnings_event_features_respect_release_and_known_since():
    price = _price_intraday()
    earnings = pd.DataFrame(
        {
            "event_timestamp": [pd.Timestamp("2024-03-01 16:00:00+00:00")],
            "known_since": [pd.Timestamp("2024-03-01 15:00:00+00:00")],
            "earnings_surprise": [0.3],
            "guidance_change_proxy": [-0.1],
        }
    )

    out = generate_fundamentals_features(price, earnings_df=earnings)

    before_known = out.loc[price["timestamp"] < pd.Timestamp("2024-03-01 15:00:00+00:00"), "earnings_days_to_event"]
    known_window = out.loc[
        (price["timestamp"] >= pd.Timestamp("2024-03-01 15:00:00+00:00"))
        & (price["timestamp"] < pd.Timestamp("2024-03-01 16:00:00+00:00")),
        "earnings_days_to_event",
    ]
    assert before_known.isna().all()
    assert (known_window > 0).all()

    before_release_surprise = out.loc[price["timestamp"] < pd.Timestamp("2024-03-01 16:00:00+00:00"), "earnings_surprise"]
    after_release_surprise = out.loc[price["timestamp"] >= pd.Timestamp("2024-03-01 16:00:00+00:00"), "earnings_surprise"]
    assert before_release_surprise.isna().all()
    assert np.allclose(after_release_surprise.dropna().values, np.array([0.3] * len(after_release_surprise.dropna())))


def test_fundamentals_graceful_fallback_when_missing_inputs():
    price = _price_intraday()
    out = generate_fundamentals_features(price, fundamentals_df=None, earnings_df=None)
    assert out.empty


def test_fundamentals_plugin_is_available_in_registry():
    price = _price_intraday()
    fundamentals = pd.DataFrame(
        {
            "filing_timestamp": [pd.Timestamp("2024-03-01 15:00:00+00:00")],
            "total_debt": [50.0],
            "total_equity": [40.0],
            "ebit": [22.0],
            "interest_expense": [4.0],
        }
    )
    earnings = pd.DataFrame(
        {
            "event_timestamp": [pd.Timestamp("2024-03-01 16:00:00+00:00")],
            "known_since": [pd.Timestamp("2024-03-01 15:00:00+00:00")],
        }
    )

    library = TechnicalFeatureLibrary(build_default_registry())
    out, meta = library.generate(
        price,
        specs=[FeatureSpec(name="fundamentals")],
        fundamentals_df=fundamentals,
        earnings_df=earnings,
        include_base_columns=False,
        drop_na=False,
    )

    assert "fund_debt_to_equity" in out.columns
    assert "earnings_days_to_event" in out.columns
    assert "fundamentals" in [item["plugin"] for item in meta["catalog"]]
