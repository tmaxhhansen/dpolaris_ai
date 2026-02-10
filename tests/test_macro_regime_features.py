from __future__ import annotations

import numpy as np
import pandas as pd

from features import FeatureSpec, TechnicalFeatureLibrary, build_default_registry, generate_macro_features
from regime.regime_classifier import RegimeClassifier


def _intraday_frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-03 13:00:00+00:00", periods=8, freq="1h")
    close = np.array([100.0, 101.0, 100.5, 101.5, 102.0, 103.0, 104.0, 103.5], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.6,
            "close": close,
            "volume": np.array([1000, 1200, 900, 1100, 1300, 1400, 1000, 950], dtype=float),
        }
    )


def test_macro_release_time_causality():
    price = _intraday_frame()
    macro = pd.DataFrame(
        {
            "release_timestamp": [
                pd.Timestamp("2024-01-03 14:00:00+00:00"),
                pd.Timestamp("2024-01-03 16:00:00+00:00"),
            ],
            "inflation_proxy": [2.1, 2.8],
            "yield_2y": [4.1, 4.3],
            "yield_10y": [4.5, 4.7],
            "vix": [16.0, 31.0],
        }
    )

    out = generate_macro_features(price, macro_df=macro)

    pre = out.loc[price["timestamp"] < pd.Timestamp("2024-01-03 14:00:00+00:00"), "macro_inflation_proxy"]
    mid = out.loc[
        (price["timestamp"] >= pd.Timestamp("2024-01-03 14:00:00+00:00"))
        & (price["timestamp"] < pd.Timestamp("2024-01-03 16:00:00+00:00")),
        "macro_inflation_proxy",
    ]
    post = out.loc[price["timestamp"] >= pd.Timestamp("2024-01-03 16:00:00+00:00"), "macro_inflation_proxy"]

    assert pre.isna().all()
    assert (mid == 2.1).all()
    assert (post == 2.8).all()


def test_macro_features_graceful_fallback_without_macro_data():
    price = _intraday_frame()
    out = generate_macro_features(price, macro_df=None)
    assert out.empty


def test_regime_classifier_graceful_fallback_without_risk_proxies():
    price = _intraday_frame()
    classifier = RegimeClassifier()
    out = classifier.classify(price)

    assert "regime_trend_flag" in out.columns
    assert "regime_high_vol_flag" in out.columns
    assert "regime_risk_off_flag" in out.columns
    assert out["regime_risk_off_flag"].isna().all()


def test_regime_plugin_uses_macro_asof_without_peek():
    price = _intraday_frame()
    macro = pd.DataFrame(
        {
            "release_timestamp": [
                pd.Timestamp("2024-01-03 13:00:00+00:00"),
                pd.Timestamp("2024-01-03 16:00:00+00:00"),
            ],
            "vix": [14.0, 34.0],
        }
    )

    library = TechnicalFeatureLibrary(build_default_registry())
    out, meta = library.generate(
        price,
        specs=[
            FeatureSpec(
                name="regime",
                params={
                    "vix_high_threshold": 25.0,
                    "trend_lookback": 3,
                    "realized_vol_window": 3,
                    "risk_z_window": 5,
                },
            )
        ],
        macro_df=macro,
        include_base_columns=False,
        drop_na=False,
    )

    assert "regime_high_vol_flag" in out.columns
    assert "regime_high_vol_flag" in meta["feature_names"]

    before = out.loc[price["timestamp"] < pd.Timestamp("2024-01-03 16:00:00+00:00"), "regime_high_vol_flag"]
    after = out.loc[price["timestamp"] >= pd.Timestamp("2024-01-03 16:00:00+00:00"), "regime_high_vol_flag"]

    assert (before == 0.0).all()
    assert (after == 1.0).all()

