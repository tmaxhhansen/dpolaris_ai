from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.prediction_inspector import (
    build_feature_snapshot,
    decide_trade_outcome,
    derive_regime,
    truncate_dataframe_asof,
)


def _assert_feature_maps_close(a: dict, b: dict) -> None:
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        av = a[key]
        bv = b[key]
        if av is None or bv is None:
            assert av is None and bv is None
            continue
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if np.isnan(av) or np.isnan(bv):
                assert np.isnan(av) and np.isnan(bv)
            else:
                assert av == pytest.approx(bv, rel=1e-9, abs=1e-12), key
        else:
            assert av == bv, key


def test_prediction_inspector_uses_causal_asof_slice(synthetic_df):
    base = synthetic_df.copy()
    inspect_time = pd.Timestamp(base.iloc[320]["date"], tz="UTC")

    base_slice = truncate_dataframe_asof(base, inspect_time)
    base_features = build_feature_snapshot(base_slice["frame"], target_horizon=5)

    mutated = base.copy()
    # Inject extreme future moves after the inspect timestamp.
    future_mask = pd.to_datetime(mutated["date"], utc=True) > inspect_time
    mutated.loc[future_mask, "close"] = mutated.loc[future_mask, "close"] * 12.0
    mutated.loc[future_mask, "open"] = mutated.loc[future_mask, "open"] * 12.0
    mutated.loc[future_mask, "high"] = mutated.loc[future_mask, "high"] * 12.0
    mutated.loc[future_mask, "low"] = mutated.loc[future_mask, "low"] * 12.0

    mutated_slice = truncate_dataframe_asof(mutated, inspect_time)
    mutated_features = build_feature_snapshot(mutated_slice["frame"], target_horizon=5)

    max_used_ts = pd.to_datetime(base_slice["frame"][base_slice["time_col"]], utc=True).max()
    assert max_used_ts <= inspect_time
    _assert_feature_maps_close(base_features["raw"], mutated_features["raw"])


def test_prediction_inspector_rejects_prehistory_timestamp(synthetic_df):
    base = synthetic_df.copy()
    first = pd.Timestamp(base.iloc[0]["date"], tz="UTC")
    before_first = first - pd.Timedelta(days=7)

    with pytest.raises(ValueError, match="No data exists at or before requested inspection time"):
        truncate_dataframe_asof(base, before_first)


def test_prediction_inspector_decision_thresholds_and_regime():
    decision = decide_trade_outcome(probability_up=0.67, confidence=0.72)
    assert decision["action"] == "BUY"

    neutral = decide_trade_outcome(probability_up=0.52, confidence=0.71)
    assert neutral["action"] == "HOLD"

    low_conf = decide_trade_outcome(probability_up=0.71, confidence=0.40)
    assert low_conf["action"] == "NO_TRADE"

    regime = derive_regime(
        {
            "price_sma20_ratio": 1.03,
            "price_sma50_ratio": 1.02,
            "price_sma200_ratio": 1.01,
            "hvol_20": 0.44,
            "roc_5": -0.01,
        }
    )
    assert regime["trend"] == "BULLISH"
    assert regime["volatility"] == "HIGH"
    assert regime["momentum"] == "NEGATIVE"
