"""
Prediction inspection helpers with strict as-of causality.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .features import FeatureEngine

TIME_COLUMNS = ("timestamp", "date", "datetime")


def parse_inspection_time(value: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        ts = pd.Timestamp(raw)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def find_time_column(df: pd.DataFrame) -> str:
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError("DataFrame must contain one of timestamp/date/datetime columns")


def truncate_dataframe_asof(
    df: pd.DataFrame,
    inspect_time: Optional[str | pd.Timestamp],
) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("No source data available for inspection")

    time_col = find_time_column(df)
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        raise ValueError("No valid timestamps available for inspection")

    requested_ts = parse_inspection_time(inspect_time)
    resolved_ts = requested_ts or pd.Timestamp(work.iloc[-1][time_col])

    if resolved_ts.tzinfo is None:
        resolved_ts = resolved_ts.tz_localize("UTC")
    else:
        resolved_ts = resolved_ts.tz_convert("UTC")

    asof = work[work[time_col] <= resolved_ts].copy().reset_index(drop=True)
    if asof.empty:
        raise ValueError("No data exists at or before requested inspection time")

    warnings: list[str] = []
    latest_source_ts = pd.Timestamp(work.iloc[-1][time_col])
    if requested_ts is not None and requested_ts > latest_source_ts:
        warnings.append(
            "Requested time is after latest available data; using latest available timestamp."
        )
    if len(asof) < len(work):
        warnings.append(
            f"Inspection is as-of {resolved_ts.isoformat()} using {len(asof)} of {len(work)} available rows."
        )

    return {
        "frame": asof,
        "time_col": time_col,
        "requested_time": requested_ts.isoformat() if requested_ts is not None else None,
        "resolved_time": pd.Timestamp(asof.iloc[-1][time_col]).isoformat(),
        "latest_source_time": latest_source_ts.isoformat(),
        "rows_total": int(len(work)),
        "rows_used": int(len(asof)),
        "warnings": warnings,
    }


def _to_native(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def latest_ohlcv_snapshot(asof_df: pd.DataFrame, time_col: str) -> dict[str, Any]:
    latest = asof_df.iloc[-1]
    snapshot: dict[str, Any] = {"timestamp": _to_native(pd.Timestamp(latest[time_col]))}
    for col in ("open", "high", "low", "close", "volume", "adj_close"):
        if col in asof_df.columns:
            snapshot[col] = _to_native(latest[col])
    return snapshot


def build_feature_snapshot(
    asof_df: pd.DataFrame,
    *,
    target_horizon: int = 5,
) -> dict[str, Any]:
    engine = FeatureEngine()
    features_df = engine.generate_features(
        asof_df.copy(),
        include_targets=False,
        target_horizon=max(1, int(target_horizon)),
    )
    if features_df.empty:
        raise ValueError("Not enough as-of history to compute features")

    feature_names = engine.get_feature_names()
    latest = features_df.iloc[-1]

    raw: dict[str, Any] = {}
    for name in feature_names:
        raw[name] = _to_native(latest.get(name))

    ranked = sorted(
        (
            (name, abs(float(value)))
            for name, value in raw.items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    feature_time_col = find_time_column(features_df)
    feature_timestamp = pd.Timestamp(features_df.iloc[-1][feature_time_col])

    return {
        "feature_names": feature_names,
        "raw": raw,
        "top_abs_features": [name for name, _ in ranked[:20]],
        "feature_timestamp": _to_native(feature_timestamp),
    }


def derive_regime(raw_features: dict[str, Any]) -> dict[str, Any]:
    def f(name: str, default: float) -> float:
        value = raw_features.get(name)
        if value is None:
            return default
        try:
            v = float(value)
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return default

    trend_votes = sum(
        1
        for value in (
            f("price_sma20_ratio", 1.0),
            f("price_sma50_ratio", 1.0),
            f("price_sma200_ratio", 1.0),
        )
        if value > 1.0
    )
    if trend_votes >= 2:
        trend = "BULLISH"
    elif trend_votes <= 1:
        trend = "BEARISH"
    else:
        trend = "MIXED"

    hvol_20 = f("hvol_20", 0.2)
    if hvol_20 >= 0.40:
        volatility = "HIGH"
    elif hvol_20 <= 0.18:
        volatility = "LOW"
    else:
        volatility = "NORMAL"

    momentum = "POSITIVE" if f("roc_5", 0.0) >= 0 else "NEGATIVE"

    return {
        "label": f"{trend.lower()}_{volatility.lower()}_{momentum.lower()}",
        "trend": trend,
        "volatility": volatility,
        "momentum": momentum,
    }


def decide_trade_outcome(
    *,
    probability_up: float,
    confidence: float,
    long_threshold: float = 0.60,
    short_threshold: float = 0.40,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    p = float(np.clip(probability_up, 0.0, 1.0))
    c = float(np.clip(confidence, 0.0, 1.0))

    if c < min_confidence:
        action = "NO_TRADE"
        reason = "Confidence below threshold"
    elif p >= long_threshold:
        action = "BUY"
        reason = "Probability above long threshold"
    elif p <= short_threshold:
        action = "SELL"
        reason = "Probability below short threshold"
    else:
        action = "HOLD"
        reason = "Probability in neutral zone"

    return {
        "action": action,
        "reason": reason,
        "probability_up": p,
        "confidence": c,
        "thresholds": {
            "long": float(long_threshold),
            "short": float(short_threshold),
            "min_confidence": float(min_confidence),
        },
    }
