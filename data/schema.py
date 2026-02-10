"""
Canonical market data schema utilities.
"""

from __future__ import annotations

from datetime import time
from typing import Optional

import numpy as np
import pandas as pd


CANONICAL_PRICE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "dividend",
    "split_factor",
    "session",
    "is_market_holiday",
    "is_synthetic",
]


_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "date", "datetime", "time"],
    "open": ["open", "Open"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
    "close": ["close", "Close"],
    "volume": ["volume", "Volume"],
    "adj_close": ["adj_close", "adj close", "Adj Close", "adjusted_close"],
    "dividend": ["dividend", "dividends", "Dividends"],
    "split_factor": ["split_factor", "stock splits", "Stock Splits", "splits"],
}


def _find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    for candidate in _COLUMN_ALIASES.get(canonical_name, []):
        if candidate in df.columns:
            return candidate
    return None


def _as_timezone_aware(
    series: pd.Series,
    source_timezone: str = "UTC",
    target_timezone: str = "America/New_York",
) -> pd.Series:
    ts = pd.to_datetime(series, utc=False, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(source_timezone, nonexistent="shift_forward", ambiguous="NaT")
    return ts.dt.tz_convert(target_timezone)


def _infer_session(timestamp: pd.Timestamp, intraday: bool) -> str:
    if not intraday:
        return "regular"

    t = timestamp.timetz().replace(tzinfo=None)
    if t < time(9, 30):
        return "pre"
    if t <= time(16, 0):
        return "regular"
    if t <= time(20, 0):
        return "after"
    return "closed"


def canonicalize_price_frame(
    raw_df: pd.DataFrame,
    *,
    source_timezone: str = "UTC",
    target_timezone: str = "America/New_York",
    intraday: bool = False,
) -> pd.DataFrame:
    """
    Convert vendor-specific dataframe into canonical schema.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)

    df = raw_df.copy()

    out = pd.DataFrame()
    for key in ["timestamp", "open", "high", "low", "close", "volume"]:
        source_col = _find_column(df, key)
        if source_col is None:
            raise ValueError(f"Missing required column '{key}' in market data")
        out[key] = df[source_col]

    out["timestamp"] = _as_timezone_aware(
        out["timestamp"],
        source_timezone=source_timezone,
        target_timezone=target_timezone,
    )

    # Optional corporate action columns.
    adj_col = _find_column(df, "adj_close")
    out["adj_close"] = df[adj_col] if adj_col else out["close"]

    div_col = _find_column(df, "dividend")
    out["dividend"] = df[div_col] if div_col else 0.0

    split_col = _find_column(df, "split_factor")
    split_raw = df[split_col] if split_col else 0.0
    out["split_factor"] = np.where(pd.to_numeric(split_raw, errors="coerce").fillna(0.0) == 0.0, 1.0, split_raw)

    for col in ["open", "high", "low", "close", "adj_close", "dividend", "split_factor", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["session"] = out["timestamp"].apply(lambda ts: _infer_session(ts, intraday=intraday))
    out["is_market_holiday"] = False
    out["is_synthetic"] = False

    out = out.sort_values("timestamp").reset_index(drop=True)
    return out[CANONICAL_PRICE_COLUMNS]


def apply_split_dividend_adjustments(
    canonical_df: pd.DataFrame,
    *,
    in_place: bool = False,
) -> pd.DataFrame:
    """
    Adjust OHLC fields using adj_close ratio.
    """
    if canonical_df is None or canonical_df.empty:
        return canonical_df.copy()

    df = canonical_df if in_place else canonical_df.copy()
    ratio = np.where(df["close"].abs() > 1e-12, df["adj_close"] / df["close"], 1.0)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * ratio

    return df
