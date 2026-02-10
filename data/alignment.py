"""
Time alignment and strict causal join utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}


def parse_timeframe(timeframe: str) -> str:
    """
    Convert common timeframe tokens to pandas offset aliases.
    """
    key = timeframe.strip().lower()
    if key not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_MAP[key]


def resample_ohlcv(
    canonical_price_df: pd.DataFrame,
    *,
    timeframe: str,
    session: Optional[str] = None,
) -> pd.DataFrame:
    """
    Resample canonical OHLCV frame to target timeframe.
    """
    if canonical_price_df is None or canonical_price_df.empty:
        return canonical_price_df.copy()

    df = canonical_price_df.copy()
    if session is not None and "session" in df.columns:
        df = df[df["session"] == session]

    freq = parse_timeframe(timeframe)
    df = df.sort_values("timestamp").set_index("timestamp")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "adj_close": "last",
        "dividend": "sum",
        "split_factor": "max",
        "is_market_holiday": "max",
        "is_synthetic": "max",
    }
    if "session" in df.columns:
        agg["session"] = "last"

    out = df.resample(freq).agg(agg).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    return out


def causal_asof_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    right_prefix: str,
    timestamp_col: str = "timestamp",
    tolerance: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """
    Strict causal as-of join (right timestamp <= left timestamp).
    """
    left = left_df.copy().sort_values(timestamp_col)
    right = right_df.copy().sort_values(timestamp_col)

    if left.empty:
        return left

    left[timestamp_col] = pd.to_datetime(left[timestamp_col], utc=True)
    right[timestamp_col] = pd.to_datetime(right[timestamp_col], utc=True)

    right_ts_col = f"{right_prefix}__asof_timestamp"
    right = right.rename(columns={timestamp_col: right_ts_col})
    rename_map = {
        col: f"{right_prefix}__{col}"
        for col in right.columns
        if col != right_ts_col
    }
    right = right.rename(columns=rename_map)

    merged = pd.merge_asof(
        left,
        right,
        left_on=timestamp_col,
        right_on=right_ts_col,
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    return merged


def align_feature_frames(
    price_df: pd.DataFrame,
    *,
    fundamentals_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
    tolerance: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """
    Align optional feature frames to price timeline using strict causal joins.
    """
    aligned = price_df.copy()
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
    aligned = aligned.sort_values("timestamp").reset_index(drop=True)

    if fundamentals_df is not None and not fundamentals_df.empty:
        aligned = causal_asof_join(
            aligned,
            fundamentals_df,
            right_prefix="fund",
            tolerance=tolerance,
        )

    if macro_df is not None and not macro_df.empty:
        aligned = causal_asof_join(
            aligned,
            macro_df,
            right_prefix="macro",
            tolerance=tolerance,
        )

    if news_df is not None and not news_df.empty:
        aligned = causal_asof_join(
            aligned,
            news_df,
            right_prefix="news",
            tolerance=tolerance,
        )

    return aligned


def assert_no_peek(aligned_df: pd.DataFrame, timestamp_col: str = "timestamp") -> bool:
    """
    Verify all as-of columns are causally aligned.
    """
    base_ts = pd.to_datetime(aligned_df[timestamp_col], utc=True)
    for col in aligned_df.columns:
        if col.endswith("__asof_timestamp"):
            rhs = pd.to_datetime(aligned_df[col], utc=True)
            mask = rhs.notna()
            if not ((rhs[mask] <= base_ts[mask]).all()):
                return False
    return True
