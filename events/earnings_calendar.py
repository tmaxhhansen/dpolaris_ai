"""
Earnings calendar feature helpers with strict release-time behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def _timestamp_col(df: pd.DataFrame) -> str:
    for candidate in ("event_timestamp", "release_timestamp", "timestamp", "date", "datetime"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Earnings calendar requires an event timestamp column")


@dataclass
class EarningsCalendarConfig:
    event_timestamp_col: str = "event_timestamp"
    known_since_col: str = "known_since"
    surprise_col: str = "earnings_surprise"
    guidance_change_col: str = "guidance_change_proxy"
    post_windows_days: tuple[int, ...] = (1, 3, 5)

    @classmethod
    def from_params(cls, params: Optional[dict[str, Any]] = None) -> "EarningsCalendarConfig":
        params = params or {}
        windows = params.get("post_windows_days", cls.post_windows_days)
        return cls(
            event_timestamp_col=str(params.get("event_timestamp_col", cls.event_timestamp_col)),
            known_since_col=str(params.get("known_since_col", cls.known_since_col)),
            surprise_col=str(params.get("surprise_col", cls.surprise_col)),
            guidance_change_col=str(params.get("guidance_change_col", cls.guidance_change_col)),
            post_windows_days=tuple(int(x) for x in windows),
        )


def _prepare_earnings_frame(
    earnings_df: pd.DataFrame,
    cfg: EarningsCalendarConfig,
) -> pd.DataFrame:
    if earnings_df is None or earnings_df.empty:
        return pd.DataFrame()

    events = earnings_df.copy()
    ts_col = cfg.event_timestamp_col if cfg.event_timestamp_col in events.columns else _timestamp_col(events)
    events["__event_timestamp__"] = pd.to_datetime(events[ts_col], utc=True, errors="coerce")
    events = events.dropna(subset=["__event_timestamp__"]).sort_values("__event_timestamp__").reset_index(drop=True)

    if cfg.known_since_col in events.columns:
        events["__known_since__"] = pd.to_datetime(events[cfg.known_since_col], utc=True, errors="coerce")
    else:
        # Default: calendar event is known from the start of history.
        events["__known_since__"] = pd.Timestamp("1970-01-01", tz="UTC")

    return events


def generate_earnings_event_features(
    price_df: pd.DataFrame,
    *,
    earnings_df: Optional[pd.DataFrame],
    params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    cfg = EarningsCalendarConfig.from_params(params)
    if price_df is None or price_df.empty:
        return pd.DataFrame(index=price_df.index if price_df is not None else None)

    work = price_df.copy()
    ts_col = "timestamp" if "timestamp" in work.columns else "date"
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work["__orig_idx__"] = np.arange(len(work))
    work = work.sort_values(ts_col).reset_index(drop=True)

    events = _prepare_earnings_frame(earnings_df if earnings_df is not None else pd.DataFrame(), cfg)
    if events.empty:
        return pd.DataFrame(index=price_df.index)

    out = pd.DataFrame(index=work.index)
    left = pd.DataFrame({"timestamp": work[ts_col]})

    # Last released event as-of current timestamp.
    past = pd.merge_asof(
        left.sort_values("timestamp"),
        events.sort_values("__event_timestamp__"),
        left_on="timestamp",
        right_on="__event_timestamp__",
        direction="backward",
        allow_exact_matches=True,
    ).reset_index(drop=True)

    # Next known earnings event at time t (requires known_since <= t).
    next_known_ts: list[pd.Timestamp | pd.NaT] = []
    event_ts = events["__event_timestamp__"].tolist()
    known_ts = events["__known_since__"].tolist()
    for current_ts in left["timestamp"].tolist():
        candidate = pd.NaT
        for i in range(len(events)):
            if pd.isna(known_ts[i]) or pd.isna(event_ts[i]):
                continue
            if known_ts[i] <= current_ts < event_ts[i]:
                candidate = event_ts[i]
                break
        next_known_ts.append(candidate)
    out["earnings_next_event_timestamp"] = pd.to_datetime(next_known_ts, utc=True, errors="coerce")

    out["earnings_days_to_event"] = (
        (out["earnings_next_event_timestamp"] - left["timestamp"]).dt.total_seconds() / 86400.0
    )

    out["earnings_last_event_timestamp"] = pd.to_datetime(
        past["__event_timestamp__"],
        utc=True,
        errors="coerce",
    )

    if cfg.surprise_col in past.columns:
        out["earnings_surprise"] = pd.to_numeric(past[cfg.surprise_col], errors="coerce")
    if cfg.guidance_change_col in past.columns:
        out["earnings_guidance_change_proxy"] = pd.to_numeric(past[cfg.guidance_change_col], errors="coerce")

    days_since = (left["timestamp"] - out["earnings_last_event_timestamp"]).dt.total_seconds() / 86400.0
    out["earnings_days_since_event"] = days_since
    out["earnings_post_window_flag"] = (days_since.between(0.0, float(max(cfg.post_windows_days)), inclusive="both")).astype(float)
    for window in cfg.post_windows_days:
        out[f"earnings_post_window_d{int(window)}"] = (days_since.between(0.0, float(window), inclusive="both")).astype(float)

    out["__orig_idx__"] = work["__orig_idx__"].values
    out = out.sort_values("__orig_idx__").drop(columns="__orig_idx__").reset_index(drop=True)
    out.index = price_df.index
    return out

