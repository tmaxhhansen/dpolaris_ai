"""
Data quality checks, auto-repair policies, and reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import pandas as pd

from .alignment import parse_timeframe


@dataclass
class QualityPolicy:
    """
    Controls automatic drop/repair behavior.
    """

    duplicate_policy: str = "drop"
    missing_timestamp_policy: str = "repair"
    negative_volume_policy: str = "repair"
    stale_price_policy: str = "drop"
    outlier_policy: str = "drop"
    stale_run_length: int = 10
    outlier_return_threshold: float = 0.35
    min_history_rows_multiplier: int = 80


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_interval_to_timedelta(interval: str) -> pd.Timedelta:
    freq = parse_timeframe(interval)
    offset = pd.tseries.frequencies.to_offset(freq)
    if hasattr(offset, "delta"):
        return offset.delta
    if hasattr(offset, "nanos"):
        return pd.Timedelta(int(offset.nanos), unit="ns")
    return pd.Timedelta(freq)


def _missing_timestamps_daily(df: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.normalize()
    start = dates.min()
    end = dates.max()

    expected: pd.DatetimeIndex
    try:
        import pandas_market_calendars as mcal  # type: ignore

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=start.date(),
            end_date=end.date(),
        )
        expected = pd.DatetimeIndex(schedule.index).tz_localize("America/New_York")
    except Exception:
        # Fallback when exchange calendar dependency is not installed.
        expected = pd.date_range(start=start, end=end, freq="B", tz="America/New_York")

    observed = pd.DatetimeIndex(dates.unique())
    missing = sorted(set(expected) - set(observed))
    return missing


def _missing_timestamps_intraday(df: pd.DataFrame, interval: str) -> list[pd.Timestamp]:
    expected_delta = _parse_interval_to_timedelta(interval)
    ts = pd.to_datetime(df["timestamp"]).sort_values()
    missing: list[pd.Timestamp] = []

    for _, day_df in ts.to_series().groupby(ts.dt.date):
        day_vals = day_df.sort_values().values
        for i in range(1, len(day_vals)):
            prev_ts = pd.Timestamp(day_vals[i - 1])
            curr_ts = pd.Timestamp(day_vals[i])
            gap = curr_ts - prev_ts
            if gap > expected_delta * 1.5:
                steps = int(gap / expected_delta) - 1
                for step in range(steps):
                    missing.append(prev_ts + (step + 1) * expected_delta)
    return missing


def _detect_stale_close(close_values: pd.Series, run_length: int) -> list[int]:
    stale_indices: list[int] = []
    if close_values.empty:
        return stale_indices

    run_start = 0
    for i in range(1, len(close_values)):
        if close_values.iloc[i] != close_values.iloc[i - 1]:
            if i - run_start >= run_length:
                stale_indices.extend(list(range(run_start, i)))
            run_start = i
    if len(close_values) - run_start >= run_length:
        stale_indices.extend(list(range(run_start, len(close_values))))
    return stale_indices


class DataQualityGate:
    """
    Runs data quality checks with optional repair and emits JSON reports.
    """

    def __init__(self, policy: Optional[QualityPolicy] = None):
        self.policy = policy or QualityPolicy()

    def run(
        self,
        canonical_df: pd.DataFrame,
        *,
        symbol: str,
        interval: str = "1d",
        horizon_days: int = 5,
        run_id: Optional[str] = None,
        report_dir: Optional[Path | str] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any], Path]:
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(report_dir) if report_dir is not None else Path(__file__).resolve().parent.parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"data_quality_{run_id}.json"

        df = canonical_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        initial_rows = len(df)
        negative_volume_count = int((df["volume"] < 0).sum())

        duplicates_mask = df.duplicated(subset=["timestamp"], keep="last")
        duplicate_count = int(duplicates_mask.sum())
        if duplicate_count > 0 and self.policy.duplicate_policy == "drop":
            df = df[~duplicates_mask].copy()

        negative_volume_mask = df["volume"] < 0
        if int(negative_volume_mask.sum()) > 0:
            if self.policy.negative_volume_policy == "repair":
                df.loc[negative_volume_mask, "volume"] = 0
            elif self.policy.negative_volume_policy == "drop":
                df = df.loc[~negative_volume_mask].copy()

        if interval.lower().endswith("d"):
            missing_ts = _missing_timestamps_daily(df)
        else:
            missing_ts = _missing_timestamps_intraday(df, interval=interval)
        missing_count = len(missing_ts)

        repaired_missing_count = 0
        if missing_count > 0 and self.policy.missing_timestamp_policy == "repair":
            synthesized_rows = []
            expected_delta = _parse_interval_to_timedelta(interval)
            for ts in missing_ts:
                ts_utc = pd.Timestamp(ts).tz_convert("UTC")
                previous = df[df["timestamp"] < ts_utc].tail(1)
                if previous.empty:
                    continue
                prev = previous.iloc[0]
                synthesized_rows.append(
                    {
                        "timestamp": ts_utc,
                        "open": prev["close"],
                        "high": prev["close"],
                        "low": prev["close"],
                        "close": prev["close"],
                        "volume": 0.0,
                        "adj_close": prev["adj_close"],
                        "dividend": 0.0,
                        "split_factor": 1.0,
                        "session": prev.get("session", "regular"),
                        "is_market_holiday": False,
                        "is_synthetic": True,
                    }
                )
            if synthesized_rows:
                df = pd.concat([df, pd.DataFrame(synthesized_rows)], ignore_index=True)
                df = df.sort_values("timestamp").reset_index(drop=True)
                repaired_missing_count = len(synthesized_rows)

        stale_indices = _detect_stale_close(df["close"], run_length=self.policy.stale_run_length)
        stale_count = len(stale_indices)
        if stale_count > 0 and self.policy.stale_price_policy == "drop":
            stale_mask = pd.Series(False, index=df.index)
            stale_mask.iloc[stale_indices] = True
            df = df.loc[~stale_mask].copy()

        close_return = df["close"].pct_change().abs().fillna(0.0)
        has_corp_action = (df["dividend"].abs() > 1e-12) | (df["split_factor"].fillna(1.0) != 1.0)
        outlier_mask = (close_return > self.policy.outlier_return_threshold) & (~has_corp_action)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0 and self.policy.outlier_policy == "drop":
            df = df.loc[~outlier_mask].copy()

        min_required_rows = max(
            int(horizon_days) * int(self.policy.min_history_rows_multiplier),
            int(horizon_days) + 1,
        )
        min_history_passed = len(df) >= min_required_rows

        report: dict[str, Any] = {
            "run_id": run_id,
            "generated_at": _timestamp_now(),
            "symbol": symbol.upper().strip(),
            "interval": interval,
            "policy": asdict(self.policy),
            "row_count_before": initial_rows,
            "row_count_after": int(len(df)),
            "checks": {
                "duplicates": {
                    "count": duplicate_count,
                    "action": self.policy.duplicate_policy,
                },
                "missing_timestamps": {
                    "count": missing_count,
                    "action": self.policy.missing_timestamp_policy,
                    "repaired_count": repaired_missing_count,
                    "examples": [str(ts) for ts in missing_ts[:5]],
                },
                "negative_volume": {
                    "count": negative_volume_count,
                    "action": self.policy.negative_volume_policy,
                },
                "stale_prices": {
                    "count": stale_count,
                    "action": self.policy.stale_price_policy,
                },
                "outliers_without_corporate_action": {
                    "count": outlier_count,
                    "action": self.policy.outlier_policy,
                    "threshold_abs_return": self.policy.outlier_return_threshold,
                },
                "minimum_history": {
                    "required_rows": min_required_rows,
                    "actual_rows": int(len(df)),
                    "passed": min_history_passed,
                },
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return df.reset_index(drop=True), report, report_path
