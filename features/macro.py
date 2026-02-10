"""
Macro feature interface with strict as-of availability semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .registry import FeaturePlugin, FeatureRegistry


def _timestamp_col(df: pd.DataFrame) -> str:
    for candidate in ("timestamp", "date", "datetime", "time"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Input DataFrame must include a timestamp column")


def _prepare_base_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ts_col = _timestamp_col(df)
    work = df.copy()
    work["__orig_idx__"] = np.arange(len(work))
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = work.sort_values(ts_col).reset_index(drop=True)
    return work, ts_col


def _restore_output_order(
    features: pd.DataFrame,
    work: pd.DataFrame,
    original_index: pd.Index,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=original_index)
    out = features.copy()
    out["__orig_idx__"] = work["__orig_idx__"].values
    out = out.sort_values("__orig_idx__").drop(columns="__orig_idx__").reset_index(drop=True)
    out.index = original_index
    return out


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b + eps)


@dataclass
class MacroFeatureConfig:
    release_timestamp_col: str = "release_timestamp"
    yield_2y_col: str = "yield_2y"
    yield_10y_col: str = "yield_10y"
    yield_slope_col: str = "yield_slope_2s10s"
    inflation_col: str = "inflation_proxy"
    jobs_col: str = "jobs_proxy"
    vix_col: str = "vix"
    credit_spread_col: str = "credit_spread"
    dollar_strength_col: str = "dollar_index"
    benchmark_rotation_col: str = "benchmark_rotation"
    fed_decision_dates: list[str] = field(default_factory=list)
    fed_release_time: str = "14:00"
    fed_timezone: str = "America/New_York"
    zscore_window: int = 63

    @classmethod
    def from_params(
        cls,
        params: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> "MacroFeatureConfig":
        params = params or {}
        context = context or {}

        fed_dates = params.get("fed_decision_dates")
        if fed_dates is None:
            fed_dates = context.get("fed_decision_dates", [])

        return cls(
            release_timestamp_col=str(params.get("release_timestamp_col", cls.release_timestamp_col)),
            yield_2y_col=str(params.get("yield_2y_col", cls.yield_2y_col)),
            yield_10y_col=str(params.get("yield_10y_col", cls.yield_10y_col)),
            yield_slope_col=str(params.get("yield_slope_col", cls.yield_slope_col)),
            inflation_col=str(params.get("inflation_col", cls.inflation_col)),
            jobs_col=str(params.get("jobs_col", cls.jobs_col)),
            vix_col=str(params.get("vix_col", cls.vix_col)),
            credit_spread_col=str(params.get("credit_spread_col", cls.credit_spread_col)),
            dollar_strength_col=str(params.get("dollar_strength_col", cls.dollar_strength_col)),
            benchmark_rotation_col=str(params.get("benchmark_rotation_col", cls.benchmark_rotation_col)),
            fed_decision_dates=[str(x) for x in (fed_dates or [])],
            fed_release_time=str(params.get("fed_release_time", cls.fed_release_time)),
            fed_timezone=str(params.get("fed_timezone", cls.fed_timezone)),
            zscore_window=int(params.get("zscore_window", cls.zscore_window)),
        )


def _prepare_macro_frame(macro_df: pd.DataFrame, cfg: MacroFeatureConfig) -> pd.DataFrame:
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    macro = macro_df.copy()
    if cfg.release_timestamp_col in macro.columns:
        release_col = cfg.release_timestamp_col
    else:
        release_col = _timestamp_col(macro)

    macro["__release_timestamp__"] = pd.to_datetime(macro[release_col], utc=True, errors="coerce")
    macro = macro.dropna(subset=["__release_timestamp__"]).sort_values("__release_timestamp__").reset_index(drop=True)
    return macro


def _get_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _add_level_delta_zscore(
    out: pd.DataFrame,
    series: Optional[pd.Series],
    name: str,
    *,
    zscore_window: int,
) -> None:
    if series is None:
        return
    out[name] = series
    out[f"{name}_diff1"] = series.diff(1)
    roll_mean = series.rolling(zscore_window, min_periods=max(10, zscore_window // 3)).mean()
    roll_std = series.rolling(zscore_window, min_periods=max(10, zscore_window // 3)).std()
    out[f"{name}_z{zscore_window}"] = _safe_div(series - roll_mean, roll_std)


def _build_fed_decision_flag(base_ts: pd.Series, cfg: MacroFeatureConfig) -> pd.Series:
    if not cfg.fed_decision_dates:
        return pd.Series(np.zeros(len(base_ts), dtype=float), index=base_ts.index)

    try:
        hour, minute = cfg.fed_release_time.split(":", maxsplit=1)
        rel_hour = int(hour)
        rel_minute = int(minute)
    except Exception:
        rel_hour = 14
        rel_minute = 0

    local_ts = pd.to_datetime(base_ts, utc=True, errors="coerce").dt.tz_convert(cfg.fed_timezone)
    flag = np.zeros(len(local_ts), dtype=float)

    for raw_day in cfg.fed_decision_dates:
        day = pd.Timestamp(raw_day)
        if day.tzinfo is not None:
            day = day.tz_convert(cfg.fed_timezone)
        else:
            day = day.tz_localize(cfg.fed_timezone)
        event_ts = day.normalize() + pd.Timedelta(hours=rel_hour, minutes=rel_minute)
        day_end = event_ts.normalize() + pd.Timedelta(days=1)
        mask = (local_ts >= event_ts) & (local_ts < day_end)
        flag = np.where(mask, 1.0, flag)

    return pd.Series(flag, index=base_ts.index)


def _macro_features_plugin(
    df: pd.DataFrame,
    params: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> pd.DataFrame:
    context = context or {}
    cfg = MacroFeatureConfig.from_params(params=params, context=context)

    work, ts_col = _prepare_base_frame(df)
    out = pd.DataFrame(index=work.index)

    macro_df = context.get("macro_df")
    if macro_df is not None and not macro_df.empty:
        macro = _prepare_macro_frame(macro_df, cfg)
        if not macro.empty:
            left = pd.DataFrame({"timestamp": work[ts_col]})
            merged = pd.merge_asof(
                left.sort_values("timestamp"),
                macro.sort_values("__release_timestamp__"),
                left_on="timestamp",
                right_on="__release_timestamp__",
                direction="backward",
                allow_exact_matches=True,
            )
            merged = merged.reset_index(drop=True)

            yield_2y = _get_series(merged, cfg.yield_2y_col)
            yield_10y = _get_series(merged, cfg.yield_10y_col)
            slope = _get_series(merged, cfg.yield_slope_col)
            if slope is None and yield_2y is not None and yield_10y is not None:
                slope = yield_10y - yield_2y

            _add_level_delta_zscore(out, yield_2y, "macro_yield_2y_level", zscore_window=cfg.zscore_window)
            _add_level_delta_zscore(out, yield_10y, "macro_yield_10y_level", zscore_window=cfg.zscore_window)
            _add_level_delta_zscore(out, slope, "macro_yield_curve_slope_10y_2y", zscore_window=cfg.zscore_window)
            _add_level_delta_zscore(
                out,
                _get_series(merged, cfg.inflation_col),
                "macro_inflation_proxy",
                zscore_window=cfg.zscore_window,
            )
            _add_level_delta_zscore(
                out,
                _get_series(merged, cfg.jobs_col),
                "macro_jobs_proxy",
                zscore_window=cfg.zscore_window,
            )
            _add_level_delta_zscore(
                out,
                _get_series(merged, cfg.vix_col),
                "macro_vix_proxy",
                zscore_window=cfg.zscore_window,
            )
            _add_level_delta_zscore(
                out,
                _get_series(merged, cfg.credit_spread_col),
                "macro_credit_spread_proxy",
                zscore_window=cfg.zscore_window,
            )
            _add_level_delta_zscore(
                out,
                _get_series(merged, cfg.dollar_strength_col),
                "macro_dollar_strength_proxy",
                zscore_window=cfg.zscore_window,
            )
            _add_level_delta_zscore(
                out,
                _get_series(merged, cfg.benchmark_rotation_col),
                "macro_benchmark_rotation_proxy",
                zscore_window=cfg.zscore_window,
            )

    if cfg.fed_decision_dates:
        out["macro_fed_decision_flag"] = _build_fed_decision_flag(work[ts_col], cfg)

    return _restore_output_order(out, work, df.index)


def generate_macro_features(
    price_df: pd.DataFrame,
    *,
    macro_df: Optional[pd.DataFrame] = None,
    params: Optional[dict[str, Any]] = None,
    fed_decision_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    context = {"macro_df": macro_df}
    params = dict(params or {})
    if fed_decision_dates is not None and "fed_decision_dates" not in params:
        params["fed_decision_dates"] = fed_decision_dates
    return _macro_features_plugin(price_df, params, context)


def register_macro_plugins(registry: FeatureRegistry) -> None:
    registry.register(
        FeaturePlugin(
            name="macro",
            generator=_macro_features_plugin,
            group="macro",
            description="Macro and event features with strict release-time as-of joins.",
        )
    )

