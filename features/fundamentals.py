"""
Fundamentals feature interface for swing/position horizons with strict release timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from events.earnings_calendar import generate_earnings_event_features
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


def _get_numeric(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


@dataclass
class FundamentalsFeatureConfig:
    release_timestamp_col: str = "filing_timestamp"
    revenue_col: str = "revenue"
    gross_margin_col: str = "gross_margin"
    operating_margin_col: str = "operating_margin"
    net_margin_col: str = "net_margin"
    fcf_col: str = "free_cash_flow"
    debt_col: str = "total_debt"
    equity_col: str = "total_equity"
    ebit_col: str = "ebit"
    interest_expense_col: str = "interest_expense"
    pe_col: str = "pe_ratio"
    ev_ebitda_col: str = "ev_ebitda"
    fcf_yield_col: str = "fcf_yield"
    market_cap_col: str = "market_cap"
    net_income_col: str = "net_income"
    enterprise_value_col: str = "enterprise_value"
    ebitda_col: str = "ebitda"
    qoq_periods: int = 1
    yoy_periods: int = 4
    trend_periods: int = 4

    @classmethod
    def from_params(cls, params: Optional[dict[str, Any]] = None) -> "FundamentalsFeatureConfig":
        params = params or {}
        return cls(
            release_timestamp_col=str(params.get("release_timestamp_col", cls.release_timestamp_col)),
            revenue_col=str(params.get("revenue_col", cls.revenue_col)),
            gross_margin_col=str(params.get("gross_margin_col", cls.gross_margin_col)),
            operating_margin_col=str(params.get("operating_margin_col", cls.operating_margin_col)),
            net_margin_col=str(params.get("net_margin_col", cls.net_margin_col)),
            fcf_col=str(params.get("fcf_col", cls.fcf_col)),
            debt_col=str(params.get("debt_col", cls.debt_col)),
            equity_col=str(params.get("equity_col", cls.equity_col)),
            ebit_col=str(params.get("ebit_col", cls.ebit_col)),
            interest_expense_col=str(params.get("interest_expense_col", cls.interest_expense_col)),
            pe_col=str(params.get("pe_col", cls.pe_col)),
            ev_ebitda_col=str(params.get("ev_ebitda_col", cls.ev_ebitda_col)),
            fcf_yield_col=str(params.get("fcf_yield_col", cls.fcf_yield_col)),
            market_cap_col=str(params.get("market_cap_col", cls.market_cap_col)),
            net_income_col=str(params.get("net_income_col", cls.net_income_col)),
            enterprise_value_col=str(params.get("enterprise_value_col", cls.enterprise_value_col)),
            ebitda_col=str(params.get("ebitda_col", cls.ebitda_col)),
            qoq_periods=int(params.get("qoq_periods", cls.qoq_periods)),
            yoy_periods=int(params.get("yoy_periods", cls.yoy_periods)),
            trend_periods=int(params.get("trend_periods", cls.trend_periods)),
        )


def _prepare_fundamentals_release_frame(
    fundamentals_df: pd.DataFrame,
    cfg: FundamentalsFeatureConfig,
) -> pd.DataFrame:
    if fundamentals_df is None or fundamentals_df.empty:
        return pd.DataFrame()

    fund = fundamentals_df.copy()
    release_col = cfg.release_timestamp_col if cfg.release_timestamp_col in fund.columns else _timestamp_col(fund)
    fund["__release_timestamp__"] = pd.to_datetime(fund[release_col], utc=True, errors="coerce")
    fund = fund.dropna(subset=["__release_timestamp__"]).sort_values("__release_timestamp__").reset_index(drop=True)
    if fund.empty:
        return fund

    revenue = _get_numeric(fund, cfg.revenue_col)
    gross_margin = _get_numeric(fund, cfg.gross_margin_col)
    op_margin = _get_numeric(fund, cfg.operating_margin_col)
    net_margin = _get_numeric(fund, cfg.net_margin_col)
    fcf = _get_numeric(fund, cfg.fcf_col)
    debt = _get_numeric(fund, cfg.debt_col)
    equity = _get_numeric(fund, cfg.equity_col)
    ebit = _get_numeric(fund, cfg.ebit_col)
    interest = _get_numeric(fund, cfg.interest_expense_col)

    out = pd.DataFrame(index=fund.index)
    out["__release_timestamp__"] = fund["__release_timestamp__"]

    if revenue is not None:
        out["fund_revenue_growth_qoq"] = revenue.pct_change(cfg.qoq_periods)
        out["fund_revenue_growth_yoy"] = revenue.pct_change(cfg.yoy_periods)
    if gross_margin is not None:
        out["fund_margin_gross_level"] = gross_margin
        out["fund_margin_gross_trend"] = gross_margin.diff(cfg.trend_periods)
    if op_margin is not None:
        out["fund_margin_operating_level"] = op_margin
        out["fund_margin_operating_trend"] = op_margin.diff(cfg.trend_periods)
    if net_margin is not None:
        out["fund_margin_net_level"] = net_margin
        out["fund_margin_net_trend"] = net_margin.diff(cfg.trend_periods)
    if fcf is not None:
        out["fund_fcf_growth_qoq"] = fcf.pct_change(cfg.qoq_periods)
        out["fund_fcf_growth_yoy"] = fcf.pct_change(cfg.yoy_periods)
        out["fund_fcf_trend"] = fcf.diff(cfg.trend_periods)

    if debt is not None and equity is not None:
        out["fund_debt_to_equity"] = _safe_div(debt, equity)
    if ebit is not None and interest is not None:
        out["fund_interest_coverage"] = _safe_div(ebit, interest.abs())

    pe_direct = _get_numeric(fund, cfg.pe_col)
    if pe_direct is not None:
        out["fund_pe_ratio"] = pe_direct
    else:
        market_cap = _get_numeric(fund, cfg.market_cap_col)
        net_income = _get_numeric(fund, cfg.net_income_col)
        if market_cap is not None and net_income is not None:
            out["fund_pe_ratio"] = _safe_div(market_cap, net_income)

    ev_ebitda_direct = _get_numeric(fund, cfg.ev_ebitda_col)
    if ev_ebitda_direct is not None:
        out["fund_ev_ebitda"] = ev_ebitda_direct
    else:
        enterprise_value = _get_numeric(fund, cfg.enterprise_value_col)
        ebitda = _get_numeric(fund, cfg.ebitda_col)
        if enterprise_value is not None and ebitda is not None:
            out["fund_ev_ebitda"] = _safe_div(enterprise_value, ebitda)

    fcf_yield_direct = _get_numeric(fund, cfg.fcf_yield_col)
    if fcf_yield_direct is not None:
        out["fund_fcf_yield"] = fcf_yield_direct
    else:
        market_cap = _get_numeric(fund, cfg.market_cap_col)
        if fcf is not None and market_cap is not None:
            out["fund_fcf_yield"] = _safe_div(fcf, market_cap)

    return out


def _fundamentals_features_plugin(
    df: pd.DataFrame,
    params: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> pd.DataFrame:
    context = context or {}
    cfg = FundamentalsFeatureConfig.from_params(params)

    work, ts_col = _prepare_base_frame(df)
    out = pd.DataFrame(index=work.index)

    fundamentals_df = context.get("fundamentals_df")
    if fundamentals_df is not None and not fundamentals_df.empty:
        release_features = _prepare_fundamentals_release_frame(fundamentals_df, cfg)
        if not release_features.empty:
            left = pd.DataFrame({"timestamp": work[ts_col]})
            merged = pd.merge_asof(
                left.sort_values("timestamp"),
                release_features.sort_values("__release_timestamp__"),
                left_on="timestamp",
                right_on="__release_timestamp__",
                direction="backward",
                allow_exact_matches=True,
            ).reset_index(drop=True)

            for col in release_features.columns:
                if col == "__release_timestamp__":
                    continue
                out[col] = pd.to_numeric(merged[col], errors="coerce")

            out["fund_last_release_timestamp"] = pd.to_datetime(
                merged["__release_timestamp__"],
                utc=True,
                errors="coerce",
            )
            out["fund_days_since_release"] = (
                (left["timestamp"] - out["fund_last_release_timestamp"]).dt.total_seconds() / 86400.0
            )

    earnings_df = context.get("earnings_df")
    earnings_params = params.get("earnings", {})
    if earnings_df is not None and not earnings_df.empty:
        earnings_features = generate_earnings_event_features(
            work[[ts_col]].rename(columns={ts_col: "timestamp"}),
            earnings_df=earnings_df,
            params=earnings_params,
        )
        if not earnings_features.empty:
            out = pd.concat([out, earnings_features.reset_index(drop=True)], axis=1)

    return _restore_output_order(out, work, df.index)


def generate_fundamentals_features(
    price_df: pd.DataFrame,
    *,
    fundamentals_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
    params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    context = {
        "fundamentals_df": fundamentals_df,
        "earnings_df": earnings_df,
    }
    return _fundamentals_features_plugin(price_df, params or {}, context)


def register_fundamentals_plugins(registry: FeatureRegistry) -> None:
    registry.register(
        FeaturePlugin(
            name="fundamentals",
            generator=_fundamentals_features_plugin,
            group="fundamentals",
            description="Release-timed fundamentals + earnings event features.",
        )
    )

