"""
Sentiment feature plugin wrapper around lightweight sentiment pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from sentiment.pipeline import (
    SentimentPipelineConfig,
    aggregate_sentiment_features,
    process_sentiment_stream,
)
from .registry import FeaturePlugin, FeatureRegistry


def _timestamp_col(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "date" in df.columns:
        return "date"
    raise ValueError("Input DataFrame must include 'timestamp' or 'date'")


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


def _sentiment_features_plugin(
    df: pd.DataFrame,
    params: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> pd.DataFrame:
    context = context or {}
    cfg = SentimentPipelineConfig.from_params(params)
    work, ts_col = _prepare_base_frame(df)

    headlines_df = context.get("headlines_df")
    if headlines_df is None:
        headlines_df = context.get("news_headlines_df")
    if headlines_df is None:
        headlines_df = context.get("news_df")

    social_df = context.get("social_df")
    if social_df is None:
        social_df = context.get("social_posts_df")

    symbol = params.get("symbol")
    if symbol is None:
        symbol = context.get("symbol")

    ticker_universe = params.get("ticker_universe")
    if ticker_universe is None:
        ticker_universe = context.get("ticker_universe")
    if ticker_universe is not None:
        ticker_universe = {str(x).upper() for x in ticker_universe}

    events_df, _meta = process_sentiment_stream(
        headlines_df=headlines_df,
        social_df=social_df,
        config=cfg,
        ticker_universe=ticker_universe,
    )
    if events_df.empty:
        return pd.DataFrame(index=df.index)

    agg = aggregate_sentiment_features(
        events_df,
        timeline=work[ts_col],
        symbol=symbol,
        config=cfg,
    )
    return _restore_output_order(agg, work, df.index)


def generate_sentiment_features(
    price_df: pd.DataFrame,
    *,
    headlines_df: Optional[pd.DataFrame] = None,
    social_df: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    ticker_universe: Optional[set[str]] = None,
    params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    context = {
        "headlines_df": headlines_df,
        "social_df": social_df,
        "symbol": symbol,
        "ticker_universe": ticker_universe,
    }
    return _sentiment_features_plugin(price_df, params or {}, context)


def register_sentiment_plugins(registry: FeatureRegistry) -> None:
    registry.register(
        FeaturePlugin(
            name="sentiment",
            generator=_sentiment_features_plugin,
            group="sentiment",
            description="Deterministic sentiment/attention aggregates with reliability weighting.",
        )
    )

