"""
Unified dataset builder for market + optional feature connectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .alignment import align_feature_frames, resample_ohlcv
from .connectors import (
    EmptyFundamentalsConnector,
    EmptyMacroConnector,
    EmptyNewsConnector,
    FundamentalsConnector,
    MacroSeriesConnector,
    NewsSentimentConnector,
    PriceConnector,
    YFinancePriceConnector,
)
from .quality import DataQualityGate, QualityPolicy
from .schema import apply_split_dividend_adjustments, canonicalize_price_frame


@dataclass
class DatasetBuildRequest:
    symbol: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    days: Optional[int] = 365
    interval: str = "1d"
    timeframe: Optional[str] = None
    include_prepost: bool = False
    adjust_prices: bool = True
    horizon_days: int = 5
    run_id: Optional[str] = None
    report_dir: Optional[Path | str] = None


class UnifiedDatasetBuilder:
    """
    Build canonical, quality-gated, causally aligned training datasets.
    """

    def __init__(
        self,
        *,
        price_connector: Optional[PriceConnector] = None,
        fundamentals_connector: Optional[FundamentalsConnector] = None,
        macro_connector: Optional[MacroSeriesConnector] = None,
        news_connector: Optional[NewsSentimentConnector] = None,
        quality_gate: Optional[DataQualityGate] = None,
    ):
        self.price_connector = price_connector or YFinancePriceConnector()
        self.fundamentals_connector = fundamentals_connector or EmptyFundamentalsConnector()
        self.macro_connector = macro_connector or EmptyMacroConnector()
        self.news_connector = news_connector or EmptyNewsConnector()
        self.quality_gate = quality_gate or DataQualityGate(QualityPolicy())

    def build(self, request: DatasetBuildRequest) -> tuple[pd.DataFrame, dict, Path]:
        symbol = request.symbol.upper().strip()
        raw_price = self.price_connector.fetch_historical(
            symbol,
            start=request.start,
            end=request.end,
            days=request.days,
            interval=request.interval,
            include_prepost=request.include_prepost,
        )
        canonical = canonicalize_price_frame(
            raw_price,
            intraday=not request.interval.lower().endswith("d"),
        )

        if request.adjust_prices:
            canonical = apply_split_dividend_adjustments(canonical)

        clean_price, quality_report, report_path = self.quality_gate.run(
            canonical,
            symbol=symbol,
            interval=request.interval,
            horizon_days=request.horizon_days,
            run_id=request.run_id,
            report_dir=request.report_dir,
        )

        if request.timeframe:
            clean_price = resample_ohlcv(
                clean_price,
                timeframe=request.timeframe,
                session=None,
            )

        fundamentals_df = self.fundamentals_connector.fetch_fundamentals(
            symbol,
            start=request.start,
            end=request.end,
        )
        macro_df = self.macro_connector.fetch_macro(start=request.start, end=request.end)
        news_df = self.news_connector.fetch_sentiment(symbol, start=request.start, end=request.end)

        dataset = align_feature_frames(
            clean_price,
            fundamentals_df=fundamentals_df,
            macro_df=macro_df,
            news_df=news_df,
        )

        return dataset.reset_index(drop=True), quality_report, report_path
