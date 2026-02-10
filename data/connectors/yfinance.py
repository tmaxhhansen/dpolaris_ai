"""
yfinance connectors for historical price and optional fundamentals.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import FundamentalsConnector, PriceConnector


try:
    import yfinance as yf

    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False


class YFinancePriceConnector(PriceConnector):
    """
    Fetch historical OHLCV + corporate actions from Yahoo Finance.
    """

    def fetch_historical(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None,
        interval: str = "1d",
        include_prepost: bool = False,
    ) -> pd.DataFrame:
        if not HAS_YFINANCE:
            raise RuntimeError("yfinance is not installed")

        ticker = yf.Ticker(symbol.upper().strip())
        if end is None:
            end = datetime.utcnow()
        if start is None:
            lookback = days if days is not None else 365
            start = end - timedelta(days=int(lookback))

        raw = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            actions=True,
            prepost=include_prepost,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.reset_index()
        rename_map = {}
        if "Date" in df.columns:
            rename_map["Date"] = "timestamp"
        if "Datetime" in df.columns:
            rename_map["Datetime"] = "timestamp"
        df = df.rename(columns=rename_map)
        return df


class YFinanceFundamentalsConnector(FundamentalsConnector):
    """
    Lightweight pluggable fundamentals connector.

    Emits snapshot rows with timestamp and selected fields.
    """

    DEFAULT_FIELDS = [
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "returnOnEquity",
        "debtToEquity",
    ]

    def __init__(self, fields: Optional[list[str]] = None):
        self.fields = fields or self.DEFAULT_FIELDS

    def fetch_fundamentals(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        _ = (start, end)
        if not HAS_YFINANCE:
            return pd.DataFrame(columns=["timestamp"])

        ticker = yf.Ticker(symbol.upper().strip())
        info = ticker.info or {}

        row = {"timestamp": pd.Timestamp.utcnow()}
        for field in self.fields:
            row[field] = info.get(field)

        return pd.DataFrame([row])
