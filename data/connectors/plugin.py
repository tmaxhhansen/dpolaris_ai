"""
Pluggable in-memory and no-op connectors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .base import FundamentalsConnector, MacroSeriesConnector, NewsSentimentConnector


class FrameConnector(FundamentalsConnector, MacroSeriesConnector, NewsSentimentConnector):
    """
    Simple pluggable connector backed by a DataFrame.
    """

    def __init__(self, frame: Optional[pd.DataFrame] = None):
        self.frame = frame.copy() if frame is not None else pd.DataFrame(columns=["timestamp"])

    def _slice(self, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
        df = self.frame.copy()
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if start is not None:
                df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
        return df.reset_index(drop=True)

    def fetch_fundamentals(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        _ = symbol
        return self._slice(start, end)

    def fetch_macro(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        return self._slice(start, end)

    def fetch_sentiment(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        _ = symbol
        return self._slice(start, end)


class EmptyFundamentalsConnector(FundamentalsConnector):
    def fetch_fundamentals(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        _ = (symbol, start, end)
        return pd.DataFrame(columns=["timestamp"])


class EmptyMacroConnector(MacroSeriesConnector):
    def fetch_macro(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        _ = (start, end)
        return pd.DataFrame(columns=["timestamp"])


class EmptyNewsConnector(NewsSentimentConnector):
    def fetch_sentiment(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        _ = (symbol, start, end)
        return pd.DataFrame(columns=["timestamp"])
