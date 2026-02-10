"""
Connector interfaces for market/fundamental/macro/news data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class PriceConnector(ABC):
    """Historical price connector interface."""

    @abstractmethod
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
        raise NotImplementedError


class FundamentalsConnector(ABC):
    """Optional fundamental-series connector interface."""

    @abstractmethod
    def fetch_fundamentals(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class MacroSeriesConnector(ABC):
    """Optional macro-series connector interface."""

    @abstractmethod
    def fetch_macro(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class NewsSentimentConnector(ABC):
    """Optional news/sentiment connector interface."""

    @abstractmethod
    def fetch_sentiment(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
