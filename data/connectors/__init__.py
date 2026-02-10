"""
Data connectors for unified dataset ingestion.
"""

from .base import (
    FundamentalsConnector,
    MacroSeriesConnector,
    NewsSentimentConnector,
    PriceConnector,
)
from .yfinance import YFinancePriceConnector, YFinanceFundamentalsConnector
from .plugin import (
    EmptyFundamentalsConnector,
    EmptyMacroConnector,
    EmptyNewsConnector,
    FrameConnector,
)

__all__ = [
    "PriceConnector",
    "FundamentalsConnector",
    "MacroSeriesConnector",
    "NewsSentimentConnector",
    "YFinancePriceConnector",
    "YFinanceFundamentalsConnector",
    "EmptyFundamentalsConnector",
    "EmptyMacroConnector",
    "EmptyNewsConnector",
    "FrameConnector",
]
