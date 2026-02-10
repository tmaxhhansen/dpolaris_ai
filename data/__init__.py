"""
Robust data layer for dPolaris AI.

This package provides:
- Canonical market schema (OHLCV + corporate actions + session metadata)
- Pluggable connectors
- Data quality gates and repair policies
- Causal alignment engine
- Unified dataset builder
"""

from .schema import (
    CANONICAL_PRICE_COLUMNS,
    apply_split_dividend_adjustments,
    canonicalize_price_frame,
)
from .quality import DataQualityGate, QualityPolicy
from .alignment import (
    causal_asof_join,
    parse_timeframe,
    resample_ohlcv,
)
from .dataset_builder import UnifiedDatasetBuilder

__all__ = [
    "CANONICAL_PRICE_COLUMNS",
    "canonicalize_price_frame",
    "apply_split_dividend_adjustments",
    "QualityPolicy",
    "DataQualityGate",
    "parse_timeframe",
    "resample_ohlcv",
    "causal_asof_join",
    "UnifiedDatasetBuilder",
]
