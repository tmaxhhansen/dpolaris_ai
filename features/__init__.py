"""
Feature library and plugin registry for price/volume behavior modeling.
"""

from .registry import FeatureRegistry, FeatureSpec, FeaturePlugin
from .fundamentals import (
    FundamentalsFeatureConfig,
    generate_fundamentals_features,
    register_fundamentals_plugins,
)
from .macro import MacroFeatureConfig, generate_macro_features, register_macro_plugins
from .sentiment import generate_sentiment_features, register_sentiment_plugins
from .technical import (
    FeatureScaler,
    TechnicalFeatureLibrary,
    build_default_registry,
)

__all__ = [
    "FeatureRegistry",
    "FeatureSpec",
    "FeaturePlugin",
    "FundamentalsFeatureConfig",
    "generate_fundamentals_features",
    "register_fundamentals_plugins",
    "MacroFeatureConfig",
    "generate_macro_features",
    "register_macro_plugins",
    "generate_sentiment_features",
    "register_sentiment_plugins",
    "FeatureScaler",
    "TechnicalFeatureLibrary",
    "build_default_registry",
]
