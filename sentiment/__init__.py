"""
Lightweight sentiment processing package.
"""

from .pipeline import (
    SentimentPipelineConfig,
    aggregate_sentiment_features,
    dedupe_near_identical,
    process_sentiment_stream,
)

__all__ = [
    "SentimentPipelineConfig",
    "dedupe_near_identical",
    "process_sentiment_stream",
    "aggregate_sentiment_features",
]

