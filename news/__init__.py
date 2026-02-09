"""
News & Sentiment Module for dPolaris

Provides news aggregation and sentiment analysis for stock market analysis.
"""

from .sentiment import (
    NewsArticle,
    SentimentResult,
    NewsFetcher,
    SentimentAnalyzer,
    NewsEngine,
    get_news_sentiment,
    RSS_FEEDS,
)

__all__ = [
    "NewsArticle",
    "SentimentResult",
    "NewsFetcher",
    "SentimentAnalyzer",
    "NewsEngine",
    "get_news_sentiment",
    "RSS_FEEDS",
]
