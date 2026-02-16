"""
News & Sentiment Module for dPolaris

Provides news aggregation and sentiment analysis for stock market analysis.
"""

try:
    from .sentiment import (
        NewsArticle,
        SentimentResult,
        NewsFetcher,
        SentimentAnalyzer,
        NewsEngine,
        get_news_sentiment,
        RSS_FEEDS,
    )
except Exception:  # pragma: no cover - optional dependencies may be missing
    NewsArticle = None
    SentimentResult = None
    NewsFetcher = None
    SentimentAnalyzer = None
    NewsEngine = None
    RSS_FEEDS = []

    def get_news_sentiment(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("News sentiment dependencies are unavailable")

__all__ = [
    "NewsArticle",
    "SentimentResult",
    "NewsFetcher",
    "SentimentAnalyzer",
    "NewsEngine",
    "get_news_sentiment",
    "RSS_FEEDS",
]
