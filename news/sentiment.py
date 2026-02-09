"""
News & Sentiment Engine for dPolaris

Fetches news from multiple sources and analyzes sentiment.
Uses FinBERT for financial sentiment analysis.
"""

import asyncio
import feedparser
import httpx
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

logger = logging.getLogger("dpolaris.news")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    source: str
    url: str
    published: datetime
    summary: Optional[str] = None
    symbols: List[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return d


@dataclass
class SentimentResult:
    """Sentiment analysis result for a symbol"""
    symbol: str
    score: float  # -1 (bearish) to +1 (bullish)
    label: str  # BEARISH, NEUTRAL, BULLISH
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    trending: bool
    top_headlines: List[str]
    last_updated: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_updated"] = self.last_updated.isoformat()
        return d


# ============================================================================
# RSS Feed Sources
# ============================================================================

RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "reuters_business": "https://www.rss.reuters.com/news/businessNews",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
    "benzinga": "https://www.benzinga.com/feed",
}


# ============================================================================
# News Fetcher
# ============================================================================

class NewsFetcher:
    """Fetches news from multiple sources"""

    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.finnhub_api_key = finnhub_api_key
        self.cache_dir = cache_dir or Path("~/dpolaris_data/news_cache").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

    async def fetch_rss_feeds(self) -> List[NewsArticle]:
        """Fetch news from all RSS feeds"""
        articles = []

        for source_name, feed_url in RSS_FEEDS.items():
            try:
                response = await self.http_client.get(feed_url)
                feed = feedparser.parse(response.text)

                for entry in feed.entries[:20]:  # Limit per source
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6])
                    else:
                        published = datetime.now()

                    article = NewsArticle(
                        title=entry.get("title", ""),
                        source=source_name,
                        url=entry.get("link", ""),
                        published=published,
                        summary=entry.get("summary", "")[:500] if entry.get("summary") else None,
                        symbols=self._extract_symbols(entry.get("title", "")),
                    )
                    articles.append(article)

                logger.info(f"Fetched {len(feed.entries)} articles from {source_name}")

            except Exception as e:
                logger.warning(f"Failed to fetch {source_name}: {e}")

        return articles

    async def fetch_finnhub_news(
        self,
        symbols: Optional[List[str]] = None,
        category: str = "general",
    ) -> List[NewsArticle]:
        """Fetch news from Finnhub API"""
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not set")
            return []

        articles = []
        base_url = "https://finnhub.io/api/v1"

        try:
            if symbols:
                # Fetch company-specific news
                for symbol in symbols[:10]:  # Limit API calls
                    url = f"{base_url}/company-news"
                    params = {
                        "symbol": symbol,
                        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                        "to": datetime.now().strftime("%Y-%m-%d"),
                        "token": self.finnhub_api_key,
                    }

                    response = await self.http_client.get(url, params=params)
                    data = response.json()

                    for item in data[:10]:  # Limit per symbol
                        article = NewsArticle(
                            title=item.get("headline", ""),
                            source="finnhub",
                            url=item.get("url", ""),
                            published=datetime.fromtimestamp(item.get("datetime", 0)),
                            summary=item.get("summary", "")[:500],
                            symbols=[symbol],
                        )
                        articles.append(article)

                    await asyncio.sleep(0.1)  # Rate limiting

            else:
                # Fetch general market news
                url = f"{base_url}/news"
                params = {
                    "category": category,
                    "token": self.finnhub_api_key,
                }

                response = await self.http_client.get(url, params=params)
                data = response.json()

                for item in data[:20]:
                    article = NewsArticle(
                        title=item.get("headline", ""),
                        source="finnhub",
                        url=item.get("url", ""),
                        published=datetime.fromtimestamp(item.get("datetime", 0)),
                        summary=item.get("summary", "")[:500],
                        symbols=self._extract_symbols(item.get("headline", "")),
                    )
                    articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Finnhub")

        except Exception as e:
            logger.error(f"Failed to fetch Finnhub news: {e}")

        return articles

    async def fetch_all_news(
        self,
        symbols: Optional[List[str]] = None,
    ) -> List[NewsArticle]:
        """Fetch news from all sources"""
        rss_task = self.fetch_rss_feeds()
        finnhub_task = self.fetch_finnhub_news(symbols)

        rss_articles, finnhub_articles = await asyncio.gather(
            rss_task, finnhub_task
        )

        all_articles = rss_articles + finnhub_articles

        # Sort by date (newest first)
        all_articles.sort(key=lambda x: x.published or datetime.min, reverse=True)

        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_key = re.sub(r"[^a-z0-9]", "", article.title.lower())[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        logger.info(f"Total unique articles: {len(unique_articles)}")
        return unique_articles

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Common patterns: $AAPL, (AAPL), AAPL:
        patterns = [
            r"\$([A-Z]{1,5})\b",
            r"\(([A-Z]{1,5})\)",
            r"\b([A-Z]{2,5})(?::|'s|\s+stock)",
        ]

        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            symbols.update(matches)

        # Filter out common false positives
        common_words = {"CEO", "IPO", "NYSE", "SEC", "FDA", "USA", "GDP", "CPI", "ETF", "AI"}
        symbols = [s for s in symbols if s not in common_words and len(s) >= 2]

        return symbols[:5]  # Limit to 5 symbols per article


# ============================================================================
# Sentiment Analyzer
# ============================================================================

class SentimentAnalyzer:
    """
    Analyzes sentiment of financial news.

    Uses either:
    - FinBERT (if transformers installed) - most accurate
    - Simple keyword-based (fallback) - faster, less accurate
    """

    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert
        self.finbert_model = None
        self.finbert_tokenizer = None

        if use_finbert:
            self._load_finbert()

    def _load_finbert(self):
        """Load FinBERT model for sentiment analysis"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            logger.info("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"

            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.finbert_model = self.finbert_model.cuda()
                logger.info("FinBERT loaded on CUDA")
            else:
                logger.info("FinBERT loaded on CPU")

        except ImportError:
            logger.warning("transformers not installed, using keyword-based sentiment")
            self.use_finbert = False
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.use_finbert = False

    def analyze(self, text: str) -> tuple:
        """
        Analyze sentiment of text.

        Returns:
            (score, label) where score is -1 to 1 and label is BEARISH/NEUTRAL/BULLISH
        """
        if self.use_finbert and self.finbert_model:
            return self._analyze_finbert(text)
        else:
            return self._analyze_keywords(text)

    def _analyze_finbert(self, text: str) -> tuple:
        """Analyze sentiment using FinBERT"""
        import torch

        # Tokenize
        inputs = self.finbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to GPU if model is on GPU
        if next(self.finbert_model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # FinBERT outputs: [positive, negative, neutral]
        positive = probs[0][0].item()
        negative = probs[0][1].item()
        neutral = probs[0][2].item()

        # Convert to score (-1 to 1)
        score = positive - negative

        # Determine label
        if score > 0.2:
            label = "BULLISH"
        elif score < -0.2:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return score, label

    def _analyze_keywords(self, text: str) -> tuple:
        """Simple keyword-based sentiment analysis (fallback)"""
        text_lower = text.lower()

        positive_words = [
            "surge", "jump", "gain", "rise", "rally", "boom", "soar",
            "beat", "exceed", "strong", "bullish", "upgrade", "buy",
            "profit", "growth", "positive", "record", "high", "best",
            "breakthrough", "innovative", "success", "win", "optimistic",
        ]

        negative_words = [
            "fall", "drop", "decline", "plunge", "crash", "sink", "tumble",
            "miss", "weak", "bearish", "downgrade", "sell", "loss",
            "negative", "concern", "risk", "warning", "low", "worst",
            "layoff", "cut", "failure", "lawsuit", "investigation", "fraud",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0, "NEUTRAL"

        score = (positive_count - negative_count) / total

        if score > 0.3:
            label = "BULLISH"
        elif score < -0.3:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return score, label

    def analyze_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze sentiment for multiple articles"""
        for article in articles:
            text = f"{article.title}. {article.summary or ''}"
            score, label = self.analyze(text)
            article.sentiment_score = score
            article.sentiment_label = label

        return articles


# ============================================================================
# News & Sentiment Engine
# ============================================================================

class NewsEngine:
    """
    Main news and sentiment engine.

    Coordinates fetching, analysis, and aggregation of news sentiment.
    """

    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        use_finbert: bool = True,
        data_dir: Optional[Path] = None,
    ):
        self.data_dir = data_dir or Path("~/dpolaris_data").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.fetcher = NewsFetcher(finnhub_api_key=finnhub_api_key)
        self.analyzer = SentimentAnalyzer(use_finbert=use_finbert)

        self.articles: List[NewsArticle] = []
        self.sentiment_by_symbol: Dict[str, SentimentResult] = {}

    async def close(self):
        """Cleanup resources"""
        await self.fetcher.close()

    async def update(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, SentimentResult]:
        """
        Fetch latest news and update sentiment analysis.

        Args:
            symbols: List of symbols to focus on

        Returns:
            Dictionary of sentiment results by symbol
        """
        logger.info("Updating news and sentiment...")

        # Fetch news
        self.articles = await self.fetcher.fetch_all_news(symbols)

        # Analyze sentiment
        self.articles = self.analyzer.analyze_articles(self.articles)

        # Aggregate by symbol
        self.sentiment_by_symbol = self._aggregate_sentiment(symbols)

        # Save to disk
        self._save_results()

        logger.info(f"Updated sentiment for {len(self.sentiment_by_symbol)} symbols")
        return self.sentiment_by_symbol

    def _aggregate_sentiment(
        self,
        target_symbols: Optional[List[str]] = None,
    ) -> Dict[str, SentimentResult]:
        """Aggregate sentiment by symbol"""
        symbol_articles = defaultdict(list)

        # Group articles by symbol
        for article in self.articles:
            if article.symbols:
                for symbol in article.symbols:
                    if target_symbols is None or symbol in target_symbols:
                        symbol_articles[symbol].append(article)

        # Calculate aggregated sentiment
        results = {}
        for symbol, articles in symbol_articles.items():
            if not articles:
                continue

            scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
            if not scores:
                continue

            avg_score = sum(scores) / len(scores)

            positive_count = sum(1 for s in scores if s > 0.2)
            negative_count = sum(1 for s in scores if s < -0.2)
            neutral_count = len(scores) - positive_count - negative_count

            # Determine label
            if avg_score > 0.2:
                label = "BULLISH"
            elif avg_score < -0.2:
                label = "BEARISH"
            else:
                label = "NEUTRAL"

            # Check if trending (many articles)
            trending = len(articles) >= 5

            # Get top headlines
            sorted_articles = sorted(
                articles,
                key=lambda a: abs(a.sentiment_score or 0),
                reverse=True,
            )
            top_headlines = [a.title for a in sorted_articles[:5]]

            results[symbol] = SentimentResult(
                symbol=symbol,
                score=avg_score,
                label=label,
                article_count=len(articles),
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                trending=trending,
                top_headlines=top_headlines,
                last_updated=datetime.now(),
            )

        return results

    def _save_results(self):
        """Save results to disk"""
        # Save articles
        articles_path = self.data_dir / "news_articles.json"
        articles_data = [a.to_dict() for a in self.articles[:100]]  # Keep last 100
        with open(articles_path, "w") as f:
            json.dump(articles_data, f, indent=2, default=str)

        # Save sentiment
        sentiment_path = self.data_dir / "sentiment.json"
        sentiment_data = {k: v.to_dict() for k, v in self.sentiment_by_symbol.items()}
        with open(sentiment_path, "w") as f:
            json.dump(sentiment_data, f, indent=2)

        logger.info(f"Saved {len(self.articles)} articles and {len(self.sentiment_by_symbol)} sentiment results")

    def get_sentiment(self, symbol: str) -> Optional[SentimentResult]:
        """Get sentiment for a specific symbol"""
        return self.sentiment_by_symbol.get(symbol.upper())

    def get_market_sentiment(self) -> dict:
        """Get overall market sentiment summary"""
        if not self.sentiment_by_symbol:
            return {"status": "no_data"}

        scores = [s.score for s in self.sentiment_by_symbol.values()]
        avg_score = sum(scores) / len(scores)

        bullish_count = sum(1 for s in self.sentiment_by_symbol.values() if s.label == "BULLISH")
        bearish_count = sum(1 for s in self.sentiment_by_symbol.values() if s.label == "BEARISH")

        if avg_score > 0.1:
            label = "BULLISH"
        elif avg_score < -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return {
            "score": avg_score,
            "label": label,
            "symbols_analyzed": len(self.sentiment_by_symbol),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "last_updated": datetime.now().isoformat(),
        }

    def get_top_movers(self, n: int = 10) -> dict:
        """Get symbols with strongest sentiment (positive and negative)"""
        if not self.sentiment_by_symbol:
            return {"bullish": [], "bearish": []}

        sorted_by_score = sorted(
            self.sentiment_by_symbol.values(),
            key=lambda s: s.score,
            reverse=True,
        )

        return {
            "bullish": [s.to_dict() for s in sorted_by_score[:n] if s.score > 0],
            "bearish": [s.to_dict() for s in sorted_by_score[-n:] if s.score < 0],
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def get_news_sentiment(
    symbols: Optional[List[str]] = None,
    finnhub_api_key: Optional[str] = None,
    use_finbert: bool = True,
) -> Dict[str, SentimentResult]:
    """
    Quick function to get news sentiment.

    Usage:
        sentiment = await get_news_sentiment(["AAPL", "NVDA", "TSLA"])
    """
    engine = NewsEngine(
        finnhub_api_key=finnhub_api_key,
        use_finbert=use_finbert,
    )

    try:
        return await engine.update(symbols)
    finally:
        await engine.close()
