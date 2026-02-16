"""Lightweight pluggable news providers with disk caching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import quote_plus

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


logger = logging.getLogger("dpolaris.news.providers")


class NewsProvider(Protocol):
    def fetch(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        ...


@dataclass
class NewsProviderContext:
    provider_name: str
    warnings: list[str]


class NoNewsProvider:
    def fetch(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        return []


class FinnhubNewsProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        if httpx is None:
            return []

        to_date = date.today()
        from_date = to_date - timedelta(days=7)
        url = (
            "https://finnhub.io/api/v1/company-news"
            f"?symbol={quote_plus(symbol)}&from={from_date.isoformat()}&to={to_date.isoformat()}&token={quote_plus(self.api_key)}"
        )
        try:
            response = httpx.get(url, timeout=20.0)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.warning("Finnhub fetch failed for %s", symbol, exc_info=True)
            return []

        if not isinstance(payload, list):
            return []

        out: list[dict[str, Any]] = []
        for item in payload[: max(1, int(limit))]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("headline") or "").strip()
            link = str(item.get("url") or "").strip()
            if not title or not link:
                continue
            published_at = _ts_from_unix(item.get("datetime"))
            out.append(
                {
                    "source": "finnhub",
                    "title": title,
                    "url": link,
                    "published_at": published_at,
                }
            )
        return out


class MarketauxNewsProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        if httpx is None:
            return []
        url = (
            "https://api.marketaux.com/v1/news/all"
            f"?symbols={quote_plus(symbol)}&filter_entities=true&limit={max(1, int(limit))}&api_token={quote_plus(self.api_key)}"
        )
        try:
            response = httpx.get(url, timeout=20.0)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.warning("Marketaux fetch failed for %s", symbol, exc_info=True)
            return []

        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []

        out: list[dict[str, Any]] = []
        for item in items[: max(1, int(limit))]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("url") or "").strip()
            if not title or not link:
                continue
            out.append(
                {
                    "source": str(item.get("source") or "marketaux"),
                    "title": title,
                    "url": link,
                    "published_at": str(item.get("published_at") or ""),
                }
            )
        return out


class NewsApiProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        if httpx is None:
            return []

        url = (
            "https://newsapi.org/v2/everything"
            f"?q={quote_plus(symbol)}&sortBy=publishedAt&pageSize={max(1, int(limit))}&apiKey={quote_plus(self.api_key)}"
        )
        try:
            response = httpx.get(url, timeout=20.0)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.warning("NewsAPI fetch failed for %s", symbol, exc_info=True)
            return []

        items = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []

        out: list[dict[str, Any]] = []
        for item in items[: max(1, int(limit))]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("url") or "").strip()
            if not title or not link:
                continue
            src_name = "newsapi"
            source_obj = item.get("source")
            if isinstance(source_obj, dict):
                src_name = str(source_obj.get("name") or src_name)
            out.append(
                {
                    "source": src_name,
                    "title": title,
                    "url": link,
                    "published_at": str(item.get("publishedAt") or ""),
                }
            )
        return out


def build_news_provider() -> tuple[NewsProvider, NewsProviderContext]:
    finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if finnhub_key:
        return FinnhubNewsProvider(finnhub_key), NewsProviderContext(provider_name="finnhub", warnings=[])

    marketaux_key = os.getenv("MARKETAUX_API_KEY", "").strip()
    if marketaux_key:
        return MarketauxNewsProvider(marketaux_key), NewsProviderContext(provider_name="marketaux", warnings=[])

    newsapi_key = os.getenv("NEWSAPI_API_KEY", "").strip()
    if newsapi_key:
        return NewsApiProvider(newsapi_key), NewsProviderContext(provider_name="newsapi", warnings=[])

    return (
        NoNewsProvider(),
        NewsProviderContext(
            provider_name="disabled",
            warnings=[
                "No news provider API key configured.",
                "Set FINNHUB_API_KEY, MARKETAUX_API_KEY, or NEWSAPI_API_KEY to enable live headlines.",
            ],
        ),
    )


def fetch_news_with_cache(
    *,
    symbol: str,
    limit: int,
    data_dir: Path,
    force_refresh: bool = False,
    cache_ttl_seconds: int = 60 * 30,
) -> tuple[list[dict[str, Any]], NewsProviderContext, dict[str, Any]]:
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return [], NewsProviderContext(provider_name="disabled", warnings=["Invalid symbol"]), {"cached": False}

    cache_path = data_dir / "news" / f"{normalized}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    if not force_refresh:
        cached_payload = _read_json(cache_path)
        if isinstance(cached_payload, dict):
            fetched_at = _parse_timestamp(cached_payload.get("fetched_at"))
            items = cached_payload.get("items")
            if fetched_at is not None and isinstance(items, list):
                age = (now - fetched_at).total_seconds()
                if age <= max(60, int(cache_ttl_seconds)):
                    return _slice_items(items, limit), NewsProviderContext(provider_name=str(cached_payload.get("provider") or "cache"), warnings=[]), {
                        "cached": True,
                        "cache_age_seconds": int(age),
                        "cache_path": str(cache_path),
                    }

    provider, context = build_news_provider()
    items = provider.fetch(normalized, max(1, int(limit)))
    normalized_items = _slice_items(items, limit)

    payload = {
        "symbol": normalized,
        "provider": context.provider_name,
        "fetched_at": now.isoformat(),
        "items": normalized_items,
    }
    try:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.debug("Unable to persist news cache for %s", normalized, exc_info=True)

    return normalized_items, context, {
        "cached": False,
        "cache_path": str(cache_path),
    }


def _slice_items(items: list[Any], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "source": str(item.get("source") or ""),
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "published_at": str(item.get("published_at") or ""),
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_timestamp(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if raw.endswith("Z"):
        candidates.append(raw[:-1] + "+00:00")
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except Exception:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _ts_from_unix(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return ""
