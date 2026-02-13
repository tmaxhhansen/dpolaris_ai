"""Daily build utilities for tradable universes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Optional, Sequence

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None


logger = logging.getLogger("dpolaris.universe")

DEFAULT_UNIVERSE_SCHEMA_VERSION = "1.0.0"
NASDAQ_TOP_FILE = "nasdaq_top_500.json"
WSB_TOP_FILE = "wsb_top_500.json"
COMBINED_TOP_FILE = "combined_1000.json"

_NASDAQ_SYMBOLS_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_REDDIT_WSB_NEW_URL = "https://www.reddit.com/r/wallstreetbets/new.json"

_TOKEN_RE = re.compile(r"(?:\$|\b)([A-Z]{1,5})\b")

_COMMON_NON_TICKERS = {
    "A",
    "AI",
    "ALL",
    "AM",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BEST",
    "BUY",
    "CEO",
    "CFO",
    "CPI",
    "DD",
    "EPS",
    "ETF",
    "FOMC",
    "FOR",
    "FROM",
    "GDP",
    "GO",
    "HOLD",
    "I",
    "IN",
    "IPO",
    "IT",
    "ITS",
    "LONG",
    "LOSS",
    "LOW",
    "MOON",
    "NEW",
    "NOW",
    "OF",
    "ON",
    "OR",
    "PUMP",
    "PUT",
    "RIP",
    "SEC",
    "SELL",
    "SHORT",
    "SO",
    "THE",
    "THIS",
    "TO",
    "USD",
    "VIX",
    "WE",
    "WITH",
    "YOU",
}

_POSITIVE_WORDS = {
    "beat",
    "beats",
    "bull",
    "bullish",
    "buy",
    "calls",
    "gain",
    "gains",
    "green",
    "long",
    "moon",
    "pump",
    "rally",
    "rip",
    "rocket",
    "strong",
    "up",
    "win",
}

_NEGATIVE_WORDS = {
    "bagholder",
    "bear",
    "bearish",
    "crash",
    "down",
    "dump",
    "loss",
    "losses",
    "miss",
    "missed",
    "puts",
    "red",
    "short",
    "weak",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_universe_dir() -> Path:
    raw = os.getenv("DPOLARIS_UNIVERSE_DIR", "universe")
    out = Path(raw).expanduser()
    if not out.is_absolute():
        out = _repo_root() / out
    return out


def _utc_now(now: Optional[datetime] = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _iso(ts: datetime) -> str:
    return _utc_now(ts).isoformat()


def _sanitize_symbol(symbol: Any) -> Optional[str]:
    if symbol is None:
        return None
    text = str(symbol).upper().strip()
    if not text:
        return None
    if any(ch in text for ch in (" ", "^", "/", "\\", "=")):
        return None
    text = text.replace(".", "-")
    if not re.match(r"^[A-Z][A-Z0-9\-]{0,9}$", text):
        return None
    return text


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _json_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _with_hash(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("universe_hash", None)
    body["universe_hash"] = _json_hash(body)
    return body


def _normalize_symbol_rows(symbol_rows: Sequence[dict[str, Any] | str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in symbol_rows:
        if isinstance(row, str):
            sym = _sanitize_symbol(row)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(
                {
                    "symbol": sym,
                    "company_name": sym,
                    "sector": None,
                    "industry": None,
                }
            )
            continue

        sym = _sanitize_symbol(row.get("symbol"))
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(
            {
                "symbol": sym,
                "company_name": row.get("company_name") or row.get("name") or sym,
                "sector": row.get("sector"),
                "industry": row.get("industry"),
            }
        )

    return out


def _fetch_nasdaq_symbols_default() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return (rows, data_sources, notes)."""
    notes: list[str] = []
    data_sources: list[dict[str, Any]] = []

    if httpx is None:
        notes.append("httpx unavailable; using fallback symbols")
        return _fallback_symbols(), _fallback_data_sources(), notes

    try:
        headers = {"User-Agent": "dpolaris-universe-builder/1.0"}
        response = httpx.get(_NASDAQ_SYMBOLS_URL, headers=headers, timeout=20.0)
        response.raise_for_status()
        rows: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            if "|" not in line or line.startswith("Symbol|"):
                continue
            if line.startswith("File Creation Time"):
                continue
            parts = line.split("|")
            if len(parts) < 8:
                continue
            symbol = _sanitize_symbol(parts[0])
            if not symbol:
                continue
            test_issue = (parts[6] or "").strip().upper()
            if test_issue == "Y":
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "company_name": (parts[1] or symbol).strip(),
                    "sector": None,
                    "industry": None,
                }
            )
        if rows:
            data_sources.append(
                {
                    "name": "nasdaqtrader_symbol_directory",
                    "url": _NASDAQ_SYMBOLS_URL,
                    "rows": len(rows),
                }
            )
            return rows, data_sources, notes
    except Exception as exc:  # pragma: no cover - network dependent
        notes.append(f"nasdaq symbol fetch failed: {exc}")

    rows = _fallback_symbols()
    data_sources.extend(_fallback_data_sources())
    return rows, data_sources, notes


def _fallback_symbols() -> list[dict[str, Any]]:
    fallback = os.getenv(
        "DPOLARIS_FALLBACK_SYMBOLS",
        "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AVGO,COST,NFLX,AMD,INTC,ADBE,CSCO,PEP",
    )
    return _normalize_symbol_rows([s.strip() for s in fallback.split(",") if s.strip()])


def _fallback_data_sources() -> list[dict[str, Any]]:
    return [{"name": "env_fallback_symbols", "notes": "DPOLARIS_FALLBACK_SYMBOLS"}]


def _fetch_symbol_profile_yfinance(symbol: str) -> dict[str, Any]:
    if yf is None:
        return {
            "symbol": symbol,
            "company_name": symbol,
            "market_cap": None,
            "avg_dollar_volume": None,
            "sector": None,
            "industry": None,
        }

    ticker = yf.Ticker(symbol)
    info: dict[str, Any] = {}
    fast_info: dict[str, Any] = {}

    try:
        fi = getattr(ticker, "fast_info", None)
        if fi is not None:
            fast_info = dict(fi)
    except Exception:
        fast_info = {}

    try:
        info = dict(ticker.info or {})
    except Exception:
        info = {}

    price = (
        _to_float(info.get("regularMarketPrice"))
        or _to_float(info.get("currentPrice"))
        or _to_float(fast_info.get("lastPrice"))
    )

    avg_volume = (
        _to_float(info.get("averageVolume"))
        or _to_float(fast_info.get("threeMonthAverageVolume"))
        or _to_float(fast_info.get("tenDayAverageVolume"))
    )

    avg_dollar_volume: Optional[float] = None
    if avg_volume and price:
        avg_dollar_volume = avg_volume * price

    if avg_dollar_volume is None:
        try:
            hist = ticker.history(period="3mo", interval="1d", auto_adjust=False)
            if not hist.empty and "Close" in hist.columns and "Volume" in hist.columns:
                avg_dollar_volume = _to_float((hist["Close"] * hist["Volume"]).tail(63).mean())
        except Exception:
            avg_dollar_volume = None

    return {
        "symbol": symbol,
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "market_cap": _to_float(info.get("marketCap") or fast_info.get("marketCap")),
        "avg_dollar_volume": avg_dollar_volume,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


def build_nasdaq_top_500(
    *,
    output_path: Optional[str | Path] = None,
    top_n: int = 500,
    min_avg_dollar_volume: float = 10_000_000.0,
    candidate_limit: int = 2000,
    symbol_rows: Optional[Sequence[dict[str, Any] | str]] = None,
    symbol_fetcher: Optional[Callable[[], tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]]] = None,
    profile_fetcher: Optional[Callable[[str], dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    generated_at = _utc_now(now)

    fetch_symbols = symbol_fetcher or _fetch_nasdaq_symbols_default
    fetch_profile = profile_fetcher or _fetch_symbol_profile_yfinance

    data_sources: list[dict[str, Any]] = []
    notes: list[str] = []

    if symbol_rows is None:
        rows, source_rows, fetch_notes = fetch_symbols()
        data_sources.extend(source_rows)
        notes.extend(fetch_notes)
    else:
        rows = _normalize_symbol_rows(symbol_rows)
        data_sources.append({"name": "provided_symbol_rows", "rows": len(rows)})

    if candidate_limit > 0:
        rows = rows[:candidate_limit]

    enriched: list[dict[str, Any]] = []
    for row in rows:
        symbol = row["symbol"]
        profile = fetch_profile(symbol) or {}
        merged = {
            "symbol": symbol,
            "company_name": profile.get("company_name") or row.get("company_name") or symbol,
            "market_cap": _to_float(profile.get("market_cap")),
            "avg_dollar_volume": _to_float(profile.get("avg_dollar_volume")),
            "sector": profile.get("sector") or row.get("sector"),
            "industry": profile.get("industry") or row.get("industry"),
        }
        enriched.append(merged)

    data_sources.append({"name": "yfinance_security_profile", "rows": len(enriched)})

    ranking_mode = "market_cap_desc_then_avg_dollar_volume_desc"
    if not any((row.get("market_cap") or 0.0) > 0.0 for row in enriched):
        ranking_mode = "avg_dollar_volume_desc_fallback"
        notes.append("market cap unavailable; used avg dollar volume ranking fallback")

    liquid = [
        row
        for row in enriched
        if (row.get("avg_dollar_volume") is not None and row.get("avg_dollar_volume", 0.0) >= min_avg_dollar_volume)
    ]
    if len(liquid) < top_n:
        notes.append(
            "liquidity filter returned fewer than requested; filling with highest-ranked remaining symbols"
        )
        liquid_symbols = {row["symbol"] for row in liquid}
        remainder = [row for row in enriched if row["symbol"] not in liquid_symbols]
        liquid.extend(remainder)

    if ranking_mode == "avg_dollar_volume_desc_fallback":
        ordered = sorted(liquid, key=lambda r: (r.get("avg_dollar_volume") or 0.0), reverse=True)
    else:
        ordered = sorted(
            liquid,
            key=lambda r: (
                r.get("market_cap") or -1.0,
                r.get("avg_dollar_volume") or -1.0,
                r.get("symbol") or "",
            ),
            reverse=True,
        )

    selected = ordered[: max(0, int(top_n))]

    payload: dict[str, Any] = {
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "criteria": {
            "exchange": "NASDAQ",
            "top_n_requested": int(top_n),
            "top_n_returned": len(selected),
            "ranking": ranking_mode,
            "liquidity_filter": {
                "metric": "average_dollar_volume",
                "min_avg_dollar_volume": float(min_avg_dollar_volume),
            },
            "candidate_limit": int(candidate_limit),
        },
        "data_sources": data_sources,
        "notes": notes,
        "tickers": selected,
    }
    payload = _with_hash(payload)

    out_path = Path(output_path) if output_path is not None else (_default_universe_dir() / NASDAQ_TOP_FILE)
    _write_json(out_path, payload)
    return payload


def _normalize_post_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _score_sentiment(text: str) -> float:
    normalized = _normalize_post_text(text)
    if not normalized:
        return 0.0
    tokens = normalized.split()
    pos = sum(1 for tok in tokens if tok in _POSITIVE_WORDS)
    neg = sum(1 for tok in tokens if tok in _NEGATIVE_WORDS)
    return float(pos - neg) / float(max(1, len(tokens)))


def _extract_tickers(text: str, valid_tickers: Optional[set[str]]) -> list[str]:
    matches: set[str] = set()
    for raw in _TOKEN_RE.findall((text or "").upper()):
        symbol = _sanitize_symbol(raw)
        if not symbol:
            continue
        if symbol in _COMMON_NON_TICKERS:
            continue
        if valid_tickers is not None and symbol not in valid_tickers:
            continue
        matches.add(symbol)
    return sorted(matches)


def _fetch_wsb_posts_default(
    *,
    window_start: datetime,
    window_end: datetime,
    max_posts: int = 2000,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    notes: list[str] = []

    if httpx is None:
        notes.append("httpx unavailable; no WSB posts fetched")
        return [], "none", notes

    headers = {"User-Agent": "dpolaris-universe-builder/1.0"}
    posts: list[dict[str, Any]] = []
    after: Optional[str] = None

    try:
        while len(posts) < max_posts:
            params: dict[str, Any] = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after
            response = httpx.get(_REDDIT_WSB_NEW_URL, headers=headers, params=params, timeout=20.0)
            response.raise_for_status()
            data = response.json() or {}
            children = ((data.get("data") or {}).get("children") or [])
            if not children:
                break

            oldest_in_page: Optional[datetime] = None
            for child in children:
                payload = child.get("data") or {}
                created = datetime.fromtimestamp(float(payload.get("created_utc", 0.0)), tz=timezone.utc)
                oldest_in_page = created if oldest_in_page is None else min(oldest_in_page, created)

                if created > window_end:
                    continue
                if created < window_start:
                    continue

                posts.append(
                    {
                        "id": payload.get("id"),
                        "title": payload.get("title") or "",
                        "body": payload.get("selftext") or "",
                        "created_at": created.isoformat(),
                    }
                )
                if len(posts) >= max_posts:
                    break

            after = ((data.get("data") or {}).get("after"))
            if not after:
                break
            if oldest_in_page is not None and oldest_in_page < window_start:
                break

    except Exception as exc:  # pragma: no cover - network dependent
        notes.append(f"reddit fetch failed: {exc}")

    return posts, "reddit_new_json", notes


def build_wsb_top_500(
    *,
    output_path: Optional[str | Path] = None,
    top_n: int = 500,
    window_days: int = 7,
    posts: Optional[Sequence[dict[str, Any]]] = None,
    post_fetcher: Optional[Callable[..., tuple[list[dict[str, Any]], str, list[str]]]] = None,
    valid_tickers: Optional[set[str]] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    generated_at = _utc_now(now)
    window_end = generated_at
    window_start = generated_at - timedelta(days=max(1, int(window_days)))

    fetcher = post_fetcher or _fetch_wsb_posts_default

    notes: list[str] = []
    source_connector = "provided_posts"
    if posts is None:
        fetched_posts, source_connector, fetch_notes = fetcher(
            window_start=window_start,
            window_end=window_end,
            max_posts=4000,
        )
        notes.extend(fetch_notes)
        raw_posts = fetched_posts
    else:
        raw_posts = list(posts)

    seen_texts: set[str] = set()
    mention_counter: dict[str, int] = {}
    sentiment_sums: dict[str, float] = {}
    example_posts: dict[str, list[dict[str, Any]]] = {}

    total_posts = 0
    deduped_posts = 0

    for post in raw_posts:
        title = str(post.get("title") or "")
        body = str(post.get("body") or post.get("selftext") or "")
        content = (title + " " + body).strip()
        if not content:
            continue
        total_posts += 1

        key = _normalize_post_text(content)
        if not key:
            continue
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped_posts += 1

        tickers = _extract_tickers(content, valid_tickers)
        if not tickers:
            continue

        sentiment = _score_sentiment(content)
        post_id = str(post.get("id") or "")

        for ticker in tickers:
            mention_counter[ticker] = mention_counter.get(ticker, 0) + 1
            sentiment_sums[ticker] = sentiment_sums.get(ticker, 0.0) + sentiment

            examples = example_posts.setdefault(ticker, [])
            if len(examples) < 3:
                examples.append(
                    {
                        "post_id": post_id,
                        "title": title[:160],
                    }
                )

    denominator_days = float(max(1, int(window_days)))

    ranking: list[dict[str, Any]] = []
    for ticker, mentions in mention_counter.items():
        avg_sentiment = sentiment_sums.get(ticker, 0.0) / float(max(1, mentions))
        examples = example_posts.get(ticker, [])
        ranking.append(
            {
                "symbol": ticker,
                "mention_count": int(mentions),
                "mention_velocity": float(mentions) / denominator_days,
                "sentiment_score": round(float(avg_sentiment), 6),
                "example_post_ids": [x.get("post_id") for x in examples if x.get("post_id")],
                "example_titles": [x.get("title") for x in examples if x.get("title")],
            }
        )

    ranking.sort(
        key=lambda row: (
            row.get("mention_count", 0),
            row.get("mention_velocity", 0.0),
            row.get("symbol", ""),
        ),
        reverse=True,
    )
    selected = ranking[: max(0, int(top_n))]

    payload: dict[str, Any] = {
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "window_start": _iso(window_start),
        "window_end": _iso(window_end),
        "criteria": {
            "top_n_requested": int(top_n),
            "top_n_returned": len(selected),
            "window_days": int(window_days),
            "dedupe": "normalized_headline_and_body",
            "ticker_linking": "regex_with_stopword_filter",
        },
        "source_connector": source_connector,
        "data_sources": [
            {
                "name": source_connector,
                "raw_posts": int(total_posts),
                "deduped_posts": int(deduped_posts),
            }
        ],
        "notes": notes,
        "tickers": selected,
    }
    payload = _with_hash(payload)

    out_path = Path(output_path) if output_path is not None else (_default_universe_dir() / WSB_TOP_FILE)
    _write_json(out_path, payload)
    return payload


def _load_universe(path: Path) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid universe payload: {path}")
    return payload


def build_combined_universe(
    *,
    output_path: Optional[str | Path] = None,
    nasdaq_payload: Optional[dict[str, Any]] = None,
    wsb_payload: Optional[dict[str, Any]] = None,
    nasdaq_path: Optional[str | Path] = None,
    wsb_path: Optional[str | Path] = None,
    top_n: int = 1000,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    generated_at = _utc_now(now)

    universe_dir = _default_universe_dir()
    ns_path = Path(nasdaq_path) if nasdaq_path is not None else (universe_dir / NASDAQ_TOP_FILE)
    ws_path = Path(wsb_path) if wsb_path is not None else (universe_dir / WSB_TOP_FILE)

    ns_payload = dict(nasdaq_payload or _load_universe(ns_path))
    ws_payload = dict(wsb_payload or _load_universe(ws_path))

    ns_rows = list(ns_payload.get("tickers") or [])
    ws_rows = list(ws_payload.get("tickers") or [])

    merged: dict[str, dict[str, Any]] = {}

    for row in ns_rows:
        symbol = _sanitize_symbol(row.get("symbol"))
        if not symbol:
            continue
        merged[symbol] = {
            "symbol": symbol,
            "company_name": row.get("company_name") or symbol,
            "market_cap": _to_float(row.get("market_cap")),
            "avg_dollar_volume": _to_float(row.get("avg_dollar_volume")),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "mention_count": 0,
            "mention_velocity": 0.0,
            "sentiment_score": None,
            "sources": ["nasdaq_top_500"],
        }

    for row in ws_rows:
        symbol = _sanitize_symbol(row.get("symbol"))
        if not symbol:
            continue
        item = merged.setdefault(
            symbol,
            {
                "symbol": symbol,
                "company_name": symbol,
                "market_cap": None,
                "avg_dollar_volume": None,
                "sector": None,
                "industry": None,
                "mention_count": 0,
                "mention_velocity": 0.0,
                "sentiment_score": None,
                "sources": [],
            },
        )
        item["mention_count"] = int(row.get("mention_count") or 0)
        item["mention_velocity"] = float(row.get("mention_velocity") or 0.0)
        if row.get("sentiment_score") is not None:
            item["sentiment_score"] = float(row.get("sentiment_score"))
        if "wsb_top_500" not in item["sources"]:
            item["sources"].append("wsb_top_500")

    merged_rows = list(merged.values())
    merged_rows.sort(
        key=lambda row: (
            2 if len(row.get("sources", [])) == 2 else 1,
            row.get("market_cap") or -1.0,
            row.get("mention_count") or -1,
            row.get("avg_dollar_volume") or -1.0,
            row.get("symbol") or "",
        ),
        reverse=True,
    )
    merged_rows = merged_rows[: max(0, int(top_n))]

    payload: dict[str, Any] = {
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "criteria": {
            "top_n_requested": int(top_n),
            "top_n_returned": len(merged_rows),
            "duplicate_resolution": "merge_by_symbol_union_sources",
        },
        "data_sources": [
            {
                "name": "nasdaq_top_500",
                "path": str(ns_path),
                "universe_hash": ns_payload.get("universe_hash"),
                "count": len(ns_rows),
            },
            {
                "name": "wsb_top_500",
                "path": str(ws_path),
                "universe_hash": ws_payload.get("universe_hash"),
                "count": len(ws_rows),
            },
        ],
        "nasdaq_top_500": ns_rows,
        "wsb_top_500": ws_rows,
        "merged": merged_rows,
    }
    payload = _with_hash(payload)

    out_path = Path(output_path) if output_path is not None else (universe_dir / COMBINED_TOP_FILE)
    _write_json(out_path, payload)
    return payload


def build_daily_universe_files(
    *,
    universe_dir: Optional[str | Path] = None,
    now: Optional[datetime] = None,
    nasdaq_count: int = 500,
    wsb_count: int = 500,
    wsb_window_days: int = 7,
    min_avg_dollar_volume: float = 10_000_000.0,
) -> dict[str, Any]:
    base_dir = Path(universe_dir).expanduser() if universe_dir is not None else _default_universe_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = _utc_now(now)
    nasdaq_path = base_dir / NASDAQ_TOP_FILE
    wsb_path = base_dir / WSB_TOP_FILE
    combined_path = base_dir / COMBINED_TOP_FILE

    nasdaq_payload = build_nasdaq_top_500(
        output_path=nasdaq_path,
        top_n=nasdaq_count,
        min_avg_dollar_volume=min_avg_dollar_volume,
        now=ts,
    )
    wsb_payload = build_wsb_top_500(
        output_path=wsb_path,
        top_n=wsb_count,
        window_days=wsb_window_days,
        valid_tickers=None,
        now=ts,
    )
    combined_payload = build_combined_universe(
        output_path=combined_path,
        nasdaq_payload=nasdaq_payload,
        wsb_payload=wsb_payload,
        top_n=nasdaq_count + wsb_count,
        now=ts,
    )

    return {
        "generated_at": _iso(ts),
        "universe_dir": str(base_dir),
        "nasdaq_top_500": {
            "path": str(nasdaq_path),
            "universe_hash": nasdaq_payload.get("universe_hash"),
            "count": len(nasdaq_payload.get("tickers") or []),
        },
        "wsb_top_500": {
            "path": str(wsb_path),
            "universe_hash": wsb_payload.get("universe_hash"),
            "count": len(wsb_payload.get("tickers") or []),
        },
        "combined_1000": {
            "path": str(combined_path),
            "universe_hash": combined_payload.get("universe_hash"),
            "count": len(combined_payload.get("merged") or []),
        },
    }
