"""Provider abstractions for universe-related external data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Protocol


logger = logging.getLogger("dpolaris.universe.providers")


class MentionsProvider(Protocol):
    """Return WSB-style posts for mention extraction."""

    def fetch_posts(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        max_posts: int,
    ) -> list[dict[str, Any]]:
        ...


@dataclass
class MentionsProviderContext:
    provider_name: str
    warnings: list[str]


class CachedMentionsProvider:
    """Load posts from a local JSON cache file."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            self.cache_path.write_text("[]\n", encoding="utf-8")

    def fetch_posts(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        max_posts: int,
    ) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []

        window_start_utc = _as_utc(window_start)
        window_end_utc = _as_utc(window_end)

        out: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            created_at = _parse_timestamp(item.get("created_at"))
            if created_at is None:
                continue
            if created_at < window_start_utc or created_at > window_end_utc:
                continue
            out.append(
                {
                    "id": item.get("id"),
                    "title": str(item.get("title") or ""),
                    "body": str(item.get("body") or item.get("selftext") or ""),
                    "created_at": created_at.isoformat(),
                }
            )
            if len(out) >= max_posts:
                break
        return out

    def persist_posts(self, posts: list[dict[str, Any]]) -> None:
        try:
            self.cache_path.write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Unable to persist mentions cache", exc_info=True)


class PrawMentionsProvider:
    """Use Reddit PRAW when credentials are configured."""

    def __init__(self, cache: CachedMentionsProvider):
        self.cache = cache

    def fetch_posts(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        max_posts: int,
    ) -> list[dict[str, Any]]:
        try:
            import praw  # type: ignore
        except Exception:
            return self.cache.fetch_posts(window_start=window_start, window_end=window_end, max_posts=max_posts)

        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "dpolaris-wsb-provider/1.0"),
            check_for_async=False,
        )

        start_utc = _as_utc(window_start)
        end_utc = _as_utc(window_end)

        posts: list[dict[str, Any]] = []
        try:
            subreddit = reddit.subreddit("wallstreetbets")
            # Pull a bounded window of recent submissions.
            for submission in subreddit.new(limit=max_posts * 2):
                created = datetime.fromtimestamp(float(submission.created_utc), tz=timezone.utc)
                if created < start_utc:
                    break
                if created > end_utc:
                    continue
                posts.append(
                    {
                        "id": getattr(submission, "id", None),
                        "title": str(getattr(submission, "title", "") or ""),
                        "body": str(getattr(submission, "selftext", "") or ""),
                        "created_at": created.isoformat(),
                    }
                )
                if len(posts) >= max_posts:
                    break
        except Exception:
            logger.warning("PRAW fetch failed; falling back to cached WSB mentions", exc_info=True)
            return self.cache.fetch_posts(window_start=window_start, window_end=window_end, max_posts=max_posts)

        if posts:
            self.cache.persist_posts(posts)
        return posts


def build_mentions_provider(data_dir: Path) -> tuple[MentionsProvider, MentionsProviderContext]:
    cache = CachedMentionsProvider((data_dir / "mentions" / "wsb_posts.json"))

    required_env = [
        os.getenv("REDDIT_CLIENT_ID", "").strip(),
        os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        os.getenv("REDDIT_USER_AGENT", "").strip(),
    ]
    if all(required_env):
        try:
            import praw  # type: ignore  # noqa: F401

            return PrawMentionsProvider(cache), MentionsProviderContext(provider_name="praw", warnings=[])
        except Exception:
            return (
                cache,
                MentionsProviderContext(
                    provider_name="cache",
                    warnings=["PRAW not installed; using cached WSB mentions file."],
                ),
            )

    return (
        cache,
        MentionsProviderContext(
            provider_name="cache",
            warnings=[
                "Reddit credentials not configured; using cached WSB mentions file.",
                "Set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET/REDDIT_USER_AGENT to enable live mentions.",
            ],
        ),
    )


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


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
