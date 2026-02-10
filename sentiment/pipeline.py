"""
Lightweight deterministic sentiment pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


_DEFAULT_CREDIBILITY = {
    "reuters": 1.0,
    "wsj": 0.98,
    "bloomberg": 0.97,
    "financial_times": 0.95,
    "cnbc": 0.9,
    "marketwatch": 0.85,
    "yahoo_finance": 0.82,
    "seeking_alpha": 0.75,
    "benzinga": 0.72,
    "x": 0.5,
    "twitter": 0.5,
    "reddit": 0.55,
}

_POSITIVE_LEXICON = {
    "beat",
    "beats",
    "upgrade",
    "upgrades",
    "growth",
    "strong",
    "bullish",
    "surge",
    "surges",
    "record",
    "profit",
    "profits",
    "outperform",
    "outperformed",
    "expands",
    "expansion",
    "raise",
    "raises",
    "raised",
    "buyback",
}

_NEGATIVE_LEXICON = {
    "miss",
    "misses",
    "downgrade",
    "downgrades",
    "weak",
    "bearish",
    "drop",
    "drops",
    "plunge",
    "plunges",
    "lawsuit",
    "investigation",
    "fraud",
    "cut",
    "cuts",
    "cutting",
    "warns",
    "warning",
    "decline",
    "declines",
    "loss",
    "losses",
}

_TICKER_STOPWORDS = {
    "CEO",
    "CFO",
    "GDP",
    "CPI",
    "PPI",
    "FOMC",
    "ETF",
    "SEC",
    "FDA",
    "USA",
    "IPO",
    "AI",
}


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _normalize_text(text).split(" ") if tok]


def _jaccard_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return float(len(a_tokens & b_tokens) / len(union))


def _extract_tickers(text: str, ticker_universe: Optional[set[str]] = None) -> list[str]:
    raw = text or ""
    candidates: list[str] = []
    candidates.extend(re.findall(r"\$([A-Z]{1,5})\b", raw))
    candidates.extend(re.findall(r"\(([A-Z]{1,5})\)", raw))
    candidates.extend(re.findall(r"\b([A-Z]{2,5})\b", raw))

    out: list[str] = []
    for c in candidates:
        if c in _TICKER_STOPWORDS:
            continue
        if ticker_universe is not None and c not in ticker_universe:
            continue
        if c not in out:
            out.append(c)
    return out


def _score_lexicon_sentiment(text: str) -> tuple[float, str]:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0, "NEUTRAL"

    pos = sum(1 for t in tokens if t in _POSITIVE_LEXICON)
    neg = sum(1 for t in tokens if t in _NEGATIVE_LEXICON)
    raw = (pos - neg) / (len(tokens) + 2.0)
    score = float(np.clip(raw * 4.0, -1.0, 1.0))
    if score > 0.08:
        label = "BULLISH"
    elif score < -0.08:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return score, label


@dataclass
class SentimentPipelineConfig:
    dedupe_similarity_threshold: float = 0.9
    dedupe_time_window: str = "48H"
    windows: tuple[str, ...] = ("4H", "1D")
    spike_lookback: int = 20
    recency_half_life_minutes: float = 360.0
    source_credibility: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_CREDIBILITY))
    default_source_weight: float = 0.7
    social_weight_multiplier: float = 0.65
    include_unlinked: bool = True

    @classmethod
    def from_params(cls, params: Optional[dict[str, Any]] = None) -> "SentimentPipelineConfig":
        params = params or {}
        windows = params.get("windows", cls.windows)
        source_credibility = dict(_DEFAULT_CREDIBILITY)
        source_credibility.update(params.get("source_credibility", {}))
        return cls(
            dedupe_similarity_threshold=float(params.get("dedupe_similarity_threshold", cls.dedupe_similarity_threshold)),
            dedupe_time_window=str(params.get("dedupe_time_window", cls.dedupe_time_window)),
            windows=tuple(str(w) for w in windows),
            spike_lookback=int(params.get("spike_lookback", cls.spike_lookback)),
            recency_half_life_minutes=float(params.get("recency_half_life_minutes", cls.recency_half_life_minutes)),
            source_credibility=source_credibility,
            default_source_weight=float(params.get("default_source_weight", cls.default_source_weight)),
            social_weight_multiplier=float(params.get("social_weight_multiplier", cls.social_weight_multiplier)),
            include_unlinked=bool(params.get("include_unlinked", cls.include_unlinked)),
        )


def _canonicalize_input(
    headlines_df: Optional[pd.DataFrame],
    social_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if headlines_df is not None and not headlines_df.empty:
        h = headlines_df.copy()
        text_col = "headline" if "headline" in h.columns else "title" if "title" in h.columns else "text"
        source_col = "source" if "source" in h.columns else None
        ticker_col = "ticker" if "ticker" in h.columns else "symbol" if "symbol" in h.columns else None
        h = h.rename(columns={text_col: "text"})
        if source_col is not None:
            h = h.rename(columns={source_col: "source"})
        else:
            h["source"] = "news_unknown"
        if ticker_col is not None:
            h = h.rename(columns={ticker_col: "provided_ticker"})
        else:
            h["provided_ticker"] = np.nan
        h["channel"] = "news"
        rows.append(h[["timestamp", "text", "source", "provided_ticker", "channel"]].copy())

    if social_df is not None and not social_df.empty:
        s = social_df.copy()
        text_col = "text" if "text" in s.columns else "headline" if "headline" in s.columns else "title"
        source_col = "source" if "source" in s.columns else "platform" if "platform" in s.columns else None
        ticker_col = "ticker" if "ticker" in s.columns else "symbol" if "symbol" in s.columns else None
        s = s.rename(columns={text_col: "text"})
        if source_col is not None:
            s = s.rename(columns={source_col: "source"})
        else:
            s["source"] = "social_unknown"
        if ticker_col is not None:
            s = s.rename(columns={ticker_col: "provided_ticker"})
        else:
            s["provided_ticker"] = np.nan
        s["channel"] = "social"
        rows.append(s[["timestamp", "text", "source", "provided_ticker", "channel"]].copy())

    if not rows:
        return pd.DataFrame(columns=["timestamp", "text", "source", "provided_ticker", "channel"])

    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["text"] = out["text"].astype(str)
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    out["provided_ticker"] = out["provided_ticker"].astype(str).str.upper().replace({"NAN": np.nan, "NONE": np.nan})
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def dedupe_near_identical(
    records_df: pd.DataFrame,
    *,
    similarity_threshold: float = 0.9,
    time_window: str = "48H",
) -> tuple[pd.DataFrame, dict[str, int]]:
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=records_df.columns if records_df is not None else []), {
            "input_count": 0,
            "deduped_count": 0,
            "output_count": 0,
        }

    df = records_df.copy().sort_values("timestamp").reset_index(drop=True)
    dedupe_td = pd.Timedelta(str(time_window).lower())

    keep_mask = np.ones(len(df), dtype=bool)
    accepted: list[dict[str, Any]] = []

    for i, row in df.iterrows():
        text_norm = _normalize_text(row.get("text", ""))
        tokens = set(_tokenize(text_norm))
        ts = pd.to_datetime(row["timestamp"], utc=True)

        is_dup = False
        for prev in reversed(accepted):
            if ts - prev["timestamp"] > dedupe_td:
                break
            if text_norm == prev["text_norm"]:
                is_dup = True
                break
            sim = _jaccard_similarity(tokens, prev["tokens"])
            if sim >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            accepted.append({"timestamp": ts, "text_norm": text_norm, "tokens": tokens})
        else:
            keep_mask[i] = False

    out = df[keep_mask].reset_index(drop=True)
    report = {
        "input_count": int(len(df)),
        "deduped_count": int((~keep_mask).sum()),
        "output_count": int(len(out)),
    }
    return out, report


def process_sentiment_stream(
    *,
    headlines_df: Optional[pd.DataFrame],
    social_df: Optional[pd.DataFrame] = None,
    config: Optional[SentimentPipelineConfig] = None,
    ticker_universe: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or SentimentPipelineConfig()
    raw = _canonicalize_input(headlines_df, social_df)
    if raw.empty:
        return pd.DataFrame(columns=["timestamp"]), {"dedupe": {"input_count": 0, "deduped_count": 0, "output_count": 0}}

    deduped, dedupe_report = dedupe_near_identical(
        raw,
        similarity_threshold=cfg.dedupe_similarity_threshold,
        time_window=cfg.dedupe_time_window,
    )

    rows: list[dict[str, Any]] = []
    for _, row in deduped.iterrows():
        text = str(row["text"])
        linked = _extract_tickers(text, ticker_universe=ticker_universe)
        provided = row.get("provided_ticker")
        if isinstance(provided, str) and provided and provided != "nan" and provided != "None":
            if ticker_universe is None or provided in ticker_universe:
                if provided not in linked:
                    linked.insert(0, provided)

        score, label = _score_lexicon_sentiment(text)
        source = str(row["source"]).lower()
        base_weight = float(cfg.source_credibility.get(source, cfg.default_source_weight))
        if row.get("channel") == "social":
            base_weight *= cfg.social_weight_multiplier

        rows.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"], utc=True),
                "text": text,
                "source": source,
                "channel": row.get("channel", "news"),
                "sentiment_score": score,
                "sentiment_label": label,
                "linked_tickers": linked,
                "primary_ticker": linked[0] if linked else np.nan,
                "source_weight": base_weight,
            }
        )

    events = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    meta = {"dedupe": dedupe_report, "event_count": int(len(events))}
    return events, meta


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        return np.nan, np.nan
    mean = float(np.sum(values * weights) / w_sum)
    var = float(np.sum(weights * ((values - mean) ** 2)) / w_sum)
    return mean, float(np.sqrt(max(var, 0.0)))


def aggregate_sentiment_features(
    events_df: pd.DataFrame,
    *,
    timeline: pd.Series,
    symbol: Optional[str] = None,
    config: Optional[SentimentPipelineConfig] = None,
) -> pd.DataFrame:
    cfg = config or SentimentPipelineConfig()
    ts = pd.to_datetime(timeline, utc=True, errors="coerce")
    out = pd.DataFrame(index=np.arange(len(ts)))

    if events_df is None or events_df.empty:
        for w in cfg.windows:
            out[f"sent_sentiment_mean_w{w}"] = np.nan
            out[f"sent_sentiment_change_w{w}"] = np.nan
            out[f"sent_sentiment_vol_w{w}"] = np.nan
            out[f"sent_attention_count_w{w}"] = 0.0
            out[f"sent_attention_weighted_w{w}"] = 0.0
            out[f"sent_attention_z_w{w}"] = np.nan
            out[f"sent_sentiment_z_w{w}"] = np.nan
            out[f"sent_spike_flag_w{w}"] = 0.0
        return out

    events = events_df.copy().sort_values("timestamp").reset_index(drop=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events["sentiment_score"] = pd.to_numeric(events["sentiment_score"], errors="coerce")
    events["source_weight"] = pd.to_numeric(events["source_weight"], errors="coerce").fillna(cfg.default_source_weight)

    symbol_norm = symbol.upper().strip() if isinstance(symbol, str) and symbol.strip() else None
    if symbol_norm is not None:
        primary = events["primary_ticker"].astype(str).str.upper()
        ticker_match = primary.eq(symbol_norm)
        if cfg.include_unlinked:
            ticker_match = ticker_match | events["primary_ticker"].isna()
        events = events[ticker_match].copy()

    if events.empty:
        return aggregate_sentiment_features(pd.DataFrame(), timeline=timeline, symbol=symbol, config=cfg)

    half_life = max(float(cfg.recency_half_life_minutes), 1.0)
    decay_lambda = np.log(2.0) / half_life

    roll_window = max(int(cfg.spike_lookback), 1)
    roll_min_periods = min(roll_window, max(3, roll_window // 2))

    for window in cfg.windows:
        win_td = pd.Timedelta(str(window).lower())
        mean_vals: list[float] = []
        vol_vals: list[float] = []
        cnt_vals: list[float] = []
        attn_weighted_vals: list[float] = []

        for current_ts in ts.tolist():
            mask = (events["timestamp"] <= current_ts) & (events["timestamp"] > (current_ts - win_td))
            subset = events[mask]
            if subset.empty:
                mean_vals.append(np.nan)
                vol_vals.append(np.nan)
                cnt_vals.append(0.0)
                attn_weighted_vals.append(0.0)
                continue

            age_min = (current_ts - subset["timestamp"]).dt.total_seconds() / 60.0
            recency = np.exp(-decay_lambda * age_min.to_numpy(dtype=float))
            weights = subset["source_weight"].to_numpy(dtype=float) * recency
            values = subset["sentiment_score"].to_numpy(dtype=float)
            mean, vol = _weighted_stats(values, weights)

            mean_vals.append(mean)
            vol_vals.append(vol)
            cnt_vals.append(float(len(subset)))
            attn_weighted_vals.append(float(np.sum(weights)))

        mean_s = pd.Series(mean_vals, index=out.index)
        vol_s = pd.Series(vol_vals, index=out.index)
        cnt_s = pd.Series(cnt_vals, index=out.index)
        attn_s = pd.Series(attn_weighted_vals, index=out.index)

        change_s = mean_s.diff(1)
        attn_roll_mean = attn_s.rolling(roll_window, min_periods=roll_min_periods).mean()
        attn_roll_std = attn_s.rolling(roll_window, min_periods=roll_min_periods).std()
        sent_roll_mean = mean_s.rolling(roll_window, min_periods=roll_min_periods).mean()
        sent_roll_std = mean_s.rolling(roll_window, min_periods=roll_min_periods).std()
        attn_z = (attn_s - attn_roll_mean) / (attn_roll_std + 1e-9)
        sent_z = (mean_s - sent_roll_mean) / (sent_roll_std + 1e-9)
        spike_flag = ((attn_z.abs() >= 2.0) | (sent_z.abs() >= 2.0)).astype(float)

        out[f"sent_sentiment_mean_w{window}"] = mean_s
        out[f"sent_sentiment_change_w{window}"] = change_s
        out[f"sent_sentiment_vol_w{window}"] = vol_s
        out[f"sent_attention_count_w{window}"] = cnt_s
        out[f"sent_attention_weighted_w{window}"] = attn_s
        out[f"sent_attention_z_w{window}"] = attn_z
        out[f"sent_sentiment_z_w{window}"] = sent_z
        out[f"sent_spike_flag_w{window}"] = spike_flag

    return out
