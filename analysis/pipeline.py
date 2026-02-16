from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import platform
from typing import Any, Optional

import numpy as np
import pandas as pd

REPORT_SECTION_ORDER = [
    "Overview",
    "Price/Volume Snapshot",
    "Technical Indicators",
    "Chart Patterns",
    "Model Signals",
    "News",
    "Risk Notes",
    "Next Steps",
]

POSITIVE_NEWS_WORDS = {
    "beat",
    "growth",
    "upgrade",
    "record",
    "strong",
    "expands",
    "launch",
    "partnership",
}
NEGATIVE_NEWS_WORDS = {
    "miss",
    "downgrade",
    "lawsuit",
    "probe",
    "cuts",
    "weak",
    "decline",
    "delay",
}

NEWS_CLUSTERS = {
    "earnings": {"earnings", "guidance", "eps", "revenue", "quarter"},
    "analyst": {"analyst", "upgrade", "downgrade", "target", "rating"},
    "macro": {"inflation", "fed", "rates", "economy", "macro"},
    "product": {"launch", "product", "chip", "ai", "platform", "service"},
    "deal": {"acquisition", "merger", "partnership", "contract", "deal"},
    "legal": {"lawsuit", "regulator", "probe", "investigation", "settlement"},
}


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published_at: Optional[str]
    url: Optional[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        parsed = float(value)
    except Exception:
        return float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    return parsed


def _to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("OHLCV history is empty")

    work = df.copy()
    if "date" not in work.columns:
        if "timestamp" in work.columns:
            work = work.rename(columns={"timestamp": "date"})
        else:
            raise ValueError("OHLCV history must include date or timestamp")

    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col not in work.columns:
            raise ValueError(f"OHLCV history missing column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if len(work) < 60:
        raise ValueError("Need at least 60 rows for report generation")

    return work


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.where(delta > 0.0, 0.0)
    losses = (-delta).where(delta < 0.0, 0.0)
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _bollinger(close: pd.Series, window: int = 20, width: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + width * std
    lower = mid - width * std
    return upper, mid, lower


def _rolling_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    coeffs = np.polyfit(x, values.astype(float), 1)
    return float(coeffs[0])


def _pivot_points(values: np.ndarray, window: int, kind: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    if len(values) < (window * 2 + 1):
        return out

    for idx in range(window, len(values) - window):
        center = float(values[idx])
        left = values[idx - window : idx]
        right = values[idx + 1 : idx + window + 1]
        if kind == "high":
            if center >= float(np.max(left)) and center >= float(np.max(right)):
                out.append((idx, center))
        else:
            if center <= float(np.min(left)) and center <= float(np.min(right)):
                out.append((idx, center))
    return out


def compute_indicator_snapshot(history: pd.DataFrame) -> dict[str, Any]:
    df = _prepare_history(history)
    close = df["close"]
    volume = df["volume"]

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    ema20 = _ema(close, 20)

    rsi14 = _rsi(close, 14)
    macd_line, macd_signal, macd_hist = _macd(close)
    atr14 = _atr(df, 14)
    boll_upper, boll_mid, boll_lower = _bollinger(close, 20, 2.0)

    vol_avg20 = volume.rolling(20, min_periods=20).mean()
    vol_avg60 = volume.rolling(60, min_periods=60).mean()
    vol_ratio = _safe_float((vol_avg20 / (vol_avg60 + 1e-9)).iloc[-1], 1.0)

    highs = _pivot_points(df["high"].to_numpy(dtype=float), window=3, kind="high")
    lows = _pivot_points(df["low"].to_numpy(dtype=float), window=3, kind="low")

    last_close = _safe_float(close.iloc[-1])
    supports = sorted({v for _, v in lows[-20:] if v <= last_close}, reverse=True)[:3]
    resistances = sorted({v for _, v in highs[-20:] if v >= last_close})[:3]

    rsi_value = _safe_float(rsi14.iloc[-1])
    if rsi_value >= 70.0:
        rsi_view = "overbought momentum"
    elif rsi_value <= 30.0:
        rsi_view = "oversold momentum"
    else:
        rsi_view = "neutral momentum"

    macd_hist_value = _safe_float(macd_hist.iloc[-1])
    macd_view = "bullish" if macd_hist_value > 0 else "bearish"

    atr_pct = _safe_float((atr14.iloc[-1] / (last_close + 1e-9)), 0.0)
    boll_pos = _safe_float((last_close - _safe_float(boll_lower.iloc[-1])) / ((_safe_float(boll_upper.iloc[-1]) - _safe_float(boll_lower.iloc[-1])) + 1e-9), 0.5)

    return {
        "sma": {
            "sma20": _safe_float(sma20.iloc[-1]),
            "sma50": _safe_float(sma50.iloc[-1]),
            "close_vs_sma20": _safe_float(last_close / (_safe_float(sma20.iloc[-1], last_close) + 1e-9) - 1.0),
            "close_vs_sma50": _safe_float(last_close / (_safe_float(sma50.iloc[-1], last_close) + 1e-9) - 1.0),
            "interpretation": "trend above moving averages" if last_close >= _safe_float(sma20.iloc[-1], last_close) else "trend below short moving average",
        },
        "ema": {
            "ema20": _safe_float(ema20.iloc[-1]),
            "close_vs_ema20": _safe_float(last_close / (_safe_float(ema20.iloc[-1], last_close) + 1e-9) - 1.0),
            "interpretation": "price leading EMA" if last_close >= _safe_float(ema20.iloc[-1], last_close) else "price lagging EMA",
        },
        "rsi": {
            "value": rsi_value,
            "interpretation": rsi_view,
        },
        "macd": {
            "line": _safe_float(macd_line.iloc[-1]),
            "signal": _safe_float(macd_signal.iloc[-1]),
            "histogram": macd_hist_value,
            "interpretation": f"{macd_view} momentum bias",
        },
        "atr": {
            "value": _safe_float(atr14.iloc[-1]),
            "atr_pct": atr_pct,
            "interpretation": "elevated realized range" if atr_pct > 0.03 else "contained realized range",
        },
        "bollinger": {
            "upper": _safe_float(boll_upper.iloc[-1]),
            "middle": _safe_float(boll_mid.iloc[-1]),
            "lower": _safe_float(boll_lower.iloc[-1]),
            "position": boll_pos,
            "interpretation": "near upper band" if boll_pos > 0.8 else "near lower band" if boll_pos < 0.2 else "inside band range",
        },
        "volume_trend": {
            "avg_volume_20": _safe_float(vol_avg20.iloc[-1]),
            "avg_volume_60": _safe_float(vol_avg60.iloc[-1]),
            "ratio_20_to_60": vol_ratio,
            "interpretation": "rising participation" if vol_ratio > 1.1 else "cooling participation" if vol_ratio < 0.9 else "stable participation",
        },
        "support_resistance": {
            "supports": [round(v, 4) for v in supports],
            "resistances": [round(v, 4) for v in resistances],
            "interpretation": "nearest levels from local pivot highs/lows",
        },
    }


def _trend_channel(close: pd.Series, window: int = 40) -> dict[str, Any]:
    lookback = close.tail(max(window, 20)).to_numpy(dtype=float)
    x = np.arange(len(lookback), dtype=float)
    slope, intercept = np.polyfit(x, lookback, 1)
    projected = slope * x + intercept
    ss_res = float(np.sum((lookback - projected) ** 2))
    ss_tot = float(np.sum((lookback - np.mean(lookback)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
    slope_pct = float(slope / (np.mean(lookback) + 1e-9))

    if slope_pct > 0.0010:
        label = "uptrend"
    elif slope_pct < -0.0010:
        label = "downtrend"
    else:
        label = "sideways"

    confidence = max(0.35, min(0.95, 0.45 + min(0.35, abs(slope_pct) * 900.0) + max(0.0, min(0.15, r2 * 0.15))))
    return {
        "name": "trend_channel",
        "signal": label,
        "confidence": round(confidence, 3),
        "description": f"{label} channel estimated from {len(lookback)} bars (slope={slope_pct:.4f}, r2={r2:.3f})",
        "metrics": {
            "slope_pct_per_bar": slope_pct,
            "r2": r2,
            "window": len(lookback),
        },
    }


def _range_breakout(df: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    if len(close) < lookback + 5:
        return {
            "name": "range_breakout",
            "signal": "insufficient_data",
            "confidence": 0.2,
            "description": "Not enough data for breakout check",
            "metrics": {},
        }

    prior_high = float(np.max(highs[-(lookback + 1) : -1]))
    prior_low = float(np.min(lows[-(lookback + 1) : -1]))
    latest = float(close[-1])
    vol_ratio = float(volume[-1] / (np.mean(volume[-20:]) + 1e-9))

    up_trigger = prior_high * 1.003
    down_trigger = prior_low * 0.997

    if latest > up_trigger:
        distance = (latest / (prior_high + 1e-9)) - 1.0
        confidence = min(0.95, 0.55 + min(0.25, distance * 20.0) + min(0.15, max(0.0, (vol_ratio - 1.0)) * 0.2))
        signal = "upside_breakout"
    elif latest < down_trigger:
        distance = 1.0 - (latest / (prior_low + 1e-9))
        confidence = min(0.95, 0.55 + min(0.25, distance * 20.0) + min(0.15, max(0.0, (vol_ratio - 1.0)) * 0.2))
        signal = "downside_breakout"
    else:
        distance = 0.0
        confidence = 0.35
        signal = "inside_range"

    return {
        "name": "range_breakout",
        "signal": signal,
        "confidence": round(confidence, 3),
        "description": f"Latest close vs prior {lookback}-bar range [{prior_low:.2f}, {prior_high:.2f}]",
        "metrics": {
            "prior_high": prior_high,
            "prior_low": prior_low,
            "latest_close": latest,
            "distance": distance,
            "volume_ratio": vol_ratio,
        },
    }


def _double_top_bottom(df: pd.DataFrame) -> list[dict[str, Any]]:
    close = df["close"].to_numpy(dtype=float)
    highs = _pivot_points(df["high"].to_numpy(dtype=float), window=4, kind="high")
    lows = _pivot_points(df["low"].to_numpy(dtype=float), window=4, kind="low")

    patterns: list[dict[str, Any]] = []

    def _build(name: str, signal: str, confidence: float, desc: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "signal": signal,
            "confidence": round(max(0.2, min(0.95, confidence)), 3),
            "description": desc,
            "metrics": metrics,
        }

    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        if i2 - i1 >= 5:
            diff_pct = abs(p2 - p1) / (max(p1, p2) + 1e-9)
            if diff_pct <= 0.03:
                trough = float(np.min(close[i1:i2 + 1]))
                confirmed = close[-1] < trough * 0.995
                confidence = 0.55 + (0.25 * (1.0 - min(1.0, diff_pct / 0.03))) + (0.1 if confirmed else 0.0)
                patterns.append(
                    _build(
                        "double_top",
                        "bearish_reversal" if confirmed else "potential_double_top",
                        confidence,
                        "Two nearby swing highs with midpoint trough",
                        {
                            "left_peak": p1,
                            "right_peak": p2,
                            "trough": trough,
                            "peak_diff_pct": diff_pct,
                            "confirmed": confirmed,
                        },
                    )
                )

    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        if i2 - i1 >= 5:
            diff_pct = abs(p2 - p1) / (min(p1, p2) + 1e-9)
            if diff_pct <= 0.03:
                crest = float(np.max(close[i1:i2 + 1]))
                confirmed = close[-1] > crest * 1.005
                confidence = 0.55 + (0.25 * (1.0 - min(1.0, diff_pct / 0.03))) + (0.1 if confirmed else 0.0)
                patterns.append(
                    _build(
                        "double_bottom",
                        "bullish_reversal" if confirmed else "potential_double_bottom",
                        confidence,
                        "Two nearby swing lows with midpoint crest",
                        {
                            "left_bottom": p1,
                            "right_bottom": p2,
                            "crest": crest,
                            "bottom_diff_pct": diff_pct,
                            "confirmed": confirmed,
                        },
                    )
                )

    return patterns


def _head_and_shoulders(df: pd.DataFrame) -> Optional[dict[str, Any]]:
    highs = _pivot_points(df["high"].to_numpy(dtype=float), window=3, kind="high")
    if len(highs) < 3:
        return None

    left, head, right = highs[-3], highs[-2], highs[-1]
    lp, hp, rp = left[1], head[1], right[1]

    shoulder_similarity = abs(lp - rp) / (max(lp, rp) + 1e-9)
    head_margin = (hp - max(lp, rp)) / (hp + 1e-9)
    spacing_ok = (head[0] - left[0] >= 3) and (right[0] - head[0] >= 3)

    if spacing_ok and shoulder_similarity <= 0.05 and head_margin >= 0.03:
        confidence = min(0.9, 0.55 + (0.2 * (1.0 - shoulder_similarity / 0.05)) + min(0.15, head_margin * 1.2))
        return {
            "name": "head_and_shoulders",
            "signal": "bearish_reversal_pattern",
            "confidence": round(confidence, 3),
            "description": "Shoulders are symmetric and head is materially higher",
            "metrics": {
                "left_shoulder": lp,
                "head": hp,
                "right_shoulder": rp,
                "shoulder_similarity": shoulder_similarity,
                "head_margin": head_margin,
            },
        }
    return None


def detect_chart_patterns(history: pd.DataFrame) -> list[dict[str, Any]]:
    df = _prepare_history(history)
    patterns = [_trend_channel(df["close"]), _range_breakout(df)]
    patterns.extend(_double_top_bottom(df))
    hs = _head_and_shoulders(df)
    if hs is not None:
        patterns.append(hs)
    return patterns


def derive_regime_context(history: pd.DataFrame, indicators: dict[str, Any]) -> dict[str, Any]:
    df = _prepare_history(history)
    close = df["close"]

    returns = close.pct_change().dropna()
    if returns.empty:
        returns = pd.Series([0.0])

    vol20 = _safe_float(returns.tail(20).std() * np.sqrt(252.0), 0.0)
    vol60 = _safe_float(returns.tail(60).std() * np.sqrt(252.0), vol20)

    rolling20 = returns.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    hist_window = rolling20.dropna().tail(252)
    if hist_window.empty:
        percentile = 0.5
    else:
        percentile = float((hist_window <= vol20).sum() / max(1, len(hist_window)))

    sma20 = _safe_float(indicators.get("sma", {}).get("sma20"), _safe_float(close.iloc[-1]))
    sma50 = _safe_float(indicators.get("sma", {}).get("sma50"), sma20)
    latest = _safe_float(close.iloc[-1])

    if latest > sma50 and sma20 > sma50:
        trend = "trend_up"
    elif latest < sma50 and sma20 < sma50:
        trend = "trend_down"
    else:
        trend = "range_transition"

    if percentile >= 0.8:
        vol_regime = "high_volatility"
    elif percentile <= 0.3:
        vol_regime = "calm_volatility"
    else:
        vol_regime = "normal_volatility"

    return {
        "trend_regime": trend,
        "volatility_regime": vol_regime,
        "volatility_annualized_20": vol20,
        "volatility_annualized_60": vol60,
        "volatility_percentile_1y": percentile,
        "summary": f"{trend.replace('_', ' ')} with {vol_regime.replace('_', ' ')} (20d vol={vol20:.2%})",
    }


def _fetch_news_disabled(symbol: str, limit: int) -> tuple[list[NewsItem], str]:
    _ = symbol, limit
    return [], "News provider is disabled (DPOLARIS_NEWS_PROVIDER=disabled)."


def _fetch_news_yfinance(symbol: str, limit: int) -> tuple[list[NewsItem], str]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        return [], f"yfinance unavailable: {exc}"

    try:
        raw_items = yf.Ticker(symbol).news or []
    except Exception as exc:
        return [], f"yfinance news fetch failed: {exc}"

    parsed: list[NewsItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        if not title:
            continue
        source = str(raw.get("publisher") or raw.get("source") or "yfinance")
        link = raw.get("link")
        ts = raw.get("providerPublishTime")
        published_at = None
        if ts is not None:
            try:
                published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            except Exception:
                published_at = None
        parsed.append(
            NewsItem(
                title=title,
                source=source,
                published_at=published_at,
                url=str(link) if link else None,
            )
        )

    parsed.sort(key=lambda item: str(item.published_at or ""), reverse=True)
    return parsed[: max(0, limit)], ""


def fetch_news(symbol: str, limit: int = 8, provider: Optional[str] = None) -> dict[str, Any]:
    requested = str(provider or os.getenv("DPOLARIS_NEWS_PROVIDER", "disabled")).strip().lower()
    if not requested:
        requested = "disabled"

    if requested in {"disabled", "none", "off"}:
        items, note = _fetch_news_disabled(symbol, limit)
        active = "disabled"
    elif requested == "yfinance":
        items, note = _fetch_news_yfinance(symbol, limit)
        active = "yfinance"
    else:
        items, note = _fetch_news_disabled(symbol, limit)
        active = "disabled"
        note = f"Unknown news provider '{requested}'. Falling back to disabled."

    refs = [
        {
            "title": item.title,
            "source": item.source,
            "published_at": item.published_at,
            "url": item.url,
        }
        for item in items
    ]

    return {
        "provider": active,
        "requested_provider": requested,
        "enabled": active != "disabled",
        "items": refs,
        "note": note,
    }


def summarize_news(news_payload: dict[str, Any]) -> dict[str, Any]:
    refs = list(news_payload.get("items") or [])
    if not refs:
        return {
            "clusters": {},
            "tone": "neutral",
            "headline_count": 0,
            "summary": news_payload.get("note") or "No recent headlines available.",
        }

    cluster_counts: dict[str, int] = {key: 0 for key in NEWS_CLUSTERS}
    cluster_counts["general"] = 0
    pos = 0
    neg = 0

    for item in refs:
        title = str(item.get("title") or "").lower()
        matched = False
        for name, words in NEWS_CLUSTERS.items():
            if any(word in title for word in words):
                cluster_counts[name] += 1
                matched = True
                break
        if not matched:
            cluster_counts["general"] += 1

        if any(word in title for word in POSITIVE_NEWS_WORDS):
            pos += 1
        if any(word in title for word in NEGATIVE_NEWS_WORDS):
            neg += 1

    tone = "mixed"
    if pos > neg + 1:
        tone = "constructive"
    elif neg > pos + 1:
        tone = "cautious"

    ordered_clusters = dict(sorted(cluster_counts.items(), key=lambda kv: kv[1], reverse=True))
    top_cluster = next((name for name, count in ordered_clusters.items() if count > 0), "general")
    summary = (
        f"{len(refs)} headlines reviewed; dominant theme: {top_cluster}. "
        f"Keyword tone is {tone}."
    )

    return {
        "clusters": ordered_clusters,
        "tone": tone,
        "headline_count": len(refs),
        "summary": summary,
    }


def _model_signals_section(
    model_signals: Optional[dict[str, Any]],
    model_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not model_signals:
        return {
            "available": False,
            "summary": "No trained model output available; report is based on price/volume and pattern context.",
            "probabilities": {},
            "calibration": {},
        }

    probabilities = {
        "probability_up": _safe_float(model_signals.get("probability_up"), 0.5),
        "probability_down": _safe_float(model_signals.get("probability_down"), 0.5),
        "confidence": _safe_float(model_signals.get("confidence"), 0.5),
    }

    metrics = (model_metadata or {}).get("metrics", {}) if isinstance(model_metadata, dict) else {}
    calibration = {
        "method": (metrics.get("probability_calibration") or {}).get("method") if isinstance(metrics, dict) else None,
        "brier_score": metrics.get("brier_score") if isinstance(metrics, dict) else None,
        "calibration_error": metrics.get("calibration_error") if isinstance(metrics, dict) else None,
    }

    summary = (
        f"Model probabilities: up={probabilities['probability_up']:.3f}, "
        f"down={probabilities['probability_down']:.3f}, confidence={probabilities['confidence']:.3f}."
    )

    return {
        "available": True,
        "summary": summary,
        "probabilities": probabilities,
        "calibration": calibration,
        "prediction_label": model_signals.get("prediction_label"),
        "model_accuracy": (model_metadata or {}).get("metrics", {}).get("accuracy") if isinstance(model_metadata, dict) else None,
    }


def _build_price_volume_snapshot(df: pd.DataFrame, indicators: dict[str, Any]) -> dict[str, Any]:
    close = df["close"]
    volume = df["volume"]
    latest = _safe_float(close.iloc[-1])
    prev = _safe_float(close.iloc[-2], latest)
    change_pct = (latest / (prev + 1e-9)) - 1.0

    avg_vol_20 = _safe_float(volume.tail(20).mean())
    today_vol = _safe_float(volume.iloc[-1])
    vol_ratio = today_vol / (avg_vol_20 + 1e-9)

    support_resistance = indicators.get("support_resistance", {})

    return {
        "latest_close": latest,
        "daily_change_pct": change_pct,
        "latest_volume": today_vol,
        "avg_volume_20": avg_vol_20,
        "volume_ratio": vol_ratio,
        "supports": support_resistance.get("supports", []),
        "resistances": support_resistance.get("resistances", []),
    }


def _risk_notes(
    indicators: dict[str, Any],
    patterns: list[dict[str, Any]],
    regime: dict[str, Any],
    model_section: dict[str, Any],
    news_summary: dict[str, Any],
) -> list[str]:
    notes: list[str] = []

    atr_pct = _safe_float(indicators.get("atr", {}).get("atr_pct"), 0.0)
    if atr_pct > 0.03:
        notes.append("Volatility is elevated (ATR > 3% of price); position sizing and stop logic should be conservative.")

    rsi_value = _safe_float(indicators.get("rsi", {}).get("value"), 50.0)
    if rsi_value >= 70.0:
        notes.append("RSI is overbought; momentum continuation is possible but pullback risk rises.")
    elif rsi_value <= 30.0:
        notes.append("RSI is oversold; reflexive bounces are possible but trend risk remains.")

    breakout = next((p for p in patterns if p.get("name") == "range_breakout"), None)
    if breakout and breakout.get("signal") in {"upside_breakout", "downside_breakout"}:
        conf = _safe_float(breakout.get("confidence"), 0.5)
        notes.append(
            f"Breakout regime detected ({breakout.get('signal')}) with confidence {conf:.2f}; monitor false-break risk on weak follow-through."
        )

    if regime.get("volatility_regime") == "high_volatility":
        notes.append("High-volatility regime can increase slippage and widen expected return dispersion.")

    if model_section.get("available"):
        confidence = _safe_float(model_section.get("probabilities", {}).get("confidence"), 0.5)
        if confidence < 0.60:
            notes.append("Model confidence is modest; treat probabilities as one input, not a standalone trigger.")
    else:
        notes.append("No trained model output was available, so this report is indicator/pattern-driven.")

    if news_summary.get("headline_count", 0) == 0:
        notes.append("No live headline feed in use; event risk may be under-represented.")

    if not notes:
        notes.append("No abnormal risk flags were detected from the configured checks.")

    return notes


def _next_steps(snapshot: dict[str, Any], patterns: list[dict[str, Any]], model_section: dict[str, Any]) -> list[str]:
    steps = [
        "Confirm whether price holds above nearest support or rejects at nearest resistance over the next 2-3 sessions.",
        "Track volume confirmation versus 20-day average; breakouts without participation are less reliable.",
        "Re-run this report after the next major catalyst (earnings, macro prints, or company guidance).",
    ]

    breakout = next((p for p in patterns if p.get("name") == "range_breakout"), None)
    if breakout and breakout.get("signal") != "inside_range":
        steps.append("Watch for retest behavior at the breakout level before increasing conviction.")

    if model_section.get("available"):
        steps.append("Compare model probabilities with indicator direction; prioritize setups where both align.")
    else:
        steps.append("If a trained model becomes available, compare probabilistic output against this technical baseline.")

    return steps[:5]


def _report_text(
    *,
    symbol: str,
    created_at: str,
    summary: str,
    snapshot: dict[str, Any],
    indicators: dict[str, Any],
    patterns: list[dict[str, Any]],
    model_section: dict[str, Any],
    news_payload: dict[str, Any],
    news_summary: dict[str, Any],
    regime: dict[str, Any],
    risk_notes: list[str],
    next_steps: list[str],
) -> str:
    lines: list[str] = []
    lines.append("## Overview")
    lines.append(f"- Symbol: {symbol}")
    lines.append(f"- Generated: {created_at}")
    lines.append(f"- Summary: {summary}")
    lines.append(f"- Regime: {regime.get('summary')}")
    lines.append("")

    lines.append("## Price/Volume Snapshot")
    lines.append(f"- Last close: {snapshot.get('latest_close'):.2f}")
    lines.append(f"- Daily change: {_to_percent(_safe_float(snapshot.get('daily_change_pct'), 0.0))}")
    lines.append(f"- Volume ratio vs 20d avg: {_safe_float(snapshot.get('volume_ratio'), 1.0):.2f}x")
    lines.append(f"- Supports: {snapshot.get('supports')}")
    lines.append(f"- Resistances: {snapshot.get('resistances')}")
    lines.append("")

    lines.append("## Technical Indicators")
    lines.append(
        "- SMA/EMA: "
        f"SMA20={_safe_float(indicators.get('sma', {}).get('sma20')):.2f}, "
        f"SMA50={_safe_float(indicators.get('sma', {}).get('sma50')):.2f}, "
        f"EMA20={_safe_float(indicators.get('ema', {}).get('ema20')):.2f}"
    )
    lines.append(
        "- RSI/MACD: "
        f"RSI14={_safe_float(indicators.get('rsi', {}).get('value')):.2f} "
        f"({indicators.get('rsi', {}).get('interpretation')}); "
        f"MACD hist={_safe_float(indicators.get('macd', {}).get('histogram')):.4f}"
    )
    lines.append(
        "- ATR/Bollinger: "
        f"ATR14={_safe_float(indicators.get('atr', {}).get('value')):.3f} "
        f"({_to_percent(_safe_float(indicators.get('atr', {}).get('atr_pct')))} of price), "
        f"Bollinger position={_safe_float(indicators.get('bollinger', {}).get('position')):.2f}"
    )
    lines.append(
        f"- Volume trend: {indicators.get('volume_trend', {}).get('interpretation')} "
        f"(20/60 ratio={_safe_float(indicators.get('volume_trend', {}).get('ratio_20_to_60'), 1.0):.2f})"
    )
    lines.append("")

    lines.append("## Chart Patterns")
    for pattern in patterns:
        lines.append(
            f"- {pattern.get('name')}: {pattern.get('signal')} "
            f"(confidence={_safe_float(pattern.get('confidence'), 0.0):.2f}) - {pattern.get('description')}"
        )
    lines.append("")

    lines.append("## Model Signals")
    if model_section.get("available"):
        probs = model_section.get("probabilities", {})
        lines.append(
            f"- Probabilities: up={_safe_float(probs.get('probability_up')):.3f}, "
            f"down={_safe_float(probs.get('probability_down')):.3f}, "
            f"confidence={_safe_float(probs.get('confidence')):.3f}"
        )
        calibration = model_section.get("calibration", {})
        lines.append(
            f"- Calibration: method={calibration.get('method') or 'none'}, "
            f"brier={calibration.get('brier_score')}, error={calibration.get('calibration_error')}"
        )
        lines.append("- Interpretation: model output is probabilistic context, not a standalone trade decision.")
    else:
        lines.append(f"- {model_section.get('summary')}")
    lines.append("")

    lines.append("## News")
    lines.append(f"- Provider: {news_payload.get('provider')}")
    if news_payload.get("note"):
        lines.append(f"- Note: {news_payload.get('note')}")
    lines.append(f"- Summary: {news_summary.get('summary')}")
    for item in list(news_payload.get("items") or [])[:5]:
        lines.append(f"- Headline: {item.get('title')} ({item.get('source')})")
    lines.append("")

    lines.append("## Risk Notes")
    for note in risk_notes:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Next Steps")
    for step in next_steps:
        lines.append(f"- {step}")

    return "\n".join(lines).strip() + "\n"


def _summary_line(symbol: str, regime: dict[str, Any], model_section: dict[str, Any]) -> str:
    if model_section.get("available"):
        probs = model_section.get("probabilities", {})
        return (
            f"{symbol} is in {regime.get('trend_regime', 'mixed')} with "
            f"{regime.get('volatility_regime', 'normal')} conditions; "
            f"model probabilities are up={_safe_float(probs.get('probability_up')):.2f}, "
            f"down={_safe_float(probs.get('probability_down')):.2f}."
        )
    return (
        f"{symbol} is in {regime.get('trend_regime', 'mixed')} with "
        f"{regime.get('volatility_regime', 'normal')} conditions; report is technical/context-driven "
        "because no model output is currently available."
    )


def generate_analysis_report(
    *,
    symbol: str,
    history: pd.DataFrame,
    model_signals: Optional[dict[str, Any]] = None,
    model_metadata: Optional[dict[str, Any]] = None,
    news_provider: Optional[str] = None,
    news_limit: int = 8,
) -> dict[str, Any]:
    ticker = str(symbol or "").strip().upper()
    if not ticker:
        raise ValueError("symbol is required")

    df = _prepare_history(history)

    indicators = compute_indicator_snapshot(df)
    patterns = detect_chart_patterns(df)
    regime = derive_regime_context(df, indicators)

    model_section = _model_signals_section(model_signals, model_metadata)
    news_payload = fetch_news(ticker, limit=max(0, int(news_limit)), provider=news_provider)
    news_summary = summarize_news(news_payload)

    snapshot = _build_price_volume_snapshot(df, indicators)
    risk_notes = _risk_notes(indicators, patterns, regime, model_section, news_summary)
    next_steps = _next_steps(snapshot, patterns, model_section)

    created_at = _utc_now_iso()
    summary = _summary_line(ticker, regime, model_section)
    report_text = _report_text(
        symbol=ticker,
        created_at=created_at,
        summary=summary,
        snapshot=snapshot,
        indicators=indicators,
        patterns=patterns,
        model_section=model_section,
        news_payload=news_payload,
        news_summary=news_summary,
        regime=regime,
        risk_notes=risk_notes,
        next_steps=next_steps,
    )

    return {
        "ticker": ticker,
        "created_at": created_at,
        "summary": summary,
        "report_text": report_text,
        "price_volume_snapshot": snapshot,
        "technical_indicators": indicators,
        "chart_patterns": patterns,
        "regime_context": regime,
        "model_signals": model_section,
        "news": {
            "provider": news_payload.get("provider"),
            "requested_provider": news_payload.get("requested_provider"),
            "enabled": news_payload.get("enabled"),
            "note": news_payload.get("note"),
            "summary": news_summary,
            "items": news_payload.get("items") or [],
        },
        "risk_notes": risk_notes,
        "next_steps": next_steps,
        "sections": list(REPORT_SECTION_ORDER),
    }


def default_version_info() -> dict[str, Any]:
    return {
        "analysis_pipeline_version": "1.0.0",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "news_provider": os.getenv("DPOLARIS_NEWS_PROVIDER", "disabled"),
    }
