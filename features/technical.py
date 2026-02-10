"""
Parameterized technical feature library focused on price/volume behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from regime.regime_classifier import register_regime_plugins
from .fundamentals import register_fundamentals_plugins
from .macro import register_macro_plugins
from .registry import FeaturePlugin, FeatureRegistry, FeatureSpec
from .sentiment import register_sentiment_plugins


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b + eps)


def _timestamp_col(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "date" in df.columns:
        return "date"
    raise ValueError("Input DataFrame must include 'timestamp' or 'date'")


def _prepare_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ts_col = _timestamp_col(df)
    work = df.copy()
    work["__orig_idx__"] = np.arange(len(work))
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = work.sort_values(ts_col).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in work.columns:
            raise ValueError(f"Missing required OHLCV column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")

    return work, ts_col


def _restore_feature_order(features: pd.DataFrame, work: pd.DataFrame, original_index) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=original_index)
    out = features.copy()
    out["__orig_idx__"] = work["__orig_idx__"].values
    out = out.sort_values("__orig_idx__").drop(columns="__orig_idx__").reset_index(drop=True)
    out.index = original_index
    return out


def _rolling_slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    y = values.astype(float)
    y_centered = y - y.mean()
    denom = (x_centered**2).sum()
    if denom < 1e-12:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)


def _asof_align_close(base_ts: pd.Series, ref_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if ref_df is None or ref_df.empty:
        return pd.DataFrame({f"{prefix}__close_asof": [np.nan] * len(base_ts)})

    ref = ref_df.copy()
    ref_ts_col = _timestamp_col(ref)
    ref[ref_ts_col] = pd.to_datetime(ref[ref_ts_col], utc=True, errors="coerce")
    if "close" not in ref.columns:
        raise ValueError("Reference frame must include 'close'")
    ref["close"] = pd.to_numeric(ref["close"], errors="coerce")
    ref = ref.sort_values(ref_ts_col)[[ref_ts_col, "close"]].rename(columns={ref_ts_col: "timestamp"})
    ref[f"{prefix}__close"] = ref["close"]
    ref = ref.drop(columns=["close"])

    left = pd.DataFrame({"timestamp": pd.to_datetime(base_ts, utc=True)})
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        ref.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.rename(columns={"timestamp": f"{prefix}__asof_timestamp"})
    return merged


def _returns_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    horizons = tuple(params.get("horizons", (1, 3, 5, 10, 20)))
    close_col = params.get("price_col", "close")

    work, _ = _prepare_ohlcv(df)
    close = pd.to_numeric(work[close_col], errors="coerce")
    out = pd.DataFrame(index=work.index)

    for h in horizons:
        h = int(h)
        out[f"ret_simple_h{h}"] = close.pct_change(h)
        out[f"ret_log_h{h}"] = np.log(_safe_div(close, close.shift(h)))

    return _restore_feature_order(out, work, df.index)


def _trend_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    sma_windows = tuple(params.get("sma_windows", (10, 20, 50, 200)))
    ema_windows = tuple(params.get("ema_windows", (12, 26)))
    slope_windows = tuple(params.get("slope_windows", (5, 10, 20)))
    adx_window = int(params.get("adx_window", 14))

    work, _ = _prepare_ohlcv(df)
    close = work["close"]
    high = work["high"]
    low = work["low"]

    out = pd.DataFrame(index=work.index)

    for w in sma_windows:
        w = int(w)
        sma = close.rolling(w, min_periods=w).mean()
        out[f"sma_w{w}"] = sma
        out[f"close_vs_sma_w{w}"] = _safe_div(close, sma) - 1.0

    for i in range(len(sma_windows) - 1):
        fast = int(sma_windows[i])
        slow = int(sma_windows[i + 1])
        fast_col = out[f"sma_w{fast}"]
        slow_col = out[f"sma_w{slow}"]
        out[f"sma_cross_{fast}_{slow}"] = (_safe_div(fast_col, slow_col) - 1.0)
        out[f"sma_cross_flag_{fast}_{slow}"] = (fast_col > slow_col).astype(float)

    for w in ema_windows:
        w = int(w)
        ema = close.ewm(span=w, adjust=False).mean()
        out[f"ema_w{w}"] = ema
        out[f"close_vs_ema_w{w}"] = _safe_div(close, ema) - 1.0

    if len(ema_windows) >= 2:
        fast = int(ema_windows[0])
        slow = int(ema_windows[1])
        out[f"ema_cross_{fast}_{slow}"] = (
            _safe_div(out[f"ema_w{fast}"], out[f"ema_w{slow}"]) - 1.0
        )

    for w in slope_windows:
        w = int(w)
        out[f"slope_close_w{w}"] = close.rolling(w, min_periods=w).apply(_rolling_slope, raw=True)

    plus_dm = high.diff()
    minus_dm = (-low).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(adx_window, min_periods=adx_window).mean()
    plus_di = 100.0 * _safe_div(plus_dm.rolling(adx_window, min_periods=adx_window).mean(), atr)
    minus_di = 100.0 * _safe_div(minus_dm.rolling(adx_window, min_periods=adx_window).mean(), atr)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = dx.rolling(adx_window, min_periods=adx_window).mean()

    out[f"plus_di_w{adx_window}"] = plus_di
    out[f"minus_di_w{adx_window}"] = minus_di
    out[f"adx_w{adx_window}"] = adx
    out[f"trend_strength_adx_w{adx_window}"] = adx
    out[f"trend_direction_w{adx_window}"] = np.sign(plus_di - minus_di)

    return _restore_feature_order(out, work, df.index)


def _momentum_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    rsi_window = int(params.get("rsi_window", 14))
    macd_fast = int(params.get("macd_fast", 12))
    macd_slow = int(params.get("macd_slow", 26))
    macd_signal = int(params.get("macd_signal", 9))
    stoch_window = int(params.get("stoch_window", 14))
    stoch_smooth = int(params.get("stoch_smooth", 3))

    work, _ = _prepare_ohlcv(df)
    close = work["close"]
    high = work["high"]
    low = work["low"]

    out = pd.DataFrame(index=work.index)

    delta = close.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = (-delta).where(delta < 0.0, 0.0)
    avg_gain = gain.rolling(rsi_window, min_periods=rsi_window).mean()
    avg_loss = loss.rolling(rsi_window, min_periods=rsi_window).mean()
    rs = _safe_div(avg_gain, avg_loss)
    out[f"rsi_w{rsi_window}"] = 100.0 - (100.0 / (1.0 + rs))

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=macd_signal, adjust=False).mean()
    out[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"] = macd
    out[f"macd_signal_{macd_fast}_{macd_slow}_{macd_signal}"] = macd_sig
    out[f"macd_hist_{macd_fast}_{macd_slow}_{macd_signal}"] = macd - macd_sig

    lowest = low.rolling(stoch_window, min_periods=stoch_window).min()
    highest = high.rolling(stoch_window, min_periods=stoch_window).max()
    stoch_k = 100.0 * _safe_div(close - lowest, highest - lowest)
    stoch_d = stoch_k.rolling(stoch_smooth, min_periods=stoch_smooth).mean()
    out[f"stoch_k_w{stoch_window}"] = stoch_k
    out[f"stoch_d_w{stoch_window}_s{stoch_smooth}"] = stoch_d

    return _restore_feature_order(out, work, df.index)


def _volatility_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    atr_window = int(params.get("atr_window", 14))
    rv_windows = tuple(params.get("rv_windows", (5, 10, 20)))
    boll_window = int(params.get("boll_window", 20))
    boll_k = float(params.get("boll_k", 2.0))
    include_parkinson = bool(params.get("include_parkinson", True))
    parkinson_windows = tuple(params.get("parkinson_windows", (10, 20)))

    work, _ = _prepare_ohlcv(df)
    close = work["close"]
    high = work["high"]
    low = work["low"]

    out = pd.DataFrame(index=work.index)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    out[f"atr_w{atr_window}"] = atr
    out[f"atr_pct_w{atr_window}"] = _safe_div(atr, close)

    log_ret = np.log(_safe_div(close, close.shift(1)))
    for w in rv_windows:
        w = int(w)
        out[f"rv_w{w}"] = log_ret.rolling(w, min_periods=w).std() * np.sqrt(252.0)

    boll_mid = close.rolling(boll_window, min_periods=boll_window).mean()
    boll_std = close.rolling(boll_window, min_periods=boll_window).std()
    boll_upper = boll_mid + (boll_k * boll_std)
    boll_lower = boll_mid - (boll_k * boll_std)
    out[f"boll_width_w{boll_window}_k{int(boll_k)}"] = _safe_div(boll_upper - boll_lower, boll_mid)
    out[f"boll_pos_w{boll_window}_k{int(boll_k)}"] = _safe_div(close - boll_lower, boll_upper - boll_lower)

    if include_parkinson:
        hl_log_sq = (np.log(_safe_div(high, low))) ** 2
        for w in parkinson_windows:
            w = int(w)
            out[f"parkinson_vol_w{w}"] = np.sqrt(
                _safe_div(hl_log_sq.rolling(w, min_periods=w).mean() * 252.0, 4.0 * np.log(2.0))
            )

    return _restore_feature_order(out, work, df.index)


def _volume_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    zscore_windows = tuple(params.get("zscore_windows", (10, 20, 50)))
    trend_windows = tuple(params.get("trend_windows", (5, 20)))

    work, _ = _prepare_ohlcv(df)
    close = work["close"]
    volume = work["volume"]
    out = pd.DataFrame(index=work.index)

    obv = (np.sign(close.diff()).fillna(0.0) * volume.fillna(0.0)).cumsum()
    out["obv"] = obv

    for w in zscore_windows:
        w = int(w)
        vol_mean = volume.rolling(w, min_periods=w).mean()
        vol_std = volume.rolling(w, min_periods=w).std()
        out[f"vol_z_w{w}"] = _safe_div(volume - vol_mean, vol_std)

    if len(trend_windows) >= 2:
        short_w = int(trend_windows[0])
        long_w = int(trend_windows[1])
        short_mean = volume.rolling(short_w, min_periods=short_w).mean()
        long_mean = volume.rolling(long_w, min_periods=long_w).mean()
        out[f"vol_trend_ratio_{short_w}_{long_w}"] = _safe_div(short_mean, long_mean)

    for w in trend_windows:
        w = int(w)
        out[f"obv_slope_w{w}"] = obv.rolling(w, min_periods=w).apply(_rolling_slope, raw=True)

    return _restore_feature_order(out, work, df.index)


def _structure_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    lookbacks = tuple(params.get("lookbacks", (20, 50)))

    work, _ = _prepare_ohlcv(df)
    close = work["close"]
    high = work["high"]
    low = work["low"]

    out = pd.DataFrame(index=work.index)
    for w in lookbacks:
        w = int(w)
        recent_high = high.rolling(w, min_periods=w).max().shift(1)
        recent_low = low.rolling(w, min_periods=w).min().shift(1)

        out[f"dist_to_high_w{w}"] = _safe_div(close, recent_high) - 1.0
        out[f"dist_to_low_w{w}"] = _safe_div(close, recent_low) - 1.0
        out[f"range_pos_w{w}"] = _safe_div(close - recent_low, recent_high - recent_low)
        out[f"breakout_up_w{w}"] = (close > recent_high).astype(float)
        out[f"breakout_dn_w{w}"] = (close < recent_low).astype(float)

    return _restore_feature_order(out, work, df.index)


def _gap_plugin(df: pd.DataFrame, params: dict[str, Any], context: Optional[dict[str, Any]]) -> pd.DataFrame:
    _ = context
    fill_windows = tuple(params.get("fill_windows", (10, 20)))
    atr_window = int(params.get("atr_window", 14))

    work, _ = _prepare_ohlcv(df)
    open_ = work["open"]
    high = work["high"]
    low = work["low"]
    close = work["close"]
    prev_close = close.shift(1)

    out = pd.DataFrame(index=work.index)
    gap = _safe_div(open_, prev_close) - 1.0
    out["gap_overnight"] = gap
    out["gap_abs"] = gap.abs()
    out["gap_up_flag"] = (gap > 0).astype(float)
    out["gap_dn_flag"] = (gap < 0).astype(float)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    out[f"gap_atr_norm_w{atr_window}"] = _safe_div(gap, _safe_div(atr, close))

    gap_fill_up = ((gap > 0) & (low <= prev_close)).astype(float)
    gap_fill_dn = ((gap < 0) & (high >= prev_close)).astype(float)
    out["gap_fill_up_flag"] = gap_fill_up
    out["gap_fill_dn_flag"] = gap_fill_dn

    for w in fill_windows:
        w = int(w)
        up_prob = gap_fill_up.where(gap > 0).shift(1).rolling(w, min_periods=max(5, w // 2)).mean()
        dn_prob = gap_fill_dn.where(gap < 0).shift(1).rolling(w, min_periods=max(5, w // 2)).mean()
        out[f"gap_fill_prob_up_w{w}"] = up_prob
        out[f"gap_fill_prob_dn_w{w}"] = dn_prob
        out[f"gap_fill_prob_w{w}"] = np.where(gap >= 0, up_prob, dn_prob)

    return _restore_feature_order(out, work, df.index)


def _relative_strength_plugin(
    df: pd.DataFrame,
    params: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> pd.DataFrame:
    context = context or {}
    benchmark_df = context.get("benchmark_df")
    sector_df = context.get("sector_df")
    horizons = tuple(params.get("horizons", (1, 5, 10, 20)))
    beta_window = int(params.get("beta_window", 60))
    corr_windows = tuple(params.get("corr_windows", (20, 60)))

    work, ts_col = _prepare_ohlcv(df)
    base_ts = pd.to_datetime(work[ts_col], utc=True)
    close = work["close"]

    out = pd.DataFrame(index=work.index)
    for h in horizons:
        h = int(h)
        out[f"ret_base_h{h}"] = close.pct_change(h)

    if benchmark_df is not None and not benchmark_df.empty:
        bench_asof = _asof_align_close(base_ts, benchmark_df, prefix="bench")
        bench_close = bench_asof["bench__close"]
        bench_ret_1 = bench_close.pct_change(1)
        out["bench_ret_h1"] = bench_ret_1

        for h in horizons:
            h = int(h)
            bench_ret_h = bench_close.pct_change(h)
            out[f"bench_ret_h{h}"] = bench_ret_h
            out[f"rel_ret_vs_bench_h{h}"] = out[f"ret_base_h{h}"] - bench_ret_h

        rolling_cov = out["ret_base_h1"].rolling(beta_window, min_periods=beta_window).cov(bench_ret_1)
        rolling_var = bench_ret_1.rolling(beta_window, min_periods=beta_window).var()
        out[f"beta_vs_bench_w{beta_window}"] = _safe_div(rolling_cov, rolling_var)

        for w in corr_windows:
            w = int(w)
            corr = out["ret_base_h1"].rolling(w, min_periods=w).corr(bench_ret_1)
            out[f"corr_vs_bench_w{w}"] = corr
            out[f"corr_regime_vs_bench_w{w}"] = np.where(corr > 0.6, 1.0, np.where(corr < 0.2, -1.0, 0.0))

    if sector_df is not None and not sector_df.empty:
        sec_asof = _asof_align_close(base_ts, sector_df, prefix="sector")
        sec_close = sec_asof["sector__close"]
        sec_ret_1 = sec_close.pct_change(1)
        out["sector_ret_h1"] = sec_ret_1

        for h in horizons:
            h = int(h)
            sec_ret_h = sec_close.pct_change(h)
            out[f"sector_ret_h{h}"] = sec_ret_h
            out[f"rel_ret_vs_sector_h{h}"] = out[f"ret_base_h{h}"] - sec_ret_h

        for w in corr_windows:
            w = int(w)
            out[f"corr_vs_sector_w{w}"] = out["ret_base_h1"].rolling(w, min_periods=w).corr(sec_ret_1)

    return _restore_feature_order(out, work, df.index)


@dataclass
class FeatureScaler:
    """
    Feature scaling helper with standard/robust options.
    """

    method: str = "none"  # one of: none, standard, robust
    _scaler: Optional[Any] = None
    _feature_names: Optional[list[str]] = None

    def fit(self, df: pd.DataFrame, feature_names: list[str]) -> "FeatureScaler":
        self._feature_names = list(feature_names)
        if self.method == "none":
            self._scaler = None
            return self
        if self.method == "standard":
            self._scaler = StandardScaler()
        elif self.method == "robust":
            self._scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler method: {self.method}")

        self._scaler.fit(df[self._feature_names].to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._feature_names is None:
            raise ValueError("Scaler is not fitted")
        out = df.copy()
        if self._scaler is None:
            return out
        values = self._scaler.transform(out[self._feature_names].to_numpy())
        out.loc[:, self._feature_names] = values
        return out

    def fit_transform(self, df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        return self.fit(df, feature_names).transform(df)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_names": self._feature_names or [],
        }


def build_default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register(
        FeaturePlugin(
            name="returns",
            generator=_returns_plugin,
            group="returns",
            description="Simple/log returns over multiple horizons.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="trend",
            generator=_trend_plugin,
            group="trend",
            description="SMA/EMA crosses, slopes, ADX-style strength.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="momentum",
            generator=_momentum_plugin,
            group="momentum",
            description="RSI, MACD, stochastic oscillators.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="volatility",
            generator=_volatility_plugin,
            group="volatility",
            description="ATR, realized vol, Bollinger, Parkinson vol.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="volume",
            generator=_volume_plugin,
            group="volume",
            description="OBV, volume z-score, volume trend.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="structure",
            generator=_structure_plugin,
            group="structure",
            description="Support/resistance and breakout proxies.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="gaps",
            generator=_gap_plugin,
            group="gaps",
            description="Overnight gaps and gap-fill heuristics.",
        )
    )
    registry.register(
        FeaturePlugin(
            name="relative_strength",
            generator=_relative_strength_plugin,
            group="relative_strength",
            description="Relative return, beta, and correlation regime.",
        )
    )
    register_fundamentals_plugins(registry)
    register_macro_plugins(registry)
    register_sentiment_plugins(registry)
    register_regime_plugins(registry)
    return registry


class TechnicalFeatureLibrary:
    """
    High-level interface around FeatureRegistry.
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self.registry = registry or build_default_registry()

    def generate(
        self,
        df: pd.DataFrame,
        *,
        specs: Optional[list[FeatureSpec]] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
        fundamentals_df: Optional[pd.DataFrame] = None,
        earnings_df: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
        headlines_df: Optional[pd.DataFrame] = None,
        social_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        ticker_universe: Optional[set[str]] = None,
        fed_decision_dates: Optional[list[str]] = None,
        include_base_columns: bool = True,
        drop_na: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Generate registered features with optional causal reference context.
        """
        ts_col = _timestamp_col(df)
        base = df.copy()
        base[ts_col] = pd.to_datetime(base[ts_col], utc=True, errors="coerce")
        base = base.sort_values(ts_col).reset_index(drop=True)

        context = {
            "benchmark_df": benchmark_df,
            "sector_df": sector_df,
            "fundamentals_df": fundamentals_df,
            "earnings_df": earnings_df,
            "macro_df": macro_df,
            "headlines_df": headlines_df,
            "social_df": social_df,
            "symbol": symbol,
            "ticker_universe": ticker_universe,
            "fed_decision_dates": fed_decision_dates or [],
        }
        out, metadata = self.registry.generate(
            base_df=base,
            specs=specs,
            context=context,
            include_base_columns=include_base_columns,
        )

        # Keep feature names for importance logging compatibility.
        metadata["feature_names"] = [
            name
            for name in metadata.get("feature_names", [])
            if name not in base.columns
        ]
        metadata["feature_groups"] = {
            item["plugin"]: item["features"] for item in metadata.get("catalog", [])
        }

        if drop_na:
            out = out.dropna()

        return out, metadata
