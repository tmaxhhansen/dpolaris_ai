"""
Market regime classifier (rule-based + optional ML override).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features.macro import generate_macro_features
from features.registry import FeaturePlugin, FeatureRegistry


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b + eps)


def _rolling_zscore(series: Optional[pd.Series], window: int) -> Optional[pd.Series]:
    if series is None:
        return None
    mean = series.rolling(window, min_periods=max(10, window // 3)).mean()
    std = series.rolling(window, min_periods=max(10, window // 3)).std()
    return _safe_div(series - mean, std)


def _numeric_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


@dataclass
class RegimeConfig:
    close_col: str = "close"
    vix_proxy_col: str = "macro_vix_proxy"
    credit_spread_col: str = "macro_credit_spread_proxy"
    dollar_strength_col: str = "macro_dollar_strength_proxy"
    benchmark_rotation_col: str = "macro_benchmark_rotation_proxy"
    trend_lookback: int = 20
    trend_return_threshold: float = 0.03
    realized_vol_window: int = 20
    trend_max_realized_vol: float = 0.35
    vix_high_threshold: float = 25.0
    realized_vol_high_threshold: float = 0.25
    risk_z_window: int = 60
    risk_off_score_threshold: float = 0.0

    @classmethod
    def from_params(cls, params: Optional[dict[str, Any]] = None) -> "RegimeConfig":
        params = params or {}
        return cls(
            close_col=str(params.get("close_col", cls.close_col)),
            vix_proxy_col=str(params.get("vix_proxy_col", cls.vix_proxy_col)),
            credit_spread_col=str(params.get("credit_spread_col", cls.credit_spread_col)),
            dollar_strength_col=str(params.get("dollar_strength_col", cls.dollar_strength_col)),
            benchmark_rotation_col=str(params.get("benchmark_rotation_col", cls.benchmark_rotation_col)),
            trend_lookback=int(params.get("trend_lookback", cls.trend_lookback)),
            trend_return_threshold=float(params.get("trend_return_threshold", cls.trend_return_threshold)),
            realized_vol_window=int(params.get("realized_vol_window", cls.realized_vol_window)),
            trend_max_realized_vol=float(params.get("trend_max_realized_vol", cls.trend_max_realized_vol)),
            vix_high_threshold=float(params.get("vix_high_threshold", cls.vix_high_threshold)),
            realized_vol_high_threshold=float(params.get("realized_vol_high_threshold", cls.realized_vol_high_threshold)),
            risk_z_window=int(params.get("risk_z_window", cls.risk_z_window)),
            risk_off_score_threshold=float(params.get("risk_off_score_threshold", cls.risk_off_score_threshold)),
        )


class RegimeClassifier:
    """
    Rule-based regime classifier with optional ML refinement.

    The rule system outputs:
    - trend vs mean-reversion state
    - high-vol vs low-vol state
    - risk-on vs risk-off state (when proxies are available)
    """

    def __init__(self, config: Optional[RegimeConfig] = None, use_ml: bool = False):
        self.config = config or RegimeConfig()
        self.use_ml = bool(use_ml)
        self._ml_model: Optional[LogisticRegression] = None
        self._ml_feature_cols = [
            "trend_return",
            "realized_vol",
            "vol_proxy",
            "credit_z",
            "dollar_z",
            "rotation_z",
        ]

    def _build_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        close = _numeric_col(df, cfg.close_col)
        if close is None and cfg.close_col != "adj_close":
            close = _numeric_col(df, "adj_close")
        if close is None:
            close = pd.Series(np.nan, index=df.index)

        log_ret = np.log(_safe_div(close, close.shift(1)))
        realized_vol = log_ret.rolling(cfg.realized_vol_window, min_periods=cfg.realized_vol_window).std() * np.sqrt(252.0)
        trend_return = close.pct_change(cfg.trend_lookback)

        vix_proxy = _numeric_col(df, cfg.vix_proxy_col)
        if vix_proxy is None:
            vol_proxy = realized_vol * 100.0
        else:
            vol_proxy = vix_proxy

        credit = _numeric_col(df, cfg.credit_spread_col)
        dollar = _numeric_col(df, cfg.dollar_strength_col)
        rotation = _numeric_col(df, cfg.benchmark_rotation_col)

        inputs = pd.DataFrame(index=df.index)
        inputs["trend_return"] = trend_return
        inputs["realized_vol"] = realized_vol
        inputs["vol_proxy"] = vol_proxy
        inputs["credit_z"] = _rolling_zscore(credit, cfg.risk_z_window)
        inputs["dollar_z"] = _rolling_zscore(dollar, cfg.risk_z_window)
        inputs["rotation_z"] = _rolling_zscore(rotation, cfg.risk_z_window)
        return inputs

    def fit(self, df: pd.DataFrame, label_col: str) -> "RegimeClassifier":
        """
        Optional ML fitting. If unavailable or under-specified, caller can ignore and use rule output.
        """
        if not self.use_ml:
            return self

        if label_col not in df.columns:
            raise ValueError(f"Missing label column for ML regime fitting: {label_col}")

        inputs = self._build_inputs(df)
        y = df[label_col]
        mask = inputs[self._ml_feature_cols].notna().all(axis=1) & y.notna()
        if int(mask.sum()) < 25:
            raise ValueError("Not enough samples to fit ML regime classifier")

        model = LogisticRegression(max_iter=1000)
        model.fit(inputs.loc[mask, self._ml_feature_cols], y.loc[mask])
        self._ml_model = model
        return self

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        inputs = self._build_inputs(df)

        out = pd.DataFrame(index=df.index)

        trend_flag = (
            (inputs["trend_return"].abs() >= cfg.trend_return_threshold)
            & (inputs["realized_vol"] <= cfg.trend_max_realized_vol)
        ).astype(float)
        out["regime_trend_flag"] = trend_flag

        has_explicit_vix = cfg.vix_proxy_col in df.columns
        if has_explicit_vix:
            high_vol_flag = (inputs["vol_proxy"] >= cfg.vix_high_threshold).astype(float)
        else:
            high_vol_flag = (inputs["realized_vol"] >= cfg.realized_vol_high_threshold).astype(float)
        out["regime_high_vol_flag"] = high_vol_flag
        out["regime_vol_proxy"] = inputs["vol_proxy"]

        risk_components: list[pd.Series] = []
        if inputs["credit_z"].notna().any():
            risk_components.append(pd.Series(np.where(inputs["credit_z"] > 0.0, 1.0, -1.0), index=df.index))
        if inputs["dollar_z"].notna().any():
            risk_components.append(pd.Series(np.where(inputs["dollar_z"] > 0.0, 1.0, -1.0), index=df.index))
        if inputs["rotation_z"].notna().any():
            # Higher benchmark rotation proxy means risk-on; invert for risk-off score.
            risk_components.append(pd.Series(np.where(inputs["rotation_z"] < 0.0, 1.0, -1.0), index=df.index))

        if risk_components:
            stacked = np.vstack([s.values for s in risk_components])
            risk_score = pd.Series(np.nanmean(stacked, axis=0), index=df.index)
            out["regime_risk_score"] = risk_score
            out["regime_risk_off_flag"] = (risk_score > cfg.risk_off_score_threshold).astype(float)
        else:
            out["regime_risk_score"] = np.nan
            out["regime_risk_off_flag"] = np.nan

        out["regime_state_code"] = (
            out["regime_trend_flag"].fillna(0.0).astype(int)
            + 2 * out["regime_high_vol_flag"].fillna(0.0).astype(int)
            + 4 * out["regime_risk_off_flag"].fillna(0.0).astype(int)
        ).astype(float)

        if self.use_ml and self._ml_model is not None:
            mask = inputs[self._ml_feature_cols].notna().all(axis=1)
            if mask.any():
                proba = self._ml_model.predict_proba(inputs.loc[mask, self._ml_feature_cols])
                if proba.shape[1] >= 2:
                    trend_prob = pd.Series(np.nan, index=df.index, dtype=float)
                    trend_prob.loc[mask] = proba[:, 1]
                    out["regime_trend_prob_ml"] = trend_prob
                    out.loc[trend_prob.notna(), "regime_trend_flag"] = (trend_prob.dropna() >= 0.5).astype(float)

        return out


def _regime_features_plugin(
    df: pd.DataFrame,
    params: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> pd.DataFrame:
    context = context or {}
    macro_df = context.get("macro_df")

    macro_params = params.get("macro_params", {})
    fed_dates = params.get("fed_decision_dates")
    macro_features = generate_macro_features(
        df,
        macro_df=macro_df,
        params=macro_params,
        fed_decision_dates=fed_dates,
    )

    if macro_features.empty:
        working = df.copy()
    else:
        working = pd.concat([df.reset_index(drop=True), macro_features.reset_index(drop=True)], axis=1)

    cfg = RegimeConfig.from_params(params)
    classifier = RegimeClassifier(
        config=cfg,
        use_ml=bool(params.get("use_ml", False)),
    )

    label_col = params.get("label_col")
    if classifier.use_ml and isinstance(label_col, str) and label_col in working.columns:
        classifier.fit(working, label_col=label_col)

    regime_frame = classifier.classify(working)

    # Keep plugin output model-friendly (numeric features only).
    numeric_cols = [
        col
        for col in regime_frame.columns
        if pd.api.types.is_numeric_dtype(regime_frame[col])
    ]
    return regime_frame[numeric_cols]


def register_regime_plugins(registry: FeatureRegistry) -> None:
    registry.register(
        FeaturePlugin(
            name="regime",
            generator=_regime_features_plugin,
            group="regime",
            description="Rule-based regime states with optional ML refinement.",
        )
    )

