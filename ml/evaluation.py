"""
Standardized evaluation metrics for prediction precision.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)])
    return arr


def _clamp_probabilities(proba: np.ndarray) -> np.ndarray:
    return np.clip(proba, 1e-6, 1 - 1e-6)


def build_reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    bins: int = 10,
) -> tuple[list[dict[str, float]], float]:
    """
    Build reliability bins and expected calibration error (ECE).
    """
    y_true = _safe_array(y_true).astype(int)
    y_proba = _clamp_probabilities(_safe_array(y_proba))

    if len(y_true) == 0:
        return [], 0.0

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(y_proba, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    curve: list[dict[str, float]] = []
    ece = 0.0
    total = len(y_true)

    for idx in range(bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        if count == 0:
            curve.append(
                {
                    "bin_low": float(bin_edges[idx]),
                    "bin_high": float(bin_edges[idx + 1]),
                    "count": 0.0,
                    "mean_predicted": 0.0,
                    "observed_frequency": 0.0,
                }
            )
            continue

        mean_pred = float(y_proba[mask].mean())
        observed = float(y_true[mask].mean())
        weight = count / total
        ece += abs(mean_pred - observed) * weight

        curve.append(
            {
                "bin_low": float(bin_edges[idx]),
                "bin_high": float(bin_edges[idx + 1]),
                "count": float(count),
                "mean_predicted": mean_pred,
                "observed_frequency": observed,
            }
        )

    return curve, float(ece)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    reliability_bins: int = 10,
) -> dict[str, Any]:
    """
    Standardized classification metrics.
    """
    y_true = _safe_array(y_true).astype(int)
    y_pred = _safe_array(y_pred).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        proba = _clamp_probabilities(_safe_array(y_proba))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except ValueError:
            metrics["roc_auc"] = None

        metrics["brier_score"] = float(brier_score_loss(y_true, proba))
        reliability_curve, ece = build_reliability_curve(
            y_true=y_true,
            y_proba=proba,
            bins=reliability_bins,
        )
        metrics["reliability_curve"] = reliability_curve
        metrics["calibration_error"] = ece
    else:
        metrics["roc_auc"] = None
        metrics["brier_score"] = None
        metrics["reliability_curve"] = []
        metrics["calibration_error"] = None

    return metrics


def compute_regression_metrics(
    y_true,
    y_pred,
    thresholds: Optional[list[float]] = None,
) -> dict[str, Any]:
    """
    Standardized regression metrics.
    """
    y_true = _safe_array(y_true)
    y_pred = _safe_array(y_pred)

    if len(y_true) == 0:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "directional_accuracy": None,
            "correlation": None,
            "hit_rate_thresholds": {},
        }

    thresholds = thresholds or [0.0, 0.005, 0.01]

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    non_zero = np.abs(y_true) > 1e-8
    if non_zero.any():
        mape = float(mean_absolute_percentage_error(y_true[non_zero], y_pred[non_zero]))
    else:
        mape = None

    directional_accuracy = float((np.sign(y_pred) == np.sign(y_true)).mean())
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else None

    hit_rate_thresholds: dict[str, float] = {}
    for threshold in thresholds:
        active = np.abs(y_pred) >= float(threshold)
        if not active.any():
            hit_rate_thresholds[f"{threshold:.4f}"] = 0.0
            continue
        hits = np.sign(y_pred[active]) == np.sign(y_true[active])
        hit_rate_thresholds[f"{threshold:.4f}"] = float(hits.mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "correlation": correlation,
        "hit_rate_thresholds": hit_rate_thresholds,
    }


def compute_trading_metrics(
    probability_up,
    realized_returns,
    long_threshold: float,
    short_threshold: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    annualization_periods: int = 252,
) -> dict[str, float]:
    """
    Convert probabilities + realized returns into slippage-adjusted trading metrics.
    """
    probability_up = _clamp_probabilities(_safe_array(probability_up))
    realized_returns = _safe_array(realized_returns)

    if len(probability_up) == 0 or len(realized_returns) == 0:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "cost_total": 0.0,
            "trades": 0.0,
            "total_return": 0.0,
        }

    positions = np.where(
        probability_up >= long_threshold,
        1.0,
        np.where(probability_up <= short_threshold, -1.0, 0.0),
    )

    prev_positions = np.concatenate(([0.0], positions[:-1]))
    position_changes = np.abs(positions - prev_positions)

    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    costs = position_changes * cost_rate

    gross_returns = positions * realized_returns
    net_returns = gross_returns - costs

    equity_curve = np.cumprod(1.0 + net_returns)
    total_return = float(equity_curve[-1] - 1.0)

    periods = len(net_returns)
    years = periods / max(annualization_periods, 1)
    if years > 0 and equity_curve[-1] > 0:
        cagr = float((equity_curve[-1] ** (1.0 / years)) - 1.0)
    else:
        cagr = 0.0

    mean_return = float(net_returns.mean())
    std_return = float(net_returns.std(ddof=0))
    if std_return > 1e-12:
        sharpe = float((mean_return / std_return) * np.sqrt(annualization_periods))
    else:
        sharpe = 0.0

    downside = net_returns[net_returns < 0.0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    if downside_std > 1e-12:
        sortino = float((mean_return / downside_std) * np.sqrt(annualization_periods))
    else:
        sortino = 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / running_max) - 1.0
    max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) else 0.0

    trade_mask = positions != 0.0
    trade_returns = net_returns[trade_mask]
    wins = trade_returns[trade_returns > 0.0]
    losses = trade_returns[trade_returns < 0.0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    if gross_loss > 1e-12:
        profit_factor = float(gross_win / gross_loss)
    elif gross_win > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    win_rate = float((trade_returns > 0.0).mean()) if len(trade_returns) else 0.0
    average_win = float(wins.mean()) if len(wins) else 0.0
    average_loss = float(losses.mean()) if len(losses) else 0.0

    exposure = float(np.mean(np.abs(positions)))
    turnover = float(np.mean(position_changes))

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "exposure": exposure,
        "turnover": turnover,
        "gross_pnl": float(gross_returns.sum()),
        "net_pnl": float(net_returns.sum()),
        "cost_total": float(costs.sum()),
        "trades": float(position_changes.sum()),
        "total_return": total_return,
    }


def fit_platt_calibration(probabilities, y_true) -> Optional[dict[str, float]]:
    """
    Fit Platt-scaling on predicted probabilities.
    """
    probabilities = _clamp_probabilities(_safe_array(probabilities))
    y_true = _safe_array(y_true).astype(int)

    if len(probabilities) < 30:
        return None
    if len(np.unique(y_true)) < 2:
        return None

    model = LogisticRegression(solver="lbfgs")
    model.fit(probabilities.reshape(-1, 1), y_true)

    return {
        "method": "platt",
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "samples": int(len(y_true)),
    }


def apply_probability_calibration(
    probabilities,
    calibration: Optional[dict[str, Any]],
) -> np.ndarray:
    """
    Apply stored calibration transform to raw probabilities.
    """
    probabilities = _clamp_probabilities(_safe_array(probabilities))
    if not calibration:
        return probabilities
    if calibration.get("method") != "platt":
        return probabilities

    coef = float(calibration.get("coef", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    logits = (coef * probabilities) + intercept
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    return _clamp_probabilities(calibrated)


def compute_primary_score(
    classification_metrics: dict[str, Any],
    trading_metrics: dict[str, Any],
    objective: str,
    metric: str,
    fallback_metric: str,
    max_drawdown: float,
    min_trades: int,
) -> float:
    """
    Compute objective-aligned primary score.
    """
    objective = (objective or "").strip().lower()
    metric = (metric or "sharpe").strip().lower()
    fallback_metric = (fallback_metric or "f1").strip().lower()

    metric_value = float(trading_metrics.get(metric, 0.0) or 0.0)
    fallback_value = float(classification_metrics.get(fallback_metric, 0.0) or 0.0)
    max_dd = float(trading_metrics.get("max_drawdown", 0.0) or 0.0)
    trades = float(trading_metrics.get("trades", 0.0) or 0.0)

    if objective == "maximize_f1":
        return float(classification_metrics.get("f1", 0.0) or 0.0)

    if objective == "maximize_sharpe_with_drawdown_cap":
        score = metric_value
        if max_dd > max_drawdown:
            score -= (max_dd - max_drawdown) * 10.0
        if trades < min_trades:
            score -= 0.5
        score += 0.1 * fallback_value
        return float(score)

    # Fallback objective
    return float(0.7 * metric_value + 0.3 * fallback_value)
