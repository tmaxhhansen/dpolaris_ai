"""
Drift/performance monitoring, retraining triggers, and structured self-critique logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from ml.evaluation import compute_classification_metrics

try:
    from scipy.stats import ks_2samp as _ks_2samp
except Exception:  # pragma: no cover - fallback path used when scipy missing
    _ks_2samp = None


@dataclass
class DriftThresholds:
    """Thresholds for feature distribution drift checks."""

    ks_stat_threshold: float = 0.15
    ks_pvalue_threshold: float = 0.01
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.20
    zscore_threshold: float = 3.0
    min_samples: int = 60
    psi_bins: int = 10


@dataclass
class PerformanceThresholds:
    """Thresholds for rolling out-of-sample and calibration monitoring."""

    window: int = 120
    min_accuracy: float = 0.50
    min_f1: float = 0.50
    min_roc_auc: float = 0.55
    max_brier_score: float = 0.25
    max_calibration_error: float = 0.12
    reliability_bins: int = 10


@dataclass
class RetrainingPolicy:
    """
    Retraining policy.

    cadence can be:
    - weekly
    - monthly
    - custom (use cadence_days)
    """

    cadence: str = "weekly"
    cadence_days: int = 7
    max_model_age_days: int = 45
    drift_triggers: tuple[str, ...] = ("critical",)
    retrain_on_performance_alert: bool = True


def compare_feature_distributions(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    *,
    feature_names: Optional[Sequence[str]] = None,
    thresholds: Optional[DriftThresholds] = None,
) -> dict[str, Any]:
    """
    Compare train vs live feature distributions and flag drift.
    """
    cfg = thresholds or DriftThresholds()
    features = list(feature_names) if feature_names else _shared_numeric_features(train_df, live_df)

    rows: list[dict[str, Any]] = []
    summary = {
        "total_features": int(len(features)),
        "stable_features": 0,
        "warning_features": 0,
        "critical_features": 0,
        "insufficient_features": 0,
    }

    for feature in features:
        train_values = _as_numeric_series(train_df.get(feature))
        live_values = _as_numeric_series(live_df.get(feature))

        row: dict[str, Any] = {
            "feature": feature,
            "train_samples": int(len(train_values)),
            "live_samples": int(len(live_values)),
            "ks_stat": None,
            "ks_pvalue": None,
            "psi": None,
            "zscore_shift": None,
            "level": "stable",
            "reasons": [],
        }

        if len(train_values) < cfg.min_samples or len(live_values) < cfg.min_samples:
            row["level"] = "insufficient_data"
            row["reasons"] = ["min_samples"]
            summary["insufficient_features"] += 1
            rows.append(row)
            continue

        ks_stat, ks_p = _ks_two_sample(train_values.to_numpy(dtype=float), live_values.to_numpy(dtype=float))
        psi = _population_stability_index(
            train_values.to_numpy(dtype=float),
            live_values.to_numpy(dtype=float),
            bins=cfg.psi_bins,
        )
        z_shift = _zscore_mean_shift(
            train_values.to_numpy(dtype=float),
            live_values.to_numpy(dtype=float),
        )

        row["ks_stat"] = float(ks_stat)
        row["ks_pvalue"] = float(ks_p) if ks_p is not None else None
        row["psi"] = float(psi)
        row["zscore_shift"] = float(z_shift) if np.isfinite(z_shift) else None

        reasons: list[str] = []
        level = "stable"

        if ks_stat >= cfg.ks_stat_threshold and (ks_p is None or ks_p <= cfg.ks_pvalue_threshold):
            reasons.append("ks")
            level = "warning"

        if psi >= cfg.psi_critical_threshold:
            reasons.append("psi_critical")
            level = "critical"
        elif psi >= cfg.psi_warning_threshold and level != "critical":
            reasons.append("psi_warning")
            level = "warning"

        if np.isfinite(z_shift) and abs(z_shift) >= cfg.zscore_threshold:
            reasons.append("zscore")
            if level == "stable":
                level = "warning"

        row["level"] = level
        row["reasons"] = reasons

        if level == "critical":
            summary["critical_features"] += 1
        elif level == "warning":
            summary["warning_features"] += 1
        else:
            summary["stable_features"] += 1

        rows.append(row)

    alert = summary["critical_features"] > 0 or summary["warning_features"] > 0

    return {
        "generated_at": _utcnow().isoformat(),
        "thresholds": asdict(cfg),
        "summary": summary,
        "features": rows,
        "alert": bool(alert),
    }


def compute_rolling_oos_metrics(
    predictions_df: pd.DataFrame,
    *,
    window: int = 120,
    step: int = 1,
    timestamp_col: str = "timestamp",
    y_true_col: str = "y_true",
    y_prob_col: str = "probability",
    y_pred_col: str = "prediction",
    reliability_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling out-of-sample classification and calibration metrics.
    """
    required = {timestamp_col, y_true_col, y_prob_col}
    missing = [c for c in required if c not in predictions_df.columns]
    if missing:
        raise ValueError(f"predictions_df missing required columns: {missing}")

    frame = predictions_df.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    frame[y_true_col] = pd.to_numeric(frame[y_true_col], errors="coerce")
    frame[y_prob_col] = pd.to_numeric(frame[y_prob_col], errors="coerce")
    if y_pred_col in frame.columns:
        frame[y_pred_col] = pd.to_numeric(frame[y_pred_col], errors="coerce")
    else:
        frame[y_pred_col] = np.where(frame[y_prob_col] >= 0.5, 1, 0)

    frame = frame.dropna(subset=[timestamp_col, y_true_col, y_prob_col, y_pred_col]).sort_values(timestamp_col)
    frame = frame.reset_index(drop=True)

    if frame.empty or len(frame) < window:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier_score",
                "calibration_error",
                "sample_size",
            ]
        )

    rows: list[dict[str, Any]] = []
    for end_idx in range(window, len(frame) + 1, max(1, int(step))):
        view = frame.iloc[end_idx - window : end_idx]
        y_true = view[y_true_col].astype(int).to_numpy()
        y_prob = view[y_prob_col].astype(float).to_numpy()
        y_pred = view[y_pred_col].astype(int).to_numpy()

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_prob,
            reliability_bins=reliability_bins,
        )
        rows.append(
            {
                "timestamp": pd.to_datetime(view[timestamp_col].iloc[-1], utc=True),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "calibration_error": metrics.get("calibration_error"),
                "sample_size": int(len(view)),
            }
        )

    return pd.DataFrame(rows)


class DriftMonitor:
    """Drift monitor with report artifacts and retraining alert helper."""

    def __init__(
        self,
        *,
        thresholds: Optional[DriftThresholds] = None,
        report_dir: Optional[Path | str] = "reports",
    ):
        self.thresholds = thresholds or DriftThresholds()
        self.report_dir = Path(report_dir).expanduser() if report_dir is not None else None

    def evaluate(
        self,
        train_df: pd.DataFrame,
        live_df: pd.DataFrame,
        *,
        feature_names: Optional[Sequence[str]] = None,
        run_id: Optional[str] = None,
        write_report: bool = True,
    ) -> dict[str, Any]:
        result = compare_feature_distributions(
            train_df=train_df,
            live_df=live_df,
            feature_names=feature_names,
            thresholds=self.thresholds,
        )
        result["run_id"] = run_id or _utcnow().strftime("%Y%m%d_%H%M%S")

        if write_report and self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.report_dir / f"drift_{result['run_id']}.json"
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            result["report_path"] = str(report_path)

        return result


class PerformanceMonitor:
    """Rolling out-of-sample and calibration monitor."""

    def __init__(
        self,
        *,
        thresholds: Optional[PerformanceThresholds] = None,
    ):
        self.thresholds = thresholds or PerformanceThresholds()

    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        *,
        window: Optional[int] = None,
    ) -> dict[str, Any]:
        cfg = self.thresholds
        rolling = compute_rolling_oos_metrics(
            predictions_df=predictions_df,
            window=int(window or cfg.window),
            reliability_bins=cfg.reliability_bins,
        )

        if rolling.empty:
            return {
                "generated_at": _utcnow().isoformat(),
                "alert": False,
                "alerts": [],
                "latest": None,
                "rolling": [],
            }

        latest = rolling.iloc[-1].to_dict()
        alerts: list[str] = []

        if float(latest.get("accuracy", 0.0) or 0.0) < cfg.min_accuracy:
            alerts.append("accuracy_below_threshold")
        if float(latest.get("f1", 0.0) or 0.0) < cfg.min_f1:
            alerts.append("f1_below_threshold")

        roc_auc = latest.get("roc_auc")
        if roc_auc is not None and float(roc_auc) < cfg.min_roc_auc:
            alerts.append("roc_auc_below_threshold")

        brier = latest.get("brier_score")
        if brier is not None and float(brier) > cfg.max_brier_score:
            alerts.append("brier_above_threshold")

        cal_err = latest.get("calibration_error")
        if cal_err is not None and float(cal_err) > cfg.max_calibration_error:
            alerts.append("calibration_error_above_threshold")

        return {
            "generated_at": _utcnow().isoformat(),
            "alert": bool(alerts),
            "alerts": alerts,
            "latest": latest,
            "rolling": rolling.to_dict(orient="records"),
        }


class RetrainingScheduler:
    """Retraining decision helper driven by time, drift, and live performance."""

    def __init__(
        self,
        policy: Optional[RetrainingPolicy] = None,
    ):
        self.policy = policy or RetrainingPolicy()

    def should_retrain(
        self,
        *,
        last_trained_at: Optional[datetime | str],
        drift_report: Optional[dict[str, Any]] = None,
        performance_report: Optional[dict[str, Any]] = None,
        now: Optional[datetime | str] = None,
    ) -> dict[str, Any]:
        policy = self.policy
        now_ts = _as_timestamp(now or _utcnow())
        reasons: list[str] = []

        if last_trained_at is None:
            reasons.append("no_previous_training")
        else:
            last_ts = _as_timestamp(last_trained_at)
            age_days = (now_ts - last_ts).total_seconds() / 86400.0
            cadence_days = _cadence_days(policy)

            if age_days >= cadence_days:
                reasons.append("scheduled_retrain_due")
            if age_days >= float(policy.max_model_age_days):
                reasons.append("max_model_age_exceeded")

        if drift_report:
            summary = drift_report.get("summary", {}) or {}
            warning_count = int(summary.get("warning_features", 0) or 0)
            critical_count = int(summary.get("critical_features", 0) or 0)
            triggers = {x.strip().lower() for x in policy.drift_triggers}

            if critical_count > 0 and "critical" in triggers:
                reasons.append("drift_critical")
            if warning_count > 0 and "warning" in triggers:
                reasons.append("drift_warning")

        if performance_report and policy.retrain_on_performance_alert:
            if performance_report.get("alert"):
                reasons.append("live_performance_degradation")

        return {
            "evaluated_at": now_ts.isoformat(),
            "retrain": bool(reasons),
            "reasons": reasons,
        }


class SelfCritiqueLogger:
    """
    Structured self-critique event logger.

    Signals and outcomes are written as append-only JSONL events and summarized weekly.
    """

    def __init__(
        self,
        *,
        log_dir: Path | str = "reports/self_critique",
    ):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_signal(
        self,
        *,
        symbol: str,
        timestamp: datetime | str,
        confidence: float,
        regime: str,
        top_features: Sequence[str | dict[str, Any]],
        prediction: Optional[float] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
        signal_id: Optional[str] = None,
    ) -> str:
        sid = signal_id or str(uuid4())
        ts = _as_timestamp(timestamp)
        payload = {
            "symbol": str(symbol).upper().strip(),
            "prediction": float(prediction) if prediction is not None else None,
            "confidence": float(confidence),
            "regime": str(regime),
            "top_features": _normalize_top_features(top_features),
            "model_name": model_name,
            "model_version": model_version,
            "extra": extra or {},
        }
        self._append_event(
            event_type="signal",
            signal_id=sid,
            timestamp=ts,
            payload=payload,
        )
        return sid

    def log_outcome(
        self,
        *,
        signal_id: str,
        outcome_timestamp: datetime | str,
        realized_return: Optional[float] = None,
        pnl: Optional[float] = None,
        outcome_label: Optional[bool] = None,
        notes: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        ts = _as_timestamp(outcome_timestamp)
        payload = {
            "realized_return": float(realized_return) if realized_return is not None else None,
            "pnl": float(pnl) if pnl is not None else None,
            "outcome_label": bool(outcome_label) if outcome_label is not None else None,
            "notes": notes,
            "extra": extra or {},
        }
        self._append_event(
            event_type="outcome",
            signal_id=str(signal_id),
            timestamp=ts,
            payload=payload,
        )

    def generate_weekly_review(
        self,
        *,
        week_ending: Optional[date | str | datetime] = None,
        output_path: Optional[Path | str] = None,
    ) -> Path:
        end_date = _as_date(week_ending or _utcnow())
        start_date = end_date - timedelta(days=6)
        events = self._load_events(start_date=start_date, end_date=end_date)

        signal_events: dict[str, dict[str, Any]] = {}
        outcome_events: dict[str, dict[str, Any]] = {}
        for event in events:
            sid = str(event.get("signal_id"))
            if event.get("event_type") == "signal":
                signal_events[sid] = event
            elif event.get("event_type") == "outcome":
                outcome_events[sid] = event

        paired: list[dict[str, Any]] = []
        for sid, signal in signal_events.items():
            out = outcome_events.get(sid)
            paired.append(
                {
                    "signal_id": sid,
                    "signal": signal,
                    "outcome": out,
                }
            )

        resolved = [x for x in paired if x["outcome"] is not None]
        win_flags = [_is_win(x["outcome"]["payload"]) for x in resolved]
        win_rate = float(np.mean(win_flags)) if win_flags else 0.0

        regime_stats = _summarize_by_regime(resolved)
        feature_stats = _summarize_feature_effectiveness(resolved)
        worked = [x for x in feature_stats if x["win_rate"] >= 0.55 and x["count"] >= 2][:5]
        failed = [x for x in feature_stats if x["win_rate"] <= 0.45 and x["count"] >= 2][:5]

        output = Path(output_path).expanduser() if output_path else Path("reports") / f"weekly_review_{end_date.isoformat()}.md"
        output.parent.mkdir(parents=True, exist_ok=True)

        md = _render_weekly_review_markdown(
            start_date=start_date,
            end_date=end_date,
            total_signals=len(paired),
            resolved_signals=len(resolved),
            win_rate=win_rate,
            regime_stats=regime_stats,
            worked=worked,
            failed=failed,
            unresolved=len(paired) - len(resolved),
        )
        output.write_text(md)
        return output

    def _append_event(
        self,
        *,
        event_type: str,
        signal_id: str,
        timestamp: pd.Timestamp,
        payload: dict[str, Any],
    ) -> None:
        path = self.log_dir / f"events_{timestamp.strftime('%Y-%m')}.jsonl"
        row = {
            "event_id": str(uuid4()),
            "event_type": event_type,
            "signal_id": signal_id,
            "timestamp": timestamp.isoformat(),
            "payload": payload,
        }
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _load_events(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.log_dir.glob("events_*.jsonl")):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    ts = _as_timestamp(event.get("timestamp"))
                    if start_date <= ts.date() <= end_date:
                        out.append(event)
        return out


def _as_numeric_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    elif values is None:
        series = pd.Series(dtype=float)
    else:
        series = pd.Series(values)
    return pd.to_numeric(series, errors="coerce").dropna()


def _shared_numeric_features(train_df: pd.DataFrame, live_df: pd.DataFrame) -> list[str]:
    shared = set(train_df.columns).intersection(set(live_df.columns))
    out: list[str] = []
    for col in sorted(shared):
        if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(live_df[col]):
            out.append(col)
    return out


def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, Optional[float]]:
    if len(a) == 0 or len(b) == 0:
        return 0.0, None

    if _ks_2samp is not None:
        result = _ks_2samp(a, b)
        return float(result.statistic), float(result.pvalue)

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    values = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, values, side="right") / max(len(a_sorted), 1)
    cdf_b = np.searchsorted(b_sorted, values, side="right") / max(len(b_sorted), 1)
    stat = float(np.max(np.abs(cdf_a - cdf_b)))

    n1 = len(a_sorted)
    n2 = len(b_sorted)
    en = np.sqrt((n1 * n2) / max(n1 + n2, 1))
    pvalue = float(min(1.0, 2.0 * np.exp(-2.0 * (en * stat) ** 2)))
    return stat, pvalue


def _population_stability_index(reference: np.ndarray, current: np.ndarray, *, bins: int) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        lo = float(np.nanmin(reference))
        hi = float(np.nanmax(reference))
        if not np.isfinite(lo):
            lo = -1.0
        if not np.isfinite(hi):
            hi = 1.0
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        edges = np.linspace(lo, hi, max(2, int(bins)) + 1)

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    epsilon = 1e-6
    ref_pct = np.maximum(ref_counts.astype(float) / max(ref_counts.sum(), 1), epsilon)
    cur_pct = np.maximum(cur_counts.astype(float) / max(cur_counts.sum(), 1), epsilon)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _zscore_mean_shift(reference: np.ndarray, current: np.ndarray) -> float:
    ref_mean = float(np.mean(reference))
    ref_std = float(np.std(reference))
    cur_mean = float(np.mean(current))

    if ref_std < 1e-12:
        if abs(cur_mean - ref_mean) < 1e-12:
            return 0.0
        return float(np.inf)

    std_err = ref_std / np.sqrt(max(len(current), 1))
    if std_err < 1e-12:
        std_err = ref_std
    return float((cur_mean - ref_mean) / std_err)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _as_date(value: date | str | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return _as_timestamp(value).date()


def _cadence_days(policy: RetrainingPolicy) -> int:
    token = policy.cadence.strip().lower()
    if token == "weekly":
        return 7
    if token == "monthly":
        return 30
    return max(1, int(policy.cadence_days))


def _normalize_top_features(items: Sequence[str | dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()
        if name:
            out.append(name)
    return out


def _is_win(outcome_payload: dict[str, Any]) -> bool:
    if outcome_payload.get("outcome_label") is not None:
        return bool(outcome_payload["outcome_label"])
    pnl = outcome_payload.get("pnl")
    if pnl is not None:
        return float(pnl) > 0.0
    realized_return = outcome_payload.get("realized_return")
    if realized_return is not None:
        return float(realized_return) > 0.0
    return False


def _summarize_by_regime(resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[bool]] = {}
    for row in resolved:
        signal_payload = row["signal"]["payload"]
        outcome_payload = row["outcome"]["payload"]
        regime = str(signal_payload.get("regime", "unknown"))
        buckets.setdefault(regime, []).append(_is_win(outcome_payload))

    out: list[dict[str, Any]] = []
    for regime, values in sorted(buckets.items()):
        out.append(
            {
                "regime": regime,
                "count": int(len(values)),
                "win_rate": float(np.mean(values)) if values else 0.0,
            }
        )
    return out


def _summarize_feature_effectiveness(resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[bool]] = {}
    for row in resolved:
        signal_payload = row["signal"]["payload"]
        outcome_payload = row["outcome"]["payload"]
        win = _is_win(outcome_payload)
        for feature in signal_payload.get("top_features", []) or []:
            buckets.setdefault(str(feature), []).append(win)

    out: list[dict[str, Any]] = []
    for feature, values in buckets.items():
        out.append(
            {
                "feature": feature,
                "count": int(len(values)),
                "win_rate": float(np.mean(values)) if values else 0.0,
            }
        )
    out.sort(key=lambda x: (x["count"], x["win_rate"]), reverse=True)
    return out


def _render_weekly_review_markdown(
    *,
    start_date: date,
    end_date: date,
    total_signals: int,
    resolved_signals: int,
    win_rate: float,
    regime_stats: list[dict[str, Any]],
    worked: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    unresolved: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Weekly Model Review ({start_date.isoformat()} to {end_date.isoformat()})")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Signals logged: **{total_signals}**")
    lines.append(f"- Outcomes resolved: **{resolved_signals}**")
    lines.append(f"- Unresolved: **{unresolved}**")
    lines.append(f"- Win rate (resolved): **{win_rate:.2%}**")
    lines.append("")
    lines.append("## What Worked")
    if worked:
        for item in worked:
            lines.append(
                f"- `{item['feature']}` | count={item['count']} | win_rate={item['win_rate']:.2%}"
            )
    else:
        lines.append("- No stable positive pattern yet (need more labeled outcomes).")
    lines.append("")
    lines.append("## What Failed")
    if failed:
        for item in failed:
            lines.append(
                f"- `{item['feature']}` | count={item['count']} | win_rate={item['win_rate']:.2%}"
            )
    else:
        lines.append("- No persistent failure cluster detected yet.")
    lines.append("")
    lines.append("## Regime Breakdown")
    if regime_stats:
        for row in regime_stats:
            lines.append(f"- `{row['regime']}` | trades={row['count']} | win_rate={row['win_rate']:.2%}")
    else:
        lines.append("- No resolved outcomes yet.")
    lines.append("")
    lines.append("## Action Items")
    lines.append("- Tighten thresholds for any feature listed under **What Failed**.")
    lines.append("- Increase position size only where feature/regime pairs remain stable.")
    lines.append("- Trigger retraining immediately on critical drift or calibration deterioration.")
    lines.append("")
    lines.append("_Generated by monitoring.self_critique weekly postmortem._")
    lines.append("")
    return "\n".join(lines)
