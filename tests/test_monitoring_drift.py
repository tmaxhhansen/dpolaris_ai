from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from monitoring.drift import (
    DriftThresholds,
    PerformanceMonitor,
    RetrainingPolicy,
    RetrainingScheduler,
    SelfCritiqueLogger,
    compare_feature_distributions,
)


def test_feature_drift_detection_flags_shifted_feature():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "feat_stable": rng.normal(0.0, 1.0, 800),
            "feat_shifted": rng.normal(0.0, 1.0, 800),
        }
    )
    live = pd.DataFrame(
        {
            "feat_stable": rng.normal(0.0, 1.0, 300),
            "feat_shifted": rng.normal(2.0, 1.0, 300),
        }
    )

    result = compare_feature_distributions(
        train,
        live,
        thresholds=DriftThresholds(zscore_threshold=2.0),
    )

    assert result["alert"] is True
    shifted = [x for x in result["features"] if x["feature"] == "feat_shifted"][0]
    assert shifted["level"] in {"warning", "critical"}


def test_performance_monitor_alerts_on_live_degradation():
    ts = pd.date_range("2025-01-01", periods=180, freq="D", tz="UTC")
    y_true = np.array([0, 1] * 90)
    y_prob = np.where(y_true == 1, 0.1, 0.9)  # confidently wrong

    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "y_true": y_true,
            "probability": y_prob,
        }
    )
    monitor = PerformanceMonitor()
    report = monitor.evaluate(frame, window=120)

    assert report["alert"] is True
    assert report["alerts"]
    assert report["latest"] is not None


def test_retraining_scheduler_triggers_on_cadence_or_drift():
    scheduler = RetrainingScheduler(policy=RetrainingPolicy(cadence="weekly"))
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)

    due = scheduler.should_retrain(
        last_trained_at=now - timedelta(days=10),
        now=now,
        drift_report=None,
        performance_report=None,
    )
    assert due["retrain"] is True
    assert "scheduled_retrain_due" in due["reasons"]

    drift_triggered = scheduler.should_retrain(
        last_trained_at=now - timedelta(days=1),
        now=now,
        drift_report={
            "summary": {
                "warning_features": 0,
                "critical_features": 2,
            }
        },
        performance_report=None,
    )
    assert drift_triggered["retrain"] is True
    assert "drift_critical" in drift_triggered["reasons"]


def test_self_critique_logger_writes_weekly_review(tmp_path):
    critique = SelfCritiqueLogger(log_dir=tmp_path / "critique_events")

    sid_1 = critique.log_signal(
        symbol="SPY",
        timestamp="2026-02-05T14:30:00Z",
        confidence=0.72,
        regime="trend",
        top_features=["rsi_14", "macd_hist_12_26_9"],
        prediction=0.68,
        model_name="SPY_direction",
        model_version="v0007",
    )
    critique.log_outcome(
        signal_id=sid_1,
        outcome_timestamp="2026-02-07T14:30:00Z",
        realized_return=0.012,
    )

    sid_2 = critique.log_signal(
        symbol="QQQ",
        timestamp="2026-02-06T14:30:00Z",
        confidence=0.66,
        regime="high_vol",
        top_features=["atr_14", "boll_width_20_2"],
        prediction=0.42,
        model_name="QQQ_direction",
        model_version="v0003",
    )
    critique.log_outcome(
        signal_id=sid_2,
        outcome_timestamp="2026-02-08T14:30:00Z",
        realized_return=-0.009,
    )

    review_path = critique.generate_weekly_review(
        week_ending="2026-02-10",
        output_path=tmp_path / "weekly_review_2026-02-10.md",
    )
    assert review_path.exists()
    text = review_path.read_text()
    assert "Weekly Model Review" in text
    assert "What Worked" in text
    assert "What Failed" in text
