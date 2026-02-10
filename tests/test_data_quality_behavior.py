from __future__ import annotations

import pandas as pd

from data.quality import DataQualityGate, QualityPolicy


def _canonical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-02 14:30:00+00:00"),
                pd.Timestamp("2024-01-03 14:30:00+00:00"),
                pd.Timestamp("2024-01-03 14:30:00+00:00"),  # duplicate
                pd.Timestamp("2024-01-05 14:30:00+00:00"),  # Jan 4 missing
            ],
            "open": [100, 101, 101, 102],
            "high": [101, 102, 102, 103],
            "low": [99, 100, 100, 101],
            "close": [100, 101, 101, 102],
            "volume": [1000, -200, 500, 1200],  # negative volume
            "adj_close": [100, 101, 101, 102],
            "dividend": [0.0, 0.0, 0.0, 0.0],
            "split_factor": [1.0, 1.0, 1.0, 1.0],
            "session": ["regular"] * 4,
            "is_market_holiday": [False] * 4,
            "is_synthetic": [False] * 4,
        }
    )


def test_quality_gate_repairs_missing_and_negative_volume(tmp_path):
    policy = QualityPolicy(
        duplicate_policy="drop",
        missing_timestamp_policy="repair",
        negative_volume_policy="repair",
        stale_price_policy="ignore",
        outlier_policy="ignore",
        min_history_rows_multiplier=1,
    )
    gate = DataQualityGate(policy=policy)

    cleaned, report, report_path = gate.run(
        _canonical_rows(),
        symbol="SPY",
        interval="1d",
        horizon_days=5,
        run_id="unit_test_run",
        report_dir=tmp_path / "reports",
    )

    assert report_path.name == "data_quality_unit_test_run.json"
    assert report_path.exists()

    # Duplicate removed
    assert report["checks"]["duplicates"]["count"] == 1
    assert cleaned["timestamp"].duplicated().sum() == 0

    # Missing daily timestamp repaired (Jan 4)
    assert report["checks"]["missing_timestamps"]["count"] >= 1
    assert report["checks"]["missing_timestamps"]["repaired_count"] >= 1
    assert cleaned["is_synthetic"].sum() >= 1

    # Negative volume repaired
    assert report["checks"]["negative_volume"]["count"] == 1
    assert (cleaned["volume"] >= 0).all()


def test_quality_gate_drops_outlier_without_corporate_action(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01 14:30:00+00:00", periods=5, freq="B"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 180, 102, 103],  # large jump at index 2
            "volume": [1000, 1000, 1000, 1000, 1000],
            "adj_close": [100, 101, 180, 102, 103],
            "dividend": [0.0] * 5,
            "split_factor": [1.0] * 5,
            "session": ["regular"] * 5,
            "is_market_holiday": [False] * 5,
            "is_synthetic": [False] * 5,
        }
    )

    policy = QualityPolicy(
        duplicate_policy="drop",
        missing_timestamp_policy="drop",
        negative_volume_policy="repair",
        stale_price_policy="ignore",
        outlier_policy="drop",
        outlier_return_threshold=0.3,
        min_history_rows_multiplier=1,
    )
    gate = DataQualityGate(policy=policy)
    cleaned, report, _ = gate.run(
        df,
        symbol="AAPL",
        interval="1d",
        horizon_days=2,
        run_id="outlier_test",
        report_dir=tmp_path / "reports",
    )

    assert report["checks"]["outliers_without_corporate_action"]["count"] >= 1
    assert len(cleaned) < len(df)
