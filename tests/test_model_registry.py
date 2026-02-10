from __future__ import annotations

from registry.model_registry import ModelRegistry


def test_model_registry_tracks_versions_and_data_window(tmp_path):
    registry = ModelRegistry(root_dir=tmp_path / "registry")
    model_blob = tmp_path / "model.pkl"
    model_blob.write_bytes(b"dummy-model-bytes")

    first = registry.register_model(
        model_name="SPY_direction",
        model_path=model_blob,
        config={"target_horizon": 5},
        metrics={"primary_score": 1.23},
        data_window={"start": "2020-01-01", "end": "2025-12-31", "rows": 1300},
        training_trigger="scheduled",
    )
    second = registry.register_model(
        model_name="SPY_direction",
        model_path=model_blob,
        config={"target_horizon": 10},
        metrics={"primary_score": 1.31},
        data_window={"start": "2021-01-01", "end": "2026-01-31", "rows": 1400},
        training_trigger="drift",
    )

    assert first["version"] == "v0001"
    assert second["version"] == "v0002"
    assert second["data_window"]["rows"] == 1400
    assert second["status"] == "active"

    listed = registry.list_models(model_name="SPY_direction")
    assert len(listed) == 2
    assert listed[0]["version"] == "v0002"
    assert listed[1]["status"] == "superseded"

    latest = registry.get_model(model_name="SPY_direction")
    assert latest is not None
    assert latest["version"] == "v0002"


def test_model_registry_attaches_monitoring_reports(tmp_path):
    registry = ModelRegistry(root_dir=tmp_path / "registry")
    model_blob = tmp_path / "model.pkl"
    model_blob.write_bytes(b"dummy-model-bytes")

    record = registry.register_model(
        model_name="QQQ_direction",
        model_path=model_blob,
        config={"target_horizon": 5},
        metrics={"primary_score": 1.05},
        data_window={"rows": 1000},
    )

    updated = registry.attach_monitoring(
        model_name="QQQ_direction",
        version=record["version"],
        drift_report={"summary": {"critical_features": 1}},
        performance_report={"alert": True, "alerts": ["f1_below_threshold"]},
        retrain_decision={"retrain": True, "reasons": ["drift_critical"]},
    )

    assert "monitoring" in updated
    assert updated["monitoring"]["drift"]["summary"]["critical_features"] == 1
    assert updated["monitoring"]["performance"]["alert"] is True
    assert updated["monitoring"]["retrain_decision"]["retrain"] is True
