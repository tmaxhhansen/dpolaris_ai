from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ml.training_artifacts import (
    compare_training_runs,
    list_run_artifact_files,
    list_training_runs,
    load_training_artifact,
    write_training_artifact,
)


def _load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _validate(schema: dict, data, schema_dir: Path) -> list[str]:
    """
    Lightweight JSON-schema validator for contract tests.
    Supports: type, required, properties, items, pattern, $ref.
    """
    errors: list[str] = []

    if "$ref" in schema:
        ref = schema["$ref"]
        ref_path = (schema_dir / ref).resolve()
        return _validate(_load_schema(ref_path), data, ref_path.parent)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        type_ok = any(_type_matches(t, data) for t in schema_type)
    elif isinstance(schema_type, str):
        type_ok = _type_matches(schema_type, data)
    else:
        type_ok = True

    if not type_ok:
        errors.append(f"type mismatch expected={schema_type} got={type(data).__name__}")
        return errors

    if isinstance(data, str) and "pattern" in schema:
        if re.match(schema["pattern"], data) is None:
            errors.append(f"pattern mismatch: {schema['pattern']} value={data}")

    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"missing required key: {key}")

        props = schema.get("properties", {})
        for key, value in data.items():
            if key in props:
                errors.extend(_validate(props[key], value, schema_dir))

    if isinstance(data, list) and "items" in schema:
        for item in data:
            errors.extend(_validate(schema["items"], item, schema_dir))

    return errors


def _type_matches(schema_type: str, value) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    return True


def test_training_artifact_schema_v1_and_run_folder_layout(tmp_path):
    run_root = tmp_path / "runs"
    artifact_info = write_training_artifact(
        run_id="contract_run_1",
        status="completed",
        model_type="lstm",
        target="target_direction",
        horizon=5,
        tickers=["SPY"],
        timeframes=["1d"],
        metrics_summary={
            "classification": {"accuracy": 0.62, "f1": 0.61, "roc_auc": 0.66},
            "regression": {},
            "trading": {"sharpe": 1.1, "max_drawdown": 0.12},
            "calibration": {"brier_score": 0.21, "calibration_error": 0.05},
        },
        run_root=run_root,
    )

    run_id = artifact_info["run_id"]
    run_dir = Path(artifact_info["run_dir"])
    assert run_dir.exists()
    assert run_dir.name == run_id
    assert (run_dir / "artifact.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "metrics_summary.json").exists()
    assert (run_dir / "universe_snapshot.json").exists()

    artifact = load_training_artifact(run_id, run_root=run_root)
    schema_dir = Path("/Users/darrenwon/my-git/dpolaris_ai/schemas")
    top_schema = _load_schema(schema_dir / "training_artifact.schema.json")
    errors = _validate(top_schema, artifact, schema_dir)
    assert errors == []


def test_training_artifact_backward_compatibility_normalization(tmp_path):
    run_root = tmp_path / "runs"
    legacy_dir = run_root / "legacy_run_1"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    legacy_payload = {
        "training_artifact_version": "0.9.0",
        "RunSummary": {
            "run_id": "legacy_run_1",
            "status": "completed",
            "created_at": "2026-02-10T00:00:00Z",
            "started_at": "2026-02-10T00:00:01Z",
            "completed_at": "2026-02-10T00:01:01Z",
            "duration_seconds": 60.0,
            "git_commit_hash": "abc123",
            "git_branch": "main",
            "environment": {"python_version": "3.12"},
            "hostname": "host",
            "user": "tester",
            "model_type": "xgboost",
            "target": "target_direction",
            "horizon": 5,
            "tickers": ["SPY"],
            "timeframes": ["1d"]
        },
        "MetricsSummary": {
            "classification": {"accuracy": 0.60}
        }
    }
    with open(legacy_dir / "artifact.json", "w") as f:
        json.dump(legacy_payload, f)

    normalized = load_training_artifact("legacy_run_1", run_root=run_root)
    assert normalized["training_artifact_version"] == "0.9.0"
    assert normalized["run_summary"]["run_id"] == "legacy_run_1"
    assert "data_summary" in normalized
    assert "feature_summary" in normalized
    assert "split_summary" in normalized
    assert "model_summary" in normalized
    assert "backtest_summary" in normalized
    assert "diagnostics_summary" in normalized
    assert "classification" in normalized["metrics_summary"]


def test_run_listing_artifact_listing_and_compare(tmp_path):
    run_root = tmp_path / "runs"

    a = write_training_artifact(
        run_id="run_a",
        status="completed",
        model_type="xgboost",
        target="target_direction",
        horizon=5,
        tickers=["SPY"],
        timeframes=["1d"],
        metrics_summary={
            "classification": {"accuracy": 0.58, "f1": 0.57, "roc_auc": 0.61},
            "regression": {},
            "trading": {"sharpe": 0.8, "max_drawdown": 0.16},
            "calibration": {}
        },
        run_root=run_root,
    )
    b = write_training_artifact(
        run_id="run_b",
        status="completed",
        model_type="lstm",
        target="target_direction",
        horizon=5,
        tickers=["SPY"],
        timeframes=["1d"],
        metrics_summary={
            "classification": {"accuracy": 0.63, "f1": 0.62, "roc_auc": 0.69},
            "regression": {},
            "trading": {"sharpe": 1.2, "max_drawdown": 0.11},
            "calibration": {}
        },
        run_root=run_root,
    )

    runs = list_training_runs(run_root=run_root, limit=10)
    ids = {x["run_id"] for x in runs}
    assert a["run_id"] in ids
    assert b["run_id"] in ids

    artifacts = list_run_artifact_files(b["run_id"], run_root=run_root)
    assert "artifact.json" in artifacts
    assert "metrics_summary.json" in artifacts

    comparison = compare_training_runs([a["run_id"], b["run_id"]], run_root=run_root)
    assert len(comparison["compared"]) == 2
    assert comparison["best"]["best_f1_run"] == b["run_id"]
    assert comparison["best"]["best_sharpe_run"] == b["run_id"]



def test_universe_snapshot_hash_is_persisted(tmp_path):
    run_root = tmp_path / "runs"

    provided_universe = {
        "schema_version": "1.0.0",
        "generated_at": "2026-02-10T00:00:00Z",
        "merged": [{"symbol": "SPY", "sources": ["nasdaq_top_500"]}],
    }

    artifact_info = write_training_artifact(
        run_id="contract_run_universe",
        status="completed",
        model_type="xgboost",
        target="target_direction",
        horizon=5,
        tickers=["SPY"],
        timeframes=["1d"],
        universe_snapshot=provided_universe,
        run_root=run_root,
    )

    run_id = artifact_info["run_id"]
    run_dir = Path(artifact_info["run_dir"])

    with open(run_dir / "universe_snapshot.json") as f:
        snapshot = json.load(f)

    canonical = dict(snapshot)
    canonical.pop("universe_hash", None)
    expected_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    assert snapshot["universe_hash"] == expected_hash

    artifact = load_training_artifact(run_id, run_root=run_root)
    assert artifact["run_summary"]["universe_hash"] == expected_hash
    listed = list_run_artifact_files(run_id, run_root=run_root)
    assert "universe_snapshot.json" in listed
