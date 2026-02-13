"""
Training observability artifact contract.

Each training run is stored in:
  runs/<run_id>/

and includes a versioned artifact payload plus section files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import getpass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any, Optional, Sequence
from uuid import uuid4

import pandas as pd


TRAINING_ARTIFACT_VERSION = "1.0.0"
ARTIFACT_FILE_NAME = "artifact.json"
CONFIG_SNAPSHOT_FILE = "config_snapshot.json"
DEPENDENCY_SNAPSHOT_FILE = "dependency_snapshot.json"
DATA_HASHES_FILE = "data_hashes.json"
REPRO_SUMMARY_FILE = "reproducibility_summary.json"
UNIVERSE_SNAPSHOT_FILE = "universe_snapshot.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _duration_seconds(started_at: Optional[str], completed_at: Optional[str]) -> Optional[float]:
    if started_at is None or completed_at is None:
        return None
    start_ts = pd.Timestamp(started_at)
    end_ts = pd.Timestamp(completed_at)
    return float(max(0.0, (end_ts - start_ts).total_seconds()))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_git(command: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(command, cwd=str(_repo_root()), stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def _git_commit_hash() -> Optional[str]:
    return _run_git(["git", "rev-parse", "HEAD"])


def _git_branch() -> Optional[str]:
    return _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _default_runs_root() -> Path:
    value = os.getenv("DPOLARIS_RUNS_DIR", "runs")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _default_universe_path() -> Path:
    value = os.getenv("DPOLARIS_ACTIVE_UNIVERSE", "universe/combined_1000.json")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _fallback_universe_payload(run_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "generated_at": _utc_now().isoformat(),
        "criteria": {
            "source": "run_tickers_fallback",
            "reason": "active_universe_file_missing_or_invalid",
        },
        "data_sources": [{"name": "run_summary_tickers"}],
        "merged": [{"symbol": s, "sources": ["run_request"]} for s in (run_summary.get("tickers") or [])],
        "notes": ["Universe snapshot fallback generated from run_summary.tickers"],
    }


def _load_universe_payload(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _write_universe_snapshot(
    *,
    run_dir: Path,
    run_summary: dict[str, Any],
    explicit_snapshot: Optional[dict[str, Any]] = None,
    source_path: Optional[str | Path] = None,
) -> tuple[dict[str, Any], str]:
    payload: Optional[dict[str, Any]] = None

    if isinstance(explicit_snapshot, dict):
        payload = dict(explicit_snapshot)

    if payload is None:
        universe_path = Path(source_path).expanduser() if source_path is not None else _default_universe_path()
        if not universe_path.is_absolute():
            universe_path = _repo_root() / universe_path
        payload = _load_universe_payload(universe_path)

    if payload is None:
        payload = _fallback_universe_payload(run_summary)

    body = dict(payload)
    body.pop("universe_hash", None)
    universe_hash = _json_sha256(body)
    body["universe_hash"] = universe_hash

    _write_json(run_dir / UNIVERSE_SNAPSHOT_FILE, body)
    return body, universe_hash


def _normalize_summary(
    payload: Optional[dict[str, Any]],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    out = dict(defaults)
    if payload:
        for key, value in payload.items():
            out[key] = value
    return out


def _build_run_summary(
    *,
    run_id: str,
    status: str,
    started_at: Optional[Any],
    completed_at: Optional[Any],
    model_type: str,
    target: str,
    horizon: int,
    tickers: Sequence[str],
    timeframes: Sequence[str],
    environment: Optional[dict[str, Any]] = None,
    universe_hash: Optional[str] = None,
) -> dict[str, Any]:
    created_at = _utc_now().isoformat()
    started_iso = _to_iso(started_at) or created_at
    completed_iso = _to_iso(completed_at)

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "torch_version": _safe_import_version("torch"),
        "numpy_version": _safe_import_version("numpy"),
        "pandas_version": _safe_import_version("pandas"),
    }
    if environment:
        env.update(environment)

    return {
        "run_id": run_id,
        "status": status,
        "created_at": created_at,
        "started_at": started_iso,
        "completed_at": completed_iso,
        "duration_seconds": _duration_seconds(started_iso, completed_iso),
        "git_commit_hash": _git_commit_hash(),
        "git_branch": _git_branch(),
        "environment": env,
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "model_type": model_type,
        "target": target,
        "horizon": int(horizon),
        "tickers": [str(x).upper() for x in tickers],
        "timeframes": [str(x) for x in timeframes],
        "universe_hash": universe_hash,
    }


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _new_run_id(prefix: str = "run") -> str:
    return f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{prefix}_{uuid4().hex[:8]}"


def _ensure_unique_run_dir(root: Path, run_id: str) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    candidate_id = run_id
    candidate_dir = root / candidate_id
    if not candidate_dir.exists():
        candidate_dir.mkdir(parents=True, exist_ok=False)
        return candidate_id, candidate_dir

    for idx in range(1, 1000):
        candidate_id = f"{run_id}_{idx:02d}"
        candidate_dir = root / candidate_id
        if not candidate_dir.exists():
            candidate_dir.mkdir(parents=True, exist_ok=False)
            return candidate_id, candidate_dir

    raise RuntimeError(f"Unable to create unique run directory for {run_id}")


def _copy_artifact_files(
    run_dir: Path,
    artifact_files: Optional[Sequence[str | Path]],
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not artifact_files:
        return copied

    target_dir = run_dir / "artifacts"
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in artifact_files:
        src = Path(item).expanduser()
        if not src.exists() or not src.is_file():
            copied.append(
                {
                    "source": str(src),
                    "copied": False,
                    "reason": "missing_or_not_file",
                }
            )
            continue

        dst = target_dir / src.name
        try:
            shutil.copy2(src, dst)
            copied.append(
                {
                    "source": str(src),
                    "path": f"artifacts/{src.name}",
                    "copied": True,
                }
            )
        except Exception as exc:
            copied.append(
                {
                    "source": str(src),
                    "copied": False,
                    "reason": str(exc),
                }
            )
    return copied


def _json_sha256(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _resolve_primary_score(metrics_summary: dict[str, Any]) -> Optional[float]:
    if not isinstance(metrics_summary, dict):
        return None

    direct = metrics_summary.get("primary_score")
    if isinstance(direct, (int, float)):
        return float(direct)

    trading = metrics_summary.get("trading") or {}
    if isinstance(trading, dict):
        sharpe = trading.get("sharpe")
        if isinstance(sharpe, (int, float)):
            return float(sharpe)

    classification = metrics_summary.get("classification") or {}
    if isinstance(classification, dict):
        f1 = classification.get("f1")
        if isinstance(f1, (int, float)):
            return float(f1)

    return None


def _build_config_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    run_summary = payload.get("run_summary", {})
    metrics_summary = payload.get("metrics_summary", {})
    data_summary = payload.get("data_summary", {})
    feature_summary = payload.get("feature_summary", {})
    split_summary = payload.get("split_summary", {})
    backtest_summary = payload.get("backtest_summary", {})

    return {
        "snapshot_version": "1.0.0",
        "generated_at": _utc_now().isoformat(),
        "training_artifact_version": payload.get("training_artifact_version"),
        "run_id": run_summary.get("run_id"),
        "status": run_summary.get("status"),
        "model": {
            "model_type": run_summary.get("model_type"),
            "target": run_summary.get("target"),
            "horizon": run_summary.get("horizon"),
            "tickers": run_summary.get("tickers", []),
            "timeframes": run_summary.get("timeframes", []),
        },
        "validation": {
            "method": split_summary.get("validation_method") or "walk_forward",
            "walk_forward_windows": split_summary.get("walk_forward_windows", []),
            "sample_sizes": split_summary.get("sample_sizes", {}),
        },
        "features": {
            "feature_registry_version": feature_summary.get("feature_registry_version"),
            "feature_count": len(feature_summary.get("features", []) or []),
            "normalization_method": feature_summary.get("normalization_method"),
            "leakage_checks_status": feature_summary.get("leakage_checks_status"),
        },
        "data": {
            "sources_used": data_summary.get("sources_used", []),
            "start": data_summary.get("start"),
            "end": data_summary.get("end"),
            "bars_count": data_summary.get("bars_count"),
            "quality_gates": data_summary.get("quality_gates", {}),
        },
        "backtest": {
            "assumptions": backtest_summary.get("assumptions", {}),
        },
        "primary_score": _resolve_primary_score(metrics_summary),
        "universe_hash": run_summary.get("universe_hash"),
    }


def _build_dependency_snapshot(run_summary: dict[str, Any]) -> dict[str, Any]:
    dependencies: list[dict[str, str]] = []
    try:
        for dist in importlib.metadata.distributions():
            name = (dist.metadata.get("Name") or "").strip() or getattr(dist, "name", "")
            version = str(getattr(dist, "version", ""))
            if not name:
                continue
            dependencies.append({"name": name, "version": version})
    except Exception:
        dependencies = []

    dependencies.sort(key=lambda x: (x.get("name", "").lower(), x.get("version", "")))

    return {
        "snapshot_version": "1.0.0",
        "generated_at": _utc_now().isoformat(),
        "run_id": run_summary.get("run_id"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
        },
        "dependencies": dependencies,
    }


def _build_data_hashes(
    payload: dict[str, Any],
    run_dir: Path,
    copied_files: Sequence[dict[str, Any]],
    universe_snapshot_path: Optional[Path] = None,
) -> dict[str, Any]:
    hashes: list[dict[str, Any]] = []

    section_ids = [
        "run_summary",
        "data_summary",
        "feature_summary",
        "split_summary",
        "model_summary",
        "metrics_summary",
        "backtest_summary",
        "diagnostics_summary",
    ]
    for section_id in section_ids:
        hashes.append(
            {
                "id": section_id,
                "kind": "section",
                "sha256": _json_sha256(payload.get(section_id, {})),
                "resolvable": True,
            }
        )

    for copied in copied_files:
        source = copied.get("source")
        relative_path = copied.get("path")
        copied_ok = bool(copied.get("copied"))

        if copied_ok and isinstance(relative_path, str) and relative_path:
            target_path = run_dir / relative_path
            sha256 = _file_sha256(target_path)
            hashes.append(
                {
                    "id": f"artifact:{Path(relative_path).name}",
                    "kind": "artifact_file",
                    "path": relative_path,
                    "source": source,
                    "sha256": sha256,
                    "resolvable": bool(sha256),
                }
            )
            continue

        hashes.append(
            {
                "id": f"artifact_source:{Path(str(source)).name}" if source else "artifact_source:unknown",
                "kind": "artifact_file",
                "path": relative_path,
                "source": source,
                "sha256": None,
                "resolvable": False,
                "reason": copied.get("reason") or "copy_failed",
            }
        )

    if universe_snapshot_path is not None:
        universe_sha = _file_sha256(universe_snapshot_path)
        hashes.append(
            {
                "id": "universe_snapshot",
                "kind": "snapshot_file",
                "path": UNIVERSE_SNAPSHOT_FILE,
                "sha256": universe_sha,
                "resolvable": bool(universe_sha),
            }
        )

    data_summary_hash = next((x.get("sha256") for x in hashes if x.get("id") == "data_summary"), None)
    dataset_hash_id = data_summary_hash[:16] if isinstance(data_summary_hash, str) and data_summary_hash else None
    all_resolvable = all(bool(item.get("resolvable")) for item in hashes)

    return {
        "hash_algorithm": "sha256",
        "generated_at": _utc_now().isoformat(),
        "dataset_hash_id": dataset_hash_id,
        "hashes": hashes,
        "all_resolvable": all_resolvable,
    }


def _build_reproducibility_summary(
    *,
    config_snapshot: dict[str, Any],
    dependency_snapshot: dict[str, Any],
    data_hashes: dict[str, Any],
) -> dict[str, Any]:
    snapshots_present = bool(config_snapshot) and bool(dependency_snapshot) and bool(data_hashes)
    data_hashes_resolvable = bool(data_hashes.get("all_resolvable"))
    re_executable = snapshots_present and data_hashes_resolvable

    checks = {
        "snapshots_present": snapshots_present,
        "data_hashes_resolvable": data_hashes_resolvable,
        "re_executable_with_same_config": re_executable,
    }
    score = int(round((sum(1 for v in checks.values() if v) * 100.0) / len(checks)))

    return {
        "snapshot_version": "1.0.0",
        "generated_at": _utc_now().isoformat(),
        "checks": checks,
        "score": score,
        "snapshots": {
            "config_snapshot": CONFIG_SNAPSHOT_FILE,
            "dependency_snapshot": DEPENDENCY_SNAPSHOT_FILE,
            "data_hashes": DATA_HASHES_FILE,
        },
        "data_hashes_resolvable": data_hashes_resolvable,
        "re_executable_with_same_config": re_executable,
    }


def normalize_training_artifact(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible normalization.

    Supports:
    - v1 section keys
    - legacy camel-case section names
    - sparse payloads missing sections
    """
    payload = dict(raw or {})
    run_summary = payload.get("run_summary") or payload.get("RunSummary") or {}
    data_summary = payload.get("data_summary") or payload.get("DataSummary") or {}
    feature_summary = payload.get("feature_summary") or payload.get("FeatureSummary") or {}
    split_summary = payload.get("split_summary") or payload.get("SplitSummary") or {}
    model_summary = payload.get("model_summary") or payload.get("ModelSummary") or {}
    metrics_summary = payload.get("metrics_summary") or payload.get("MetricsSummary") or {}
    backtest_summary = payload.get("backtest_summary") or payload.get("BacktestSummary") or {}
    diagnostics_summary = payload.get("diagnostics_summary") or payload.get("DiagnosticsSummary") or {}
    reproducibility_summary = payload.get("reproducibility_summary") or payload.get("ReproducibilitySummary") or {}

    run_defaults = {
        "run_id": payload.get("run_id") or "unknown",
        "status": payload.get("status") or "unknown",
        "created_at": payload.get("created_at") or _utc_now().isoformat(),
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "git_commit_hash": None,
        "git_branch": None,
        "environment": {},
        "hostname": None,
        "user": None,
        "model_type": payload.get("model_type") or "unknown",
        "target": payload.get("target") or "target_direction",
        "horizon": int(payload.get("horizon") or 5),
        "tickers": payload.get("tickers") or [],
        "timeframes": payload.get("timeframes") or [],
        "reproducibility_score": None,
        "universe_hash": None,
    }
    data_defaults = {
        "sources_used": [],
        "start": None,
        "end": None,
        "bars_count": None,
        "missingness_report": {},
        "corporate_actions_applied": [],
        "adjustments": [],
        "outliers_detected": {},
        "drop_or_repair_decisions": [],
        "quality_gates": {},
        "data_hashes": {},
    }
    feature_defaults = {
        "feature_registry_version": None,
        "features": [],
        "missingness_per_feature": {},
        "normalization_method": None,
        "leakage_checks_status": "unknown",
    }
    split_defaults = {
        "walk_forward_windows": [],
        "train_ranges": [],
        "val_ranges": [],
        "test_ranges": [],
        "sample_sizes": {},
    }
    model_defaults = {
        "algorithm": "unknown",
        "hyperparameters": {},
        "feature_importance": [],
        "calibration_method": None,
    }
    metrics_defaults = {
        "classification": {},
        "regression": {},
        "trading": {},
        "calibration": {},
    }
    backtest_defaults = {
        "assumptions": {},
        "equity_curve_stats": {},
        "trade_list_artifact": None,
    }
    diagnostics_defaults = {
        "drift_baseline_stats": {},
        "regime_distribution": {},
        "error_analysis": {},
        "top_failure_cases": [],
    }
    reproducibility_defaults = {
        "checks": {
            "snapshots_present": False,
            "data_hashes_resolvable": False,
            "re_executable_with_same_config": False,
        },
        "score": 0,
        "snapshots": {
            "config_snapshot": CONFIG_SNAPSHOT_FILE,
            "dependency_snapshot": DEPENDENCY_SNAPSHOT_FILE,
            "data_hashes": DATA_HASHES_FILE,
        },
        "data_hashes_resolvable": False,
        "re_executable_with_same_config": False,
    }

    normalized = {
        "training_artifact_version": str(payload.get("training_artifact_version") or "0.0.0"),
        "run_summary": _normalize_summary(run_summary, run_defaults),
        "data_summary": _normalize_summary(data_summary, data_defaults),
        "feature_summary": _normalize_summary(feature_summary, feature_defaults),
        "split_summary": _normalize_summary(split_summary, split_defaults),
        "model_summary": _normalize_summary(model_summary, model_defaults),
        "metrics_summary": _normalize_summary(metrics_summary, metrics_defaults),
        "backtest_summary": _normalize_summary(backtest_summary, backtest_defaults),
        "diagnostics_summary": _normalize_summary(diagnostics_summary, diagnostics_defaults),
        "reproducibility_summary": _normalize_summary(reproducibility_summary, reproducibility_defaults),
        "artifacts": payload.get("artifacts") or {"files": []},
    }

    rs = normalized["run_summary"]
    if not rs.get("duration_seconds"):
        rs["duration_seconds"] = _duration_seconds(rs.get("started_at"), rs.get("completed_at"))

    repro = normalized["reproducibility_summary"]
    if rs.get("reproducibility_score") is None:
        score = repro.get("score")
        if isinstance(score, (int, float)):
            rs["reproducibility_score"] = int(score)

    if not normalized["data_summary"].get("data_hashes"):
        hashes = payload.get("data_hashes")
        if isinstance(hashes, dict):
            normalized["data_summary"]["data_hashes"] = hashes

    return normalized


def write_training_artifact(
    *,
    run_id: Optional[str],
    status: str,
    model_type: str,
    target: str,
    horizon: int,
    tickers: Sequence[str],
    timeframes: Sequence[str],
    data_summary: Optional[dict[str, Any]] = None,
    feature_summary: Optional[dict[str, Any]] = None,
    split_summary: Optional[dict[str, Any]] = None,
    model_summary: Optional[dict[str, Any]] = None,
    metrics_summary: Optional[dict[str, Any]] = None,
    backtest_summary: Optional[dict[str, Any]] = None,
    diagnostics_summary: Optional[dict[str, Any]] = None,
    started_at: Optional[Any] = None,
    completed_at: Optional[Any] = None,
    environment: Optional[dict[str, Any]] = None,
    artifact_files: Optional[Sequence[str | Path]] = None,
    run_root: Optional[str | Path] = None,
    universe_snapshot: Optional[dict[str, Any]] = None,
    universe_source_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    root = Path(run_root).expanduser() if run_root is not None else _default_runs_root()
    rid = run_id or _new_run_id(model_type.replace(" ", "_").lower())
    rid, run_dir = _ensure_unique_run_dir(root, rid)

    payload = normalize_training_artifact(
        {
            "training_artifact_version": TRAINING_ARTIFACT_VERSION,
            "run_summary": _build_run_summary(
                run_id=rid,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                model_type=model_type,
                target=target,
                horizon=horizon,
                tickers=tickers,
                timeframes=timeframes,
                environment=environment,
                universe_hash=None,
            ),
            "data_summary": data_summary or {},
            "feature_summary": feature_summary or {},
            "split_summary": split_summary or {},
            "model_summary": model_summary or {},
            "metrics_summary": metrics_summary or {},
            "backtest_summary": backtest_summary or {},
            "diagnostics_summary": diagnostics_summary or {},
            "reproducibility_summary": {},
            "artifacts": {"files": []},
        }
    )

    universe_payload, universe_hash = _write_universe_snapshot(
        run_dir=run_dir,
        run_summary=payload["run_summary"],
        explicit_snapshot=universe_snapshot,
        source_path=universe_source_path,
    )
    payload["run_summary"]["universe_hash"] = universe_hash

    copied = _copy_artifact_files(run_dir, artifact_files)
    payload["artifacts"]["files"] = copied

    config_snapshot = _build_config_snapshot(payload)
    dependency_snapshot = _build_dependency_snapshot(payload["run_summary"])
    data_hashes = _build_data_hashes(
        payload,
        run_dir,
        copied,
        universe_snapshot_path=(run_dir / UNIVERSE_SNAPSHOT_FILE),
    )
    reproducibility_summary = _build_reproducibility_summary(
        config_snapshot=config_snapshot,
        dependency_snapshot=dependency_snapshot,
        data_hashes=data_hashes,
    )

    payload["reproducibility_summary"] = reproducibility_summary
    payload["run_summary"]["reproducibility_score"] = reproducibility_summary.get("score")
    payload["data_summary"]["data_hashes"] = {
        "dataset_hash_id": data_hashes.get("dataset_hash_id"),
        "hash_algorithm": data_hashes.get("hash_algorithm"),
        "hash_count": len(data_hashes.get("hashes", [])),
        "all_resolvable": data_hashes.get("all_resolvable"),
    }

    _write_json(run_dir / ARTIFACT_FILE_NAME, payload)
    _write_json(run_dir / "run_summary.json", payload["run_summary"])
    _write_json(run_dir / "data_summary.json", payload["data_summary"])
    _write_json(run_dir / "feature_summary.json", payload["feature_summary"])
    _write_json(run_dir / "split_summary.json", payload["split_summary"])
    _write_json(run_dir / "model_summary.json", payload["model_summary"])
    _write_json(run_dir / "metrics_summary.json", payload["metrics_summary"])
    _write_json(run_dir / "backtest_summary.json", payload["backtest_summary"])
    _write_json(run_dir / "diagnostics_summary.json", payload["diagnostics_summary"])
    _write_json(run_dir / CONFIG_SNAPSHOT_FILE, config_snapshot)
    _write_json(run_dir / DEPENDENCY_SNAPSHOT_FILE, dependency_snapshot)
    _write_json(run_dir / DATA_HASHES_FILE, data_hashes)
    _write_json(run_dir / REPRO_SUMMARY_FILE, reproducibility_summary)
    _write_json(run_dir / UNIVERSE_SNAPSHOT_FILE, universe_payload)
    _write_json(
        run_dir / "manifest.json",
        {
            "training_artifact_version": TRAINING_ARTIFACT_VERSION,
            "run_id": rid,
            "generated_at": _utc_now().isoformat(),
            "files": sorted(
                [p.name for p in run_dir.iterdir() if p.is_file()]
                + [f.get("path") for f in copied if f.get("path")]
            ),
        },
    )

    return {
        "run_id": rid,
        "run_dir": str(run_dir),
        "artifact_path": str(run_dir / ARTIFACT_FILE_NAME),
        "artifact": payload,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_training_artifact(run_id: str, *, run_root: Optional[str | Path] = None) -> dict[str, Any]:
    root = Path(run_root).expanduser() if run_root is not None else _default_runs_root()
    run_dir = root / run_id
    artifact_path = run_dir / ARTIFACT_FILE_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Run artifact not found: {artifact_path}")
    with open(artifact_path) as f:
        raw = json.load(f)
    return normalize_training_artifact(raw)


def list_training_runs(*, run_root: Optional[str | Path] = None, limit: int = 50) -> list[dict[str, Any]]:
    root = Path(run_root).expanduser() if run_root is not None else _default_runs_root()
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted([x for x in root.iterdir() if x.is_dir()], reverse=True):
        artifact_path = run_dir / ARTIFACT_FILE_NAME
        if not artifact_path.exists():
            continue
        try:
            with open(artifact_path) as f:
                raw = json.load(f)
            normalized = normalize_training_artifact(raw)
            summary = normalized.get("run_summary", {})
            data_summary = normalized.get("data_summary", {})
            metrics = normalized.get("metrics_summary", {})
            reproducibility = normalized.get("reproducibility_summary", {})
            runs.append(
                {
                    "run_id": summary.get("run_id", run_dir.name),
                    "status": summary.get("status"),
                    "created_at": summary.get("created_at"),
                    "completed_at": summary.get("completed_at"),
                    "duration_seconds": summary.get("duration_seconds"),
                    "model_type": summary.get("model_type"),
                    "target": summary.get("target"),
                    "horizon": summary.get("horizon"),
                    "tickers": summary.get("tickers", []),
                    "timeframes": summary.get("timeframes", []),
                    "data_start": data_summary.get("start"),
                    "data_end": data_summary.get("end"),
                    "primary_score": _resolve_primary_score(metrics),
                    "reproducibility_score": reproducibility.get("score"),
                    "universe_hash": summary.get("universe_hash"),
                    "path": str(run_dir),
                }
            )
        except Exception:
            continue

    runs.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return runs[: max(1, int(limit))]


def list_run_artifact_files(run_id: str, *, run_root: Optional[str | Path] = None) -> list[str]:
    root = Path(run_root).expanduser() if run_root is not None else _default_runs_root()
    run_dir = root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    files: list[str] = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(run_dir)))
    return files


def resolve_run_artifact_path(
    run_id: str,
    artifact_name: str,
    *,
    run_root: Optional[str | Path] = None,
) -> Path:
    root = Path(run_root).expanduser() if run_root is not None else _default_runs_root()
    run_dir = (root / run_id).resolve()
    target = (run_dir / artifact_name).resolve()

    if run_dir not in target.parents and target != run_dir:
        raise ValueError("Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact_name}")
    return target


def compare_training_runs(
    run_ids: Sequence[str],
    *,
    run_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    compared: list[dict[str, Any]] = []
    for run_id in run_ids:
        artifact = load_training_artifact(run_id, run_root=run_root)
        run_summary = artifact["run_summary"]
        metrics = artifact["metrics_summary"]
        row = {
            "run_id": run_summary.get("run_id"),
            "status": run_summary.get("status"),
            "completed_at": run_summary.get("completed_at"),
            "model_type": run_summary.get("model_type"),
            "target": run_summary.get("target"),
            "horizon": run_summary.get("horizon"),
            "duration_seconds": run_summary.get("duration_seconds"),
            "classification_accuracy": metrics.get("classification", {}).get("accuracy"),
            "classification_f1": metrics.get("classification", {}).get("f1"),
            "classification_roc_auc": metrics.get("classification", {}).get("roc_auc"),
            "trading_sharpe": metrics.get("trading", {}).get("sharpe"),
            "trading_max_drawdown": metrics.get("trading", {}).get("max_drawdown"),
            "primary_score": _resolve_primary_score(metrics),
        }
        compared.append(row)

    best = {
        "best_f1_run": _best_run(compared, "classification_f1"),
        "best_sharpe_run": _best_run(compared, "trading_sharpe"),
        "lowest_drawdown_run": _best_run(compared, "trading_max_drawdown", reverse=False),
    }

    return {
        "run_ids": list(run_ids),
        "compared": compared,
        "best": best,
    }


def _best_run(rows: list[dict[str, Any]], key: str, reverse: bool = True) -> Optional[str]:
    valid = [r for r in rows if isinstance(r.get(key), (int, float))]
    if not valid:
        return None
    sorted_rows = sorted(valid, key=lambda x: float(x.get(key) or 0.0), reverse=reverse)
    return sorted_rows[0].get("run_id")
