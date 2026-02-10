"""
Versioned model registry with config/metrics/data-window lineage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Optional
from uuid import uuid4


class ModelRegistry:
    """
    Lightweight file-backed model registry.

    Stores:
    - model version records
    - config snapshots
    - metrics snapshots
    - data-window metadata for reproducibility
    """

    def __init__(
        self,
        *,
        root_dir: Path | str = "~/dpolaris_data/model_registry",
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.models_dir = self.root_dir / "models"
        self.state_path = self.root_dir / "registry.json"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state({"records": []})

    def register_model(
        self,
        *,
        model_name: str,
        model_path: Path | str,
        config: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        data_window: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        framework: Optional[str] = None,
        notes: str = "",
        training_trigger: str = "manual",
        version: Optional[str] = None,
        status: str = "active",
    ) -> dict[str, Any]:
        """
        Register one model artifact version.
        """
        name = str(model_name).strip()
        if not name:
            raise ValueError("model_name is required")

        artifact_path = Path(model_path).expanduser()
        if not artifact_path.exists():
            raise FileNotFoundError(f"model_path not found: {artifact_path}")

        state = self._read_state()
        records = list(state.get("records", []))

        version_token = str(version).strip() if version is not None else self._next_version(name, records)
        created_at = datetime.now(timezone.utc).isoformat()

        if status == "active":
            for rec in records:
                if rec.get("model_name") == name and rec.get("status") == "active":
                    rec["status"] = "superseded"
                    rec["superseded_at"] = created_at

        record = {
            "id": str(uuid4()),
            "model_name": name,
            "version": version_token,
            "created_at": created_at,
            "status": status,
            "framework": framework,
            "model_path": str(artifact_path),
            "model_sha256": self._sha256_file(artifact_path),
            "config": config or {},
            "metrics": metrics or {},
            "data_window": data_window or {},
            "tags": tags or [],
            "notes": notes,
            "training_trigger": training_trigger,
            "monitoring": {},
        }

        snapshot_dir = self.models_dir / name / version_token
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(snapshot_dir / "registry_record.json", record)
        self._write_json(snapshot_dir / "config_snapshot.json", record["config"])
        self._write_json(snapshot_dir / "metrics_snapshot.json", record["metrics"])
        self._write_json(snapshot_dir / "data_window.json", record["data_window"])

        records.append(record)
        self._write_state({"records": records})
        return record

    def list_models(
        self,
        *,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        records = list(self._read_state().get("records", []))

        if model_name is not None:
            token = str(model_name).strip()
            records = [r for r in records if r.get("model_name") == token]
        if status is not None:
            token = str(status).strip()
            records = [r for r in records if r.get("status") == token]

        records.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
        if limit is not None:
            records = records[: max(0, int(limit))]
        return records

    def get_model(
        self,
        *,
        model_name: str,
        version: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        candidates = self.list_models(model_name=model_name)
        if not candidates:
            return None

        if version is None:
            active = [x for x in candidates if x.get("status") == "active"]
            return active[0] if active else candidates[0]

        token = str(version).strip()
        for row in candidates:
            if row.get("version") == token:
                return row
        return None

    def set_status(
        self,
        *,
        model_name: str,
        version: str,
        status: str,
    ) -> dict[str, Any]:
        state = self._read_state()
        records = list(state.get("records", []))

        found: Optional[dict[str, Any]] = None
        for rec in records:
            if rec.get("model_name") == model_name and rec.get("version") == version:
                rec["status"] = status
                rec["status_updated_at"] = datetime.now(timezone.utc).isoformat()
                found = rec

        if found is None:
            raise KeyError(f"model record not found: {model_name} {version}")

        if status == "active":
            for rec in records:
                if (
                    rec.get("model_name") == model_name
                    and rec.get("version") != version
                    and rec.get("status") == "active"
                ):
                    rec["status"] = "superseded"
                    rec["superseded_at"] = datetime.now(timezone.utc).isoformat()

        self._write_state({"records": records})
        return found

    def attach_monitoring(
        self,
        *,
        model_name: str,
        version: str,
        drift_report: Optional[dict[str, Any]] = None,
        performance_report: Optional[dict[str, Any]] = None,
        retrain_decision: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Attach drift/performance diagnostics to a model record.
        """
        state = self._read_state()
        records = list(state.get("records", []))

        found: Optional[dict[str, Any]] = None
        for rec in records:
            if rec.get("model_name") == model_name and rec.get("version") == version:
                monitoring = dict(rec.get("monitoring", {}))
                monitoring["updated_at"] = datetime.now(timezone.utc).isoformat()
                if drift_report is not None:
                    monitoring["drift"] = drift_report
                if performance_report is not None:
                    monitoring["performance"] = performance_report
                if retrain_decision is not None:
                    monitoring["retrain_decision"] = retrain_decision
                rec["monitoring"] = monitoring
                found = rec
                break

        if found is None:
            raise KeyError(f"model record not found: {model_name} {version}")

        self._write_state({"records": records})
        return found

    def _next_version(self, model_name: str, records: list[dict[str, Any]]) -> str:
        pattern = re.compile(r"^v(\d+)$")
        candidates: list[int] = []
        for rec in records:
            if rec.get("model_name") != model_name:
                continue
            token = str(rec.get("version", ""))
            match = pattern.match(token)
            if match:
                candidates.append(int(match.group(1)))

        next_idx = max(candidates) + 1 if candidates else 1
        return f"v{next_idx:04d}"

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"records": []}
        with open(self.state_path) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            return {"records": []}
        if "records" not in loaded or not isinstance(loaded["records"], list):
            loaded["records"] = []
        return loaded

    def _write_state(self, state: dict[str, Any]) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.state_path, state)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
