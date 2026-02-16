from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Optional
from uuid import uuid4

ANALYSIS_ARTIFACT_VERSION = "1.0.0"
_ANALYSIS_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_id(value: str) -> str:
    cleaned = _ANALYSIS_ID_RE.sub("-", str(value or "")).strip("-._")
    return cleaned or f"analysis-{uuid4().hex[:10]}"


def _analysis_root(data_dir: Optional[str | Path] = None) -> Path:
    configured = os.getenv("DPOLARIS_ANALYSIS_DIR")
    if configured:
        root = Path(configured).expanduser()
        if root.is_absolute():
            return root
        base = Path(data_dir).expanduser() if data_dir is not None else Path("~/dpolaris_data").expanduser()
        return base / root

    base = Path(data_dir).expanduser() if data_dir is not None else Path("~/dpolaris_data").expanduser()
    return base / "analysis"


def _analysis_file(root: Path, analysis_id: str) -> Path:
    return root / f"{_sanitize_id(analysis_id)}.json"


def _analysis_index_file(root: Path) -> Path:
    return root / "index.jsonl"


def _build_default_id(ticker: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"analysis_{ticker.lower()}_{stamp}_{uuid4().hex[:8]}"


def _extract_confidence(payload: dict[str, Any]) -> Optional[float]:
    signals = payload.get("signals")
    if isinstance(signals, dict):
        raw = signals.get("confidence")
        try:
            return float(raw) if raw is not None else None
        except Exception:
            pass
    report = payload.get("report")
    if isinstance(report, dict):
        model_signals = report.get("model_signals")
        if isinstance(model_signals, dict):
            raw = model_signals.get("confidence")
            try:
                return float(raw) if raw is not None else None
            except Exception:
                pass
    return None


def _summary_view(payload: dict[str, Any], *, path: Optional[Path] = None) -> dict[str, Any]:
    ticker = payload.get("ticker")
    analysis_date = payload.get("analysis_date") or payload.get("created_at")
    return {
        "id": payload.get("id"),
        "ticker": ticker,
        "symbol": ticker,
        "created_at": payload.get("created_at"),
        "analysis_date": analysis_date,
        "model_type": payload.get("model_type"),
        "device": payload.get("device"),
        "summary": payload.get("summary"),
        "confidence": _extract_confidence(payload),
        "training_window": payload.get("training_window"),
        "source": payload.get("source"),
        "path": str(path) if path is not None else payload.get("path"),
        "url": payload.get("url"),
    }


def write_analysis_artifact(
    payload: dict[str, Any],
    *,
    data_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    ticker = str(payload.get("ticker") or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")

    body = dict(payload)
    body["ticker"] = ticker
    body["id"] = _sanitize_id(str(body.get("id") or _build_default_id(ticker)))
    body["created_at"] = str(body.get("created_at") or _utc_now_iso())
    body["analysis_date"] = str(body.get("analysis_date") or body["created_at"])
    body["summary"] = str(body.get("summary") or "")
    body["report_text"] = str(body.get("report_text") or "")
    body.setdefault("model_type", "none")
    body.setdefault("device", "cpu")
    body.setdefault("training_window", {})
    body.setdefault("version_info", {})
    body.setdefault("signals", {})
    body.setdefault("indicators", {})
    body.setdefault("news_refs", [])
    body.setdefault("artifact_version", ANALYSIS_ARTIFACT_VERSION)

    root = _analysis_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _analysis_file(root, body["id"])
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_analysis_index_entry(root=root, payload=body, path=path)
    return body


def _append_analysis_index_entry(*, root: Path, payload: dict[str, Any], path: Path) -> None:
    entry = {
        "id": payload.get("id"),
        "symbol": payload.get("ticker"),
        "ticker": payload.get("ticker"),
        "created_at": payload.get("created_at"),
        "analysis_date": payload.get("analysis_date") or payload.get("created_at"),
        "model_type": payload.get("model_type"),
        "run_id": payload.get("run_id"),
        "summary": payload.get("summary"),
        "summary_path": str(path),
        "full_path": str(path),
    }
    index_path = _analysis_index_file(root)
    try:
        with index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Keep artifact writes resilient even if index append fails.
        pass


def load_analysis_artifact(
    analysis_id: str,
    *,
    data_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    root = _analysis_root(data_dir)
    path = _analysis_file(root, analysis_id)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Analysis artifact not found: {analysis_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid analysis artifact: {analysis_id}")
    return payload


def list_analysis_artifacts(
    *,
    data_dir: Optional[str | Path] = None,
    limit: int = 200,
    ticker: Optional[str] = None,
) -> list[dict[str, Any]]:
    root = _analysis_root(data_dir)
    if not root.exists() or not root.is_dir():
        return []

    ticker_filter = str(ticker or "").strip().upper()
    rows: list[dict[str, Any]] = []

    for path in root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        symbol = str(payload.get("ticker") or "").strip().upper()
        if ticker_filter and symbol != ticker_filter:
            continue

        rows.append(_summary_view(payload, path=path))

    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    safe_limit = max(1, min(int(limit), 5000))
    return rows[:safe_limit]


def latest_analysis_for_symbol(
    ticker: str,
    *,
    data_dir: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None

    rows = list_analysis_artifacts(data_dir=data_dir, ticker=symbol, limit=1)
    if not rows:
        return None

    analysis_id = str(rows[0].get("id") or "").strip()
    if not analysis_id:
        return None

    try:
        return load_analysis_artifact(analysis_id, data_dir=data_dir)
    except Exception:
        return None
