"""
FastAPI Server for dPolaris

Provides REST API and WebSocket endpoints for the Mac app.
"""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
from contextlib import asynccontextmanager
import logging
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

from core.config import Config, get_config
from core.database import Database
from core.memory import DPolarisMemory
from core.ai import DPolarisAI
from core.llm_provider import LLMUnavailableError
from tools.market_data import MarketDataService, get_market_overview

try:
    from monitoring.drift import SelfCritiqueLogger
except Exception:  # pragma: no cover - keep API startup resilient
    SelfCritiqueLogger = None

try:
    from ml.training_artifacts import (
        compare_training_runs,
        list_run_artifact_files,
        list_training_runs,
        load_training_artifact,
        resolve_run_artifact_path,
        write_training_artifact,
    )
except Exception:  # pragma: no cover
    compare_training_runs = None
    list_run_artifact_files = None
    list_training_runs = None
    load_training_artifact = None
    resolve_run_artifact_path = None
    write_training_artifact = None

try:
    from ml.prediction_inspector import (
        build_feature_snapshot,
        decide_trade_outcome,
        derive_regime,
        latest_ohlcv_snapshot,
        truncate_dataframe_asof,
    )
except Exception:  # pragma: no cover
    build_feature_snapshot = None
    decide_trade_outcome = None
    derive_regime = None
    latest_ohlcv_snapshot = None
    truncate_dataframe_asof = None

logger = logging.getLogger("dpolaris.api")

# Global instances
config: Optional[Config] = None
db: Optional[Database] = None
memory: Optional[DPolarisMemory] = None
ai: Optional[DPolarisAI] = None
market_service: Optional[MarketDataService] = None
self_critique_logger = None
training_jobs: dict[str, dict] = {}
training_job_order: list[str] = []
training_job_queue: Optional[asyncio.Queue[str]] = None
training_job_worker_task: Optional[asyncio.Task] = None
scan_jobs: dict[str, dict] = {}
scan_job_order: list[str] = []
scan_job_queue: Optional[asyncio.Queue[str]] = None
scan_job_worker_task: Optional[asyncio.Task] = None
server_started_at: Optional[datetime] = None

SUPPORTED_DL_MODELS = {"lstm", "transformer"}
MAX_TRAINING_JOBS = 200
MAX_TRAINING_JOB_LOG_LINES = 500
MAX_SCAN_JOBS = 100
MAX_SCAN_JOB_LOG_LINES = 2000
SCAN_STATE_FILE = "scan_job.json"
SCAN_INDEX_FILE = "scan_results_index.json"
SCAN_REQUEST_FILE = "scan_request.json"
SCAN_RESULTS_DIR = "scan_results"
DEFAULT_UNIVERSE_SCHEMA_VERSION = "1.0.0"
KNOWN_UNIVERSE_NAMES = {"nasdaq_top_500", "wsb_top_500", "combined_1000"}
SUPPORTED_UNIVERSE_EXTENSIONS = {".json", ".yaml", ".yml", ".txt"}
LISTABLE_UNIVERSE_EXTENSIONS = {".json", ".yaml", ".yml", ".txt"}
FALLBACK_UNIVERSE_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "NFLX",
    "AMD", "INTC", "ADBE", "CSCO", "PEP", "QCOM", "TXN", "AMAT", "INTU", "BKNG",
    "MU", "LRCX", "ADI", "PANW", "KLAC", "CRWD", "MELI", "MAR", "MDLZ", "ADP",
    "SBUX", "AMGN", "ISRG", "PYPL", "GILD", "REGN", "VRTX", "ABNB", "DASH", "SNPS",
    "CDNS", "FTNT", "ORLY", "CTAS", "CSX", "ROP", "CMCSA", "TMUS", "PDD", "ASML",
]

SECTOR_ETF_MAP = {
    "basic materials": "XLB",
    "communication services": "XLC",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "energy": "XLE",
    "financial services": "XLF",
    "health care": "XLV",
    "industrials": "XLI",
    "real estate": "XLRE",
    "technology": "XLK",
    "utilities": "XLU",
}


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    candidates = [normalized]
    if "Z" in normalized:
        candidates.append(normalized.replace("Z", "+00:00"))

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    return None


def _format_uptime(started_at: Optional[datetime]) -> Optional[str]:
    if started_at is None:
        return None

    elapsed_seconds = max(0, int((datetime.utcnow() - started_at).total_seconds()))
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _pid_cmdline(pid: int) -> Optional[str]:
    if pid <= 0:
        return None
    if psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        parts = proc.cmdline()
        return " ".join(parts) if parts else None
    except Exception:
        return None


def _find_pid_on_port(port: int) -> Optional[int]:
    try:
        if psutil is not None:
            for conn in psutil.net_connections(kind="tcp"):
                laddr = getattr(conn, "laddr", None)
                status = str(getattr(conn, "status", "")).upper()
                if not laddr or status != "LISTEN":
                    continue
                if int(getattr(laddr, "port", -1)) == int(port):
                    pid = getattr(conn, "pid", None)
                    if pid:
                        return int(pid)
    except Exception:
        pass

    try:
        if os.name == "nt":
            proc = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
            needle = f":{int(port)}"
            for line in proc.stdout.splitlines():
                if needle not in line or "LISTENING" not in line.upper():
                    continue
                parts = line.split()
                if parts:
                    return int(parts[-1])
    except Exception:
        return None
    return None


def _port_owner_via_psutil(port: int) -> tuple[Optional[int], Optional[str]]:
    if psutil is None:
        return None, None
    try:
        for conn in psutil.net_connections(kind="tcp"):
            laddr = getattr(conn, "laddr", None)
            status = str(getattr(conn, "status", "")).upper()
            if not laddr or status != "LISTEN":
                continue
            if int(getattr(laddr, "port", -1)) != int(port):
                continue
            pid = getattr(conn, "pid", None)
            if not pid:
                return None, None
            cmdline = _pid_cmdline(int(pid))
            return int(pid), cmdline
    except Exception:
        return None, None
    return None, None


def _port_owner_via_netstat_cim(port: int) -> tuple[Optional[int], Optional[str]]:
    pid = _find_pid_on_port(int(port))
    if not pid:
        return None, None

    cmdline = None
    if os.name == "nt":
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "$p = Get-CimInstance Win32_Process -Filter \"ProcessId = "
                        f"{int(pid)}\"; if ($p) {{ $p.CommandLine }}"
                    ),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                text = (proc.stdout or "").strip()
                if text:
                    cmdline = text
        except Exception:
            cmdline = None
    return int(pid), cmdline


def _is_repo_server_cmdline(cmdline: Optional[str], port: int = 8420) -> bool:
    if not cmdline:
        return False
    cmd = str(cmdline).lower()
    repo = str(_repo_root()).lower()
    return (
        repo in cmd
        and "-m cli.main server" in cmd
        and f"--port {int(port)}" in cmd
    )


def _orchestrator_runtime_status(
    *,
    data_dir: Path,
    default_host: str = "127.0.0.1",
    default_port: int = 8420,
    heartbeat_stale_seconds: int = 180,
) -> dict[str, Any]:
    run_dir = data_dir / "run"
    pid_path = run_dir / "orchestrator.pid"
    hb_path = run_dir / "orchestrator.heartbeat.json"
    backend_pid_path = run_dir / "backend.pid"

    pid_exists = pid_path.exists()
    pid = None
    if pid_exists:
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None

    hb_payload: dict[str, Any] = {}
    heartbeat_exists = hb_path.exists()
    if heartbeat_exists:
        try:
            with open(hb_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                hb_payload = loaded
        except Exception:
            hb_payload = {}

    host = str(hb_payload.get("host") or default_host)
    port = int(hb_payload.get("port") or default_port)

    last_hb_raw = hb_payload.get("last_heartbeat")
    last_hb_dt = _parse_timestamp(last_hb_raw) if isinstance(last_hb_raw, str) else None
    heartbeat_age_seconds = None
    heartbeat_recent = False
    if last_hb_dt is not None:
        heartbeat_age_seconds = max(0, int((datetime.utcnow() - last_hb_dt).total_seconds()))
        heartbeat_recent = heartbeat_age_seconds <= int(heartbeat_stale_seconds)

    pid_alive = _is_pid_alive(int(pid)) if pid else False
    running = bool(pid_exists and pid and pid_alive and heartbeat_recent)

    backend_pid = None
    if backend_pid_path.exists():
        try:
            backend_pid = int(backend_pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            backend_pid = None
    backend_pid_alive = _is_pid_alive(int(backend_pid)) if backend_pid else False
    port_owner_pid = _find_pid_on_port(port)
    port_owner_unknown = bool(
        port_owner_pid
        and (not backend_pid or int(port_owner_pid) != int(backend_pid))
    )

    return {
        "running": running,
        "pid_exists": pid_exists,
        "pid": pid,
        "pid_alive": pid_alive,
        "heartbeat_exists": heartbeat_exists,
        "heartbeat_file": str(hb_path),
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "heartbeat_recent": heartbeat_recent,
        "heartbeat_stale_threshold_seconds": int(heartbeat_stale_seconds),
        "started_at": hb_payload.get("started_at"),
        "last_heartbeat": hb_payload.get("last_heartbeat"),
        "host": host,
        "port": port,
        "pid_file": str(pid_path),
        "orchestrator_cmdline": _pid_cmdline(int(pid)) if pid else None,
        "port_owner_unknown": port_owner_unknown,
        "backend_state": {
            "pid": backend_pid,
            "pid_file": str(backend_pid_path),
            "running": bool(backend_pid and backend_pid_alive),
            "pid_alive": backend_pid_alive,
            "port_owner_pid": port_owner_pid,
            "port_owner_unknown": port_owner_unknown,
            "health_url": f"http://{host}:{port}/health",
        },
        "heartbeat_payload": hb_payload,
    }


def _public_training_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "type": job["type"],
        "symbol": job["symbol"],
        "model_type": job["model_type"],
        "epochs": job["epochs"],
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "logs": job.get("logs", []),
    }


def _llm_disabled_detail() -> dict[str, Any]:
    provider = "none"
    reason = "LLM provider is disabled."
    if ai is not None:
        provider = ai.llm_provider_name
        reason = ai.llm_disabled_reason or reason
    return {
        "error": "llm_disabled",
        "provider": provider,
        "detail": reason,
    }


def _require_llm_enabled() -> None:
    if ai is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "detail": "AI engine is not initialized."},
        )
    try:
        ai.require_llm_enabled()
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail=_llm_disabled_detail())


def _scheduler_dependency_detail() -> dict[str, str]:
    return {
        "detail": "scheduler_disabled",
        "reason": "apscheduler not installed",
        "fix": "pip install apscheduler",
    }


def _is_apscheduler_missing(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return exc.name == "apscheduler"
    if isinstance(exc, ImportError):
        return "apscheduler" in str(exc).lower()
    return False


def _raise_if_scheduler_dependency_missing(exc: Exception) -> bool:
    if not _is_apscheduler_missing(exc):
        return False
    raise HTTPException(status_code=503, detail=_scheduler_dependency_detail())


def _scheduler_dependency_response() -> JSONResponse:
    return JSONResponse(status_code=503, content=_scheduler_dependency_detail())


def _trim_training_jobs() -> None:
    if len(training_job_order) <= MAX_TRAINING_JOBS:
        return

    overflow = len(training_job_order) - MAX_TRAINING_JOBS
    for _ in range(overflow):
        old_job_id = training_job_order.pop(0)
        job = training_jobs.get(old_job_id)

        if job and job.get("status") in {"queued", "running"}:
            training_job_order.append(old_job_id)
            continue

        training_jobs.pop(old_job_id, None)


def _append_training_job_log(job: dict, message: str) -> None:
    cleaned = str(message).strip()
    if not cleaned:
        return

    logs = job.setdefault("logs", [])
    logs.append(f"{utc_now_iso()} | {cleaned}")
    if len(logs) > MAX_TRAINING_JOB_LOG_LINES:
        del logs[:-MAX_TRAINING_JOB_LOG_LINES]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runs_root() -> Path:
    raw = os.getenv("DPOLARIS_RUNS_DIR", "runs")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _scan_run_dir(run_id: str) -> Path:
    return _runs_root() / run_id


def _scan_results_dir(run_id: str) -> Path:
    return _scan_run_dir(run_id) / SCAN_RESULTS_DIR


def _scan_state_path(run_id: str) -> Path:
    return _scan_run_dir(run_id) / SCAN_STATE_FILE


def _scan_index_path(run_id: str) -> Path:
    return _scan_run_dir(run_id) / SCAN_INDEX_FILE


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _json_load(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return None
    return None


def _sanitize_symbol(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if any(ch in text for ch in (" ", "/", "\\", "=")):
        return None
    text = text.replace(".", "-")
    if not text[0].isalpha():
        return None
    if len(text) > 10:
        return None
    for ch in text:
        if not (ch.isalnum() or ch == "-"):
            return None
    return text


def _json_sha256(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _universe_with_hash(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("universe_hash", None)
    body["universe_hash"] = _json_sha256(body)
    return body


def _fallback_symbols() -> list[str]:
    raw = os.getenv("DPOLARIS_FALLBACK_SYMBOLS", "")
    if raw.strip():
        parsed = [_sanitize_symbol(x) for x in raw.split(",")]
        cleaned = [x for x in parsed if x]
        if cleaned:
            return cleaned
    return list(FALLBACK_UNIVERSE_SYMBOLS)


def _build_fallback_nasdaq_payload(symbols: list[str]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "symbol": symbol,
            "company_name": symbol,
            "market_cap": None,
            "avg_dollar_volume": None,
            "sector": None,
            "industry": None,
        }
        for symbol in symbols
    ]
    payload = {
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "criteria": {
            "exchange": "NASDAQ",
            "top_n_requested": 500,
            "top_n_returned": len(rows),
            "ranking": "fallback_symbol_list",
            "liquidity_filter": {
                "metric": "average_dollar_volume",
                "min_avg_dollar_volume": 0.0,
            },
            "candidate_limit": len(rows),
        },
        "data_sources": [
            {
                "name": "fallback_symbol_list",
                "notes": "generated because universe file was missing",
            }
        ],
        "notes": [
            "Fallback universe generated in API runtime due missing universe file.",
            "Set DPOLARIS_FALLBACK_SYMBOLS for a custom symbol list.",
        ],
        "tickers": rows,
    }
    return _universe_with_hash(payload)


def _build_fallback_wsb_payload(symbols: list[str]) -> dict[str, Any]:
    generated = datetime.now(timezone.utc)
    rows = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols, start=1):
        mentions = max(1, total - idx + 1)
        rows.append(
            {
                "symbol": symbol,
                "mention_count": mentions,
                "mention_velocity": round(float(mentions) / 7.0, 6),
                "sentiment_score": 0.0,
                "example_post_ids": [],
                "example_titles": [],
            }
        )
    payload = {
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "window_start": (generated - timedelta(days=7)).isoformat(),
        "window_end": generated.isoformat(),
        "criteria": {
            "top_n_requested": 500,
            "top_n_returned": len(rows),
            "window_days": 7,
            "dedupe": "fallback",
            "ticker_linking": "fallback",
        },
        "source_connector": "fallback_mentions",
        "data_sources": [
            {
                "name": "fallback_mentions",
                "raw_posts": 0,
                "deduped_posts": 0,
            }
        ],
        "notes": [
            "Fallback WSB universe generated in API runtime due missing universe file.",
            "Replace with scheduled universe build for real mention data.",
        ],
        "tickers": rows,
    }
    return _universe_with_hash(payload)


def _build_fallback_combined_payload(ns_payload: dict[str, Any], ws_payload: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    merged: dict[str, dict[str, Any]] = {}
    for row in (ns_payload.get("tickers") or []):
        symbol = _sanitize_symbol((row or {}).get("symbol"))
        if not symbol:
            continue
        merged[symbol] = {
            "symbol": symbol,
            "company_name": row.get("company_name") or symbol,
            "market_cap": row.get("market_cap"),
            "avg_dollar_volume": row.get("avg_dollar_volume"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "mention_count": 0,
            "mention_velocity": 0.0,
            "sentiment_score": None,
            "sources": ["nasdaq_top_500"],
        }
    for row in (ws_payload.get("tickers") or []):
        symbol = _sanitize_symbol((row or {}).get("symbol"))
        if not symbol:
            continue
        item = merged.setdefault(
            symbol,
            {
                "symbol": symbol,
                "company_name": symbol,
                "market_cap": None,
                "avg_dollar_volume": None,
                "sector": None,
                "industry": None,
                "mention_count": 0,
                "mention_velocity": 0.0,
                "sentiment_score": None,
                "sources": [],
            },
        )
        item["mention_count"] = int(row.get("mention_count") or 0)
        item["mention_velocity"] = float(row.get("mention_velocity") or 0.0)
        if row.get("sentiment_score") is not None:
            item["sentiment_score"] = float(row.get("sentiment_score"))
        if "wsb_top_500" not in item["sources"]:
            item["sources"].append("wsb_top_500")

    merged_rows = sorted(
        merged.values(),
        key=lambda x: (
            float(x.get("market_cap") or 0.0),
            float(x.get("avg_dollar_volume") or 0.0),
            int(x.get("mention_count") or 0),
            str(x.get("symbol") or ""),
        ),
        reverse=True,
    )[:1000]

    payload = {
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "criteria": {
            "top_n_requested": 1000,
            "top_n_returned": len(merged_rows),
            "duplicate_resolution": "merge_on_symbol",
        },
        "data_sources": [
            {"name": "nasdaq_top_500", "count": len(ns_payload.get("tickers") or [])},
            {"name": "wsb_top_500", "count": len(ws_payload.get("tickers") or [])},
        ],
        "notes": [
            "Fallback combined universe generated in API runtime due missing universe files.",
        ],
        "nasdaq_top_500": ns_payload.get("tickers") or [],
        "wsb_top_500": ws_payload.get("tickers") or [],
        "merged": merged_rows,
    }
    return _universe_with_hash(payload)


def _ensure_default_universe_file(universe_name: str) -> Optional[Path]:
    base_name = (universe_name or "").strip()
    if base_name.endswith(".json"):
        base_name = base_name[:-5]
    if base_name not in KNOWN_UNIVERSE_NAMES:
        return None

    universe_dir = _repo_root() / "universe"
    universe_dir.mkdir(parents=True, exist_ok=True)
    target = universe_dir / f"{base_name}.json"
    if target.exists() and target.is_file():
        return target

    symbols = _fallback_symbols()
    nasdaq_path = universe_dir / "nasdaq_top_500.json"
    wsb_path = universe_dir / "wsb_top_500.json"
    combined_path = universe_dir / "combined_1000.json"

    if base_name == "nasdaq_top_500":
        _json_dump(nasdaq_path, _build_fallback_nasdaq_payload(symbols))
        return nasdaq_path

    if base_name == "wsb_top_500":
        _json_dump(wsb_path, _build_fallback_wsb_payload(symbols))
        return wsb_path

    if not nasdaq_path.exists():
        _json_dump(nasdaq_path, _build_fallback_nasdaq_payload(symbols))
    if not wsb_path.exists():
        _json_dump(wsb_path, _build_fallback_wsb_payload(symbols))

    ns_payload = _json_load(nasdaq_path) or _build_fallback_nasdaq_payload(symbols)
    ws_payload = _json_load(wsb_path) or _build_fallback_wsb_payload(symbols)
    _json_dump(combined_path, _build_fallback_combined_payload(ns_payload, ws_payload))
    return combined_path


def _resolve_universe_path(universe_name: str) -> Path:
    name = (universe_name or "").strip()
    if not name:
        raise ValueError("universe is required")

    candidate = Path(name).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    root = _repo_root()
    if not candidate.is_absolute():
        repo_candidate = (root / candidate).resolve()
        if repo_candidate.exists() and repo_candidate.is_file():
            return repo_candidate

    universe_dir = root / "universe"
    direct_name = f"{name}.json" if not name.endswith(".json") else name
    for possible in (
        universe_dir / direct_name,
        universe_dir / name,
    ):
        if possible.exists() and possible.is_file():
            return possible.resolve()

    generated = _ensure_default_universe_file(name)
    if generated is not None and generated.exists() and generated.is_file():
        return generated.resolve()

    raise FileNotFoundError(f"Universe file not found for '{universe_name}'")


def _load_scan_universe(universe_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    path = _resolve_universe_path(universe_name)
    payload = _json_load(path)
    if payload is None:
        raise ValueError(f"Invalid universe JSON: {path}")

    rows: list[Any] = []
    if isinstance(payload.get("merged"), list):
        rows = payload.get("merged") or []
    elif isinstance(payload.get("tickers"), list):
        rows = payload.get("tickers") or []
    else:
        # Fallback to merged list from known keys.
        rows = (payload.get("nasdaq_top_500") or []) + (payload.get("wsb_top_500") or [])

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if isinstance(raw, str):
            symbol = _sanitize_symbol(raw)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(
                {
                    "symbol": symbol,
                    "company_name": symbol,
                    "sector": None,
                    "industry": None,
                    "market_cap": None,
                    "avg_dollar_volume": None,
                    "mention_count": None,
                    "mention_velocity": None,
                }
            )
            continue

        if not isinstance(raw, dict):
            continue
        symbol = _sanitize_symbol(raw.get("symbol"))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(
            {
                "symbol": symbol,
                "company_name": raw.get("company_name") or raw.get("name") or symbol,
                "sector": raw.get("sector"),
                "industry": raw.get("industry"),
                "market_cap": raw.get("market_cap"),
                "avg_dollar_volume": raw.get("avg_dollar_volume"),
                "mention_count": raw.get("mention_count"),
                "mention_velocity": raw.get("mention_velocity"),
            }
        )

    if not normalized:
        raise ValueError(f"No tickers found in universe: {path}")
    return payload, normalized, path


def _normalize_universe_name(universe_name: str) -> str:
    name = (universe_name or "").strip()
    if not name:
        raise ValueError("Universe name is required")
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Invalid universe name. Use a simple name without path separators.")
    return name


def _configured_universe_dirs() -> list[Path]:
    roots: list[Path] = []
    repo_universe = (_repo_root() / "universe").resolve()
    roots.append(repo_universe)

    raw = str(os.getenv("DPOLARIS_UNIVERSE_DIR", "universe")).strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (_repo_root() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if all(candidate != existing for existing in roots):
            roots.append(candidate)
    return roots


def _discover_universe_definitions() -> list[dict[str, str]]:
    discovered: dict[str, dict[str, str]] = {}
    for root in _configured_universe_dirs():
        if not root.exists() or not root.is_dir():
            continue
        for entry in root.iterdir():
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            if not entry.is_file() or entry.suffix.lower() not in LISTABLE_UNIVERSE_EXTENSIONS:
                continue
            name = entry.stem
            current = discovered.get(name)
            payload = {"name": name, "source": "file", "path": str(entry.resolve())}
            if current is None:
                discovered[name] = payload
                continue
            # Prefer files from the primary repo universe directory.
            if str(current.get("path", "")).startswith(str((_repo_root() / "universe").resolve())):
                continue
            if str(entry.resolve()).startswith(str((_repo_root() / "universe").resolve())):
                discovered[name] = payload
    return [discovered[name] for name in sorted(discovered.keys())]


def _universe_file_modified_iso(path: Path) -> Optional[str]:
    try:
        ts = path.stat().st_mtime
    except Exception:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _load_universe_symbols_from_path(path: Path) -> list[str]:
    payload = _load_universe_file_payload(path)
    return _extract_universe_tickers(payload, path)


def _build_all_universe_entry(universe_entries: list[dict[str, Any]]) -> dict[str, Any]:
    symbols: set[str] = set()
    for item in universe_entries:
        name = str(item.get("name") or "").strip().lower()
        if not name or name == "all":
            continue
        path_str = str(item.get("path") or "").strip()
        if not path_str:
            continue
        try:
            tickers = _load_universe_symbols_from_path(Path(path_str))
        except Exception:
            continue
        for ticker in tickers:
            cleaned = _sanitize_symbol(ticker)
            if cleaned:
                symbols.add(cleaned)
    return {
        "name": "all",
        "count": len(symbols),
        "path": "dynamic:all",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _list_universe_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    has_all = False
    for item in _discover_universe_definitions():
        name = str(item.get("name") or "").strip()
        path_str = str(item.get("path") or "").strip()
        if not name or not path_str:
            continue
        path = Path(path_str)
        try:
            count = len(_load_universe_symbols_from_path(path))
        except Exception:
            count = 0
        entry = {
            "name": name,
            "count": int(count),
            "path": str(path),
            "updated_at": _universe_file_modified_iso(path),
        }
        entries.append(entry)
        if name.lower() == "all":
            has_all = True

    if not has_all:
        entries.append(_build_all_universe_entry(entries))

    entries.sort(key=lambda x: str(x.get("name") or "").lower())
    return entries


def _list_universe_definitions() -> list[str]:
    return [str(item.get("name") or "") for item in _list_universe_entries() if str(item.get("name") or "")]


def _extract_tickers_from_container(values: list[Any], seen: set[str], output: list[str]) -> None:
    for item in values:
        if isinstance(item, str):
            symbol = _sanitize_symbol(item)
        elif isinstance(item, dict):
            symbol = _sanitize_symbol(item.get("symbol") or item.get("ticker"))
        else:
            symbol = None
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        output.append(symbol)


def _extract_universe_tickers(payload: Any, source_path: Path) -> list[str]:
    candidates: list[list[Any]] = []
    if isinstance(payload, list):
        candidates.append(payload)
    elif isinstance(payload, dict):
        for key in ("tickers", "symbols", "merged", "universe", "nasdaq_top_500", "wsb_top_500", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.append(value)
    else:
        raise ValueError(f"Unsupported universe structure in {source_path}. Expected JSON/YAML list or object.")

    if not candidates:
        raise ValueError(
            f"Unknown universe format in {source_path}. Expected keys like 'tickers', 'symbols', or 'merged'."
        )

    symbols: list[str] = []
    seen: set[str] = set()
    for values in candidates:
        _extract_tickers_from_container(values, seen, symbols)
    if not symbols:
        raise ValueError(f"No tickers found in universe definition: {source_path}")
    return symbols


def _load_universe_file_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ValueError("YAML support unavailable. Install dependency with: pip install PyYAML")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            tickers = []
            seen: set[str] = set()
            for raw_line in f.readlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                symbol = _sanitize_symbol(line)
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                tickers.append(symbol)
        return {"tickers": tickers, "format": "text"}
    raise ValueError(
        f"Unknown universe file format '{suffix}' for {path}. Supported formats: .json, .yaml, .yml, .txt"
    )


def _resolve_universe_definition_path(universe_name: str) -> tuple[str, Path]:
    requested = _normalize_universe_name(universe_name)
    display_name = Path(requested).stem
    roots = _configured_universe_dirs()
    if roots:
        roots[0].mkdir(parents=True, exist_ok=True)

    for universe_dir in roots:
        direct = universe_dir / requested
        if direct.exists() and direct.is_file():
            if direct.suffix.lower() not in SUPPORTED_UNIVERSE_EXTENSIONS:
                raise ValueError(
                    f"Unknown universe file format for {direct}. Supported formats: .json, .yaml, .yml, .txt"
                )
            return Path(requested).stem, direct.resolve()

        for ext in (".json", ".yaml", ".yml", ".txt"):
            candidate = universe_dir / f"{display_name}{ext}"
            if candidate.exists() and candidate.is_file():
                return display_name, candidate.resolve()

    generated = _ensure_default_universe_file(display_name)
    if generated is not None and generated.exists() and generated.is_file():
        return display_name, generated.resolve()

    searched = [str(path) for path in roots]
    supported = sorted(list(SUPPORTED_UNIVERSE_EXTENSIONS))
    raise FileNotFoundError(
        f"Universe '{display_name}' not found. Searched: {searched}. Supported extensions: {supported}"
    )


def _normalize_universe_alias(universe_name: str) -> str:
    normalized = (universe_name or "").strip().lower()
    if normalized in {"all", "default", "*"}:
        return "combined_1000"
    return universe_name


def _short_run_id(run_id: str) -> str:
    return run_id[:8]


def _append_scan_job_log(job: dict[str, Any], message: str) -> None:
    cleaned = str(message).strip()
    if not cleaned:
        return

    logs = job.setdefault("logs", [])
    logs.append(f"{utc_now_iso()} | {cleaned}")
    if len(logs) > MAX_SCAN_JOB_LOG_LINES:
        del logs[:-MAX_SCAN_JOB_LOG_LINES]


def _trim_scan_jobs() -> None:
    if len(scan_job_order) <= MAX_SCAN_JOBS:
        return

    overflow = len(scan_job_order) - MAX_SCAN_JOBS
    for _ in range(overflow):
        old_job_id = scan_job_order.pop(0)
        job = scan_jobs.get(old_job_id)
        if job and job.get("status") in {"queued", "running"}:
            scan_job_order.append(old_job_id)
            continue
        scan_jobs.pop(old_job_id, None)


def _scan_status_counts(job: dict[str, Any]) -> dict[str, int]:
    status_counter = Counter()
    for item in (job.get("ticker_status") or {}).values():
        status_counter[str((item or {}).get("status") or "queued")] += 1
    return {
        "queued": int(status_counter.get("queued", 0)),
        "running": int(status_counter.get("running", 0)),
        "completed": int(status_counter.get("completed", 0)),
        "failed": int(status_counter.get("failed", 0)),
    }


def _refresh_scan_progress(job: dict[str, Any]) -> None:
    counts = _scan_status_counts(job)
    total = int(job.get("total_tickers") or 0)
    processed = counts["completed"] + counts["failed"]
    progress = (processed / total * 100.0) if total > 0 else 0.0

    job["queued_tickers"] = counts["queued"]
    job["running_tickers"] = counts["running"]
    job["completed_tickers"] = counts["completed"]
    job["failed_tickers"] = counts["failed"]
    job["progress_percent"] = round(progress, 2)


def _public_scan_job(job: dict[str, Any]) -> dict[str, Any]:
    counts = _scan_status_counts(job)
    total = int(job.get("total_tickers") or 0)
    processed = counts["completed"] + counts["failed"]
    warnings = job.get("warnings") or []
    errors = job.get("errors") or {}
    result_index = job.get("result_index") if isinstance(job.get("result_index"), list) else []
    primary_score = None
    if result_index:
        scores = [
            _safe_float(item.get("primary_score"), np.nan)
            for item in result_index
            if isinstance(item, dict)
        ]
        finite = [float(x) for x in scores if np.isfinite(x)]
        if finite:
            primary_score = max(finite)

    horizon_days = job.get("horizon_days")
    options_mode = bool(job.get("options_mode"))
    concurrency = job.get("concurrency")
    config_summary = f"h={horizon_days}d, options={'on' if options_mode else 'off'}, workers={concurrency}"

    return {
        "id": job.get("id"),
        "run_id": job.get("id"),
        "runId": job.get("id"),
        "shortRunId": _short_run_id(str(job.get("id") or "")),
        "status": job.get("status"),
        "runMode": job.get("run_mode", "scan"),
        "universe": job.get("universe_name"),
        "universeHash": job.get("universe_hash"),
        "universe_hash": job.get("universe_hash"),
        "horizonDays": horizon_days,
        "horizon_days": horizon_days,
        "optionsMode": options_mode,
        "options_mode": options_mode,
        "concurrency": concurrency,
        "config_summary": config_summary,
        "primary_score": primary_score,
        "progressPercent": round((processed / total * 100.0), 2) if total > 0 else 0.0,
        "processedTickers": processed,
        "totalTickers": total,
        "completedTickers": counts["completed"],
        "failedTickers": counts["failed"],
        "queuedTickers": counts["queued"],
        "runningTickers": counts["running"],
        "currentTicker": job.get("current_ticker"),
        "current_ticker": job.get("current_ticker"),
        "createdAt": job.get("created_at"),
        "created_at": job.get("created_at"),
        "updatedAt": job.get("updated_at"),
        "updated_at": job.get("updated_at"),
        "startedAt": job.get("started_at"),
        "started_at": job.get("started_at"),
        "completedAt": job.get("completed_at"),
        "completed_at": job.get("completed_at"),
        "runDir": job.get("run_dir"),
        "warnings": warnings[-20:],
        "errorsSummary": {
            "count": len(errors),
            "tickers": sorted(list(errors.keys()))[:20],
        },
        "logs": (job.get("logs") or [])[-200:],
    }


def _persist_scan_job_state(job: dict[str, Any]) -> None:
    run_id = str(job.get("id") or "")
    if not run_id:
        return
    run_dir = _scan_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = copy.deepcopy(job)
    # Keep state payload compact; run artifacts hold full per-ticker details.
    state.pop("result_index", None)
    _json_dump(_scan_state_path(run_id), state)
    try:
        _json_dump(run_dir / "run_summary.json", _build_scan_run_summary(job))
    except Exception:
        pass


def _load_scan_job_state(run_id: str) -> Optional[dict[str, Any]]:
    payload = _json_load(_scan_state_path(run_id))
    if payload is None:
        return None
    if payload.get("id") is None:
        payload["id"] = run_id
    payload.setdefault("result_index", [])
    payload.setdefault("ticker_status", {})
    payload.setdefault("warnings", [])
    payload.setdefault("errors", {})
    payload.setdefault("logs", [])
    _refresh_scan_progress(payload)
    return payload


def _load_scan_index(run_id: str) -> list[dict[str, Any]]:
    path = _scan_index_path(run_id)
    payload = _json_load(path)
    if payload and isinstance(payload.get("items"), list):
        return [x for x in payload.get("items") if isinstance(x, dict)]

    result_dir = _scan_results_dir(run_id)
    if not result_dir.exists():
        return []

    items: list[dict[str, Any]] = []
    for file in sorted(result_dir.glob("*.json")):
        loaded = _json_load(file)
        if not loaded:
            continue
        summary = loaded.get("summary")
        if isinstance(summary, dict):
            items.append(summary)
    return items


def _persist_scan_index(run_id: str, items: list[dict[str, Any]]) -> None:
    payload = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "count": len(items),
        "items": items,
    }
    _json_dump(_scan_index_path(run_id), payload)


def _latest_time_col(df) -> Optional[str]:
    for col in ("timestamp", "date", "datetime"):
        if col in df.columns:
            return col
    return None


def _extract_close_series(df) -> tuple[list[datetime], list[float]]:
    if df is None or len(df) == 0:
        return [], []
    time_col = _latest_time_col(df)
    if time_col is None or "close" not in df.columns:
        return [], []

    work = df.copy()
    work[time_col] = work[time_col].apply(_parse_timestamp)
    work["close"] = work["close"].apply(lambda x: _safe_float(x, np.nan))
    work = work.dropna(subset=[time_col, "close"]).sort_values(time_col)
    if work.empty:
        return [], []
    return list(work[time_col]), [float(x) for x in work["close"].tolist()]


def _pct_return(closes: list[float], lookback: int) -> Optional[float]:
    if len(closes) <= int(lookback):
        return None
    prev = closes[-(int(lookback) + 1)]
    if prev == 0:
        return None
    return (closes[-1] / prev) - 1.0


def _pattern_analogs(
    times: list[datetime],
    closes: list[float],
    *,
    window: int = 30,
    horizon: int = 5,
    top_k: int = 5,
) -> dict[str, Any]:
    if len(closes) < (window + horizon + 15):
        return {
            "window": window,
            "horizon": horizon,
            "top_k": top_k,
            "analogs": [],
            "aggregated_outcomes": {
                "count": 0,
                "mean_forward_return": None,
                "median_forward_return": None,
                "win_rate": None,
            },
        }

    arr = np.asarray(closes, dtype=float)
    rets = np.diff(np.log(np.clip(arr, 1e-9, None)))
    if len(rets) < (window + horizon + 5):
        return {
            "window": window,
            "horizon": horizon,
            "top_k": top_k,
            "analogs": [],
            "aggregated_outcomes": {
                "count": 0,
                "mean_forward_return": None,
                "median_forward_return": None,
                "win_rate": None,
            },
        }

    latest_start = len(rets) - window
    latest_vec = rets[latest_start:]

    candidates: list[dict[str, Any]] = []
    max_end = len(rets) - horizon
    for end_idx in range(window, max_end):
        # Prevent overlap with the current window to keep the analog set historical.
        if end_idx >= latest_start:
            break
        start_idx = end_idx - window
        cand_vec = rets[start_idx:end_idx]
        if len(cand_vec) != window:
            continue

        distance = float(np.linalg.norm(cand_vec - latest_vec))
        anchor_px = arr[end_idx]
        future_px = arr[end_idx + horizon]
        if anchor_px == 0:
            continue
        forward_return = float((future_px / anchor_px) - 1.0)
        candidates.append(
            {
                "distance": distance,
                "window_start": times[start_idx + 1].isoformat() if (start_idx + 1) < len(times) else None,
                "window_end": times[end_idx].isoformat() if end_idx < len(times) else None,
                "forward_return_horizon": forward_return,
                "outcome": "up" if forward_return > 0 else "down",
            }
        )

    candidates.sort(key=lambda x: x["distance"])
    selected = candidates[: max(1, int(top_k))]

    fwd_returns = [float(x.get("forward_return_horizon")) for x in selected if x.get("forward_return_horizon") is not None]
    aggregated = {
        "count": len(fwd_returns),
        "mean_forward_return": float(np.mean(fwd_returns)) if fwd_returns else None,
        "median_forward_return": float(np.median(fwd_returns)) if fwd_returns else None,
        "win_rate": float(np.mean([x > 0 for x in fwd_returns])) if fwd_returns else None,
    }

    return {
        "window": int(window),
        "horizon": int(horizon),
        "top_k": int(top_k),
        "analogs": selected,
        "aggregated_outcomes": aggregated,
    }


def _sector_etf_for_row(row: dict[str, Any]) -> Optional[str]:
    raw_sector = str(row.get("sector") or "").strip().lower()
    if not raw_sector:
        return None
    return SECTOR_ETF_MAP.get(raw_sector)


def _safe_price(value: Any) -> Optional[float]:
    numeric = _safe_float(value, np.nan)
    if np.isfinite(numeric):
        return float(numeric)
    return None


def _mid_price(bid: Any, ask: Any) -> Optional[float]:
    b = _safe_price(bid)
    a = _safe_price(ask)
    if b is None and a is None:
        return None
    if b is None:
        return a
    if a is None:
        return b
    return (b + a) / 2.0


def _options_candidates(
    *,
    bias: str,
    confidence: float,
    iv_rank: Optional[float],
    latest_price: float,
    horizon_days: int,
) -> list[dict[str, Any]]:
    confidence = _clamp(float(confidence), 0.0, 1.0)
    iv_rank = None if iv_rank is None else float(iv_rank)
    horizon_days = int(max(1, horizon_days))
    dte = max(14, min(60, horizon_days * 4))
    expected_move = latest_price * max(0.01, min(0.2, 0.015 + (0.12 * (iv_rank or 0) / 100.0)))

    candidates: list[dict[str, Any]] = []
    if bias == "LONG":
        long_call_pop = _clamp(0.45 + (confidence * 0.35), 0.05, 0.95)
        spread_pop = _clamp(0.52 + (confidence * 0.30), 0.05, 0.95)
        candidates.extend(
            [
                {
                    "rank": 1,
                    "strategy_type": "bull_call_spread" if (iv_rank is not None and iv_rank >= 55.0) else "long_call",
                    "expiry_dte": dte,
                    "strikes": [round(latest_price, 2), round(latest_price + expected_move, 2)],
                    "debit_credit": "debit",
                    "mid_estimate": round(max(0.25, expected_move * 0.35), 2),
                    "max_loss": round(max(0.25, expected_move * 0.35), 2),
                    "max_gain": round(max(0.5, expected_move), 2),
                    "breakevens": [round(latest_price + max(0.25, expected_move * 0.35), 2)],
                    "pop": round(spread_pop if iv_rank is not None and iv_rank >= 55.0 else long_call_pop, 4),
                    "ev": round((spread_pop * expected_move) - ((1 - spread_pop) * (expected_move * 0.35)), 4),
                    "confidence": round(confidence, 4),
                },
                {
                    "rank": 2,
                    "strategy_type": "cash_secured_put",
                    "expiry_dte": dte,
                    "strikes": [round(latest_price * 0.97, 2)],
                    "debit_credit": "credit",
                    "mid_estimate": round(max(0.2, expected_move * 0.20), 2),
                    "max_loss": round(latest_price * 0.97, 2),
                    "max_gain": round(max(0.2, expected_move * 0.20), 2),
                    "breakevens": [round((latest_price * 0.97) - max(0.2, expected_move * 0.20), 2)],
                    "pop": round(_clamp(0.55 + confidence * 0.25, 0.05, 0.98), 4),
                    "ev": round(expected_move * 0.12, 4),
                    "confidence": round(_clamp(confidence * 0.95, 0.01, 0.99), 4),
                },
            ]
        )
    elif bias == "SHORT":
        put_pop = _clamp(0.45 + (confidence * 0.35), 0.05, 0.95)
        spread_pop = _clamp(0.52 + (confidence * 0.30), 0.05, 0.95)
        candidates.extend(
            [
                {
                    "rank": 1,
                    "strategy_type": "bear_put_spread" if (iv_rank is not None and iv_rank >= 55.0) else "long_put",
                    "expiry_dte": dte,
                    "strikes": [round(latest_price, 2), round(max(0.01, latest_price - expected_move), 2)],
                    "debit_credit": "debit",
                    "mid_estimate": round(max(0.25, expected_move * 0.35), 2),
                    "max_loss": round(max(0.25, expected_move * 0.35), 2),
                    "max_gain": round(max(0.5, expected_move), 2),
                    "breakevens": [round(latest_price - max(0.25, expected_move * 0.35), 2)],
                    "pop": round(spread_pop if iv_rank is not None and iv_rank >= 55.0 else put_pop, 4),
                    "ev": round((spread_pop * expected_move) - ((1 - spread_pop) * (expected_move * 0.35)), 4),
                    "confidence": round(confidence, 4),
                },
                {
                    "rank": 2,
                    "strategy_type": "bear_call_spread",
                    "expiry_dte": dte,
                    "strikes": [round(latest_price * 1.02, 2), round(latest_price * 1.05, 2)],
                    "debit_credit": "credit",
                    "mid_estimate": round(max(0.2, expected_move * 0.18), 2),
                    "max_loss": round(max(0.35, expected_move * 0.50), 2),
                    "max_gain": round(max(0.2, expected_move * 0.18), 2),
                    "breakevens": [round((latest_price * 1.02) + max(0.2, expected_move * 0.18), 2)],
                    "pop": round(_clamp(0.55 + confidence * 0.22, 0.05, 0.98), 4),
                    "ev": round(expected_move * 0.08, 4),
                    "confidence": round(_clamp(confidence * 0.93, 0.01, 0.99), 4),
                },
            ]
        )
    else:
        candidates.append(
            {
                "rank": 1,
                "strategy_type": "no_trade",
                "expiry_dte": dte,
                "strikes": [],
                "debit_credit": "n/a",
                "mid_estimate": 0.0,
                "max_loss": 0.0,
                "max_gain": 0.0,
                "breakevens": [],
                "pop": 0.0,
                "ev": 0.0,
                "confidence": round(confidence, 4),
            }
        )
    return candidates


async def _build_options_decision_support(
    *,
    symbol: str,
    bias: str,
    confidence: float,
    latest_price: float,
    horizon_days: int,
    market: MarketDataService,
) -> dict[str, Any]:
    warnings: list[str] = []
    iv_metrics = await market.get_iv_metrics(symbol)
    option_chain = await market.get_options(symbol)

    iv_level = _safe_float((iv_metrics or {}).get("current_iv"), np.nan)
    if not np.isfinite(iv_level):
        iv_level = None

    iv_rank = _safe_float((iv_metrics or {}).get("iv_rank"), np.nan)
    if not np.isfinite(iv_rank):
        iv_rank = None

    term_structure_slope = None
    if option_chain and isinstance(option_chain.get("expirations_available"), list):
        expirations = [str(x) for x in (option_chain.get("expirations_available") or []) if str(x).strip()]
        if len(expirations) >= 2:
            # Approximate: compare current expiration IV with a farther expiration IV if available.
            near = await market.get_options(symbol, expiration=expirations[0])
            far = await market.get_options(symbol, expiration=expirations[min(len(expirations) - 1, 2)])
            near_iv = _safe_float(((near or {}).get("calls") or [{}])[0].get("implied_volatility"), np.nan)
            far_iv = _safe_float(((far or {}).get("calls") or [{}])[0].get("implied_volatility"), np.nan)
            if np.isfinite(near_iv) and np.isfinite(far_iv):
                term_structure_slope = float(far_iv - near_iv)

    if not option_chain:
        warnings.append("Options chain unavailable; strategy metrics are heuristic estimates.")

    avg_spread = None
    liquid = True
    calls = (option_chain or {}).get("calls") or []
    sampled = calls[:25]
    spreads = []
    volumes = []
    for row in sampled:
        bid = _safe_price(row.get("bid"))
        ask = _safe_price(row.get("ask"))
        mid = _mid_price(bid, ask)
        vol = _safe_float(row.get("volume"), np.nan)
        if bid is not None and ask is not None and mid and mid > 0:
            spreads.append((ask - bid) / mid)
        if np.isfinite(vol):
            volumes.append(vol)
    if spreads:
        avg_spread = float(np.mean(spreads))
        if avg_spread > 0.35:
            liquid = False
            warnings.append("Wide option spreads detected; execution quality risk is high.")
    if volumes and float(np.mean(volumes)) < 20:
        liquid = False
        warnings.append("Low average options volume; liquidity risk is elevated.")

    candidates = _options_candidates(
        bias=bias,
        confidence=confidence,
        iv_rank=iv_rank,
        latest_price=latest_price,
        horizon_days=horizon_days,
    )

    scenario_moves = [-0.05, -0.02, 0.0, 0.02, 0.05]
    scenario_rows = []
    for move in scenario_moves:
        pnl = 0.0
        if bias == "LONG":
            pnl = max(-1.0, move * 40.0)
        elif bias == "SHORT":
            pnl = max(-1.0, (-move) * 40.0)
        scenario_rows.append(
            {
                "underlying_move_pct": round(move, 4),
                "pnl_estimate_r": round(pnl, 4),
            }
        )

    return {
        "enabled": True,
        "iv_level": iv_level,
        "iv_rank_percentile": iv_rank,
        "term_structure_slope": term_structure_slope,
        "earnings_proximity_days": None,
        "major_event_flags": [],
        "candidates": candidates,
        "scenario_table": scenario_rows,
        "warnings": warnings,
        "liquidity_ok": liquid,
        "avg_relative_spread": avg_spread,
    }


def _scan_summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return {
        "ticker": payload.get("ticker"),
        "status": payload.get("status", "unknown"),
        "primary_score": None,
    }


def _update_scan_index(run_id: str, row: dict[str, Any]) -> None:
    items = _load_scan_index(run_id)
    ticker = str(row.get("ticker") or "").upper()
    if ticker:
        replaced = False
        for idx, item in enumerate(items):
            if str(item.get("ticker") or "").upper() == ticker:
                items[idx] = row
                replaced = True
                break
        if not replaced:
            items.append(row)
    _persist_scan_index(run_id, items)


def _write_scan_result(run_id: str, ticker: str, payload: dict[str, Any]) -> Path:
    symbol = _sanitize_symbol(ticker)
    if not symbol:
        raise ValueError(f"Invalid ticker symbol: {ticker}")
    out_path = _scan_results_dir(run_id) / f"{symbol}.json"
    _json_dump(out_path, payload)
    return out_path


async def _execute_deep_learning_subprocess(
    symbol: str,
    model_type: str,
    epochs: int,
    on_log: Optional[Callable[[str], None]] = None,
) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")
    env.setdefault("KMP_BLOCKTIME", "0")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    command = [
        sys.executable,
        "-m",
        "ml.deep_learning_worker",
        "--symbol",
        symbol,
        "--model-type",
        model_type,
        "--epochs",
        str(epochs),
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(repo_root),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def _drain_stream(stream, sink: list[str], stream_name: str) -> None:
        while True:
            raw_line = await stream.readline()
            if not raw_line:
                break

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            sink.append(line)

            # Final JSON payload is emitted on stdout; keep it out of live logs.
            if stream_name == "stdout":
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        continue
                except json.JSONDecodeError:
                    pass

            if on_log is not None:
                on_log(line)

    assert process.stdout is not None
    assert process.stderr is not None

    stdout_task = asyncio.create_task(_drain_stream(process.stdout, stdout_lines, "stdout"))
    stderr_task = asyncio.create_task(_drain_stream(process.stderr, stderr_lines, "stderr"))

    return_code = await process.wait()
    await asyncio.gather(stdout_task, stderr_task)

    stdout_text = "\n".join(stdout_lines).strip()
    stderr_text = "\n".join(stderr_lines).strip()
    stderr_summary = stderr_lines[-1] if stderr_lines else ""

    if return_code != 0:
        if return_code < 0:
            signal_number = -return_code
            if stderr_summary:
                raise RuntimeError(
                    f"Deep-learning worker crashed with signal {signal_number}: {stderr_summary}"
                )
            raise RuntimeError(f"Deep-learning worker crashed with signal {signal_number}")

        raise RuntimeError(
            stderr_summary or stderr_text or f"Deep-learning worker failed with exit code {return_code}"
        )

    if not stdout_text:
        raise RuntimeError("Deep-learning worker returned empty output")

    # Worker may emit logs; parse the last JSON line.
    payload: Optional[dict] = None
    for line in reversed(stdout_text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            payload = loaded
            break

    if payload is None:
        excerpt = stdout_text[-400:]
        raise RuntimeError(f"Deep-learning worker returned invalid output: {excerpt}")

    return payload


def _classify_deep_learning_error(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    if ("no module named 'torch'" in lowered) or ('no module named "torch"' in lowered):
        return "dependency_missing: torch", "PyTorch is not installed. Install with: pip install torch"
    if "module not found" in lowered and "torch" in lowered:
        return "dependency_missing: torch", "PyTorch is not installed. Install with: pip install torch"
    if ("no module named 'sklearn'" in lowered) or ('no module named "sklearn"' in lowered):
        return "dependency_missing: scikit-learn", "scikit-learn is not installed. Install with: pip install scikit-learn"
    if "module not found" in lowered and "sklearn" in lowered:
        return "dependency_missing: scikit-learn", "scikit-learn is not installed. Install with: pip install scikit-learn"
    return "runtime_error", message


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_tags(tags_value) -> list[str]:
    if tags_value is None:
        return []
    if isinstance(tags_value, list):
        return [str(x) for x in tags_value]
    if isinstance(tags_value, str):
        raw = tags_value.strip()
        if not raw:
            return []
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return [str(x) for x in loaded]
        except json.JSONDecodeError:
            pass
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _extract_signal_id_from_tags(tags_value) -> Optional[str]:
    tags = _parse_tags(tags_value)
    for tag in tags:
        if str(tag).startswith("signal_id:"):
            value = str(tag).split("signal_id:", 1)[1].strip()
            if value:
                return value
    return None


def _rank_signal_features(latest_features, *, top_n: int = 5) -> list[str]:
    feature_scores: dict[str, float] = {}

    def _score(name: str, value: float) -> None:
        if np.isfinite(value):
            feature_scores[name] = abs(float(value))

    _score("roc_5", _safe_float(latest_features.get("roc_5"), 0.0))
    _score("hvol_20", _safe_float(latest_features.get("hvol_20"), 0.0))
    _score("atr_14", _safe_float(latest_features.get("atr_14"), 0.0))
    _score("adx", _safe_float(latest_features.get("adx"), 0.0) - 20.0)
    _score("rsi_14", _safe_float(latest_features.get("rsi_14"), 50.0) - 50.0)
    _score("vol_ratio_20", _safe_float(latest_features.get("vol_ratio_20"), 1.0) - 1.0)
    _score("price_sma20_ratio", _safe_float(latest_features.get("price_sma20_ratio"), 1.0) - 1.0)
    _score("price_sma50_ratio", _safe_float(latest_features.get("price_sma50_ratio"), 1.0) - 1.0)
    _score("price_sma200_ratio", _safe_float(latest_features.get("price_sma200_ratio"), 1.0) - 1.0)

    ranked = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[: max(1, int(top_n))]]


def _build_signal_from_features(
    symbol: str,
    latest_price: float,
    latest_features,
    prediction: dict,
    horizon_days: int,
) -> dict:
    atr_14 = max(_safe_float(latest_features.get("atr_14"), latest_price * 0.01), latest_price * 0.004)
    atr_percent = atr_14 / latest_price if latest_price > 0 else 0.0
    rsi_14 = _safe_float(latest_features.get("rsi_14"), 50.0)
    roc_5 = _safe_float(latest_features.get("roc_5"), 0.0)
    hvol_20 = _safe_float(latest_features.get("hvol_20"), 0.2)
    vol_ratio_20 = _safe_float(latest_features.get("vol_ratio_20"), 1.0)
    adx = _safe_float(latest_features.get("adx"), 18.0)

    price_sma20_ratio = _safe_float(latest_features.get("price_sma20_ratio"), 1.0)
    price_sma50_ratio = _safe_float(latest_features.get("price_sma50_ratio"), 1.0)
    price_sma200_ratio = _safe_float(latest_features.get("price_sma200_ratio"), 1.0)

    trend_votes = sum(
        1 for ratio in [price_sma20_ratio, price_sma50_ratio, price_sma200_ratio] if ratio > 1.0
    )
    if trend_votes >= 2:
        trend = "BULLISH"
    elif trend_votes <= 1:
        trend = "BEARISH"
    else:
        trend = "MIXED"

    if hvol_20 >= 0.40:
        volatility_regime = "HIGH"
    elif hvol_20 <= 0.18:
        volatility_regime = "LOW"
    else:
        volatility_regime = "NORMAL"

    momentum = "POSITIVE" if roc_5 > 0 else "NEGATIVE"

    probability_up = _clamp(_safe_float(prediction.get("probability_up"), 0.5), 0.0, 1.0)
    probability_down = _clamp(_safe_float(prediction.get("probability_down"), 1.0 - probability_up), 0.0, 1.0)
    model_confidence = _clamp(_safe_float(prediction.get("confidence"), max(probability_up, probability_down)), 0.0, 1.0)
    model_accuracy = prediction.get("model_accuracy")
    model_accuracy_float = _safe_float(model_accuracy, 0.55) if model_accuracy is not None else None

    edge_strength = abs(probability_up - 0.5) * 2.0
    accuracy_component = (
        _clamp((model_accuracy_float - 0.5) / 0.25, 0.0, 1.0)
        if model_accuracy_float is not None
        else 0.5
    )
    setup_confidence = _clamp((0.65 * edge_strength) + (0.35 * accuracy_component), 0.05, 0.95)

    if probability_up >= 0.60:
        bias = "LONG"
    elif probability_up <= 0.40:
        bias = "SHORT"
    else:
        bias = "NO_TRADE"

    entry_trigger = None
    entry_zone_low = None
    entry_zone_high = None
    stop_loss = None
    target_1 = None
    target_2 = None
    setup_type = "WAIT_FOR_EDGE"
    entry_condition = "Wait for directional edge above 60/40 probability."
    invalidation = "No trade until signal quality improves."

    if bias == "LONG":
        setup_type = "BREAKOUT_CONTINUATION" if trend == "BULLISH" else "PULLBACK_LONG"
        entry_trigger = latest_price + (0.25 * atr_14)
        entry_zone_low = latest_price + (0.10 * atr_14)
        entry_zone_high = latest_price + (0.50 * atr_14)
        stop_loss = entry_trigger - (1.6 * atr_14)
        target_1 = entry_trigger + (2.0 * atr_14)
        target_2 = entry_trigger + (3.5 * atr_14)
        entry_condition = "Break and hold above trigger with expanding volume."
        invalidation = f"Close below stop ({stop_loss:.2f}) or momentum rollover."
    elif bias == "SHORT":
        setup_type = "BREAKDOWN_CONTINUATION" if trend == "BEARISH" else "RALLY_FADE"
        entry_trigger = latest_price - (0.25 * atr_14)
        entry_zone_low = latest_price - (0.50 * atr_14)
        entry_zone_high = latest_price - (0.10 * atr_14)
        stop_loss = entry_trigger + (1.6 * atr_14)
        target_1 = entry_trigger - (2.0 * atr_14)
        target_2 = entry_trigger - (3.5 * atr_14)
        entry_condition = "Break and hold below trigger with weak breadth."
        invalidation = f"Close above stop ({stop_loss:.2f}) or momentum reversal."

    portfolio_snapshot = db.get_latest_portfolio() if db is not None else None
    portfolio_value = _safe_float(
        (portfolio_snapshot or {}).get("total_value"),
        _safe_float(config.goal.starting_capital if config else 100000.0, 100000.0),
    )
    portfolio_value = max(portfolio_value, 1.0)
    max_risk_percent = _safe_float(config.risk.max_portfolio_risk_percent if config else 2.0, 2.0)
    max_position_percent = _safe_float(config.risk.max_position_size_percent if config else 5.0, 5.0)
    max_risk_dollars = portfolio_value * (max_risk_percent / 100.0)
    max_position_dollars = portfolio_value * (max_position_percent / 100.0)

    risk_per_share = None
    suggested_shares = 0
    suggested_notional = 0.0
    suggested_position_percent = 0.0
    rr_target_1 = None
    rr_target_2 = None
    targets = []

    if entry_trigger is not None and stop_loss is not None:
        risk_per_share = abs(entry_trigger - stop_loss)
        if risk_per_share > 0:
            shares_by_risk = int(max_risk_dollars / risk_per_share)
            shares_by_position = int(max_position_dollars / entry_trigger) if entry_trigger > 0 else 0
            suggested_shares = max(0, min(shares_by_risk, shares_by_position))

        suggested_notional = suggested_shares * entry_trigger
        suggested_position_percent = (suggested_notional / portfolio_value) * 100.0

        if target_1 is not None and risk_per_share and risk_per_share > 0:
            rr_target_1 = abs(target_1 - entry_trigger) / risk_per_share
        if target_2 is not None and risk_per_share and risk_per_share > 0:
            rr_target_2 = abs(target_2 - entry_trigger) / risk_per_share

        if target_1 is not None:
            targets.append({
                "label": "TP1",
                "price": round(target_1, 4),
                "r_multiple": round(rr_target_1, 3) if rr_target_1 is not None else None,
            })
        if target_2 is not None:
            targets.append({
                "label": "TP2",
                "price": round(target_2, 4),
                "r_multiple": round(rr_target_2, 3) if rr_target_2 is not None else None,
            })

    reasons = [
        f"Model probability up={probability_up * 100:.1f}% (confidence={model_confidence * 100:.1f}%).",
        f"Trend regime: {trend}; momentum: {momentum}; RSI={rsi_14:.1f}.",
        f"Volatility regime: {volatility_regime} (ATR={atr_percent * 100:.2f}% of price).",
    ]
    if vol_ratio_20 >= 1.2:
        reasons.append(f"Volume is expanding ({vol_ratio_20:.2f}x 20-day average).")
    if adx >= 20:
        reasons.append(f"Trend strength is constructive (ADX {adx:.1f}).")

    risk_flags = []
    if bias == "NO_TRADE":
        risk_flags.append("No-trade zone: directional edge is weak (probability near 50/50).")
    if volatility_regime == "HIGH":
        risk_flags.append("High volatility regime: widen stops and reduce size.")
    if adx < 16:
        risk_flags.append("Weak trend strength (ADX < 16) increases chop risk.")
    if rsi_14 >= 72:
        risk_flags.append("Overbought RSI may increase pullback risk for longs.")
    if rsi_14 <= 28:
        risk_flags.append("Oversold RSI may increase squeeze risk for shorts.")
    if bias != "NO_TRADE" and suggested_shares <= 0:
        risk_flags.append("Risk budget too small for this setup; skip or reduce instrument size.")

    if bias == "LONG":
        options_strategy = "Bull Call Spread" if volatility_regime == "HIGH" else "Long Call / Call Spread"
        options_stance = "Bullish"
    elif bias == "SHORT":
        options_strategy = "Bear Put Spread" if volatility_regime == "HIGH" else "Long Put / Put Spread"
        options_stance = "Bearish"
    else:
        options_strategy = "No options trade"
        options_stance = "Neutral"

    options_plan = {
        "stance": options_stance,
        "strategy": options_strategy,
        "dte_range": "21-45",
        "delta_range": "0.30-0.45",
        "max_premium_pct_of_portfolio": round(max_risk_percent * 0.5, 2),
    }

    top_features = _rank_signal_features(latest_features, top_n=5)
    signal_id: Optional[str] = None
    if self_critique_logger is not None:
        try:
            signal_id = self_critique_logger.log_signal(
                symbol=symbol,
                timestamp=utc_now_iso(),
                confidence=setup_confidence,
                regime=f"{trend.lower()}_{volatility_regime.lower()}",
                top_features=top_features,
                prediction=probability_up,
                model_name=prediction.get("model_name"),
                model_version=prediction.get("model_version"),
                extra={
                    "bias": bias,
                    "setup_type": setup_type,
                    "horizon_days": int(horizon_days),
                },
            )
        except Exception as exc:
            logger.warning("Self-critique signal logging failed for %s: %s", symbol, exc)

    return {
        "symbol": symbol,
        "signal_id": signal_id,
        "generated_at": utc_now_iso(),
        "bias": bias,
        "setup_type": setup_type,
        "time_horizon_days": horizon_days,
        "confidence": round(setup_confidence, 6),
        "model_confidence": round(model_confidence, 6),
        "probability_up": round(probability_up, 6),
        "probability_down": round(probability_down, 6),
        "entry": {
            "trigger": round(entry_trigger, 4) if entry_trigger is not None else None,
            "zone_low": round(entry_zone_low, 4) if entry_zone_low is not None else None,
            "zone_high": round(entry_zone_high, 4) if entry_zone_high is not None else None,
            "condition": entry_condition,
        },
        "risk": {
            "stop_loss": round(stop_loss, 4) if stop_loss is not None else None,
            "invalidation": invalidation,
            "risk_per_share": round(risk_per_share, 4) if risk_per_share is not None else None,
            "max_risk_dollars": round(max_risk_dollars, 2),
            "max_portfolio_risk_percent": round(max_risk_percent, 3),
            "suggested_shares": suggested_shares,
            "suggested_notional": round(suggested_notional, 2),
            "suggested_position_percent": round(suggested_position_percent, 3),
        },
        "targets": targets,
        "reasons": reasons,
        "risk_flags": risk_flags,
        "insights": [
            {
                "title": "Trend vs MAs",
                "detail": f"Price/SMA20={price_sma20_ratio:.3f}, Price/SMA50={price_sma50_ratio:.3f}, Price/SMA200={price_sma200_ratio:.3f}.",
            },
            {
                "title": "Volatility Regime",
                "detail": f"HV20={hvol_20:.3f}, ATR={atr_14:.3f} ({atr_percent * 100:.2f}% of price).",
            },
            {
                "title": "Execution Focus",
                "detail": "Only execute if trigger is hit and sizing stays within risk budget.",
            },
        ],
        "options_plan": options_plan,
        "model": {
            "source": prediction.get("source", "unknown"),
            "name": prediction.get("model_name", "unknown"),
            "type": prediction.get("model_type", "unknown"),
            "version": prediction.get("model_version"),
            "accuracy": round(_safe_float(model_accuracy_float, 0.0), 6) if model_accuracy_float is not None else None,
        },
        "top_features": top_features,
        "market_snapshot": {
            "last_price": round(latest_price, 4),
            "rsi_14": round(rsi_14, 4),
            "atr_14": round(atr_14, 4),
            "atr_percent": round(atr_percent, 6),
            "hvol_20": round(hvol_20, 6),
            "trend": trend,
            "momentum": momentum,
            "volatility_regime": volatility_regime,
            "volume_ratio_20": round(vol_ratio_20, 6),
            "adx": round(adx, 4),
        },
    }


async def _fetch_history_cached(
    market: MarketDataService,
    cache: dict[str, Any],
    symbol: str,
    days: int,
):
    key = f"{symbol}:{int(days)}"
    if key not in cache:
        cache[key] = await market.get_historical(symbol, days=days)
    frame = cache.get(key)
    if frame is None:
        return None
    try:
        return frame.copy()
    except Exception:
        return frame


async def _build_scan_ticker_payload(
    *,
    run_id: str,
    symbol: str,
    row_meta: dict[str, Any],
    horizon_days: int,
    history_days: int,
    options_mode: bool,
    market: MarketDataService,
    history_cache: dict[str, Any],
    risk_config: dict[str, Any],
) -> dict[str, Any]:
    from data.quality import DataQualityGate
    from data.schema import apply_split_dividend_adjustments, canonicalize_price_frame
    from ml.features import FeatureEngine

    raw_df = await _fetch_history_cached(market, history_cache, symbol, history_days)
    if raw_df is None or len(raw_df) < max(260, horizon_days * 60):
        raise RuntimeError(f"Not enough historical data for {symbol}")

    canonical = canonicalize_price_frame(raw_df, source_timezone="UTC", target_timezone="UTC", intraday=False)
    canonical = apply_split_dividend_adjustments(canonical)

    quality_gate = DataQualityGate()
    quality_report_id = f"{run_id}_{symbol}".replace("-", "_")
    quality_df, quality_report, quality_report_path = quality_gate.run(
        canonical,
        symbol=symbol,
        interval="1d",
        horizon_days=horizon_days,
        run_id=quality_report_id,
        report_dir=_scan_run_dir(run_id) / "reports",
    )

    if not quality_report.get("checks", {}).get("minimum_history", {}).get("passed", False):
        required = quality_report.get("checks", {}).get("minimum_history", {}).get("required_rows")
        actual = quality_report.get("checks", {}).get("minimum_history", {}).get("actual_rows")
        raise RuntimeError(f"Data quality gate failed minimum history ({actual}/{required}) for {symbol}")

    feature_input = quality_df.rename(columns={"timestamp": "date"})
    feature_engine = FeatureEngine()
    feature_df = feature_engine.generate_features(
        feature_input[["date", "open", "high", "low", "close", "volume"]],
        include_targets=False,
    )
    if feature_df.empty:
        raise RuntimeError(f"Feature generation failed for {symbol}")

    latest_features = feature_df.iloc[-1]
    latest_price = _safe_float(quality_df.iloc[-1].get("close"), 0.0)
    if latest_price <= 0:
        raise RuntimeError(f"Invalid latest price for {symbol}")

    prediction = _predict_symbol_direction(symbol, feature_input)
    signal = _build_signal_from_features(
        symbol=symbol,
        latest_price=latest_price,
        latest_features=latest_features,
        prediction=prediction,
        horizon_days=horizon_days,
    )

    raw_feature_map = {}
    for name in feature_engine.get_feature_names():
        raw_feature_map[name] = latest_features.get(name)

    regime = derive_regime(raw_feature_map) if derive_regime is not None else {
        "label": "unknown",
        "trend": "UNKNOWN",
        "volatility": "UNKNOWN",
        "momentum": "UNKNOWN",
    }

    times, closes = _extract_close_series(feature_input)
    ticker_ret_20 = _pct_return(closes, 20)
    ticker_ret_60 = _pct_return(closes, 60)

    qqq_df = await _fetch_history_cached(market, history_cache, "QQQ", history_days)
    qqq_times, qqq_closes = _extract_close_series(qqq_df)
    qqq_ret_20 = _pct_return(qqq_closes, 20)
    rel_vs_qqq = None
    if ticker_ret_20 is not None and qqq_ret_20 is not None:
        rel_vs_qqq = ticker_ret_20 - qqq_ret_20

    sector_symbol = _sector_etf_for_row(row_meta)
    sector_ret_20 = None
    rel_vs_sector = None
    if sector_symbol:
        sec_df = await _fetch_history_cached(market, history_cache, sector_symbol, history_days)
        _, sec_closes = _extract_close_series(sec_df)
        sector_ret_20 = _pct_return(sec_closes, 20)
        if ticker_ret_20 is not None and sector_ret_20 is not None:
            rel_vs_sector = ticker_ret_20 - sector_ret_20

    adx_value = _safe_float(latest_features.get("adx"), 0.0)
    vol_20 = _safe_float(latest_features.get("hvol_20"), 0.0)
    trend_state = "trend" if adx_value >= 20 else "chop"
    vol_state = "high_vol" if vol_20 >= 0.40 else ("low_vol" if vol_20 <= 0.18 else "mid_vol")
    risk_proxy = "risk_on"
    if qqq_ret_20 is not None and qqq_ret_20 < 0:
        risk_proxy = "risk_off"

    gap = None
    if len(closes) >= 2:
        prev_close = closes[-2]
        if prev_close > 0:
            gap = (latest_price / prev_close) - 1.0

    recent_high = max(closes[-20:]) if len(closes) >= 20 else max(closes)
    recent_low = min(closes[-20:]) if len(closes) >= 20 else min(closes)
    dist_to_high = ((latest_price / recent_high) - 1.0) if recent_high > 0 else None
    dist_to_low = ((latest_price / recent_low) - 1.0) if recent_low > 0 else None

    pattern_analogs = _pattern_analogs(
        times=times,
        closes=closes,
        window=max(20, min(45, horizon_days * 6)),
        horizon=max(1, horizon_days),
        top_k=5,
    )

    options_support = {
        "enabled": False,
        "warnings": [],
        "candidates": [],
    }
    if options_mode:
        options_support = await _build_options_decision_support(
            symbol=symbol,
            bias=str(signal.get("bias") or "NO_TRADE"),
            confidence=_safe_float(signal.get("confidence"), 0.5),
            latest_price=latest_price,
            horizon_days=horizon_days,
            market=market,
        )

    risk_mode = "standard"
    confidence = _safe_float(signal.get("confidence"), 0.5)
    if confidence < 0.58 or vol_state == "high_vol":
        risk_mode = "conservative"
    elif confidence >= 0.75 and vol_state == "low_vol":
        risk_mode = "aggressive"

    risk_summary = {
        "position_sizing_guidance": risk_mode,
        "stop_suggestion": (signal.get("risk") or {}).get("stop_loss"),
        "profit_take_suggestions": [x.get("price") for x in (signal.get("targets") or []) if x.get("price") is not None],
        "hard_warnings": list((signal.get("risk_flags") or [])) + list(options_support.get("warnings") or []),
        "risk_config": risk_config or {},
    }

    now_iso = utc_now_iso()
    price_asof = times[-1].isoformat() if times else now_iso
    traceability = {
        "model": {
            "source": prediction.get("source"),
            "name": prediction.get("model_name"),
            "type": prediction.get("model_type"),
            "version": prediction.get("model_version"),
            "training_run_id": prediction.get("run_id"),
        },
        "features_used": feature_engine.get_feature_names(),
        "top_contributing_features": signal.get("top_features", []),
        "data_quality_flags": quality_report.get("checks", {}),
        "missing_data_fallbacks": options_support.get("warnings", []),
        "asof_timestamps": {
            "price": price_asof,
            "benchmark_qqq": qqq_times[-1].isoformat() if qqq_times else None,
            "sector": sector_symbol,
        },
        "quality_report_path": str(quality_report_path),
    }

    pattern_section = {
        "recent_trend_stats": {
            "return_20d": ticker_ret_20,
            "return_60d": ticker_ret_60,
            "adx": adx_value,
            "hvol_20": vol_20,
            "rsi_14": _safe_float(latest_features.get("rsi_14"), np.nan),
        },
        "momentum_stats": {
            "roc_5": _safe_float(latest_features.get("roc_5"), np.nan),
            "roc_10": _safe_float(latest_features.get("roc_10"), np.nan),
            "roc_20": _safe_float(latest_features.get("roc_20"), np.nan),
        },
        "anomaly_signals": {
            "volume_spike": bool(_safe_float(latest_features.get("vol_ratio_20"), 1.0) >= 1.8),
            "gap_risk_proxy": gap,
            "gap_flag": bool(abs(gap) >= 0.02) if gap is not None else False,
        },
        "support_resistance": {
            "recent_high_20": recent_high,
            "recent_low_20": recent_low,
            "distance_to_high": dist_to_high,
            "distance_to_low": dist_to_low,
            "breakout_flag": bool(dist_to_high is not None and dist_to_high > 0),
            "breakdown_flag": bool(dist_to_low is not None and dist_to_low < 0),
        },
        "pattern_analogs": pattern_analogs,
    }

    market_context = {
        "regime_label": regime.get("label"),
        "regime_components": {
            "trend_state": trend_state,
            "volatility_state": vol_state,
            "risk_proxy": risk_proxy,
        },
        "relative_strength": {
            "vs_qqq_20d": rel_vs_qqq,
            "vs_sector_20d": rel_vs_sector,
            "benchmark_return_20d": qqq_ret_20,
            "sector_return_20d": sector_ret_20,
            "sector_etf": sector_symbol,
        },
    }

    summary = {
        "ticker": symbol,
        "status": "completed",
        "primary_score": confidence,
        "bias": signal.get("bias"),
        "confidence": confidence,
        "model_type": prediction.get("model_type"),
        "target_horizon": horizon_days,
        "dataset_range": {
            "start": times[0].isoformat() if times else None,
            "end": times[-1].isoformat() if times else None,
            "bars": len(times),
        },
        "regime_label": market_context.get("regime_label"),
        "updated_at": now_iso,
    }

    return {
        "run_id": run_id,
        "ticker": symbol,
        "status": "completed",
        "generated_at": now_iso,
        "market_context": market_context,
        "price_volume_pattern_analysis": pattern_section,
        "options_decision_support": options_support,
        "risk_summary": risk_summary,
        "traceability": traceability,
        "signal": signal,
        "prediction": prediction,
        "summary": summary,
    }


def _mark_scan_ticker_status(
    job: dict[str, Any],
    symbol: str,
    *,
    status: str,
    error: Optional[str] = None,
) -> None:
    ticker_status = job.setdefault("ticker_status", {})
    entry = dict(ticker_status.get(symbol) or {})
    now = utc_now_iso()
    entry["status"] = status
    entry["updated_at"] = now
    if status == "running":
        entry["started_at"] = now
    else:
        entry.setdefault("started_at", now)
    if status in {"completed", "failed"}:
        entry["completed_at"] = now
    if error:
        entry["error"] = str(error)
    elif status == "completed":
        entry["error"] = None
    ticker_status[symbol] = entry
    _refresh_scan_progress(job)


async def _run_scan_job(run_id: str) -> None:
    job = scan_jobs.get(run_id)
    if job is None:
        job = _load_scan_job_state(run_id)
        if job is None:
            return
        scan_jobs[run_id] = job
        if run_id not in scan_job_order:
            scan_job_order.append(run_id)
            _trim_scan_jobs()

    run_dir = _scan_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _scan_results_dir(run_id).mkdir(parents=True, exist_ok=True)

    job["status"] = "running"
    job["started_at"] = job.get("started_at") or utc_now_iso()
    job["updated_at"] = utc_now_iso()
    _append_scan_job_log(job, "Started deep-learning scan job")
    _persist_scan_job_state(job)

    rows = list(job.get("universe_rows") or [])
    if not rows:
        job["status"] = "failed"
        job["error"] = "Universe rows are empty"
        job["completed_at"] = utc_now_iso()
        job["updated_at"] = job["completed_at"]
        _append_scan_job_log(job, "Scan failed: no universe rows")
        _persist_scan_job_state(job)
        return

    options_mode = bool(job.get("options_mode", False))
    horizon_days = int(job.get("horizon_days", 5))
    history_days = int(job.get("history_days", 3650))
    concurrency = int(job.get("concurrency", 8))
    risk_cfg = dict(job.get("risk_config") or {})
    force_recompute = bool(job.get("force_recompute", False))

    market = market_service or MarketDataService()
    history_cache: dict[str, Any] = {}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, min(concurrency, 64)))

    async def _process(row_meta: dict[str, Any]) -> None:
        symbol = _sanitize_symbol(row_meta.get("symbol"))
        if not symbol:
            return
        existing_status = ((job.get("ticker_status") or {}).get(symbol) or {}).get("status")
        out_path = _scan_results_dir(run_id) / f"{symbol}.json"
        if (not force_recompute) and existing_status == "completed" and out_path.exists():
            async with lock:
                _mark_scan_ticker_status(job, symbol, status="completed")
                _append_scan_job_log(job, f"Skipped {symbol} (already completed)")
                job["updated_at"] = utc_now_iso()
                _persist_scan_job_state(job)
            return

        async with lock:
            _mark_scan_ticker_status(job, symbol, status="running")
            job["current_ticker"] = symbol
            job["updated_at"] = utc_now_iso()
            _persist_scan_job_state(job)

        async with sem:
            try:
                payload = await _build_scan_ticker_payload(
                    run_id=run_id,
                    symbol=symbol,
                    row_meta=row_meta,
                    horizon_days=horizon_days,
                    history_days=history_days,
                    options_mode=options_mode,
                    market=market,
                    history_cache=history_cache,
                    risk_config=risk_cfg,
                )
                _write_scan_result(run_id, symbol, payload)
                row = _scan_summary_row(payload)
                async with lock:
                    _mark_scan_ticker_status(job, symbol, status="completed")
                    (job.setdefault("errors", {})).pop(symbol, None)
                    _update_scan_index(run_id, row)
                    _append_scan_job_log(job, f"Completed {symbol} (score={_safe_float(row.get('primary_score'), 0.0):.3f})")
                    job["updated_at"] = utc_now_iso()
                    _persist_scan_job_state(job)
            except Exception as exc:
                err_payload = {
                    "run_id": run_id,
                    "ticker": symbol,
                    "status": "failed",
                    "generated_at": utc_now_iso(),
                    "error": str(exc),
                    "summary": {
                        "ticker": symbol,
                        "status": "failed",
                        "primary_score": None,
                        "bias": None,
                        "confidence": None,
                        "model_type": None,
                        "target_horizon": horizon_days,
                        "dataset_range": {},
                        "regime_label": None,
                        "updated_at": utc_now_iso(),
                    },
                }
                _write_scan_result(run_id, symbol, err_payload)
                async with lock:
                    _mark_scan_ticker_status(job, symbol, status="failed", error=str(exc))
                    errors = job.setdefault("errors", {})
                    errors[symbol] = str(exc)
                    _update_scan_index(run_id, _scan_summary_row(err_payload))
                    _append_scan_job_log(job, f"Failed {symbol}: {exc}")
                    job["updated_at"] = utc_now_iso()
                    _persist_scan_job_state(job)

    await asyncio.gather(*[_process(row) for row in rows])

    counts = _scan_status_counts(job)
    now = utc_now_iso()
    if counts["completed"] > 0:
        job["status"] = "completed"
    else:
        job["status"] = "failed"
    job["current_ticker"] = None
    job["completed_at"] = now
    job["updated_at"] = now
    _append_scan_job_log(
        job,
        f"Scan finished: completed={counts['completed']} failed={counts['failed']} total={job.get('total_tickers')}",
    )
    _persist_scan_job_state(job)


async def _scan_job_worker() -> None:
    logger.info("Deep-learning scan worker online")
    while True:
        assert scan_job_queue is not None
        run_id = await scan_job_queue.get()
        try:
            await _run_scan_job(run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected error while processing scan job %s", run_id)
        finally:
            scan_job_queue.task_done()


async def _run_deep_learning_job(job_id: str) -> None:
    job = training_jobs.get(job_id)
    if job is None:
        return

    started_at = utc_now_iso()
    job["status"] = "running"
    job["started_at"] = started_at
    job["updated_at"] = started_at

    symbol = job["symbol"]
    model_type = job["model_type"]
    epochs = job["epochs"]
    _append_training_job_log(
        job,
        f"Started deep-learning job for {symbol} ({model_type.upper()}, epochs={epochs})",
    )

    try:
        result = await _execute_deep_learning_subprocess(
            symbol=symbol,
            model_type=model_type,
            epochs=epochs,
            on_log=lambda line: _append_training_job_log(job, line),
        )

        completed_at = utc_now_iso()
        job["status"] = "completed"
        job["updated_at"] = completed_at
        job["completed_at"] = completed_at
        job["result"] = {
            "symbol": symbol,
            "model_name": result.get("model_name", symbol),
            "model_type": result.get("model_type", model_type),
            "metrics": result.get("metrics"),
            "epochs_trained": result.get("epochs_trained", epochs),
            "device": result.get("device", "unknown"),
            "data_quality_report": result.get("data_quality_report"),
            "data_quality_summary": result.get("data_quality_summary"),
            "run_id": result.get("run_id"),
            "run_dir": result.get("run_dir"),
        }
        run_dir = result.get("run_dir")
        if run_dir and job.get("logs"):
            try:
                artifacts_dir = Path(run_dir) / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                (artifacts_dir / "training.log").write_text("\n".join(job["logs"]) + "\n")
            except Exception as exc:
                logger.warning("Unable to persist training logs in run artifact %s: %s", run_dir, exc)
        _append_training_job_log(job, "Training completed successfully")

        logger.info("Deep-learning job %s completed for %s", job_id, symbol)

    except Exception as e:
        completed_at = utc_now_iso()
        error_code, error_message = _classify_deep_learning_error(e)
        job["status"] = "failed"
        job["updated_at"] = completed_at
        job["completed_at"] = completed_at
        job["error"] = f"{error_code}: {error_message}"
        job["result"] = {
            "symbol": symbol,
            "model_name": symbol,
            "model_type": model_type,
            "error_code": error_code,
            "error_message": error_message,
        }
        if write_training_artifact is not None:
            try:
                failure_artifact = write_training_artifact(
                    run_id=job_id,
                    status="failed",
                    model_type=model_type,
                    target="target_direction",
                    horizon=5,
                    tickers=[symbol],
                    timeframes=["1d"],
                    started_at=job.get("started_at"),
                    completed_at=completed_at,
                    diagnostics_summary={
                        "drift_baseline_stats": {},
                        "regime_distribution": {},
                        "error_analysis": {"message": error_message, "error_code": error_code},
                        "top_failure_cases": [{"stage": "deep_learning_job", "error": error_message}],
                    },
                )
                job["result"] = {
                    "symbol": symbol,
                    "model_name": symbol,
                    "model_type": model_type,
                    "run_id": failure_artifact.get("run_id"),
                    "run_dir": failure_artifact.get("run_dir"),
                    "error_code": error_code,
                    "error_message": error_message,
                }
            except Exception as artifact_exc:
                logger.warning("Failed to write failure run artifact for job %s: %s", job_id, artifact_exc)
        _append_training_job_log(job, f"Training failed ({error_code}): {error_message}")
        logger.exception("Deep-learning job %s failed for %s", job_id, symbol)


async def _deep_learning_job_worker() -> None:
    logger.info("Deep-learning job worker online")
    while True:
        assert training_job_queue is not None
        job_id = await training_job_queue.get()
        try:
            await _run_deep_learning_job(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected error while processing deep-learning job %s", job_id)
        finally:
            training_job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global config, db, memory, ai, market_service, self_critique_logger
    global training_jobs, training_job_order, training_job_queue, training_job_worker_task
    global scan_jobs, scan_job_order, scan_job_queue, scan_job_worker_task
    global server_started_at

    # Startup
    logger.info("Starting dPolaris API...")
    config = get_config()
    db = Database()
    memory = DPolarisMemory(db)
    ai = DPolarisAI(config, db, memory)
    market_service = MarketDataService()
    if SelfCritiqueLogger is not None:
        self_critique_logger = SelfCritiqueLogger(log_dir=config.data_dir / "reports" / "self_critique")
    training_jobs = {}
    training_job_order = []
    training_job_queue = asyncio.Queue()
    training_job_worker_task = asyncio.create_task(_deep_learning_job_worker())
    scan_jobs = {}
    scan_job_order = []
    scan_job_queue = asyncio.Queue()
    scan_job_worker_task = asyncio.create_task(_scan_job_worker())
    server_started_at = datetime.utcnow()
    logger.info("dPolaris API started")

    yield

    # Shutdown
    if training_job_worker_task is not None:
        training_job_worker_task.cancel()
        try:
            await training_job_worker_task
        except asyncio.CancelledError:
            pass
        training_job_worker_task = None
    training_job_queue = None

    if scan_job_worker_task is not None:
        scan_job_worker_task.cancel()
        try:
            await scan_job_worker_task
        except asyncio.CancelledError:
            pass
        scan_job_worker_task = None
    scan_job_queue = None

    server_started_at = None
    self_critique_logger = None

    logger.info("Shutting down dPolaris API...")


app = FastAPI(
    title="dPolaris API",
    description="Trading Intelligence System API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Mac app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class PortfolioUpdate(BaseModel):
    cash: float
    invested: float
    total_value: float
    daily_pnl: float = 0
    goal_progress: float = 0


class PositionCreate(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    position_type: str = "stock"
    option_details: Optional[dict] = None
    notes: str = ""


class TradeCreate(BaseModel):
    symbol: str
    strategy: str
    direction: str
    entry_price: float
    quantity: float
    thesis: str = ""
    iv_at_entry: Optional[float] = None
    market_regime: Optional[str] = None
    conviction_score: Optional[float] = None
    tags: list[str] = Field(default_factory=list)
    signal_id: Optional[str] = None


class TradeClose(BaseModel):
    exit_price: float
    outcome_notes: str = ""
    lessons: str = ""


class AlertCreate(BaseModel):
    symbol: str
    alert_type: str
    condition: str
    threshold: float
    message: str = ""


class WatchlistAdd(BaseModel):
    symbol: str
    thesis: str = ""
    target_entry: Optional[float] = None
    priority: int = 5


class ChatMessage(BaseModel):
    message: str


class AnalyzeRequest(BaseModel):
    symbol: str
    analysis_type: str = "full"


class MemoryCreate(BaseModel):
    category: str
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class DeepLearningTrainJobRequest(BaseModel):
    symbol: str
    model_type: str = Field(default="lstm")
    epochs: int = Field(default=50, ge=1, le=500)


class ScanStartRequest(BaseModel):
    universe: str = Field(default="combined_1000")
    run_mode: str = Field(default="scan", alias="runMode")
    horizon_config: dict[str, Any] = Field(default_factory=dict, alias="horizonConfig")
    options_mode: bool = Field(default=False, alias="optionsMode")
    strategy_universe_config: dict[str, Any] = Field(default_factory=dict, alias="strategyUniverseConfig")
    risk_config: dict[str, Any] = Field(default_factory=dict, alias="riskConfig")
    run_id: Optional[str] = Field(default=None, alias="runId")
    force_recompute: bool = Field(default=False, alias="forceRecompute")
    concurrency: Optional[int] = None
    tickers: Optional[list[str]] = None

    model_config = {"populate_by_name": True}


# ==================== REST Endpoints ====================

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/debug/port-owner")
async def debug_port_owner(port: int = Query(8420, ge=1, le=65535)):
    """Inspect TCP port owner for local backend diagnostics."""
    owner_pid = None
    owner_cmdline = None
    source = "none"

    if psutil is not None:
        owner_pid, owner_cmdline = _port_owner_via_psutil(int(port))
        source = "psutil"
    if owner_pid is None:
        owner_pid, owner_cmdline = _port_owner_via_netstat_cim(int(port))
        source = "netstat+cim"

    return {
        "port": int(port),
        "owner_pid": owner_pid,
        "owner_cmdline": owner_cmdline,
        "matches_repo_server_signature": _is_repo_server_cmdline(owner_cmdline, port=int(port)),
        "source": source,
    }


# --- Portfolio ---
@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio summary"""
    portfolio = db.get_latest_portfolio()
    if not portfolio:
        return {
            "total_value": 0,
            "cash": 0,
            "invested": 0,
            "daily_pnl": 0,
            "goal_progress": 0,
        }
    return portfolio


@app.post("/api/portfolio")
async def update_portfolio(update: PortfolioUpdate):
    """Update portfolio snapshot"""
    snapshot_id = db.save_portfolio_snapshot(
        cash=update.cash,
        invested=update.invested,
        total_value=update.total_value,
        daily_pnl=update.daily_pnl,
        goal_progress=update.goal_progress,
    )
    return {"id": snapshot_id, "status": "saved"}


@app.get("/api/portfolio/history")
async def get_portfolio_history(days: int = 30):
    """Get portfolio value history"""
    return db.get_portfolio_history(days)


# --- Goal ---
@app.get("/api/goal")
async def get_goal_progress():
    """Get goal progress details"""
    portfolio = db.get_latest_portfolio()
    if not portfolio:
        return {
            "current_value": 0,
            "target": config.goal.target,
            "progress_percent": 0,
        }

    current = portfolio.get("total_value", 0)
    target = config.goal.target
    start = config.goal.starting_capital

    progress = ((current - start) / (target - start) * 100) if target > start else 0

    return {
        "current_value": current,
        "target": target,
        "starting_capital": start,
        "progress_percent": progress,
        "profit_to_date": current - start,
        "profit_remaining": target - current,
    }


# --- Positions ---
@app.get("/api/positions")
async def get_positions():
    """Get open positions"""
    return db.get_open_positions()


@app.post("/api/positions")
async def create_position(position: PositionCreate):
    """Add new position"""
    position_id = db.add_position(
        symbol=position.symbol.upper(),
        quantity=position.quantity,
        entry_price=position.entry_price,
        position_type=position.position_type,
        option_details=position.option_details,
        notes=position.notes,
    )
    return {"id": position_id, "status": "created"}


@app.delete("/api/positions/{position_id}")
async def close_position(position_id: int, exit_price: float = Query(...)):
    """Close a position"""
    result = db.close_position(position_id, exit_price)
    if result:
        return {"status": "closed", **result}
    raise HTTPException(status_code=404, detail="Position not found")


# --- Journal ---
@app.get("/api/journal")
async def get_journal(
    limit: int = 50,
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Get trade journal entries"""
    return db.get_trades(limit=limit, strategy=strategy, symbol=symbol)


@app.post("/api/journal")
async def create_trade(trade: TradeCreate):
    """Add trade to journal"""
    tags = list(trade.tags)
    if trade.signal_id:
        signal_tag = f"signal_id:{trade.signal_id}"
        if signal_tag not in tags:
            tags.append(signal_tag)

    trade_id = db.add_trade(
        symbol=trade.symbol.upper(),
        strategy=trade.strategy,
        direction=trade.direction,
        entry_price=trade.entry_price,
        quantity=trade.quantity,
        thesis=trade.thesis,
        iv_at_entry=trade.iv_at_entry,
        market_regime=trade.market_regime,
        conviction_score=trade.conviction_score,
        tags=tags,
    )
    return {"id": trade_id, "status": "created"}


@app.put("/api/journal/{trade_id}/close")
async def close_trade(trade_id: int, close_data: TradeClose):
    """Close a trade"""
    close_result = db.close_trade(
        trade_id=trade_id,
        exit_price=close_data.exit_price,
        outcome_notes=close_data.outcome_notes,
        lessons=close_data.lessons,
    )

    trade_row = db.get_trade(trade_id)
    if trade_row and close_result and self_critique_logger is not None:
        signal_id = _extract_signal_id_from_tags(trade_row.get("tags"))
        if signal_id:
            try:
                self_critique_logger.log_outcome(
                    signal_id=signal_id,
                    outcome_timestamp=utc_now_iso(),
                    realized_return=_safe_float(close_result.get("pnl_percent"), 0.0) / 100.0,
                    pnl=_safe_float(close_result.get("pnl"), 0.0),
                    notes=close_data.outcome_notes,
                    extra={
                        "trade_id": trade_id,
                        "symbol": trade_row.get("symbol"),
                    },
                )
            except Exception as exc:
                logger.warning("Self-critique outcome logging failed for trade %s: %s", trade_id, exc)

    # Learn from the trade
    trades = db.get_trades(limit=1)
    if trades:
        memory.learn_from_trade(trades[0])

    return {"status": "closed"}


@app.get("/api/journal/stats")
async def get_trade_stats():
    """Get trading statistics"""
    return db.get_trade_stats()


# --- Watchlist ---
@app.get("/api/watchlist")
async def get_watchlist():
    """Get watchlist"""
    return db.get_watchlist()


@app.post("/api/watchlist")
async def add_to_watchlist(item: WatchlistAdd):
    """Add to watchlist"""
    item_id = db.add_to_watchlist(
        symbol=item.symbol.upper(),
        thesis=item.thesis,
        target_entry=item.target_entry,
        priority=item.priority,
    )
    return {"id": item_id, "status": "added"}


@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Remove from watchlist"""
    db.remove_from_watchlist(symbol.upper())
    return {"status": "removed"}


# --- Alerts ---
@app.get("/api/alerts")
async def get_alerts(active_only: bool = True):
    """Get alerts"""
    if active_only:
        return db.get_active_alerts()
    # Would need a get_all_alerts method
    return db.get_active_alerts()


@app.post("/api/alerts")
async def create_alert(alert: AlertCreate):
    """Create alert"""
    alert_id = db.create_alert(
        symbol=alert.symbol.upper(),
        alert_type=alert.alert_type,
        condition=alert.condition,
        threshold=alert.threshold,
        message=alert.message,
    )
    return {"id": alert_id, "status": "created"}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete alert"""
    # Would need a delete_alert method
    return {"status": "deleted"}


# --- AI Status & Memory ---
@app.get("/api/status")
async def get_ai_status():
    """Get AI daemon/status summary for the app dashboard."""
    daemon_running = False
    scheduler_available = False
    scheduler_error: Optional[dict[str, str]] = None
    scheduler_activity_times: list[datetime] = []

    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler_status = scheduler.get_status()
        scheduler_available = True
        daemon_running = bool(scheduler_status.get("running", False))

        for key in ("last_training", "last_news_scan", "last_prediction", "last_sync"):
            parsed = _parse_timestamp(scheduler_status.get(key))
            if parsed is not None:
                scheduler_activity_times.append(parsed)
    except Exception as exc:
        if _is_apscheduler_missing(exc):
            scheduler_error = _scheduler_dependency_detail()
        # Scheduler is optional in local/dev flows.
        pass

    with db.get_connection() as conn:
        memories_row = conn.execute(
            "SELECT COUNT(*) AS count FROM ai_memory WHERE is_active = 1"
        ).fetchone()
        trades_row = conn.execute("SELECT COUNT(*) AS count FROM trades").fetchone()
        memory_last_row = conn.execute(
            "SELECT MAX(created_at) AS timestamp FROM ai_memory WHERE is_active = 1"
        ).fetchone()
        trades_last_row = conn.execute(
            "SELECT MAX(created_at) AS timestamp FROM trades"
        ).fetchone()
        models_row = conn.execute(
            "SELECT COUNT(*) AS count FROM ml_models WHERE is_active = 1"
        ).fetchone()

    total_memories = int((memories_row["count"] if memories_row else 0) or 0)
    total_trades = int((trades_row["count"] if trades_row else 0) or 0)
    models_available = int((models_row["count"] if models_row else 0) or 0)

    # Prefer filesystem-backed model discovery because deep-learning models are file-based.
    try:
        from ml import Predictor

        models_available = len(Predictor().list_available_models())
    except Exception:
        pass

    win_rate: Optional[float] = None
    try:
        stats = db.get_trade_stats()
        raw_win_rate = stats.get("win_rate")
        if raw_win_rate is not None:
            win_rate = float(raw_win_rate)
    except Exception:
        pass

    db_activity_times = [
        _parse_timestamp(memory_last_row["timestamp"] if memory_last_row else None),
        _parse_timestamp(trades_last_row["timestamp"] if trades_last_row else None),
    ]
    all_activity_times = [
        ts for ts in scheduler_activity_times + db_activity_times if ts is not None
    ]
    last_activity = max(all_activity_times).isoformat() if all_activity_times else None

    return {
        "daemon_running": daemon_running,
        "scheduler_available": scheduler_available,
        "scheduler_error": scheduler_error,
        "last_activity": last_activity,
        "total_memories": total_memories,
        "total_trades": total_trades,
        "models_available": models_available,
        "llm_provider": ai.llm_provider_name if ai else "none",
        "llm_enabled": ai.llm_enabled if ai else False,
        "llm_detail": None if (ai and ai.llm_enabled) else _llm_disabled_detail()["detail"],
        "uptime": _format_uptime(server_started_at),
        "win_rate": win_rate,
    }


@app.get("/api/orchestrator/status")
async def get_orchestrator_status():
    """Get orchestrator and backend self-healing status."""
    try:
        cfg = config or get_config()
        status = _orchestrator_runtime_status(
            data_dir=cfg.data_dir,
            default_host="127.0.0.1",
            default_port=8420,
            heartbeat_stale_seconds=180,
        )
        return status
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/orchestrator/restart-backend")
async def orchestrator_restart_backend():
    """Restart backend using orchestrator process manager."""
    try:
        runtime = _orchestrator_runtime_status(
            data_dir=(config or get_config()).data_dir,
            default_host="127.0.0.1",
            default_port=8420,
            heartbeat_stale_seconds=180,
        )
        hb = runtime.get("heartbeat_payload") if isinstance(runtime, dict) else {}
        if not isinstance(hb, dict):
            hb = {}

        host = str(hb.get("host") or "127.0.0.1")
        port = int(hb.get("port") or 8420)
        python_exe = str(hb.get("python_executable") or sys.executable)
        workdir = str(hb.get("workdir") or Path(__file__).resolve().parent.parent)

        from daemon.backend_process import BackendProcessConfig, BackendProcessManager

        manager = BackendProcessManager(
            BackendProcessConfig(
                host=host,
                port=port,
                python_exe=Path(python_exe).resolve(),
                workdir=Path(workdir).resolve(),
                data_dir=(config or get_config()).data_dir,
            )
        )
        manager.restart_backend(reason="api_request")
        healthy = manager.wait_until_healthy(timeout_seconds=20)
        result = {
            "status": "ok" if healthy else "error",
            "healthy": healthy,
            "backend": manager.get_state(),
            "orchestrator_running": runtime.get("running", False),
        }
        if result.get("status") != "ok":
            raise HTTPException(status_code=500, detail=result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/memories")
async def get_memories(category: Optional[str] = None, limit: int = 50):
    """Get AI memories with optional category filter."""
    safe_limit = max(1, min(limit, 500))
    return memory.recall(category=category, limit=safe_limit, min_importance=0)


@app.post("/api/memories")
async def add_memory(entry: MemoryCreate):
    """Add a memory entry."""
    memory_id = memory.learn(
        category=entry.category,
        content=entry.content,
        importance=entry.importance,
    )
    return {"id": memory_id, "status": "saved"}


# --- AI ---
@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Chat with dPolaris AI"""
    _require_llm_enabled()
    response = await ai.chat(message.message)
    return {"response": response, "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze a symbol"""
    _require_llm_enabled()
    response = await ai.chat(f"@analyze {request.symbol}")
    return {
        "symbol": request.symbol,
        "analysis": response,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/scout")
async def scout():
    """Run opportunity scanner"""
    _require_llm_enabled()
    response = await ai.chat("@scout")
    return {"report": response, "timestamp": datetime.now().isoformat()}


# --- Deep Learning Scan ---
def _normalize_run_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
    normalized = "".join(allowed)
    if not normalized:
        return None
    return normalized[:64]


def _int_from_map(source: dict[str, Any], keys: list[str], default: int) -> int:
    for key in keys:
        raw = source.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return int(default)


def _build_scan_run_summary(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": job.get("id"),
        "type": "deep_learning_scan",
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "universe": job.get("universe_name"),
        "universe_hash": job.get("universe_hash"),
        "run_mode": job.get("run_mode"),
        "horizon_days": job.get("horizon_days"),
        "history_days": job.get("history_days"),
        "options_mode": job.get("options_mode"),
        "concurrency": job.get("concurrency"),
        "total_tickers": job.get("total_tickers"),
        "progress_percent": job.get("progress_percent"),
    }


def _public_scan_run_from_summary(run_id: str, summary: dict[str, Any], index_items: list[dict[str, Any]]) -> dict[str, Any]:
    completed = 0
    failed = 0
    primary_score = None
    scores: list[float] = []
    for row in index_items:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").lower()
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        score = _safe_float(row.get("primary_score"), np.nan)
        if np.isfinite(score):
            scores.append(float(score))
    if scores:
        primary_score = max(scores)

    total = int(summary.get("total_tickers") or len(index_items) or 0)
    if total <= 0:
        total = max(completed + failed, len(index_items))
    processed = completed + failed
    progress = summary.get("progress_percent")
    if not isinstance(progress, (int, float)):
        progress = (processed / total * 100.0) if total > 0 else 0.0
    progress = float(progress)
    queued = max(0, total - processed)

    run_mode = summary.get("run_mode") or "scan"
    horizon_days = summary.get("horizon_days")
    options_mode = bool(summary.get("options_mode"))
    concurrency = summary.get("concurrency")
    config_summary = f"h={horizon_days}d, options={'on' if options_mode else 'off'}, workers={concurrency}"
    status = str(summary.get("status") or ("completed" if total > 0 and processed >= total else "unknown"))

    return {
        "id": run_id,
        "run_id": run_id,
        "runId": run_id,
        "shortRunId": _short_run_id(run_id),
        "status": status,
        "runMode": run_mode,
        "run_mode": run_mode,
        "universe": summary.get("universe"),
        "universeHash": summary.get("universe_hash"),
        "universe_hash": summary.get("universe_hash"),
        "horizonDays": horizon_days,
        "horizon_days": horizon_days,
        "optionsMode": options_mode,
        "options_mode": options_mode,
        "concurrency": concurrency,
        "config_summary": config_summary,
        "primary_score": primary_score,
        "progressPercent": round(progress, 2),
        "processedTickers": processed,
        "totalTickers": total,
        "completedTickers": completed,
        "failedTickers": failed,
        "queuedTickers": queued,
        "runningTickers": 0,
        "currentTicker": None,
        "current_ticker": None,
        "createdAt": summary.get("created_at"),
        "created_at": summary.get("created_at"),
        "updatedAt": summary.get("updated_at"),
        "updated_at": summary.get("updated_at"),
        "startedAt": summary.get("started_at"),
        "started_at": summary.get("started_at"),
        "completedAt": summary.get("completed_at"),
        "completed_at": summary.get("completed_at"),
        "runDir": str(_scan_run_dir(run_id)),
        "warnings": [],
        "errorsSummary": {"count": 0, "tickers": []},
        "logs": [],
    }


def _collect_scan_run_ids() -> list[str]:
    run_ids: set[str] = {str(x) for x in scan_job_order if str(x).strip()}
    root = _runs_root()
    if root.exists() and root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            run_id = child.name
            if (
                _scan_state_path(run_id).exists()
                or _scan_index_path(run_id).exists()
                or (child / "run_summary.json").exists()
            ):
                run_ids.add(run_id)
    return sorted(run_ids)


def _list_scan_runs(limit: int, status_filter: Optional[str]) -> list[dict[str, Any]]:
    wanted = str(status_filter or "").strip().lower()
    rows: list[dict[str, Any]] = []
    for run_id in _collect_scan_run_ids():
        job = scan_jobs.get(run_id) or _load_scan_job_state(run_id)
        if job is not None:
            public = _public_scan_job(job)
        else:
            run_dir = _scan_run_dir(run_id)
            summary = _json_load(run_dir / "run_summary.json") or {}
            if not summary:
                continue
            index_items = _load_scan_index(run_id)
            public = _public_scan_run_from_summary(run_id, summary, index_items)

        state = str(public.get("status") or "").strip().lower()
        if wanted and state != wanted:
            continue
        rows.append(public)

    def _sort_key(item: dict[str, Any]) -> datetime:
        for key in ("updatedAt", "completedAt", "startedAt", "createdAt", "updated_at", "completed_at", "started_at", "created_at"):
            parsed = _parse_timestamp(item.get(key))
            if parsed is not None:
                return parsed
        return datetime.min

    rows.sort(key=_sort_key, reverse=True)
    return rows[: max(1, int(limit))]


@app.get("/universe/list")
@app.get("/api/universe/list")
@app.get("/scan/universe/list")
@app.get("/api/scan/universe/list")
async def list_universe_definitions():
    """List available universe definitions."""
    universes = _list_universe_entries()
    payload: dict[str, Any] = {"universes": universes, "count": len(universes)}
    if not universes:
        payload["detail"] = "No universe files found in configured universe directories."
    return payload


@app.get("/scan/universe")
@app.get("/api/scan/universe")
async def get_scan_universe_by_name(name: str = Query(..., min_length=1)):
    return await get_scan_universe(_normalize_universe_alias(name))


@app.get("/scan/universe/{universe_name}")
@app.get("/api/scan/universe/{universe_name}")
async def get_scan_universe(universe_name: str):
    universe_name = _normalize_universe_alias(universe_name)
    try:
        normalized_name, path = _resolve_universe_definition_path(universe_name)
        payload = _load_universe_file_payload(path)
        tickers = _extract_universe_tickers(payload, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response = {
        "name": normalized_name,
        "path": str(path),
        "tickers": tickers,
        "count": len(tickers),
    }
    if isinstance(payload, dict):
        response["generated_at"] = payload.get("generated_at")
        response["universe_hash"] = payload.get("universe_hash")
        response["schema_version"] = payload.get("schema_version")
        response["universe"] = payload
    return response


@app.get("/universe/{universe_name}")
@app.get("/api/universe/{universe_name}")
async def get_universe_definition(universe_name: str):
    normalized_input = (universe_name or "").strip().lower()
    if normalized_input == "list":
        universes = _list_universe_entries()
        payload: dict[str, Any] = {"universes": universes, "count": len(universes)}
        if not universes:
            payload["detail"] = "No universe files found in configured universe directories."
        return payload

    if normalized_input in {"all", "default", "*"}:
        entries = _list_universe_entries()
        all_entry = next((x for x in entries if str(x.get("name") or "").lower() == "all"), None)
        symbols: set[str] = set()
        for item in entries:
            name = str(item.get("name") or "").strip().lower()
            if not name or name == "all":
                continue
            try:
                tickers = _load_universe_symbols_from_path(Path(str(item.get("path") or "")))
            except Exception:
                continue
            for ticker in tickers:
                cleaned = _sanitize_symbol(ticker)
                if cleaned:
                    symbols.add(cleaned)
        tickers_sorted = sorted(symbols)
        return {
            "name": "all",
            "tickers": tickers_sorted,
            "count": len(tickers_sorted),
            "meta": {
                "source": "dynamic_union",
                "path": str((all_entry or {}).get("path") or "dynamic:all"),
                "updated_at": (all_entry or {}).get("updated_at"),
            },
        }

    universe_name = _normalize_universe_alias(universe_name)
    known_names = _list_universe_definitions()
    try:
        normalized_name, path = _resolve_universe_definition_path(universe_name)
        payload = _load_universe_file_payload(path)
        tickers = _extract_universe_tickers(payload, path)
    except FileNotFoundError:
        looked_in = [str(path) for path in _configured_universe_dirs()]
        supported = sorted(list(SUPPORTED_UNIVERSE_EXTENSIONS))
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Universe file not found",
                "name": str(universe_name),
                "looked_in": looked_in,
                "supported_extensions": supported,
                "searched_paths": looked_in,
                "known_universes": known_names,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    meta: dict[str, Any] = {"source": "file", "path": str(path)}
    if isinstance(payload, dict):
        for key in ("generated_at", "universe_hash", "schema_version", "format"):
            if key in payload:
                meta[key] = payload.get(key)

    return {
        "name": normalized_name,
        "tickers": tickers,
        "count": len(tickers),
        "meta": meta,
    }


@app.get("/scan/runs")
@app.get("/api/scan/runs")
async def list_scan_runs_endpoint(
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
):
    rows = _list_scan_runs(limit=limit, status_filter=status)
    return {"runs": rows, "count": len(rows)}


@app.post("/scan/start")
@app.post("/api/scan/start")
async def start_scan(request: ScanStartRequest):
    """Start (or resume) a deep-learning scan over a configured universe."""
    if scan_job_queue is None:
        raise HTTPException(status_code=503, detail="Scan queue is not ready")

    try:
        universe_payload, universe_rows, universe_path = _load_scan_universe(request.universe)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if request.tickers:
        wanted = {_sanitize_symbol(x) for x in request.tickers}
        wanted = {x for x in wanted if x}
        if wanted:
            universe_rows = [row for row in universe_rows if row.get("symbol") in wanted]
        if not universe_rows:
            raise HTTPException(status_code=400, detail="Requested tickers are not present in selected universe")

    requested_run_id = _normalize_run_id(request.run_id)
    run_id = requested_run_id or f"scan_{uuid4().hex[:12]}"
    if requested_run_id is None:
        while _scan_run_dir(run_id).exists():
            run_id = f"scan_{uuid4().hex[:12]}"

    run_dir = _scan_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _scan_results_dir(run_id).mkdir(parents=True, exist_ok=True)

    existing = _load_scan_job_state(run_id)
    force_recompute = bool(request.force_recompute)
    if existing and not force_recompute:
        status = str(existing.get("status") or "unknown")
        if status == "completed":
            return {
                "runId": run_id,
                "status": "completed",
                "statusUrl": f"/scan/status/{run_id}",
                "resultsUrl": f"/scan/results/{run_id}",
            }

        existing["status"] = "queued"
        existing["updated_at"] = utc_now_iso()
        scan_jobs[run_id] = existing
        if run_id not in scan_job_order:
            scan_job_order.append(run_id)
            _trim_scan_jobs()
        await scan_job_queue.put(run_id)
        _persist_scan_job_state(existing)
        _json_dump(run_dir / "run_summary.json", _build_scan_run_summary(existing))
        return {
            "runId": run_id,
            "status": "queued",
            "statusUrl": f"/scan/status/{run_id}",
            "resultsUrl": f"/scan/results/{run_id}",
        }

    if force_recompute:
        results_dir = _scan_results_dir(run_id)
        for old in results_dir.glob("*.json"):
            try:
                old.unlink()
            except Exception:
                pass

    horizon_cfg = dict(request.horizon_config or {})
    strategy_cfg = dict(request.strategy_universe_config or {})
    risk_cfg = dict(request.risk_config or {})
    history_days = _int_from_map(
        source=horizon_cfg,
        keys=["historyDays", "history_days", "lookbackDays", "lookback_days"],
        default=int(os.getenv("DPOLARIS_SCAN_HISTORY_DAYS", "3650")),
    )
    horizon_days = _int_from_map(
        source=horizon_cfg,
        keys=["horizonDays", "horizon_days", "days", "horizon"],
        default=5,
    )
    workers = request.concurrency
    if workers is None:
        workers = _int_from_map(
            source=strategy_cfg,
            keys=["concurrency", "workers", "workerCount"],
            default=int(os.getenv("DPOLARIS_SCAN_WORKERS", "8")),
        )

    now = utc_now_iso()
    ticker_status = {
        row["symbol"]: {
            "status": "queued",
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
            "error": None,
        }
        for row in universe_rows
    }

    job = {
        "id": run_id,
        "status": "queued",
        "type": "deep_learning_scan",
        "run_mode": request.run_mode,
        "universe_name": request.universe,
        "universe_hash": universe_payload.get("universe_hash"),
        "universe_path": str(universe_path),
        "run_dir": str(run_dir),
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "current_ticker": None,
        "horizon_days": max(1, min(horizon_days, 90)),
        "history_days": max(120, min(history_days, 10000)),
        "options_mode": bool(request.options_mode),
        "risk_config": risk_cfg,
        "strategy_universe_config": strategy_cfg,
        "horizon_config": horizon_cfg,
        "force_recompute": force_recompute,
        "concurrency": max(1, min(int(workers), 64)),
        "total_tickers": len(universe_rows),
        "ticker_status": ticker_status,
        "warnings": [],
        "errors": {},
        "logs": [],
        "result_index": [],
        "request": request.model_dump(by_alias=True),
        "universe_rows": universe_rows,
    }
    _refresh_scan_progress(job)
    _append_scan_job_log(job, f"Job queued for {len(universe_rows)} tickers from universe '{request.universe}'")

    _json_dump(run_dir / "universe_snapshot.json", universe_payload)
    _json_dump(run_dir / SCAN_REQUEST_FILE, request.model_dump(by_alias=True))
    _persist_scan_index(run_id, [])
    _persist_scan_job_state(job)
    _json_dump(run_dir / "run_summary.json", _build_scan_run_summary(job))

    scan_jobs[run_id] = job
    if run_id not in scan_job_order:
        scan_job_order.append(run_id)
    _trim_scan_jobs()
    await scan_job_queue.put(run_id)

    return {
        "runId": run_id,
        "status": "queued",
        "statusUrl": f"/scan/status/{run_id}",
        "resultsUrl": f"/scan/results/{run_id}",
    }


@app.get("/scan/status/{run_id}")
@app.get("/api/scan/status/{run_id}")
async def get_scan_status(run_id: str):
    job = scan_jobs.get(run_id) or _load_scan_job_state(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Scan run not found: {run_id}")
    return _public_scan_job(job)


@app.get("/scan/results/{run_id}")
@app.get("/api/scan/results/{run_id}")
async def get_scan_results(
    run_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, alias="pageSize", ge=1, le=500),
    status: Optional[str] = Query(None),
):
    state = scan_jobs.get(run_id) or _load_scan_job_state(run_id)
    run_dir = _scan_run_dir(run_id)
    if state is None and not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Scan run not found: {run_id}")

    items = _load_scan_index(run_id)
    if status:
        wanted = status.strip().lower()
        items = [x for x in items if str(x.get("status", "")).lower() == wanted]

    def _sort_key(item: dict[str, Any]):
        score = item.get("primary_score")
        if isinstance(score, (int, float)):
            return (0, -float(score), str(item.get("ticker") or ""))
        return (1, 0.0, str(item.get("ticker") or ""))

    items = sorted(items, key=_sort_key)
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    paged = items[start:end]

    return {
        "runId": run_id,
        "status": (state or {}).get("status"),
        "page": page,
        "pageSize": page_size,
        "total": total,
        "count": len(paged),
        "items": paged,
    }


@app.get("/scan/result/{run_id}/{ticker}")
@app.get("/api/scan/result/{run_id}/{ticker}")
async def get_scan_result_detail(run_id: str, ticker: str):
    symbol = _sanitize_symbol(ticker)
    if not symbol:
        raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}")
    path = _scan_results_dir(run_id) / f"{symbol}.json"
    loaded = _json_load(path)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"Scan result not found for {symbol} in run {run_id}")
    return loaded


# --- Market Data ---
@app.get("/api/market/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote"""
    quote = await market_service.get_quote(symbol.upper())
    if quote:
        return quote
    raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")


@app.get("/api/market/overview")
async def market_overview():
    """Get market overview"""
    return await get_market_overview()


@app.get("/api/market/regime")
async def get_regime():
    """Get market regime"""
    snapshot = db.get_latest_market_snapshot()
    if snapshot:
        return snapshot
    # Generate fresh assessment
    response = await ai.chat("@regime")
    return {"regime": response, "timestamp": datetime.now().isoformat()}


# --- Performance ---
@app.get("/api/performance")
async def get_performance(days: int = 90):
    """Get performance metrics"""
    return db.get_performance_history(days)


@app.get("/api/performance/strategies")
async def get_strategy_performance():
    """Get performance by strategy"""
    return memory.get_strategy_rankings()


# --- ML Models ---
@app.get("/api/models")
async def list_models():
    """List available ML models"""
    try:
        from ml import Predictor
        predictor = Predictor()
        return predictor.list_available_models()
    except Exception as e:
        return {"error": str(e), "models": []}


@app.post("/api/predict/{symbol}")
async def predict(symbol: str):
    """Get ML prediction for symbol"""
    response = await ai.chat(f"@predict {symbol}")
    return {"symbol": symbol, "prediction": response}


@app.post("/api/train/{symbol}")
async def train_model(symbol: str):
    """Train model for symbol"""
    response = await ai.chat(f"@train {symbol}")
    return {"symbol": symbol, "result": response}


# --- Deep Learning ---
@app.get("/api/deep-learning/status")
async def get_dl_status():
    """Get deep learning system status"""
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    forced_py313 = os.getenv("DPOLARIS_ALLOW_PY313_TORCH") == "1"
    py313_guard_active = (py_major, py_minor) >= (3, 13) and not forced_py313

    torch_importable = False
    torch_error = None
    sklearn_importable = False
    sklearn_error = None
    cuda_available = False
    cuda_device = None
    mps_available = False
    device = "unknown"
    torch_version = None

    try:
        import torch  # type: ignore

        torch_importable = True
        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception as exc:
        torch_error = str(exc)

    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401

        sklearn_importable = True
    except Exception as exc:
        sklearn_error = str(exc)

    if torch_importable:
        try:
            from ml.deep_learning import DEVICE

            device = str(DEVICE)
        except Exception:
            if cuda_available:
                device = "cuda"
            elif mps_available:
                device = "mps"
            else:
                device = "cpu"

    deep_learning_enabled = torch_importable and sklearn_importable and not py313_guard_active
    if not torch_importable:
        deep_learning_reason = "torch is not installed; run requirements-windows.txt or scripts/install_torch_gpu.ps1"
    elif not sklearn_importable:
        deep_learning_reason = "scikit-learn is not installed; run requirements-windows.txt"
    elif py313_guard_active:
        deep_learning_reason = (
            "disabled on Python 3.13 for stability; use Python 3.11/3.12 or set DPOLARIS_ALLOW_PY313_TORCH=1"
        )
    else:
        deep_learning_reason = "enabled"

    return {
        "device": device,
        "python_executable": sys.executable,
        "python_version": f"{py_major}.{py_minor}",
        "torch_importable": torch_importable,
        "torch_version": torch_version,
        "torch_error": torch_error,
        "sklearn_importable": sklearn_importable,
        "sklearn_error": sklearn_error,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "mps_available": mps_available,
        "deep_learning_enabled": deep_learning_enabled,
        "deep_learning_reason": deep_learning_reason,
    }


@app.post("/api/jobs/deep-learning/train")
async def enqueue_deep_learning_training(job_request: DeepLearningTrainJobRequest):
    """Submit deep-learning training as an async job."""
    if training_job_queue is None:
        raise HTTPException(status_code=503, detail="Training queue is not ready")

    symbol = job_request.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    model_type = job_request.model_type.strip().lower()
    if model_type not in SUPPORTED_DL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_type '{model_type}'. Choose one of {sorted(SUPPORTED_DL_MODELS)}",
        )

    now = utc_now_iso()
    job_id = str(uuid4())
    job = {
        "id": job_id,
        "status": "queued",
        "type": "deep_learning_train",
        "symbol": symbol,
        "model_type": model_type,
        "epochs": job_request.epochs,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "logs": [],
    }
    _append_training_job_log(
        job,
        f"Job queued for {symbol} ({model_type.upper()}, epochs={job_request.epochs})",
    )

    training_jobs[job_id] = job
    training_job_order.append(job_id)
    _trim_training_jobs()
    await training_job_queue.put(job_id)

    logger.info("Queued deep-learning job %s for %s (%s)", job_id, symbol, model_type)
    return _public_training_job(job)


@app.get("/api/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get training job status/result."""
    job = training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _public_training_job(job)


@app.get("/api/jobs")
async def list_training_jobs(limit: int = 20):
    """List recent training jobs (newest first)."""
    safe_limit = max(1, min(limit, 100))
    ordered_ids = list(reversed(training_job_order[:]))[:safe_limit]
    jobs = [
        _public_training_job(training_jobs[job_id])
        for job_id in ordered_ids
        if job_id in training_jobs
    ]
    return {"jobs": jobs, "count": len(jobs)}


# --- Training Run Artifacts ---
@app.get("/runs")
async def list_runs(limit: int = Query(50, ge=1, le=500)):
    """List training runs from run artifacts."""
    if list_training_runs is None:
        raise HTTPException(status_code=503, detail="Training artifact registry unavailable")
    runs = list_training_runs(limit=limit)
    return {"runs": runs, "count": len(runs)}


@app.get("/runs/compare")
async def compare_runs(run_ids: str = Query(..., description="Comma-separated run IDs")):
    """Compare multiple runs on headline metrics."""
    if compare_training_runs is None:
        raise HTTPException(status_code=503, detail="Training artifact comparator unavailable")
    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two run IDs")
    try:
        return compare_training_runs(ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    """Get full training run artifact payload."""
    if load_training_artifact is None:
        raise HTTPException(status_code=503, detail="Training artifact loader unavailable")
    try:
        artifact = load_training_artifact(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return artifact


@app.get("/runs/{run_id}/artifacts")
async def get_run_artifacts(run_id: str):
    """List files available under runs/<run_id>/."""
    if list_run_artifact_files is None:
        raise HTTPException(status_code=503, detail="Training artifact loader unavailable")
    try:
        files = list_run_artifact_files(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {"run_id": run_id, "artifacts": files, "count": len(files)}


@app.get("/runs/{run_id}/artifact/{artifact_name:path}")
async def get_run_artifact_file(run_id: str, artifact_name: str):
    """Download one artifact file from a run folder."""
    if resolve_run_artifact_path is None:
        raise HTTPException(status_code=503, detail="Training artifact loader unavailable")
    try:
        artifact_path = resolve_run_artifact_path(run_id, artifact_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    return FileResponse(path=str(artifact_path), filename=artifact_path.name)


@app.post("/api/deep-learning/train/{symbol}")
async def train_deep_learning(symbol: str, model_type: str = "lstm", epochs: int = 50):
    """Train deep learning model for a symbol"""
    started_at = utc_now_iso()
    try:
        model_type = model_type.strip().lower()
        if model_type not in SUPPORTED_DL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model_type '{model_type}'. Choose one of {sorted(SUPPORTED_DL_MODELS)}",
            )

        result = await _execute_deep_learning_subprocess(
            symbol=symbol.upper(),
            model_type=model_type,
            epochs=epochs,
        )

        return {
            "symbol": symbol.upper(),
            "model_name": result.get("model_name", symbol.upper()),
            "model_type": result.get("model_type", model_type),
            "metrics": result.get("metrics"),
            "epochs_trained": result.get("epochs_trained", epochs),
            "device": result.get("device", "unknown"),
            "data_quality_report": result.get("data_quality_report"),
            "data_quality_summary": result.get("data_quality_summary"),
            "run_id": result.get("run_id"),
            "run_dir": result.get("run_dir"),
        }

    except HTTPException:
        raise
    except Exception as e:
        if write_training_artifact is not None:
            try:
                failure_artifact = write_training_artifact(
                    run_id=None,
                    status="failed",
                    model_type=model_type,
                    target="target_direction",
                    horizon=5,
                    tickers=[symbol.upper()],
                    timeframes=["1d"],
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    diagnostics_summary={
                        "drift_baseline_stats": {},
                        "regime_distribution": {},
                        "error_analysis": {"message": str(e)},
                        "top_failure_cases": [{"stage": "api_deep_learning_train", "error": str(e)}],
                    },
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(e),
                        "run_id": failure_artifact.get("run_id"),
                        "run_dir": failure_artifact.get("run_dir"),
                    },
                )
            except HTTPException:
                raise
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/deep-learning/predict/{symbol}")
async def deep_learning_predict(symbol: str):
    """Get deep learning prediction for symbol"""
    try:
        from ml.deep_learning import DeepLearningTrainer
        from tools.market_data import MarketDataService

        trainer = DeepLearningTrainer()
        market = MarketDataService()

        # Load model
        try:
            model, scaler, metadata = trainer.load_model(symbol.upper())
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"No trained model for {symbol}")

        # Get recent data
        df = await market.get_historical(
            symbol.upper(),
            days=int(os.getenv("DPOLARIS_DL_PREDICT_LOOKBACK_DAYS", "365")),
        )
        if df is None or len(df) < 60:
            raise HTTPException(status_code=400, detail="Not enough recent data")

        # Predict
        result = trainer.predict(
            model,
            scaler,
            df,
            probability_calibration=metadata.get("metrics", {}).get("probability_calibration"),
        )

        return {
            "symbol": symbol.upper(),
            "prediction": result["prediction_label"],
            "confidence": result["confidence"],
            "probability_up": result["probability_up"],
            "probability_down": result["probability_down"],
            "model_type": metadata.get("model_type", "unknown"),
            "model_accuracy": metadata.get("metrics", {}).get("accuracy"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _predict_symbol_direction(symbol: str, df) -> dict:
    deep_learning_error: Optional[str] = None

    try:
        from ml.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer()
        model, scaler, metadata = trainer.load_model(symbol)
        result = trainer.predict(
            model,
            scaler,
            df,
            probability_calibration=metadata.get("metrics", {}).get("probability_calibration"),
        )

        return {
            "source": "deep_learning",
            "model_name": f"{symbol}_dl",
            "model_type": metadata.get("model_type", "lstm"),
            "model_version": metadata.get("version"),
            "model_accuracy": metadata.get("metrics", {}).get("accuracy"),
            "prediction_label": result["prediction_label"],
            "confidence": _safe_float(result.get("confidence"), 0.5),
            "probability_up": _safe_float(result.get("probability_up"), 0.5),
            "probability_down": _safe_float(result.get("probability_down"), 0.5),
        }
    except Exception as exc:
        deep_learning_error = str(exc)

    try:
        from ml import Predictor

        predictor = Predictor()
        for model_name in [f"{symbol}_direction", symbol]:
            try:
                prediction = predictor.predict(model_name, df)
                metadata = predictor.loaded_models.get(model_name, {}).get("metadata", {})
                prediction_label = prediction.get("prediction_label", "UP")
                probability_up = prediction.get(
                    "probability_up",
                    0.55 if str(prediction_label).upper() == "UP" else 0.45,
                )
                confidence = prediction.get(
                    "confidence",
                    max(probability_up, 1.0 - _safe_float(probability_up, 0.5)),
                )

                return {
                    "source": "classic_ml",
                    "model_name": model_name,
                    "model_type": metadata.get("model_type", "unknown"),
                    "model_version": metadata.get("version"),
                    "model_accuracy": metadata.get("metrics", {}).get("accuracy"),
                    "prediction_label": str(prediction_label).upper(),
                    "confidence": _safe_float(confidence, 0.5),
                    "probability_up": _safe_float(probability_up, 0.5),
                    "probability_down": _safe_float(
                        prediction.get("probability_down"),
                        1.0 - _safe_float(probability_up, 0.5),
                    ),
                }
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Classic model fallback failed for %s: %s", symbol, exc)

    raise RuntimeError(
        f"No usable trained model found for {symbol}. "
        f"Deep-learning error: {deep_learning_error or 'unknown'}"
    )


@app.get("/predict/inspect")
async def inspect_prediction(
    ticker: str = Query(..., min_length=1),
    time: Optional[str] = Query(None, description="As-of timestamp (ISO8601)"),
    horizon: int = Query(5, ge=1, le=60),
    run_id: Optional[str] = Query(None, alias="runId"),
):
    """Inspect one prediction trace using strict as-of (causal) data."""
    if (
        truncate_dataframe_asof is None
        or build_feature_snapshot is None
        or latest_ohlcv_snapshot is None
        or derive_regime is None
        or decide_trade_outcome is None
    ):
        raise HTTPException(status_code=503, detail="Prediction inspector utilities unavailable")

    symbol = ticker.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Ticker is required")

    warnings: list[str] = []

    try:
        lookback_days = max(
            int(os.getenv("DPOLARIS_INSPECT_HISTORY_DAYS", "3650")),
            int(config.ml.training_data_days if config else 730),
        )
        market = market_service or MarketDataService()
        source_df = await market.get_historical(symbol, days=lookback_days)

        if source_df is None or len(source_df) < 80:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data for {symbol} to inspect prediction",
            )

        slice_info = truncate_dataframe_asof(source_df, time)
        asof_df = slice_info["frame"]
        warnings.extend(slice_info.get("warnings") or [])

        feature_snapshot = build_feature_snapshot(asof_df, target_horizon=horizon)
        raw_features = feature_snapshot.get("raw") or {}
        raw_input = latest_ohlcv_snapshot(asof_df, slice_info["time_col"])

        prediction = _predict_symbol_direction(symbol, asof_df)
        probability_up = _safe_float(prediction.get("probability_up"), 0.5)
        confidence = _safe_float(
            prediction.get("confidence"),
            max(probability_up, 1.0 - probability_up),
        )

        decision = decide_trade_outcome(
            probability_up=probability_up,
            confidence=confidence,
            long_threshold=0.60,
            short_threshold=0.40,
            min_confidence=0.55,
        )
        regime = derive_regime(raw_features)

        macro_values = {
            str(k): v
            for k, v in raw_features.items()
            if str(k).startswith("macro_")
        }
        if not macro_values:
            macro_values = {"available": False}
            warnings.append("No macro features available for this as-of timestamp.")

        sentiment_counts = {
            str(k): v
            for k, v in raw_features.items()
            if str(k).startswith("sent_")
            or "sentiment" in str(k)
            or "attention" in str(k)
        }
        if not sentiment_counts:
            sentiment_counts = {"available": False}
            warnings.append("No sentiment features available for this as-of timestamp.")

        normalized_features: dict[str, Optional[float]] = {}
        normalization = {
            "method": "none",
            "scaler_applied": False,
            "feature_count": 0,
            "source": "raw",
        }

        if str(prediction.get("source", "")).lower() == "classic_ml":
            try:
                from ml import Predictor

                predictor = Predictor()
                model_name = str(prediction.get("model_name") or "")
                if model_name and predictor.load_model(model_name):
                    model_data = predictor.loaded_models.get(model_name, {})
                    scaler = model_data.get("scaler")
                    metadata = model_data.get("metadata", {}) or {}
                    model_feature_names = list(metadata.get("feature_names") or feature_snapshot.get("feature_names") or [])

                    prepared_values: list[float] = []
                    prepared_names: list[str] = []
                    for name in model_feature_names:
                        raw_value = raw_features.get(name)
                        numeric_value = None
                        try:
                            candidate = float(raw_value)
                            if np.isfinite(candidate):
                                numeric_value = candidate
                        except Exception:
                            numeric_value = None
                        prepared_names.append(name)
                        prepared_values.append(np.nan if numeric_value is None else numeric_value)

                    if prepared_names:
                        normalization["feature_count"] = len(prepared_names)
                        raw_matrix = np.asarray([prepared_values], dtype=float)

                        if scaler is not None and np.isfinite(raw_matrix).all():
                            scaled_matrix = scaler.transform(raw_matrix)
                            scaled_values = scaled_matrix[0]
                            normalized_features = {
                                name: (float(scaled_values[idx]) if np.isfinite(scaled_values[idx]) else None)
                                for idx, name in enumerate(prepared_names)
                            }
                            normalization["method"] = "model_scaler"
                            normalization["scaler_applied"] = True
                            normalization["source"] = "model"
                        else:
                            normalized_features = {
                                name: (float(prepared_values[idx]) if np.isfinite(prepared_values[idx]) else None)
                                for idx, name in enumerate(prepared_names)
                            }
                            normalization["method"] = "raw_fallback"
                            normalization["source"] = "model"
                            if scaler is not None:
                                warnings.append(
                                    "Scaler exists but could not be applied because one or more model inputs are missing/non-finite."
                                )

                        precision_cfg = metadata.get("precision_config") if isinstance(metadata, dict) else None
                        if isinstance(precision_cfg, dict):
                            feature_cfg = precision_cfg.get("features")
                            if isinstance(feature_cfg, dict) and feature_cfg.get("scaling_method"):
                                normalization["method"] = str(feature_cfg.get("scaling_method"))
                    else:
                        warnings.append("Model metadata did not expose feature_names for normalization.")
                else:
                    warnings.append("Model context unavailable; returning raw feature vector only.")
            except Exception as exc:
                warnings.append(f"Normalization context unavailable: {exc}")

        if not normalized_features:
            normalization["feature_count"] = len(raw_features)

        explanation_top = _rank_signal_features(raw_features, top_n=8)
        if not explanation_top:
            explanation_top = list(feature_snapshot.get("top_abs_features") or [])[:8]

        explanation_notes = [
            f"Regime: {regime.get('label', 'unknown')}.",
            f"Model source: {prediction.get('source', 'unknown')}.",
            f"Decision: {decision.get('action', 'unknown')} ({decision.get('reason', 'n/a')}).",
        ]

        run_context = {}
        resolved_run_id = run_id.strip() if run_id else None
        if resolved_run_id and load_training_artifact is not None:
            try:
                artifact = load_training_artifact(resolved_run_id)
                run_summary = artifact.get("run_summary", {}) if isinstance(artifact, dict) else {}
                run_context = {
                    "run_id": run_summary.get("run_id", resolved_run_id),
                    "status": run_summary.get("status"),
                    "model_type": run_summary.get("model_type"),
                    "target": run_summary.get("target"),
                    "horizon": run_summary.get("horizon"),
                }
            except FileNotFoundError:
                warnings.append(f"Run {resolved_run_id} was not found; continuing without run artifact context.")
            except Exception as exc:
                warnings.append(f"Run artifact context unavailable: {exc}")

        return {
            "ticker": symbol,
            "requested_time": slice_info.get("requested_time"),
            "resolved_time": slice_info.get("resolved_time"),
            "horizon": int(horizon),
            "run_id": resolved_run_id,
            "run_context": run_context,
            "trace_meta": {
                "rows_total": int(slice_info.get("rows_total", 0)),
                "rows_used": int(slice_info.get("rows_used", 0)),
                "latest_source_time": slice_info.get("latest_source_time"),
                "feature_timestamp": feature_snapshot.get("feature_timestamp"),
                "causal_asof": True,
            },
            "raw_input_snapshot": {
                "ohlcv": raw_input,
                "macro_values": macro_values,
                "sentiment_counts": sentiment_counts,
            },
            "feature_vector": {
                "raw": raw_features,
                "normalized": normalized_features,
                "normalization": normalization,
            },
            "regime": regime,
            "model_output": prediction,
            "decision": decision,
            "explanation": {
                "top_features": explanation_top,
                "notes": explanation_notes,
            },
            "warnings": warnings,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction inspect failed for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/predict/inspect")
async def inspect_prediction_legacy(
    ticker: str = Query(..., min_length=1),
    time: Optional[str] = Query(None, description="As-of timestamp (ISO8601)"),
    horizon: int = Query(5, ge=1, le=60),
    run_id: Optional[str] = Query(None, alias="runId"),
):
    """Legacy alias for /predict/inspect."""
    return await inspect_prediction(ticker=ticker, time=time, horizon=horizon, run_id=run_id)


@app.post("/api/signals/{symbol}")
async def generate_trade_setup(symbol: str, horizon_days: int = Query(5, ge=1, le=20)):
    """Generate actionable trade setup with entries/stops/targets and insights."""
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    try:
        from ml.features import FeatureEngine

        lookback_days = max(
            int(os.getenv("DPOLARIS_SIGNAL_HISTORY_DAYS", "3650")),
            int(config.ml.training_data_days if config else 730),
        )
        market = market_service or MarketDataService()
        df = await market.get_historical(normalized_symbol, days=lookback_days)

        if df is None or len(df) < 260:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data for {normalized_symbol}",
            )

        latest_price = _safe_float(df.iloc[-1].get("close"))
        if latest_price <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid latest price for {normalized_symbol}")

        feature_engine = FeatureEngine()
        feature_df = feature_engine.generate_features(df, include_targets=False)
        if feature_df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Could not generate technical features for {normalized_symbol}",
            )

        latest_features = feature_df.iloc[-1]
        prediction = _predict_symbol_direction(normalized_symbol, df)
        return _build_signal_from_features(
            symbol=normalized_symbol,
            latest_price=latest_price,
            latest_features=latest_features,
            prediction=prediction,
            horizon_days=horizon_days,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate trade setup for %s", normalized_symbol)
        raise HTTPException(status_code=500, detail=str(exc))


# --- News & Sentiment ---
@app.get("/api/news/sentiment")
async def get_news_sentiment(symbols: Optional[str] = None):
    """Get news sentiment for symbols"""
    try:
        from news import NewsEngine

        engine = NewsEngine(use_finbert=False)  # Use keyword for speed

        symbol_list = symbols.split(",") if symbols else None
        results = await engine.update(symbols=symbol_list)

        await engine.close()

        return {
            "market_sentiment": engine.get_market_sentiment(),
            "top_movers": engine.get_top_movers(5),
            "symbols": {k: v.to_dict() for k, v in results.items()},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str):
    """Get sentiment for a specific symbol"""
    try:
        from news import NewsEngine

        engine = NewsEngine(use_finbert=False)
        results = await engine.update(symbols=[symbol.upper()])
        await engine.close()

        sentiment = results.get(symbol.upper())
        if not sentiment:
            return {"symbol": symbol.upper(), "sentiment": None, "message": "No news found"}

        return {
            "symbol": symbol.upper(),
            "sentiment": sentiment.to_dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/articles")
async def get_news_articles(limit: int = 20):
    """Get recent news articles"""
    try:
        from news import NewsEngine

        engine = NewsEngine(use_finbert=False)
        await engine.update()
        await engine.close()

        articles = [a.to_dict() for a in engine.articles[:limit]]
        return {"articles": articles, "count": len(articles)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Scheduler ---
@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        return scheduler.get_status()

    except Exception as exc:
        if _is_apscheduler_missing(exc):
            return _scheduler_dependency_response()
        return {"running": False, "error": str(exc)}


@app.post("/api/scheduler/start")
async def start_scheduler():
    """Start the scheduler"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler.start()
        return {"status": "started"}

    except Exception as exc:
        if _is_apscheduler_missing(exc):
            return _scheduler_dependency_response()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/scheduler/stop")
async def stop_scheduler():
    """Stop the scheduler"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler.stop()
        return {"status": "stopped"}

    except Exception as exc:
        if _is_apscheduler_missing(exc):
            return _scheduler_dependency_response()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/scheduler/run/{job_id}")
async def run_scheduler_job(job_id: str):
    """Manually run a scheduler job"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        await scheduler.run_now(job_id)
        return {"status": "completed", "job": job_id}

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        if _is_apscheduler_missing(exc):
            return _scheduler_dependency_response()
        raise HTTPException(status_code=500, detail=str(exc))


# --- Cloud Sync ---
@app.get("/api/cloud/predictions")
async def get_cloud_predictions(symbols: Optional[str] = None, limit: int = 50):
    """Get predictions from cloud"""
    try:
        from cloud import SupabaseSync

        sync = SupabaseSync.from_env()
        symbol_list = symbols.split(",") if symbols else None

        predictions = await sync.pull_predictions(symbols=symbol_list, limit=limit)
        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        return {"predictions": [], "error": str(e)}


@app.get("/api/cloud/sentiment")
async def get_cloud_sentiment(symbols: Optional[str] = None, limit: int = 50):
    """Get sentiment from cloud"""
    try:
        from cloud import SupabaseSync

        sync = SupabaseSync.from_env()
        symbol_list = symbols.split(",") if symbols else None

        sentiment = await sync.pull_sentiment(symbols=symbol_list, limit=limit)
        return {"sentiment": sentiment, "count": len(sentiment)}

    except Exception as e:
        return {"sentiment": [], "error": str(e)}


# ==================== WebSocket Endpoints ====================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {
            "portfolio": [],
            "alerts": [],
            "chat": [],
            "prices": [],
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        if websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, message: dict):
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """Real-time portfolio updates"""
    await manager.connect(websocket, "portfolio")
    try:
        while True:
            portfolio = db.get_latest_portfolio()
            await websocket.send_json({
                "type": "portfolio_update",
                "data": portfolio or {},
                "timestamp": datetime.now().isoformat(),
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "portfolio")


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Real-time alert notifications"""
    await manager.connect(websocket, "alerts")
    try:
        while True:
            # Check for triggered alerts
            alerts = db.get_active_alerts()
            if alerts:
                await websocket.send_json({
                    "type": "alerts_update",
                    "alerts": alerts,
                    "timestamp": datetime.now().isoformat(),
                })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "alerts")


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time chat with AI"""
    await manager.connect(websocket, "chat")
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            # Process message
            response = await ai.chat(user_message)

            await websocket.send_json({
                "type": "ai_response",
                "message": response,
                "timestamp": datetime.now().isoformat(),
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, "chat")


@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """Real-time price updates"""
    await manager.connect(websocket, "prices")
    try:
        while True:
            # Get watchlist symbols
            watchlist = db.get_watchlist()
            symbols = [w["symbol"] for w in watchlist] if watchlist else config.watchlist

            # Fetch quotes
            quotes = await market_service.get_multiple_quotes(symbols)

            await websocket.send_json({
                "type": "price_update",
                "quotes": quotes,
                "timestamp": datetime.now().isoformat(),
            })
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "prices")


@app.websocket("/ws/stream")
async def websocket_stream_chat(websocket: WebSocket):
    """Streaming chat with AI"""
    await manager.connect(websocket, "chat")
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            # Stream response
            async for chunk in ai.stream_chat(user_message):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk,
                })

            await websocket.send_json({
                "type": "done",
                "timestamp": datetime.now().isoformat(),
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, "chat")


def run_server(host: str = "127.0.0.1", port: int = 8420):
    """Run the API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
