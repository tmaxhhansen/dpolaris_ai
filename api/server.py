"""
FastAPI Server for dPolaris

Provides REST API and WebSocket endpoints for the Mac app.
"""

import asyncio
import copy
import hashlib
import json
import os
import platform
import signal
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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
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
from daemon.backend_control import (
    BackendControlConfig,
    BackendControlManager,
    BackendControlError,
    PortInUseByUnknownProcessError,
    UnsafeForceKillError,
)

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

try:
    from analysis.artifacts import (
        latest_analysis_for_symbol as load_latest_analysis_artifact,
        list_analysis_artifacts as list_analysis_artifacts_store,
        load_analysis_artifact as load_analysis_artifact_store,
        write_analysis_artifact as write_analysis_artifact_store,
    )
    from analysis.pipeline import default_version_info, generate_analysis_report
except Exception:  # pragma: no cover
    load_latest_analysis_artifact = None
    list_analysis_artifacts_store = None
    load_analysis_artifact_store = None
    write_analysis_artifact_store = None
    default_version_info = None
    generate_analysis_report = None

try:
    from universe.builder import (
        build_combined_universe,
        build_nasdaq_top_500,
        build_wsb_top_500,
    )
    from universe.providers import build_mentions_provider
except Exception:  # pragma: no cover
    build_combined_universe = None
    build_nasdaq_top_500 = None
    build_wsb_top_500 = None
    build_mentions_provider = None

try:
    from news.providers import fetch_news_with_cache
except Exception:  # pragma: no cover
    fetch_news_with_cache = None

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
backend_control_manager: Optional[BackendControlManager] = None
backend_heartbeat_task: Optional[asyncio.Task] = None

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
UNIVERSE_CANONICAL_NAMES = ("nasdaq500", "watchlist", "combined", "wsb100")
UNIVERSE_ALIAS_MAP = {
    "nasdaq500": "nasdaq500",
    "nasdaq300": "nasdaq500",
    "nasdaq_top_500": "nasdaq500",
    "nasdaqtop500": "nasdaq500",
    "nasdaq_top500": "nasdaq500",
    "wsb100": "wsb100",
    "wsb_top_500": "wsb100",
    "wsbtop500": "wsb100",
    "wsb_top500": "wsb100",
    "wsb_favorites": "wsb100",
    "wsbfavorites": "wsb100",
    "combined": "combined",
    "combined400": "combined",
    "combined_400": "combined",
    "combined_1000": "combined",
    "combined1000": "combined",
    "watchlist": "watchlist",
    "watch_list": "watchlist",
    "watch": "watchlist",
    "custom": "watchlist",
    "customstocks": "watchlist",
    "custom_stocks": "watchlist",
}
UNIVERSE_FILE_CANDIDATES = {
    "nasdaq500": ("nasdaq500.json", "nasdaq300.json", "nasdaq_top_500.json", "nasdaq_top500.json"),
    "wsb100": ("wsb100.json", "wsb_top_500.json", "wsb_top500.json"),
    "combined": ("combined.json", "combined400.json", "combined_1000.json"),
    "watchlist": ("watchlist.json", "custom.json"),
}
FALLBACK_UNIVERSE_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "NFLX",
    "AMD", "INTC", "ADBE", "CSCO", "PEP", "QCOM", "TXN", "AMAT", "INTU", "BKNG",
    "MU", "LRCX", "ADI", "PANW", "KLAC", "CRWD", "MELI", "MAR", "MDLZ", "ADP",
    "SBUX", "AMGN", "ISRG", "PYPL", "GILD", "REGN", "VRTX", "ABNB", "DASH", "SNPS",
    "CDNS", "FTNT", "ORLY", "CTAS", "CSX", "ROP", "CMCSA", "TMUS", "PDD", "ASML",
    "AMZN", "PLTR", "HOOD", "COIN", "RIVN", "SOFI", "ROKU", "SHOP", "UBER", "LYFT",
    "BABA", "JD", "BIDU", "MRNA", "BIIB", "ILMN", "DXCM", "IDXX", "TEAM", "DDOG",
    "NET", "ZS", "OKTA", "DOCU", "MDB", "SNOW", "CRSP", "EXAS", "TTD", "WDAY",
    "ANSS", "PAYX", "MNST", "KDP", "CHTR", "EA", "ATVI", "TTWO", "MTCH", "EBAY",
    "FAST", "EXC", "XEL", "AEP", "PCAR", "ODFL", "UAL", "LUV", "DAL", "CCL",
    "NCLH", "UAL", "FSLR", "ENPH", "SEDG", "RUN", "CHWY", "CVNA", "NDAQ", "ICE",
    "MSCI", "SPGI", "BLK", "SCHW", "GS", "MS", "JPM", "BAC", "C", "WFC",
    "V", "MA", "AXP", "PYPL", "SQ", "AFRM", "UPST", "MSTR", "RIOT", "MARA",
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "DVN", "APA", "OXY", "HAL",
    "JNJ", "PFE", "ABBV", "LLY", "UNH", "CVS", "HUM", "CNC", "ELV", "CI",
    "WMT", "TGT", "HD", "LOW", "NKE", "DIS", "SBUX", "KO", "PEP", "MCD",
    "BA", "GE", "CAT", "DE", "HON", "RTX", "LMT", "NOC", "GD", "EMR",
]
FALLBACK_SYMBOL_POOL_TARGET = 420
FALLBACK_SYNTHETIC_PREFIX = "ZZ"
UNIVERSE_METADATA_CACHE_FILE = "universe_metadata_cache.json"
UNIVERSE_METADATA_CACHE_TTL_SECONDS = max(300, int(os.getenv("DPOLARIS_UNIVERSE_METADATA_TTL_SECONDS", str(12 * 3600))))
UNIVERSE_METADATA_REFRESH_LIMIT = max(0, int(os.getenv("DPOLARIS_UNIVERSE_METADATA_REFRESH_LIMIT", "40")))

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


def _resolve_backend_bind_host_port() -> tuple[str, int]:
    host = os.getenv("DPOLARIS_BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_port = os.getenv("DPOLARIS_BACKEND_PORT", "8420").strip()
    try:
        port = int(raw_port)
    except Exception:
        port = 8420
    return host, port


def _ensure_backend_control_manager() -> BackendControlManager:
    global backend_control_manager
    if backend_control_manager is not None:
        return backend_control_manager

    cfg = config or get_config()
    host, port = _resolve_backend_bind_host_port()
    backend_control_manager = BackendControlManager(
        BackendControlConfig(
            host=host,
            port=port,
            data_dir=cfg.data_dir,
            python_executable=Path(sys.executable).resolve(),
            workdir=Path(__file__).resolve().parent.parent,
        )
    )
    return backend_control_manager


def _control_error_payload(exc: Exception) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, PortInUseByUnknownProcessError):
        payload = exc.as_dict()
        payload["force_supported"] = True
        return 409, payload
    if isinstance(exc, UnsafeForceKillError):
        payload = exc.as_dict()
        payload["force_supported"] = True
        return 409, payload
    if isinstance(exc, BackendControlError):
        return 500, exc.as_dict()
    return 500, {
        "error": "backend_control_exception",
        "message": str(exc),
        "details": {},
    }


async def _backend_heartbeat_worker() -> None:
    while True:
        manager = _ensure_backend_control_manager()
        try:
            manager.touch_current_process_heartbeat(started_at=server_started_at, healthy=True)
        except Exception as exc:
            logger.warning("Failed to update backend heartbeat: %s", exc)
        await asyncio.sleep(3)


async def _signal_current_process_later(sig: int = signal.SIGTERM, delay_seconds: float = 0.25) -> None:
    await asyncio.sleep(max(0.05, float(delay_seconds)))
    try:
        os.kill(os.getpid(), int(sig))
    except Exception as exc:
        logger.warning("Failed to signal current process: %s", exc)


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
        "error": "missing_dependency",
        "dependency": "apscheduler",
        "install": "pip install apscheduler",
    }


def _torch_dependency_detail() -> dict[str, Any]:
    return {
        "error": "missing_dependency",
        "dependency": "torch",
        "install": "pip install torch torchvision",
        "message": "PyTorch is required for deep-learning endpoints.",
    }


def _sklearn_dependency_detail() -> dict[str, Any]:
    return {
        "error": "missing_dependency",
        "dependency": "scikit-learn",
        "install": "pip install scikit-learn",
        "message": "scikit-learn is required for ML endpoints.",
    }


def _require_torch() -> None:
    """Raise HTTPException 503 if torch is not available."""
    try:
        import torch  # type: ignore # noqa: F401
    except ImportError:
        raise HTTPException(status_code=503, detail=_torch_dependency_detail())
    except Exception as exc:
        detail = _torch_dependency_detail()
        detail["message"] = f"PyTorch import failed: {exc}"
        raise HTTPException(status_code=503, detail=detail)


def _require_sklearn() -> None:
    """Raise HTTPException 503 if sklearn is not available."""
    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError:
        raise HTTPException(status_code=503, detail=_sklearn_dependency_detail())
    except Exception as exc:
        detail = _sklearn_dependency_detail()
        detail["message"] = f"scikit-learn import failed: {exc}"
        raise HTTPException(status_code=503, detail=detail)


def _require_deep_learning() -> None:
    """Raise HTTPException 503 if deep-learning deps are missing."""
    _require_torch()
    _require_sklearn()


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


def _universe_root() -> Path:
    raw = os.getenv("DPOLARIS_UNIVERSE_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _repo_root() / path
        return path

    if config is not None and getattr(config, "data_dir", None) is not None:
        try:
            return Path(config.data_dir).expanduser() / "universe"
        except Exception:
            pass

    path = Path("~/dpolaris_data/universe").expanduser()
    return path


def _normalize_universe_key(name: str) -> str:
    return "".join(ch.lower() for ch in str(name or "") if ch.isalnum() or ch == "_")


def _canonical_universe_name(name: str) -> Optional[str]:
    raw = str(name or "").strip()
    if not raw:
        return None
    norm = _normalize_universe_key(raw)
    return UNIVERSE_ALIAS_MAP.get(norm) or UNIVERSE_ALIAS_MAP.get(raw.lower())


def _runs_root() -> Path:
    raw = os.getenv("DPOLARIS_RUNS_DIR", "runs")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _analysis_data_dir() -> Path:
    if config is not None and getattr(config, "data_dir", None) is not None:
        try:
            return Path(config.data_dir).expanduser()
        except Exception:
            pass
    return Path("~/dpolaris_data").expanduser()


def _training_window_from_history(df: Any) -> dict[str, Any]:
    if df is None:
        return {}
    try:
        rows = int(len(df))
    except Exception:
        return {}
    if rows <= 0:
        return {}

    time_col = None
    for candidate in ("date", "timestamp"):
        if candidate in getattr(df, "columns", []):
            time_col = candidate
            break

    if not time_col:
        return {"rows": rows}

    try:
        start = str(df[time_col].iloc[0])
        end = str(df[time_col].iloc[-1])
        return {"rows": rows, "start": start, "end": end}
    except Exception:
        return {"rows": rows}


def _analysis_version_info(extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if default_version_info is not None:
        try:
            base = default_version_info()
            if isinstance(base, dict):
                info.update(base)
        except Exception:
            pass
    info.setdefault("python_version", platform.python_version())
    info.setdefault("platform", platform.platform())
    info.setdefault("python_executable", sys.executable)
    info.setdefault("llm_provider", os.getenv("LLM_PROVIDER", "none"))
    if extra:
        info.update(extra)
    return info


def _detected_runtime_device() -> str:
    try:
        from core.device import detect_device

        detected = detect_device()
        return str(detected.get("device") or "cpu")
    except Exception:
        return "cpu"


async def _fetch_analysis_history(symbol: str, history_days: int) -> Any:
    market = market_service or MarketDataService()
    return await market.get_historical(symbol, days=max(120, int(history_days)))


def _extract_model_signals(prediction: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(prediction, dict):
        return None
    if prediction.get("probability_up") is None and prediction.get("confidence") is None:
        return None
    return {
        "prediction_label": prediction.get("prediction_label"),
        "confidence": _safe_float(prediction.get("confidence"), 0.5),
        "probability_up": _safe_float(prediction.get("probability_up"), 0.5),
        "probability_down": _safe_float(prediction.get("probability_down"), 0.5),
        "raw_probability_up": prediction.get("raw_probability_up"),
        "raw_probability_down": prediction.get("raw_probability_down"),
    }


def _build_analysis_payload(
    *,
    symbol: str,
    report: dict[str, Any],
    model_type: str,
    device: str,
    history_df: Any,
    source: str,
    run_id: Optional[str],
) -> dict[str, Any]:
    created_at = report.get("created_at") or utc_now_iso()
    return {
        "ticker": symbol,
        "created_at": created_at,
        "analysis_date": created_at,
        "model_type": model_type or "none",
        "training_window": _training_window_from_history(history_df),
        "device": device or "cpu",
        "version_info": _analysis_version_info({"source": source, "run_id": run_id}),
        "summary": str(report.get("summary") or ""),
        "report_text": str(report.get("report_text") or ""),
        "signals": report.get("model_signals") or {},
        "indicators": report.get("technical_indicators") or {},
        "news_refs": ((report.get("news") or {}).get("items") or []) if isinstance(report.get("news"), dict) else [],
        "report": report,
        "source": source,
        "run_id": run_id,
    }


async def _generate_and_store_analysis(
    *,
    symbol: str,
    history_df: Any,
    model_signals: Optional[dict[str, Any]] = None,
    model_metadata: Optional[dict[str, Any]] = None,
    model_type: str = "none",
    device: Optional[str] = None,
    source: str = "analysis",
    run_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if generate_analysis_report is None or write_analysis_artifact_store is None:
        return None
    report = generate_analysis_report(
        symbol=symbol,
        history=history_df,
        model_signals=model_signals,
        model_metadata=model_metadata,
    )
    payload = _build_analysis_payload(
        symbol=symbol,
        report=report,
        model_type=model_type or "none",
        device=device or _detected_runtime_device(),
        history_df=history_df,
        source=source,
        run_id=run_id,
    )
    return write_analysis_artifact_store(payload, data_dir=_analysis_data_dir())


async def _load_model_context_for_analysis(symbol: str, history_df: Any) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], str, str]:
    model_signals: Optional[dict[str, Any]] = None
    model_metadata: Optional[dict[str, Any]] = None
    model_type = "none"
    device = _detected_runtime_device()

    try:
        from ml.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer()
        model, scaler, metadata = trainer.load_model(symbol)
        prediction = trainer.predict(
            model,
            scaler,
            history_df,
            probability_calibration=(metadata.get("metrics") or {}).get("probability_calibration")
            if isinstance(metadata, dict)
            else None,
        )
        model_signals = _extract_model_signals(prediction)
        model_metadata = metadata if isinstance(metadata, dict) else {}
        model_type = str((model_metadata or {}).get("model_type") or "deep_learning")
        device = str(getattr(trainer, "device", device))
    except Exception:
        model_signals = None
        model_metadata = None

    return model_signals, model_metadata, model_type, device


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


def _dedupe_symbols(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        symbol = _sanitize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _symbols_from_universe_payload(payload: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for key in (
        "tickers",
        "merged",
        "nasdaq500",
        "nasdaq300",
        "wsb100",
        "watchlist",
        "custom",
        "nasdaq_top_500",
        "wsb_top_500",
        "items",
        "rows",
        "data",
    ):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                symbols.append(item)
                continue
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("ticker")
                if symbol:
                    symbols.append(str(symbol))
    return _dedupe_symbols(symbols)


def _existing_universe_symbols() -> list[str]:
    universe_dir = _universe_root()
    paths = [
        universe_dir / "nasdaq500.json",
        universe_dir / "nasdaq300.json",
        universe_dir / "wsb100.json",
        universe_dir / "combined.json",
        universe_dir / "combined400.json",
        universe_dir / "watchlist.json",
        universe_dir / "custom.json",
        universe_dir / "nasdaq_top_500.json",
        universe_dir / "wsb_top_500.json",
        universe_dir / "combined_1000.json",
    ]
    collected: list[str] = []
    for path in paths:
        payload = _json_load(path)
        if not isinstance(payload, dict):
            continue
        collected.extend(_symbols_from_universe_payload(payload))
    return _dedupe_symbols(collected)


def _with_synthetic_tail(symbols: list[str], minimum: int) -> list[str]:
    target = max(1, int(minimum))
    out = list(symbols)
    next_id = 1
    while len(out) < target:
        out.append(f"{FALLBACK_SYNTHETIC_PREFIX}{next_id:04d}")
        next_id += 1
    return out


def _fallback_symbols(minimum: int = FALLBACK_SYMBOL_POOL_TARGET) -> list[str]:
    raw = os.getenv("DPOLARIS_FALLBACK_SYMBOLS", "")
    env_symbols: list[str] = []
    if raw.strip():
        env_symbols = _dedupe_symbols([x for x in raw.split(",") if x.strip()])

    base = env_symbols if env_symbols else _dedupe_symbols(_existing_universe_symbols() + FALLBACK_UNIVERSE_SYMBOLS)
    return _with_synthetic_tail(base, minimum)


def _build_fallback_nasdaq_payload(symbols: list[str], *, top_n: int = 500) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    limited = symbols[: max(1, int(top_n))]
    rows = [
        {
            "symbol": symbol,
            "name": symbol,
            "company_name": symbol,
            "market_cap": None,
            "avg_volume_7d": None,
            "avg_dollar_volume": None,
            "change_pct_1d": None,
            "change_percent_1d": None,
            "sector": None,
            "industry": None,
        }
        for symbol in limited
    ]
    payload = {
        "name": "nasdaq500",
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "updated_at": generated_at,
        "criteria": {
            "exchange": "NASDAQ",
            "top_n_requested": int(top_n),
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


def _build_fallback_wsb_payload(symbols: list[str], *, top_n: int = 100) -> dict[str, Any]:
    generated = datetime.now(timezone.utc)
    limited = symbols[: max(1, int(top_n))]
    rows = []
    total = len(limited)
    for idx, symbol in enumerate(limited, start=1):
        mentions = max(1, total - idx + 1)
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "company_name": symbol,
                "sector": None,
                "market_cap": None,
                "avg_volume_7d": None,
                "change_pct_1d": None,
                "mention_count": mentions,
                "mention_velocity": round(float(mentions) / 7.0, 6),
                "sentiment_score": 0.0,
                "example_post_ids": [],
                "example_titles": [],
            }
        )
    payload = {
        "name": "wsb100",
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "updated_at": generated.isoformat(),
        "window_start": (generated - timedelta(days=7)).isoformat(),
        "window_end": generated.isoformat(),
        "criteria": {
            "top_n_requested": int(top_n),
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


def _build_fallback_combined_payload(
    ns_payload: dict[str, Any],
    ws_payload: dict[str, Any],
    watchlist_payload: Optional[dict[str, Any]] = None,
    *,
    top_n: int = 700,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    merged: dict[str, dict[str, Any]] = {}
    for row in (ns_payload.get("tickers") or []):
        symbol = _sanitize_symbol((row or {}).get("symbol"))
        if not symbol:
            continue
        merged[symbol] = {
            "symbol": symbol,
            "name": row.get("name") or row.get("company_name") or symbol,
            "company_name": row.get("company_name") or symbol,
            "market_cap": row.get("market_cap"),
            "avg_volume_7d": row.get("avg_volume_7d"),
            "avg_dollar_volume": row.get("avg_dollar_volume"),
            "change_pct_1d": (
                row.get("change_pct_1d")
                if row.get("change_pct_1d") is not None
                else row.get("change_percent_1d")
            ),
            "change_percent_1d": (
                row.get("change_percent_1d")
                if row.get("change_percent_1d") is not None
                else row.get("change_pct_1d")
            ),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "mention_count": 0,
            "mention_velocity": 0.0,
            "sentiment_score": None,
            "sources": ["nasdaq_top_500"],
        }
    for item in (watchlist_payload or {}).get("tickers") or []:
        if isinstance(item, dict):
            symbol = _sanitize_symbol(item.get("symbol") or item.get("ticker"))
            row = item
        else:
            symbol = _sanitize_symbol(item)
            row = {}
        if not symbol:
            continue
        merged_item = merged.setdefault(
            symbol,
            {
                "symbol": symbol,
                "name": symbol,
                "company_name": symbol,
                "market_cap": None,
                "avg_volume_7d": None,
                "avg_dollar_volume": None,
                "change_pct_1d": None,
                "change_percent_1d": None,
                "sector": None,
                "industry": None,
                "mention_count": 0,
                "mention_velocity": 0.0,
                "sentiment_score": None,
                "sources": [],
            },
        )
        merged_item["name"] = row.get("name") or merged_item.get("name") or symbol
        merged_item["company_name"] = row.get("company_name") or merged_item.get("company_name") or symbol
        merged_item["sector"] = row.get("sector") if row.get("sector") is not None else merged_item.get("sector")
        if "watchlist" not in merged_item["sources"]:
            merged_item["sources"].append("watchlist")

    merged_rows = sorted(
        merged.values(),
        key=lambda x: (
            float(x.get("market_cap") or 0.0),
            float(x.get("avg_dollar_volume") or 0.0),
            str(x.get("symbol") or ""),
        ),
        reverse=True,
    )[: max(1, int(top_n))]

    payload = {
        "name": "combined",
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "updated_at": generated_at,
        "criteria": {
            "top_n_requested": int(top_n),
            "top_n_returned": len(merged_rows),
            "duplicate_resolution": "merge_on_symbol",
        },
        "data_sources": [
            {"name": "nasdaq_top_500", "count": len(ns_payload.get("tickers") or [])},
            {"name": "wsb_top_500", "count": len(ws_payload.get("tickers") or [])},
            {"name": "watchlist", "count": len((watchlist_payload or {}).get("tickers") or [])},
        ],
        "notes": [
            "Combined universe generated from NASDAQ 500 + watchlist.",
        ],
        "nasdaq500": ns_payload.get("tickers") or [],
        "nasdaq300": ns_payload.get("tickers") or [],
        "wsb100": ws_payload.get("tickers") or [],
        "watchlist": (watchlist_payload or {}).get("tickers") or [],
        "custom": (watchlist_payload or {}).get("tickers") or [],
        "nasdaq_top_500": ns_payload.get("tickers") or [],
        "wsb_top_500": ws_payload.get("tickers") or [],
        "tickers": merged_rows,
        "merged": merged_rows,
    }
    return _universe_with_hash(payload)


def _candidate_universe_paths(canonical_name: str) -> list[Path]:
    universe_dir = _universe_root()
    names = UNIVERSE_FILE_CANDIDATES.get(canonical_name, ())
    return [universe_dir / name for name in names]


def _watchlist_store_path() -> Path:
    return _analysis_data_dir() / "watchlist.json"


def _watchlist_universe_path() -> Path:
    return _universe_root() / "watchlist.json"


def _custom_universe_path() -> Path:
    # Legacy alias path for older clients expecting /api/universe/custom.
    return _universe_root() / "custom.json"


def _normalize_watchlist_entries(raw_entries: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_entries if isinstance(raw_entries, list) else []:
        if isinstance(item, dict):
            symbol = _sanitize_symbol(item.get("symbol") or item.get("ticker"))
            added_at = str(item.get("added_at") or item.get("created_at") or "").strip()
        else:
            symbol = _sanitize_symbol(item)
            added_at = ""
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        entries.append(
            {
                "symbol": symbol,
                "added_at": added_at or utc_now_iso(),
            }
        )
    return entries


def _load_watchlist_store_payload() -> dict[str, Any]:
    path = _watchlist_store_path()
    payload: Any = None
    if path.exists() and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            payload = None
    tickers_source: Any = []
    updated_at = utc_now_iso()

    if isinstance(payload, dict):
        tickers_source = payload.get("tickers") or []
        updated_at = str(payload.get("updated_at") or payload.get("generated_at") or updated_at)
    elif isinstance(payload, list):
        tickers_source = payload

    entries = _normalize_watchlist_entries(tickers_source)
    body = {
        "updated_at": updated_at,
        "tickers": entries,
    }

    if not path.exists():
        _json_dump(path, body)
    return body


def _save_watchlist_store_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "updated_at": utc_now_iso(),
        "tickers": _normalize_watchlist_entries(entries),
    }
    _json_dump(_watchlist_store_path(), payload)
    return payload


def _watchlist_symbols(payload: Optional[dict[str, Any]] = None) -> list[str]:
    source = payload or _load_watchlist_store_payload()
    symbols: list[str] = []
    for entry in source.get("tickers") or []:
        if isinstance(entry, dict):
            symbol = _sanitize_symbol(entry.get("symbol") or entry.get("ticker"))
        else:
            symbol = _sanitize_symbol(entry)
        if symbol:
            symbols.append(symbol)
    return _dedupe_symbols(symbols)


def _build_watchlist_universe_payload(store_payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    source = store_payload or _load_watchlist_store_payload()
    rows: list[dict[str, Any]] = []
    for entry in source.get("tickers") or []:
        if isinstance(entry, dict):
            symbol = _sanitize_symbol(entry.get("symbol") or entry.get("ticker"))
            added_at = str(entry.get("added_at") or "").strip() or None
        else:
            symbol = _sanitize_symbol(entry)
            added_at = None
        if not symbol:
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "company_name": symbol,
                "sector": None,
                "industry": None,
                "market_cap": None,
                "avg_volume_7d": None,
                "avg_dollar_volume": None,
                "change_pct_1d": None,
                "change_percent_1d": None,
                "analysis_date": None,
                "last_analysis_date": None,
                "mention_count": 0,
                "mentions": 0,
                "added_at": added_at,
            }
        )

    payload = {
        "name": "watchlist",
        "schema_version": DEFAULT_UNIVERSE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "updated_at": str(source.get("updated_at") or utc_now_iso()),
        "criteria": {
            "source": "user_watchlist",
            "top_n_requested": len(rows),
            "top_n_returned": len(rows),
        },
        "data_sources": [
            {
                "name": "watchlist_store",
                "path": str(_watchlist_store_path()),
                "count": len(rows),
            }
        ],
        "tickers": rows,
    }
    return _universe_with_hash(payload)


def _refresh_watchlist_universe_file(*, refresh_metadata: bool = True) -> dict[str, Any]:
    payload = _build_watchlist_universe_payload()
    rows = list(payload.get("tickers") or [])
    if refresh_metadata and rows:
        rows = _enrich_universe_rows_with_metadata(rows)
    payload["tickers"] = rows
    payload["updated_at"] = utc_now_iso()
    payload = _universe_with_hash(payload)
    _json_dump(_watchlist_universe_path(), payload)
    # Keep legacy custom alias file in sync for older clients.
    _json_dump(_custom_universe_path(), payload)
    return payload


def _load_watchlist_universe_payload() -> dict[str, Any]:
    path = _watchlist_universe_path()
    payload = _json_load(path)
    if isinstance(payload, dict):
        tickers = payload.get("tickers")
        if isinstance(tickers, list):
            return payload
    return _refresh_watchlist_universe_file(refresh_metadata=True)


def _load_custom_universe_payload() -> dict[str, Any]:
    # Legacy alias for older helper names.
    return _load_watchlist_universe_payload()


def _save_custom_universe_symbols(symbols: list[str]) -> dict[str, Any]:
    existing = _load_watchlist_store_payload()
    prior_by_symbol: dict[str, dict[str, Any]] = {}
    for entry in existing.get("tickers") or []:
        if not isinstance(entry, dict):
            continue
        symbol = _sanitize_symbol(entry.get("symbol") or entry.get("ticker"))
        if not symbol:
            continue
        prior_by_symbol[symbol] = entry

    entries: list[dict[str, Any]] = []
    for symbol in _dedupe_symbols(symbols):
        prev = prior_by_symbol.get(symbol) or {}
        entries.append(
            {
                "symbol": symbol,
                "added_at": str(prev.get("added_at") or utc_now_iso()),
            }
        )

    saved = _save_watchlist_store_entries(entries)
    _refresh_watchlist_universe_file(refresh_metadata=True)
    return {
        "updated_at": saved.get("updated_at"),
        "tickers": [entry.get("symbol") for entry in (saved.get("tickers") or []) if isinstance(entry, dict)],
    }


def _refresh_combined_universe_file() -> None:
    universe_dir = _universe_root()
    universe_dir.mkdir(parents=True, exist_ok=True)

    ns_path = _ensure_default_universe_file("nasdaq500") or (universe_dir / "nasdaq500.json")
    ws_path = _ensure_default_universe_file("wsb100") or (universe_dir / "wsb100.json")
    watchlist_payload = _load_watchlist_universe_payload()

    ns_payload = _json_load(ns_path) if ns_path is not None else None
    ws_payload = _json_load(ws_path) if ws_path is not None else None
    if not isinstance(ns_payload, dict):
        ns_payload = _build_fallback_nasdaq_payload(_fallback_symbols(), top_n=500)
    if not isinstance(ws_payload, dict):
        ws_payload = _build_fallback_wsb_payload(_fallback_symbols(), top_n=100)

    combined_payload = _build_fallback_combined_payload(
        ns_payload,
        ws_payload,
        watchlist_payload,
        top_n=max(
            1,
            len(ns_payload.get("tickers") or [])
            + len(watchlist_payload.get("tickers") or []),
        ),
    )
    combined_payload["name"] = "combined"
    combined_payload["updated_at"] = utc_now_iso()
    combined_payload = _universe_with_hash(combined_payload)
    _json_dump(universe_dir / "combined.json", combined_payload)
    _json_dump(universe_dir / "combined400.json", combined_payload)


def _ensure_default_universe_file(universe_name: str) -> Optional[Path]:
    canonical = _canonical_universe_name(universe_name)
    if canonical is None:
        return None

    universe_dir = _universe_root()
    universe_dir.mkdir(parents=True, exist_ok=True)

    for candidate in _candidate_universe_paths(canonical):
        if candidate.exists() and candidate.is_file():
            return candidate

    symbols = _fallback_symbols()
    nasdaq_path = universe_dir / "nasdaq500.json"
    wsb_path = universe_dir / "wsb100.json"
    combined_path = universe_dir / "combined.json"
    legacy_combined_path = universe_dir / "combined400.json"
    watchlist_path = _watchlist_universe_path()
    custom_path = _custom_universe_path()

    if canonical == "nasdaq500":
        payload = _build_fallback_nasdaq_payload(symbols, top_n=500)
        _json_dump(nasdaq_path, payload)
        # keep legacy alias file in sync
        _json_dump(universe_dir / "nasdaq300.json", payload)
        return nasdaq_path

    if canonical == "wsb100":
        _json_dump(wsb_path, _build_fallback_wsb_payload(symbols, top_n=100))
        return wsb_path

    if canonical == "watchlist":
        if watchlist_path.exists() and watchlist_path.is_file():
            return watchlist_path
        payload = _refresh_watchlist_universe_file(refresh_metadata=True)
        return _watchlist_universe_path() if payload else watchlist_path

    ns_existing = next((p for p in _candidate_universe_paths("nasdaq500") if p.exists() and p.is_file()), nasdaq_path)
    ws_existing = next((p for p in _candidate_universe_paths("wsb100") if p.exists() and p.is_file()), wsb_path)
    watchlist_existing = next((p for p in _candidate_universe_paths("watchlist") if p.exists() and p.is_file()), watchlist_path)

    if not ns_existing.exists():
        payload = _build_fallback_nasdaq_payload(symbols, top_n=500)
        _json_dump(nasdaq_path, payload)
        _json_dump(universe_dir / "nasdaq300.json", payload)
        ns_existing = nasdaq_path
    if not ws_existing.exists():
        _json_dump(wsb_path, _build_fallback_wsb_payload(symbols, top_n=100))
        ws_existing = wsb_path
    if not watchlist_existing.exists():
        _refresh_watchlist_universe_file(refresh_metadata=True)
        watchlist_existing = _watchlist_universe_path()
    if not custom_path.exists():
        payload = _json_load(watchlist_existing) or {"updated_at": utc_now_iso(), "tickers": []}
        _json_dump(custom_path, payload)

    ns_payload = _json_load(ns_existing) or _build_fallback_nasdaq_payload(symbols, top_n=500)
    ws_payload = _json_load(ws_existing) or _build_fallback_wsb_payload(symbols, top_n=100)
    watchlist_payload = _json_load(watchlist_existing) or _build_watchlist_universe_payload()
    combined_payload = _build_fallback_combined_payload(
        ns_payload,
        ws_payload,
        watchlist_payload,
        top_n=max(
            1,
            len(ns_payload.get("tickers") or [])
            + len(watchlist_payload.get("tickers") or []),
        ),
    )
    _json_dump(combined_path, combined_payload)
    _json_dump(legacy_combined_path, combined_payload)
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

    canonical = _canonical_universe_name(name)
    if canonical:
        for candidate in _candidate_universe_paths(canonical):
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        generated = _ensure_default_universe_file(canonical)
        if generated is not None and generated.exists() and generated.is_file():
            return generated.resolve()

    universe_dir = _universe_root()
    direct_name = f"{name}.json" if not name.endswith(".json") else name
    for possible in (
        universe_dir / direct_name,
        universe_dir / name,
    ):
        if possible.exists() and possible.is_file():
            return possible.resolve()

    raise FileNotFoundError(f"Universe file not found for '{universe_name}'")


def _universe_metadata_cache_path() -> Path:
    return _analysis_data_dir() / "cache" / UNIVERSE_METADATA_CACHE_FILE


def _load_universe_metadata_cache() -> dict[str, dict[str, Any]]:
    path = _universe_metadata_cache_path()
    payload = _json_load(path)
    if not isinstance(payload, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        symbol = _sanitize_symbol(key)
        if symbol:
            out[symbol] = value
    return out


def _save_universe_metadata_cache(payload: dict[str, dict[str, Any]]) -> None:
    try:
        _json_dump(_universe_metadata_cache_path(), payload)
    except Exception:
        logger.debug("Failed to persist universe metadata cache", exc_info=True)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _name_is_placeholder(name: Any, symbol: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return True
    normalized = text.upper()
    return normalized in {symbol.upper(), "—", "-"}


def _is_synthetic_fallback_symbol(symbol: str) -> bool:
    normalized = (symbol or "").strip().upper()
    if not normalized.startswith(FALLBACK_SYNTHETIC_PREFIX):
        return False
    suffix = normalized[len(FALLBACK_SYNTHETIC_PREFIX):]
    return bool(suffix) and suffix.isdigit()


def _row_needs_metadata(row: dict[str, Any]) -> bool:
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return False
    if _is_synthetic_fallback_symbol(symbol):
        return False
    if _name_is_placeholder(row.get("name"), symbol):
        return True
    if _name_is_placeholder(row.get("company_name"), symbol):
        return True
    if not str(row.get("sector") or "").strip():
        return True
    if _coerce_optional_float(row.get("market_cap")) is None:
        return True
    if _coerce_optional_float(row.get("avg_volume_7d")) is None:
        return True
    return _coerce_optional_float(row.get("change_pct_1d")) is None


def _cached_metadata_entry(cache: dict[str, dict[str, Any]], symbol: str) -> Optional[dict[str, Any]]:
    entry = cache.get(symbol.upper())
    if not isinstance(entry, dict):
        return None

    cached_at = _parse_timestamp(entry.get("cached_at"))
    if cached_at is None:
        return None
    age_seconds = (datetime.utcnow() - cached_at).total_seconds()
    if age_seconds > UNIVERSE_METADATA_CACHE_TTL_SECONDS:
        return None

    data = entry.get("data")
    if isinstance(data, dict):
        return data
    return None


def _normalize_metadata_payload(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "name": payload.get("name") or payload.get("company_name"),
        "sector": payload.get("sector"),
        "market_cap": _coerce_optional_float(payload.get("market_cap")),
        "avg_volume_7d": _coerce_optional_float(payload.get("avg_volume_7d")),
        "change_pct_1d": _coerce_optional_float(
            payload.get("change_pct_1d")
            if payload.get("change_pct_1d") is not None
            else payload.get("change_percent_1d")
        ),
    }


def _fetch_universe_symbol_metadata(symbol: str) -> dict[str, Any]:
    try:
        import yfinance  # noqa: F401
    except Exception:
        return {}

    try:
        payload = _fetch_single_stock_metadata(symbol)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return _normalize_metadata_payload(symbol, payload)


def _apply_metadata_to_row(row: dict[str, Any], metadata: dict[str, Any]) -> None:
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return

    candidate_name = metadata.get("name")
    if candidate_name and _name_is_placeholder(row.get("name"), symbol):
        row["name"] = str(candidate_name)
    if candidate_name and _name_is_placeholder(row.get("company_name"), symbol):
        row["company_name"] = str(candidate_name)

    if metadata.get("sector") and not str(row.get("sector") or "").strip():
        row["sector"] = metadata.get("sector")

    if _coerce_optional_float(row.get("market_cap")) is None and metadata.get("market_cap") is not None:
        row["market_cap"] = _coerce_optional_float(metadata.get("market_cap"))

    if _coerce_optional_float(row.get("avg_volume_7d")) is None and metadata.get("avg_volume_7d") is not None:
        avg_volume = _coerce_optional_float(metadata.get("avg_volume_7d"))
        row["avg_volume_7d"] = avg_volume
        if _coerce_optional_float(row.get("avg_dollar_volume")) is None:
            row["avg_dollar_volume"] = avg_volume

    if _coerce_optional_float(row.get("change_pct_1d")) is None and metadata.get("change_pct_1d") is not None:
        change = _coerce_optional_float(metadata.get("change_pct_1d"))
        row["change_pct_1d"] = change
        row["change_percent_1d"] = change


def _enrich_universe_rows_with_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    cache = _load_universe_metadata_cache()
    cache_dirty = False

    row_by_symbol: dict[str, dict[str, Any]] = {}
    refresh_candidates: list[str] = []

    for row in rows:
        symbol = _sanitize_symbol(row.get("symbol"))
        if not symbol:
            continue
        row_by_symbol[symbol] = row
        cached = _cached_metadata_entry(cache, symbol)
        skip_refresh = False
        if cached is not None:
            if cached.get("_missing"):
                skip_refresh = True
            else:
                _apply_metadata_to_row(row, cached)
        if not skip_refresh and _row_needs_metadata(row):
            refresh_candidates.append(symbol)

    refresh_limit = max(0, UNIVERSE_METADATA_REFRESH_LIMIT)
    for symbol in refresh_candidates[:refresh_limit]:
        metadata = _fetch_universe_symbol_metadata(symbol)
        if not metadata:
            cache[symbol] = {
                "cached_at": utc_now_iso(),
                "data": {"_missing": True},
            }
            cache_dirty = True
            continue
        row = row_by_symbol.get(symbol)
        if row is None:
            continue
        _apply_metadata_to_row(row, metadata)
        cache[symbol] = {
            "cached_at": utc_now_iso(),
            "data": metadata,
        }
        cache_dirty = True

    if cache_dirty:
        _save_universe_metadata_cache(cache)

    analysis_dates = _latest_analysis_dates_by_symbol(limit=5000)
    if analysis_dates:
        for row in rows:
            symbol = _sanitize_symbol(row.get("symbol"))
            if not symbol:
                continue
            date_text = analysis_dates.get(symbol)
            if date_text:
                row["analysis_date"] = date_text
                row["last_analysis_date"] = date_text

    return rows


def _latest_analysis_dates_by_symbol(limit: int = 5000) -> dict[str, str]:
    if list_analysis_artifacts_store is None:
        return {}
    try:
        items = list_analysis_artifacts_store(data_dir=_analysis_data_dir(), limit=max(1, int(limit)))
    except Exception:
        return {}
    if not isinstance(items, list):
        return {}

    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = _sanitize_symbol(item.get("ticker") or item.get("symbol"))
        if not symbol or symbol in out:
            continue
        analysis_date = item.get("analysis_date") or item.get("created_at")
        if analysis_date:
            out[symbol] = str(analysis_date)
    return out


def _load_scan_universe(universe_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    path = _resolve_universe_path(universe_name)
    canonical_name = (
        _canonical_universe_name(universe_name)
        or _canonical_universe_name(Path(path).stem)
        or Path(path).stem
    )
    payload = _json_load(path)
    if payload is None:
        raise ValueError(f"Invalid universe JSON: {path}")

    rows: list[Any] = []
    if isinstance(payload.get("merged"), list):
        rows = payload.get("merged") or []
    elif isinstance(payload.get("tickers"), list):
        rows = payload.get("tickers") or []
    else:
        rows = (
            (payload.get("nasdaq500") or [])
            + (payload.get("nasdaq300") or [])
            + (payload.get("wsb100") or [])
            + (payload.get("watchlist") or [])
            + (payload.get("custom") or [])
            + (payload.get("nasdaq_top_500") or [])
            + (payload.get("wsb_top_500") or [])
        )

    if canonical_name == "combined" and not rows:
        ns_payload, ns_rows, _ = _load_scan_universe("nasdaq500")
        ws_payload, ws_rows, _ = _load_scan_universe("wsb100")
        watchlist_payload, watchlist_rows, _ = _load_scan_universe("watchlist")
        payload = _build_fallback_combined_payload(
            ns_payload or {"tickers": ns_rows},
            ws_payload or {"tickers": ws_rows},
            watchlist_payload or {"tickers": watchlist_rows},
            top_n=max(1, len(ns_rows) + len(watchlist_rows)),
        )
        rows = payload.get("tickers") or payload.get("merged") or []

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
                    "name": symbol,
                    "company_name": symbol,
                    "sector": None,
                    "industry": None,
                    "market_cap": None,
                    "avg_volume_7d": None,
                    "avg_dollar_volume": None,
                    "change_pct_1d": None,
                    "change_percent_1d": None,
                    "analysis_date": None,
                    "last_analysis_date": None,
                    "mention_count": None,
                    "mentions": None,
                    "mention_velocity": None,
                }
            )
            continue

        if not isinstance(raw, dict):
            continue
        symbol = _sanitize_symbol(raw.get("symbol") or raw.get("ticker"))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(
            {
                "symbol": symbol,
                "name": raw.get("name") or raw.get("company_name") or symbol,
                "company_name": raw.get("company_name") or raw.get("name") or symbol,
                "sector": raw.get("sector"),
                "industry": raw.get("industry"),
                "market_cap": raw.get("market_cap"),
                "avg_volume_7d": (
                    raw.get("avg_volume_7d")
                    if raw.get("avg_volume_7d") is not None
                    else (
                        raw.get("average_volume_7d")
                        if raw.get("average_volume_7d") is not None
                        else (
                            raw.get("averageVolume")
                            if raw.get("averageVolume") is not None
                            else raw.get("averageDailyVolume10Day")
                        )
                    )
                ),
                "avg_dollar_volume": raw.get("avg_dollar_volume") if raw.get("avg_dollar_volume") is not None else raw.get("avg_volume_7d"),
                "change_pct_1d": (
                    raw.get("change_pct_1d")
                    if raw.get("change_pct_1d") is not None
                    else raw.get("change_percent_1d")
                ),
                "change_percent_1d": (
                    raw.get("change_percent_1d")
                    if raw.get("change_percent_1d") is not None
                    else raw.get("change_pct_1d")
                ),
                "analysis_date": raw.get("analysis_date") or raw.get("last_analysis_date"),
                "last_analysis_date": raw.get("last_analysis_date") or raw.get("analysis_date"),
                "mention_count": raw.get("mention_count") if raw.get("mention_count") is not None else raw.get("mentions"),
                "mentions": raw.get("mentions") if raw.get("mentions") is not None else raw.get("mention_count"),
                "mention_velocity": raw.get("mention_velocity"),
            }
        )

    if canonical_name == "nasdaq500" and len(normalized) < 500:
        seen_symbols = {item.get("symbol") for item in normalized if isinstance(item, dict)}
        for fallback in _fallback_symbols(minimum=520):
            symbol = _sanitize_symbol(fallback)
            if not symbol or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            normalized.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "company_name": symbol,
                    "sector": None,
                    "industry": None,
                    "market_cap": None,
                    "avg_volume_7d": None,
                    "avg_dollar_volume": None,
                    "change_pct_1d": None,
                    "change_percent_1d": None,
                    "analysis_date": None,
                    "last_analysis_date": None,
                    "mention_count": None,
                    "mentions": None,
                    "mention_velocity": None,
                }
            )
            if len(normalized) >= 500:
                break

    if canonical_name == "watchlist":
        payload["name"] = "watchlist"
        if normalized:
            normalized = _enrich_universe_rows_with_metadata(normalized)
        if "updated_at" not in payload:
            payload["updated_at"] = payload.get("generated_at") or payload.get("updated_at") or utc_now_iso()
        payload.setdefault("generated_at", payload.get("updated_at"))
        return payload, normalized, path

    if not normalized:
        raise ValueError(f"No tickers found in universe: {path}")
    normalized = _enrich_universe_rows_with_metadata(normalized)
    payload["name"] = canonical_name
    if "updated_at" not in payload:
        payload["updated_at"] = payload.get("generated_at")
    return payload, normalized, path


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
        "tickerStatus": job.get("ticker_status") or {},
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
        return "missing_dependency:torch", message
    if "module not found" in lowered and "torch" in lowered:
        return "missing_dependency:torch", message
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
        try:
            analysis_history_days = int(os.getenv("DPOLARIS_ANALYSIS_HISTORY_DAYS", "730"))
            history_df = await _fetch_analysis_history(symbol, analysis_history_days)
            if history_df is not None and len(history_df) >= 60:
                model_signals, model_metadata, resolved_model_type, device_hint = await _load_model_context_for_analysis(
                    symbol,
                    history_df,
                )
                analysis_artifact = await _generate_and_store_analysis(
                    symbol=symbol,
                    history_df=history_df,
                    model_signals=model_signals,
                    model_metadata=model_metadata,
                    model_type=str(result.get("model_type") or resolved_model_type),
                    device=str(result.get("device") or device_hint),
                    source="deep_learning_job",
                    run_id=str(result.get("run_id") or job_id),
                )
                if analysis_artifact:
                    job_result = job.setdefault("result", {})
                    job_result["analysis_id"] = analysis_artifact.get("id")
                    job_result["analysis_summary"] = analysis_artifact.get("summary")
                    job_result["analysis_report"] = analysis_artifact.get("report_text")
                    _append_training_job_log(
                        job,
                        f"Analysis artifact saved ({analysis_artifact.get('id')})",
                    )
        except Exception as analysis_exc:
            logger.warning("Unable to persist analysis artifact for job %s: %s", job_id, analysis_exc)

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
        try:
            analysis_history_days = int(os.getenv("DPOLARIS_ANALYSIS_HISTORY_DAYS", "730"))
            history_df = await _fetch_analysis_history(symbol, analysis_history_days)
            if history_df is not None and len(history_df) >= 60:
                analysis_artifact = await _generate_and_store_analysis(
                    symbol=symbol,
                    history_df=history_df,
                    model_signals=None,
                    model_metadata={"error": error_message},
                    model_type=model_type,
                    device=_detected_runtime_device(),
                    source="deep_learning_job_failed",
                    run_id=job_id,
                )
                if analysis_artifact:
                    job_result = job.setdefault("result", {})
                    job_result["analysis_id"] = analysis_artifact.get("id")
                    job_result["analysis_summary"] = analysis_artifact.get("summary")
        except Exception as analysis_exc:
            logger.warning("Unable to persist failure analysis artifact for job %s: %s", job_id, analysis_exc)
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
    global server_started_at, backend_control_manager, backend_heartbeat_task

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
    backend_control_manager = _ensure_backend_control_manager()
    backend_control_manager.touch_current_process_heartbeat(started_at=server_started_at, healthy=True)
    backend_heartbeat_task = asyncio.create_task(_backend_heartbeat_worker())
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

    if backend_heartbeat_task is not None:
        backend_heartbeat_task.cancel()
        try:
            await backend_heartbeat_task
        except asyncio.CancelledError:
            pass
        backend_heartbeat_task = None

    if backend_control_manager is not None:
        backend_control_manager.clear_current_process_runtime_files()
        backend_control_manager = None

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


class CustomUniverseSymbolRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)


class ScanStartRequest(BaseModel):
    universe: str = Field(default="combined")
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
    try:
        _ensure_backend_control_manager().touch_current_process_heartbeat(started_at=server_started_at, healthy=True)
    except Exception:
        pass
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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
    """Get persisted watchlist with enriched metadata."""
    return await get_scan_universe("watchlist")


@app.post("/api/watchlist/add")
async def add_watchlist_symbol(symbol: str = Query(..., min_length=1, max_length=10)):
    """Add ticker to watchlist (idempotent) and refresh cached metadata."""
    safe_symbol = _sanitize_symbol(symbol)
    if not safe_symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    saved = _watchlist_add_symbol(safe_symbol)
    return {
        "status": "ok",
        "symbol": safe_symbol,
        **saved,
    }


@app.post("/api/watchlist/remove")
async def remove_watchlist_symbol(symbol: str = Query(..., min_length=1, max_length=10)):
    """Remove ticker from watchlist."""
    safe_symbol = _sanitize_symbol(symbol)
    if not safe_symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    saved = _watchlist_remove_symbol(safe_symbol)
    return {
        "status": "ok",
        "symbol": safe_symbol,
        **saved,
    }


@app.post("/api/watchlist")
async def add_to_watchlist(item: WatchlistAdd):
    """Legacy body-based watchlist add route."""
    symbol = _sanitize_symbol(item.symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    saved = _watchlist_add_symbol(symbol)
    return {
        "status": "ok",
        "symbol": symbol,
        **saved,
    }


@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Legacy delete watchlist route."""
    safe_symbol = _sanitize_symbol(symbol)
    if not safe_symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    saved = _watchlist_remove_symbol(safe_symbol)
    return {
        "status": "ok",
        "symbol": safe_symbol,
        **saved,
    }


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

    # Device summary for deep learning
    device_summary: dict[str, Any] = {
        "device": "unknown",
        "torch_available": False,
        "sklearn_available": False,
        "deep_learning_ready": False,
    }
    try:
        from ml.device import get_device_info, get_dependency_status

        device_info = get_device_info()
        dep_status = get_dependency_status()
        device_summary = {
            "device": device_info.get("device", "cpu"),
            "reason": device_info.get("reason"),
            "torch_available": dep_status.get("torch_available", False),
            "sklearn_available": dep_status.get("sklearn_available", False),
            "deep_learning_ready": dep_status.get("deep_learning_ready", False),
            "cuda_available": device_info.get("cuda_available", False),
            "mps_available": device_info.get("mps_available", False),
        }
    except Exception:
        pass

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
        "device_summary": device_summary,
    }


@app.get("/api/control/backend/status")
async def control_backend_status():
    """Get managed backend process status for external control-center clients."""
    manager = _ensure_backend_control_manager()
    status = manager.get_status(include_health_check=True)
    status["status"] = "ok"
    return status


@app.post("/api/control/backend/start")
async def control_backend_start(force: bool = Query(False)):
    """Start managed backend process if not already running."""
    manager = _ensure_backend_control_manager()
    try:
        return manager.start_backend(force=bool(force), wait_for_health_seconds=25)
    except Exception as exc:
        status_code, payload = _control_error_payload(exc)
        raise HTTPException(status_code=status_code, detail=payload)


@app.post("/api/control/backend/stop")
async def control_backend_stop(force: bool = Query(False)):
    """Stop managed backend process."""
    manager = _ensure_backend_control_manager()
    try:
        result = manager.stop_backend(force=bool(force))
    except Exception as exc:
        status_code, payload = _control_error_payload(exc)
        raise HTTPException(status_code=status_code, detail=payload)

    if result.get("status") == "self_stop_required":
        result["status"] = "stopping"
        result["self_stop"] = True
        asyncio.create_task(_signal_current_process_later(signal.SIGTERM, 0.3))
        return result

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error") or result)

    return result


@app.post("/api/control/backend/restart")
async def control_backend_restart(force: bool = Query(False)):
    """Restart managed backend process."""
    manager = _ensure_backend_control_manager()
    try:
        result = manager.restart_backend(force=bool(force), wait_for_health_seconds=25)
    except Exception as exc:
        status_code, payload = _control_error_payload(exc)
        raise HTTPException(status_code=status_code, detail=payload)

    if result.get("status") == "self_stop_required":
        helper_pid = manager.spawn_restart_helper(old_pid=os.getpid(), force=bool(force))
        restarting = {
            "status": "restarting",
            "helper_pid": helper_pid,
            "backend": manager.get_status(include_health_check=False),
        }
        asyncio.create_task(_signal_current_process_later(signal.SIGTERM, 0.35))
        return restarting

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error") or result)

    return result


@app.get("/api/control/orchestrator/status")
async def control_orchestrator_status():
    """Control-center orchestrator status surface."""
    cfg = config or get_config()
    return _orchestrator_runtime_status(
        data_dir=cfg.data_dir,
        default_host="127.0.0.1",
        default_port=8420,
        heartbeat_stale_seconds=180,
    )


@app.post("/api/control/orchestrator/start")
async def control_orchestrator_start(
    force: bool = Query(False),
    dry_run: bool = Query(False),
    interval_health: int = Query(60, ge=5, le=3600),
    interval_scan: str = Query("30m"),
):
    """Start orchestrator process for dPolaris_ops integration."""
    cfg = config or get_config()
    runtime = _orchestrator_runtime_status(
        data_dir=cfg.data_dir,
        default_host="127.0.0.1",
        default_port=8420,
        heartbeat_stale_seconds=180,
    )
    if runtime.get("running"):
        return {"status": "already_running", "orchestrator": runtime}

    run_dir = cfg.data_dir / "run"
    pid_path = run_dir / "orchestrator.pid"
    heartbeat_path = run_dir / "orchestrator.heartbeat.json"
    if runtime.get("pid") and not _is_pid_alive(int(runtime["pid"])):
        try:
            if pid_path.exists():
                pid_path.unlink()
        except Exception:
            pass
        try:
            if heartbeat_path.exists():
                heartbeat_path.unlink()
        except Exception:
            pass

    if runtime.get("port_owner_unknown") and not force:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "backend_port_owned_by_unmanaged_process",
                "message": "Backend port is currently owned by a non-managed process.",
                "details": runtime.get("backend_state", {}),
            },
        )

    cmd = [
        str(Path(sys.executable).resolve()),
        "-m",
        "cli.main",
        "orchestrator",
        "--host",
        str(runtime.get("host") or "127.0.0.1"),
        "--port",
        str(int(runtime.get("port") or 8420)),
        "--interval-health",
        str(int(interval_health)),
        "--interval-scan",
        str(interval_scan),
    ]
    if dry_run:
        cmd.append("--dry-run")

    logs_dir = cfg.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    log_path = logs_dir / f"orchestrator-control-{stamp}.log"
    with open(log_path, "a", encoding="utf-8") as log_handle:
        kwargs: dict[str, Any] = {
            "cwd": str(Path(__file__).resolve().parent.parent),
            "env": os.environ.copy(),
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kwargs)

    for _ in range(20):
        await asyncio.sleep(0.25)
        runtime = _orchestrator_runtime_status(
            data_dir=cfg.data_dir,
            default_host="127.0.0.1",
            default_port=8420,
            heartbeat_stale_seconds=180,
        )
        if runtime.get("running") or runtime.get("pid") == proc.pid:
            break

    return {
        "status": "started",
        "pid": proc.pid,
        "orchestrator": runtime,
    }


@app.post("/api/control/orchestrator/stop")
async def control_orchestrator_stop(force: bool = Query(False)):
    """Stop orchestrator process for dPolaris_ops integration."""
    cfg = config or get_config()
    runtime = _orchestrator_runtime_status(
        data_dir=cfg.data_dir,
        default_host="127.0.0.1",
        default_port=8420,
        heartbeat_stale_seconds=180,
    )
    pid = runtime.get("pid")
    run_dir = cfg.data_dir / "run"
    pid_path = run_dir / "orchestrator.pid"
    heartbeat_path = run_dir / "orchestrator.heartbeat.json"

    if not pid:
        return {"status": "not_running", "orchestrator": runtime}

    if not _is_pid_alive(int(pid)):
        try:
            if pid_path.exists():
                pid_path.unlink()
        except Exception:
            pass
        try:
            if heartbeat_path.exists():
                heartbeat_path.unlink()
        except Exception:
            pass
        return {"status": "not_running", "orchestrator": _orchestrator_runtime_status(data_dir=cfg.data_dir)}

    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "stop_failed", "message": str(exc)})

    for _ in range(30):
        if not _is_pid_alive(int(pid)):
            break
        await asyncio.sleep(0.2)

    if _is_pid_alive(int(pid)) and force:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass
        await asyncio.sleep(0.3)

    stopped = not _is_pid_alive(int(pid))
    if stopped:
        try:
            if pid_path.exists():
                pid_path.unlink()
        except Exception:
            pass
        try:
            if heartbeat_path.exists():
                heartbeat_path.unlink()
        except Exception:
            pass

    return {
        "status": "stopped" if stopped else "error",
        "orchestrator": _orchestrator_runtime_status(
            data_dir=cfg.data_dir,
            default_host="127.0.0.1",
            default_port=8420,
            heartbeat_stale_seconds=180,
        ),
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


def _int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _enforce_wsb_min_mentions(
    payload: dict[str, Any],
    *,
    min_mentions: int,
    top_n: int,
    fallback_symbols: list[str],
) -> dict[str, Any]:
    rows = list(payload.get("tickers") or [])
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = _sanitize_symbol(row.get("symbol"))
        if not symbol or symbol in seen:
            continue
        mentions = _coerce_optional_int(row.get("mention_count")) or 0
        if mentions < min_mentions:
            continue
        seen.add(symbol)
        clean = dict(row)
        clean["symbol"] = symbol
        clean["mention_count"] = mentions
        clean.setdefault("name", symbol)
        clean.setdefault("company_name", symbol)
        clean.setdefault("sector", None)
        clean.setdefault("market_cap", None)
        clean.setdefault("avg_volume_7d", None)
        clean.setdefault("change_pct_1d", None)
        filtered.append(clean)

    for symbol in fallback_symbols:
        safe_symbol = _sanitize_symbol(symbol)
        if not safe_symbol or safe_symbol in seen:
            continue
        seen.add(safe_symbol)
        filtered.append(
            {
                "symbol": safe_symbol,
                "name": safe_symbol,
                "company_name": safe_symbol,
                "sector": None,
                "market_cap": None,
                "avg_volume_7d": None,
                "change_pct_1d": None,
                "mention_count": 0,
                "mention_velocity": 0.0,
                "sentiment_score": 0.0,
                "example_post_ids": [],
                "example_titles": [],
            }
        )
        if len(filtered) >= max(1, int(top_n)):
            break

    payload["tickers"] = filtered[: max(1, int(top_n))]
    criteria = dict(payload.get("criteria") or {})
    criteria["top_n_requested"] = int(top_n)
    criteria["top_n_returned"] = len(payload["tickers"])
    criteria["min_mentions"] = int(min_mentions)
    payload["criteria"] = criteria

    notes = list(payload.get("notes") or [])
    if len(filtered) < top_n:
        notes.append("WSB mention rows below target; filled with fallback symbols at mention_count=0.")
    payload["notes"] = notes
    return payload


def _normalize_nasdaq500_payload(
    payload: dict[str, Any],
    *,
    top_n: int,
    fallback_symbols: list[str],
) -> dict[str, Any]:
    rows = list(payload.get("tickers") or [])
    enriched: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = _sanitize_symbol(row.get("symbol"))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        clean = dict(row)
        clean["symbol"] = symbol
        clean.setdefault("name", clean.get("company_name") or symbol)
        clean.setdefault("company_name", clean.get("name") or symbol)
        clean.setdefault("sector", None)
        clean.setdefault("market_cap", None)
        clean.setdefault("avg_volume_7d", clean.get("avg_dollar_volume"))
        clean.setdefault("change_pct_1d", clean.get("change_percent_1d"))
        enriched.append(clean)

    ranked = [row for row in enriched if _coerce_optional_float(row.get("market_cap")) is not None]
    unknown = [row for row in enriched if _coerce_optional_float(row.get("market_cap")) is None]

    ranked.sort(
        key=lambda row: (
            float(_coerce_optional_float(row.get("market_cap")) or 0.0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    unknown.sort(key=lambda row: str(row.get("symbol") or ""))

    ordered = ranked + unknown

    for symbol in fallback_symbols:
        safe_symbol = _sanitize_symbol(symbol)
        if not safe_symbol or safe_symbol in seen:
            continue
        seen.add(safe_symbol)
        ordered.append(
            {
                "symbol": safe_symbol,
                "name": safe_symbol,
                "company_name": safe_symbol,
                "sector": None,
                "market_cap": None,
                "avg_volume_7d": None,
                "change_pct_1d": None,
            }
        )
        if len(ordered) >= max(1, int(top_n)):
            break

    payload["tickers"] = ordered[: max(1, int(top_n))]
    criteria = dict(payload.get("criteria") or {})
    criteria["top_n_requested"] = int(top_n)
    criteria["top_n_returned"] = len(payload["tickers"])
    criteria["ranking"] = "market_cap_desc_then_symbol"
    payload["criteria"] = criteria
    return payload


def _rebuild_universe_payloads(force: bool = False) -> dict[str, Any]:
    universe_dir = _universe_root()
    universe_dir.mkdir(parents=True, exist_ok=True)
    watchlist_payload = _load_watchlist_universe_payload()

    if build_nasdaq_top_500 is None or build_wsb_top_500 is None or build_combined_universe is None:
        warnings = ["Universe builder dependency unavailable; generated deterministic fallback payloads."]
        symbols = _fallback_symbols()
        nasdaq_payload = _build_fallback_nasdaq_payload(symbols, top_n=500)
        wsb_payload = _build_fallback_wsb_payload(symbols, top_n=100)
        combined_payload = _build_fallback_combined_payload(
            nasdaq_payload,
            wsb_payload,
            watchlist_payload,
            top_n=max(1, len(nasdaq_payload.get("tickers") or []) + len(watchlist_payload.get("tickers") or [])),
        )
        _json_dump(universe_dir / "nasdaq500.json", nasdaq_payload)
        _json_dump(universe_dir / "nasdaq300.json", nasdaq_payload)
        _json_dump(universe_dir / "wsb100.json", wsb_payload)
        _json_dump(universe_dir / "watchlist.json", watchlist_payload)
        _json_dump(universe_dir / "custom.json", watchlist_payload)
        _json_dump(universe_dir / "combined.json", combined_payload)
        _json_dump(universe_dir / "combined400.json", combined_payload)
        return {
            "status": "ok",
            "warnings": warnings,
            "universes": {
                "nasdaq500": {"count": len(nasdaq_payload.get("tickers") or [])},
                "wsb100": {"count": len(wsb_payload.get("tickers") or [])},
                "combined": {"count": len(combined_payload.get("tickers") or [])},
                "watchlist": {"count": len(watchlist_payload.get("tickers") or [])},
            },
            "generated_at": utc_now_iso(),
        }

    warnings: list[str] = []
    cache = _load_universe_metadata_cache()
    cache_dirty = False

    def profile_fetcher(symbol: str) -> dict[str, Any]:
        nonlocal cache_dirty
        safe_symbol = _sanitize_symbol(symbol) or symbol.upper()
        cached = _cached_metadata_entry(cache, safe_symbol)
        if cached:
            metadata = cached
        else:
            metadata = _fetch_universe_symbol_metadata(safe_symbol)
            if metadata:
                cache[safe_symbol] = {
                    "cached_at": utc_now_iso(),
                    "data": metadata,
                }
                cache_dirty = True

        return {
            "symbol": safe_symbol,
            "company_name": metadata.get("name") or safe_symbol,
            "market_cap": metadata.get("market_cap"),
            # Use avg volume as a liquidity proxy when dollar volume is unavailable.
            "avg_dollar_volume": metadata.get("avg_volume_7d"),
            "sector": metadata.get("sector"),
            "industry": None,
        }

    nasdaq_path = universe_dir / "nasdaq500.json"
    wsb_path = universe_dir / "wsb100.json"
    combined_path = universe_dir / "combined.json"
    combined_legacy_path = universe_dir / "combined400.json"

    nasdaq_payload = build_nasdaq_top_500(
        output_path=nasdaq_path,
        top_n=500,
        min_avg_dollar_volume=0.0,
        candidate_limit=max(520, _int_env("DPOLARIS_NASDAQ_CANDIDATE_LIMIT", 1200)),
        profile_fetcher=profile_fetcher,
    )
    nasdaq_payload = _normalize_nasdaq500_payload(
        nasdaq_payload,
        top_n=500,
        fallback_symbols=_fallback_symbols(minimum=520),
    )
    nasdaq_payload["name"] = "nasdaq500"
    nasdaq_payload["updated_at"] = utc_now_iso()
    nasdaq_payload = _universe_with_hash(nasdaq_payload)
    _json_dump(nasdaq_path, nasdaq_payload)
    _json_dump(universe_dir / "nasdaq300.json", nasdaq_payload)

    valid_tickers = {
        _sanitize_symbol((row or {}).get("symbol"))
        for row in (nasdaq_payload.get("tickers") or [])
        if isinstance(row, dict)
    }
    valid_tickers = {x for x in valid_tickers if x}

    provider_name = "cache"
    if build_mentions_provider is not None:
        mentions_provider, mentions_context = build_mentions_provider(_analysis_data_dir())
        provider_name = mentions_context.provider_name
        warnings.extend(list(mentions_context.warnings or []))
    else:
        mentions_provider = None
        warnings.append("Mentions provider dependency unavailable; using fallback WSB symbols.")

    def post_fetcher(*, window_start: datetime, window_end: datetime, max_posts: int = 4000) -> tuple[list[dict[str, Any]], str, list[str]]:
        local_notes = list(warnings)
        if mentions_provider is None:
            return [], provider_name, local_notes
        try:
            posts = mentions_provider.fetch_posts(window_start=window_start, window_end=window_end, max_posts=max_posts)
            return posts, provider_name, local_notes
        except Exception as exc:
            local_notes.append(f"mentions fetch failed: {exc}")
            return [], provider_name, local_notes

    wsb_payload = build_wsb_top_500(
        output_path=wsb_path,
        top_n=100,
        window_days=max(1, _int_env("DPOLARIS_WSB_WINDOW_DAYS", 1)),
        post_fetcher=post_fetcher,
        valid_tickers=valid_tickers,
    )
    wsb_payload["name"] = "wsb100"
    wsb_payload["updated_at"] = utc_now_iso()
    wsb_payload = _enforce_wsb_min_mentions(
        wsb_payload,
        min_mentions=max(0, _int_env("DPOLARIS_WSB_MIN_MENTIONS", 2)),
        top_n=100,
        fallback_symbols=[str(x.get("symbol")) for x in (nasdaq_payload.get("tickers") or []) if isinstance(x, dict)],
    )
    wsb_payload = _universe_with_hash(wsb_payload)
    _json_dump(wsb_path, wsb_payload)

    combined_payload = _build_fallback_combined_payload(
        nasdaq_payload,
        wsb_payload,
        watchlist_payload,
        top_n=max(
            1,
            len(nasdaq_payload.get("tickers") or [])
            + len(watchlist_payload.get("tickers") or []),
        ),
    )
    combined_payload["name"] = "combined"
    combined_payload["updated_at"] = utc_now_iso()
    merged_rows = list(combined_payload.get("merged") or combined_payload.get("tickers") or [])
    combined_payload["tickers"] = merged_rows
    combined_payload = _universe_with_hash(combined_payload)
    _json_dump(combined_path, combined_payload)
    _json_dump(combined_legacy_path, combined_payload)
    _json_dump(universe_dir / "watchlist.json", watchlist_payload)
    _json_dump(universe_dir / "custom.json", watchlist_payload)

    if cache_dirty:
        _save_universe_metadata_cache(cache)

    return {
        "status": "ok",
        "warnings": warnings,
        "provider_status": {
            "mentions": provider_name,
        },
        "universes": {
            "nasdaq500": {
                "path": str(nasdaq_path),
                "count": len(nasdaq_payload.get("tickers") or []),
                "universe_hash": nasdaq_payload.get("universe_hash"),
            },
            "wsb100": {
                "path": str(wsb_path),
                "count": len(wsb_payload.get("tickers") or []),
                "universe_hash": wsb_payload.get("universe_hash"),
            },
            "combined": {
                "path": str(combined_path),
                "count": len(combined_payload.get("tickers") or []),
                "universe_hash": combined_payload.get("universe_hash"),
            },
            "watchlist": {
                "path": str(_watchlist_store_path()),
                "count": len(watchlist_payload.get("tickers") or []),
                "universe_hash": None,
            },
        },
        "generated_at": utc_now_iso(),
        "force": bool(force),
    }


def _list_available_universes() -> list[dict[str, Any]]:
    """List canonical universe definitions with metadata."""
    universes: list[dict[str, Any]] = []

    for name in UNIVERSE_CANONICAL_NAMES:
        _ensure_default_universe_file(name)
        try:
            path = _resolve_universe_path(name)
            payload = _json_load(path) or {}
            rows_obj = payload.get("tickers")
            if not isinstance(rows_obj, list):
                rows_obj = payload.get("merged")
            if not isinstance(rows_obj, list):
                rows_obj = []
            row_count = len(rows_obj)
        except Exception:
            continue
        universes.append(
            {
                "name": name,
                "path": str(path),
                "ticker_count": row_count,
                "schema_version": payload.get("schema_version"),
                "generated_at": payload.get("generated_at"),
                "updated_at": payload.get("updated_at") or payload.get("generated_at"),
                "universe_hash": payload.get("universe_hash"),
            }
        )

    return universes


@app.get("/api/universe/list")
async def list_universe_names():
    """Return canonical universe names for control-center clients."""
    for canonical in UNIVERSE_CANONICAL_NAMES:
        _ensure_default_universe_file(canonical)
    return list(UNIVERSE_CANONICAL_NAMES)


@app.post("/api/universe/rebuild")
async def rebuild_universes_endpoint(force: bool = Query(False)):
    """Rebuild NASDAQ/WSB/combined universes and persist under runtime data dir."""
    try:
        result = await asyncio.to_thread(_rebuild_universe_payloads, force)
        return result
    except Exception as exc:
        logger.exception("Universe rebuild failed")
        return {
            "status": "error",
            "detail": str(exc),
            "warnings": ["Universe rebuild failed; existing universe files were kept."],
            "generated_at": utc_now_iso(),
        }


def _watchlist_response_from_store(payload: dict[str, Any]) -> dict[str, Any]:
    symbols = _watchlist_symbols(payload)
    return {
        "updated_at": payload.get("updated_at"),
        "count": len(symbols),
        "tickers": symbols,
    }


def _watchlist_add_symbol(symbol: str) -> dict[str, Any]:
    payload = _load_watchlist_store_payload()
    existing_by_symbol: dict[str, dict[str, Any]] = {}
    for entry in payload.get("tickers") or []:
        if not isinstance(entry, dict):
            continue
        entry_symbol = _sanitize_symbol(entry.get("symbol") or entry.get("ticker"))
        if not entry_symbol:
            continue
        existing_by_symbol[entry_symbol] = entry

    if symbol not in existing_by_symbol:
        existing_by_symbol[symbol] = {
            "symbol": symbol,
            "added_at": utc_now_iso(),
        }

    saved = _save_watchlist_store_entries(list(existing_by_symbol.values()))
    _refresh_watchlist_universe_file(refresh_metadata=True)
    _refresh_combined_universe_file()
    return _watchlist_response_from_store(saved)


def _watchlist_remove_symbol(symbol: str) -> dict[str, Any]:
    payload = _load_watchlist_store_payload()
    entries = []
    for entry in payload.get("tickers") or []:
        if not isinstance(entry, dict):
            continue
        entry_symbol = _sanitize_symbol(entry.get("symbol") or entry.get("ticker"))
        if not entry_symbol or entry_symbol == symbol:
            continue
        entries.append(
            {
                "symbol": entry_symbol,
                "added_at": str(entry.get("added_at") or utc_now_iso()),
            }
        )

    saved = _save_watchlist_store_entries(entries)
    _refresh_watchlist_universe_file(refresh_metadata=True)
    _refresh_combined_universe_file()
    return _watchlist_response_from_store(saved)


@app.get("/api/universe/watchlist")
async def get_watchlist_universe_endpoint():
    return await get_scan_universe("watchlist")


@app.get("/api/universe/custom")
async def get_custom_universe_endpoint():
    # Legacy route alias.
    return await get_scan_universe("watchlist")


@app.post("/api/universe/watchlist/add")
async def add_watchlist_universe_symbol(request: CustomUniverseSymbolRequest):
    symbol = _sanitize_symbol(request.symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    saved = _watchlist_add_symbol(symbol)
    return {
        "status": "ok",
        "symbol": symbol,
        **saved,
    }


@app.post("/api/universe/custom/add")
async def add_custom_universe_symbol(request: CustomUniverseSymbolRequest):
    # Legacy route alias.
    return await add_watchlist_universe_symbol(request)


@app.post("/api/universe/watchlist/remove")
async def remove_watchlist_universe_symbol(request: CustomUniverseSymbolRequest):
    symbol = _sanitize_symbol(request.symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    saved = _watchlist_remove_symbol(symbol)
    return {
        "status": "ok",
        "symbol": symbol,
        **saved,
    }


@app.post("/api/universe/custom/remove")
async def remove_custom_universe_symbol(request: CustomUniverseSymbolRequest):
    # Legacy route alias.
    return await remove_watchlist_universe_symbol(request)


@app.get("/universe/list")
async def list_universes_legacy():
    """Legacy metadata-rich universe listing."""
    universes = _list_available_universes()
    return {
        "universes": universes,
        "count": len(universes),
    }


@app.get("/scan/universe")
@app.get("/api/scan/universe")
async def get_scan_universe_by_name(name: str = Query(..., min_length=1)):
    return await get_scan_universe(name)


@app.get("/scan/universe/{universe_name}")
@app.get("/api/scan/universe/{universe_name}")
@app.get("/universe/{universe_name}")
@app.get("/api/universe/{universe_name}")
async def get_scan_universe(universe_name: str):
    if _normalize_universe_key(universe_name) in {"list", "names"}:
        return await list_universe_names()

    try:
        payload, rows, path = _load_scan_universe(universe_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    canonical_name = (
        _canonical_universe_name(universe_name)
        or _canonical_universe_name(payload.get("name"))
        or _canonical_universe_name(Path(path).stem)
        or universe_name
    )
    response_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        symbol = _sanitize_symbol(row.get("symbol") or row.get("ticker"))
        if not symbol:
            continue
        mentions = row.get("mentions")
        if mentions is None:
            mentions = row.get("mention_count")
        analysis_date = row.get("last_analysis_date") or row.get("analysis_date")
        response_rows.append(
            {
                "symbol": symbol,
                "name": row.get("name") or row.get("company_name"),
                "sector": row.get("sector"),
                "market_cap": _coerce_optional_float(row.get("market_cap")),
                "avg_volume_7d": _coerce_optional_float(
                    row.get("avg_volume_7d")
                    if row.get("avg_volume_7d") is not None
                    else (
                        row.get("average_volume_7d")
                        if row.get("average_volume_7d") is not None
                        else (
                            row.get("averageVolume")
                            if row.get("averageVolume") is not None
                            else row.get("averageDailyVolume10Day")
                        )
                    )
                ),
                "change_pct_1d": _coerce_optional_float(
                    row.get("change_pct_1d")
                    if row.get("change_pct_1d") is not None
                    else row.get("change_percent_1d")
                ),
                "mentions": _coerce_optional_int(mentions),
                "last_analysis_date": str(analysis_date) if analysis_date else None,
            }
        )

    response = {
        "name": canonical_name,
        "requested_name": universe_name,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "updated_at": payload.get("updated_at") or payload.get("generated_at"),
        "universe_hash": payload.get("universe_hash"),
        "schema_version": payload.get("schema_version"),
        "count": len(response_rows),
        "tickers": response_rows,
        "universe": payload,
    }
    return response


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

    # Check for tickers filter - can be in request.tickers or strategy_universe_config.tickers
    ticker_filter: Optional[list[str]] = None
    if request.tickers:
        ticker_filter = list(request.tickers)
    elif request.strategy_universe_config and request.strategy_universe_config.get("tickers"):
        raw_tickers = request.strategy_universe_config.get("tickers")
        if isinstance(raw_tickers, list):
            ticker_filter = [str(t) for t in raw_tickers]

    if ticker_filter:
        wanted = {_sanitize_symbol(x) for x in ticker_filter}
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


@app.get("/api/deep-learning/device")
async def get_dl_device():
    """
    Get detailed device information for deep-learning workloads.

    Returns device selection info including:
    - device: Selected device (cuda/mps/cpu)
    - reason: Why this device was selected
    - torch_version: PyTorch version
    - cuda_available: Whether CUDA is available
    - mps_available: Whether Apple MPS is available
    - gpu_name: GPU name if CUDA available
    """
    try:
        from ml.device import get_device_info, get_dependency_status

        device_info = get_device_info()
        dep_status = get_dependency_status()

        return {
            "device": device_info.get("device", "cpu"),
            "reason": device_info.get("reason", "Unknown"),
            "torch_version": device_info.get("torch_version"),
            "cuda_available": device_info.get("cuda_available", False),
            "mps_available": device_info.get("mps_available", False),
            "gpu_name": device_info.get("gpu_name"),
            "requested": device_info.get("requested", "auto"),
            "warning": device_info.get("warning"),
            "torch_importable": dep_status.get("torch_available", False),
            "sklearn_importable": dep_status.get("sklearn_available", False),
            "deep_learning_ready": dep_status.get("deep_learning_ready", False),
        }
    except ImportError:
        # ml.device module not available - provide basic info
        return {
            "device": "cpu",
            "reason": "ml.device module unavailable",
            "torch_version": None,
            "cuda_available": False,
            "mps_available": False,
            "gpu_name": None,
            "requested": os.getenv("DPOLARIS_DEVICE", "auto"),
            "warning": "Device detection unavailable",
            "torch_importable": False,
            "sklearn_importable": False,
            "deep_learning_ready": False,
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

        response = {
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
        try:
            history_df = await _fetch_analysis_history(
                symbol.upper(),
                int(os.getenv("DPOLARIS_ANALYSIS_HISTORY_DAYS", "730")),
            )
            if history_df is not None and len(history_df) >= 60:
                model_signals, model_metadata, resolved_model_type, device_hint = await _load_model_context_for_analysis(
                    symbol.upper(),
                    history_df,
                )
                analysis_artifact = await _generate_and_store_analysis(
                    symbol=symbol.upper(),
                    history_df=history_df,
                    model_signals=model_signals,
                    model_metadata=model_metadata,
                    model_type=str(result.get("model_type") or resolved_model_type),
                    device=str(result.get("device") or device_hint),
                    source="deep_learning_train_endpoint",
                    run_id=str(result.get("run_id") or ""),
                )
                if analysis_artifact:
                    response["analysis_id"] = analysis_artifact.get("id")
                    response["analysis_summary"] = analysis_artifact.get("summary")
                    response["analysis_report"] = analysis_artifact.get("report_text")
        except Exception as analysis_exc:
            logger.warning("Unable to persist analysis artifact for train endpoint %s: %s", symbol, analysis_exc)
        return response

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

        response = {
            "symbol": symbol.upper(),
            "prediction": result["prediction_label"],
            "confidence": result["confidence"],
            "probability_up": result["probability_up"],
            "probability_down": result["probability_down"],
            "model_type": metadata.get("model_type", "unknown"),
            "model_accuracy": metadata.get("metrics", {}).get("accuracy"),
        }
        try:
            model_signals = _extract_model_signals(result)
            analysis_artifact = await _generate_and_store_analysis(
                symbol=symbol.upper(),
                history_df=df,
                model_signals=model_signals,
                model_metadata=metadata if isinstance(metadata, dict) else {},
                model_type=str((metadata or {}).get("model_type") or "deep_learning"),
                device=str(getattr(trainer, "device", _detected_runtime_device())),
                source="deep_learning_predict",
                run_id=str((metadata or {}).get("run_id") or ""),
            )
            if analysis_artifact:
                response["analysis_id"] = analysis_artifact.get("id")
                response["analysis_summary"] = analysis_artifact.get("summary")
                response["analysis_report"] = analysis_artifact.get("report_text")
        except Exception as analysis_exc:
            logger.warning("Unable to persist analysis artifact for predict %s: %s", symbol, analysis_exc)

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeepLearningRunRequest(BaseModel):
    tickers: list[str]
    fetch_data: bool = True
    train_if_missing: bool = True
    model_type: str = "lstm"
    epochs: int = 50


@app.post("/deep-learning/run")
@app.post("/api/deep-learning/run")
@app.post("/dl/run")
async def run_deep_learning(request: DeepLearningRunRequest):
    """
    Run deep learning on selected tickers.
    Fetches data from yfinance, trains models if missing, and returns predictions.
    """
    from ml.deep_learning import DeepLearningTrainer
    from tools.market_data import MarketDataService

    results = {}
    errors = {}
    trainer = DeepLearningTrainer()
    market = market_service or MarketDataService()

    for ticker in request.tickers:
        symbol = ticker.strip().upper()
        if not symbol:
            continue

        try:
            logger.info("Processing deep learning for %s", symbol)

            # Fetch historical data
            df = await market.get_historical(
                symbol,
                days=int(os.getenv("DPOLARIS_DL_TRAINING_DAYS", "730")),
            )
            if df is None or len(df) < 200:
                errors[symbol] = f"Not enough historical data for {symbol} (got {len(df) if df is not None else 0} rows)"
                continue

            # Try to load existing model, or train new one
            model = None
            scaler = None
            metadata = {}

            try:
                model, scaler, metadata = trainer.load_model(symbol)
                logger.info("Loaded existing model for %s", symbol)
            except (FileNotFoundError, Exception) as load_err:
                if request.train_if_missing:
                    logger.info("Training new model for %s (load error: %s)", symbol, load_err)
                    try:
                        train_result = trainer.train_full_pipeline(
                            df,
                            model_name=symbol,
                            model_type=request.model_type,
                            epochs=request.epochs,
                        )
                        model, scaler, metadata = trainer.load_model(symbol)
                        logger.info("Trained new model for %s", symbol)
                    except Exception as train_err:
                        errors[symbol] = f"Training failed: {train_err}"
                        continue
                else:
                    errors[symbol] = f"No trained model for {symbol}"
                    continue

            # Make prediction
            try:
                prediction = trainer.predict(
                    model,
                    scaler,
                    df,
                    probability_calibration=metadata.get("metrics", {}).get("probability_calibration"),
                )
                result_row = {
                    "prediction": prediction.get("prediction_label", "UNKNOWN"),
                    "confidence": prediction.get("confidence", 0.5),
                    "probability_up": prediction.get("probability_up", 0.5),
                    "probability_down": prediction.get("probability_down", 0.5),
                    "model_type": metadata.get("model_type", request.model_type),
                    "model_accuracy": metadata.get("metrics", {}).get("accuracy"),
                    "status": "success",
                }
                try:
                    model_signals = _extract_model_signals(prediction)
                    analysis_artifact = await _generate_and_store_analysis(
                        symbol=symbol,
                        history_df=df,
                        model_signals=model_signals,
                        model_metadata=metadata if isinstance(metadata, dict) else {},
                        model_type=str(metadata.get("model_type", request.model_type)),
                        device=str(getattr(trainer, "device", _detected_runtime_device())),
                        source="deep_learning_run",
                        run_id=str((metadata or {}).get("run_id") or ""),
                    )
                    if analysis_artifact:
                        result_row["analysis_id"] = analysis_artifact.get("id")
                        result_row["analysis_summary"] = analysis_artifact.get("summary")
                        result_row["analysis_report"] = analysis_artifact.get("report_text")
                except Exception as analysis_exc:
                    logger.warning("Unable to persist analysis artifact for %s: %s", symbol, analysis_exc)
                results[symbol] = result_row
            except Exception as pred_err:
                errors[symbol] = f"Prediction failed: {pred_err}"

        except Exception as e:
            errors[symbol] = str(e)
            logger.exception("Error processing %s", symbol)

    return {
        "status": "completed",
        "total_requested": len(request.tickers),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


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
@app.get("/api/news/{symbol}")
async def get_symbol_news(
    symbol: str,
    limit: int = Query(20, ge=1, le=100),
    force: bool = Query(False),
):
    """Return recent headline items for one ticker with resilient provider fallbacks."""
    normalized = _sanitize_symbol(symbol)
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid symbol")

    if fetch_news_with_cache is None:
        return {
            "symbol": normalized,
            "provider": "disabled",
            "count": 0,
            "items": [],
            "warnings": ["News provider module unavailable; returning empty list."],
            "cached": False,
            "updated_at": utc_now_iso(),
        }

    try:
        items, context, meta = await asyncio.to_thread(
            fetch_news_with_cache,
            symbol=normalized,
            limit=limit,
            data_dir=_analysis_data_dir(),
            force_refresh=bool(force),
        )
        return {
            "symbol": normalized,
            "provider": context.provider_name,
            "count": len(items),
            "items": items,
            "warnings": list(context.warnings or []),
            "cached": bool(meta.get("cached")),
            "cache_path": meta.get("cache_path"),
            "updated_at": utc_now_iso(),
        }
    except Exception as exc:
        logger.exception("News fetch failed for %s", normalized)
        return {
            "symbol": normalized,
            "provider": "error",
            "count": 0,
            "items": [],
            "warnings": ["News fetch failed; returning empty list."],
            "detail": str(exc),
            "cached": False,
            "updated_at": utc_now_iso(),
        }


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
        return {
            "market_sentiment": {"status": "unavailable"},
            "top_movers": [],
            "symbols": {},
            "warnings": ["News sentiment unavailable; optional dependencies may be missing."],
            "detail": str(e),
        }


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
        return {
            "symbol": symbol.upper(),
            "sentiment": None,
            "warnings": ["News sentiment unavailable; optional dependencies may be missing."],
            "detail": str(e),
        }


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
        return {
            "articles": [],
            "count": 0,
            "warnings": ["News articles unavailable; optional dependencies may be missing."],
            "detail": str(e),
        }


# --- Scheduler ---
@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        return scheduler.get_status()

    except Exception as exc:
        _raise_if_scheduler_dependency_missing(exc)
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
        _raise_if_scheduler_dependency_missing(exc)
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
        _raise_if_scheduler_dependency_missing(exc)
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
        _raise_if_scheduler_dependency_missing(exc)
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


# ==================== Stock Metadata + Analysis History for Java App ====================

# In-memory cache for stock metadata (60-second TTL)
_stock_metadata_cache: dict[str, dict[str, Any]] = {}
_stock_metadata_cache_expiry: dict[str, float] = {}
STOCK_METADATA_CACHE_TTL_SECONDS = 60


def _yfinance_dependency_detail() -> dict[str, Any]:
    return {
        "error": "missing_dependency",
        "dependency": "yfinance",
        "install": "pip install yfinance",
        "message": "yfinance is required for stock metadata endpoints.",
    }


def _require_yfinance() -> None:
    """Raise HTTPException 503 if yfinance is not available."""
    try:
        import yfinance  # noqa: F401
    except ImportError:
        raise HTTPException(status_code=503, detail=_yfinance_dependency_detail())


def _fetch_single_stock_metadata(symbol: str) -> dict[str, Any]:
    """Fetch metadata for a single stock from yfinance (blocking call)."""
    import time
    import yfinance as yf

    now = time.time()

    # Check cache
    cache_key = symbol.upper()
    if cache_key in _stock_metadata_cache:
        expiry = _stock_metadata_cache_expiry.get(cache_key, 0)
        if now < expiry:
            return _stock_metadata_cache[cache_key]

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "name": None,
        "sector": None,
        "market_cap": None,
        "avg_volume_7d": None,
        "change_percent_1d": None,
        "as_of": datetime.utcnow().isoformat(),
        "source": "yfinance",
        "error": None,
    }

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        fast_info = {}
        try:
            fi = getattr(ticker, "fast_info", None)
            if fi is not None:
                fast_info = dict(fi)
        except Exception:
            fast_info = {}

        result["name"] = info.get("shortName") or info.get("longName")
        result["sector"] = info.get("sector")
        result["market_cap"] = info.get("marketCap")

        # Prefer direct percentage field when available.
        change_pct = info.get("regularMarketChangePercent")
        if change_pct is None:
            change_pct = fast_info.get("regularMarketChangePercent")
        if change_pct is not None:
            result["change_percent_1d"] = round(float(change_pct), 4)
        else:
            current_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or fast_info.get("lastPrice")
            )
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            if current_price is not None and prev_close is not None and prev_close > 0:
                result["change_percent_1d"] = round(((float(current_price) - float(prev_close)) / float(prev_close)) * 100, 4)

        # 7-day average volume from recent bars, then fall back to provider summary fields.
        history_avg_volume: Optional[float] = None
        try:
            hist = ticker.history(period="1mo", interval="1d")
            if hist is not None and not hist.empty and "Volume" in hist.columns:
                recent_vol = hist["Volume"].tail(7)
                if len(recent_vol) > 0:
                    history_avg_volume = float(recent_vol.mean())
        except Exception:
            history_avg_volume = None

        if history_avg_volume is not None and np.isfinite(history_avg_volume):
            result["avg_volume_7d"] = int(round(history_avg_volume))
        else:
            avg_vol = (
                info.get("averageVolume")
                or info.get("averageDailyVolume10Day")
                or fast_info.get("tenDayAverageVolume")
                or fast_info.get("threeMonthAverageVolume")
            )
            if avg_vol is not None:
                try:
                    result["avg_volume_7d"] = int(float(avg_vol))
                except Exception:
                    pass

    except Exception as exc:
        result["error"] = str(exc)

    # Update cache
    _stock_metadata_cache[cache_key] = result
    _stock_metadata_cache_expiry[cache_key] = now + STOCK_METADATA_CACHE_TTL_SECONDS

    return result


@app.get("/api/stocks/metadata")
async def get_stocks_metadata(symbols: str = Query(..., min_length=1, description="Comma-separated stock symbols")):
    """
    Get stock metadata for multiple symbols.

    Returns sector, market cap, 7-day average volume, and 1-day change percent.
    Uses yfinance as data source with 60-second in-memory cache.
    """
    _require_yfinance()

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail={"error": "invalid_symbols", "message": "No valid symbols provided"})

    # Limit to 50 symbols per request
    if len(symbol_list) > 50:
        symbol_list = symbol_list[:50]

    # Fetch metadata (run in thread pool to avoid blocking)
    loop = asyncio.get_event_loop()
    results: dict[str, dict[str, Any]] = {}

    for sym in symbol_list:
        try:
            meta = await loop.run_in_executor(None, _fetch_single_stock_metadata, sym)
            results[sym] = meta
        except Exception as exc:
            results[sym] = {
                "symbol": sym,
                "name": None,
                "sector": None,
                "market_cap": None,
                "avg_volume_7d": None,
                "change_percent_1d": None,
                "as_of": datetime.utcnow().isoformat(),
                "source": "yfinance",
                "error": str(exc),
            }

    return results


def _find_latest_analysis_for_symbol(symbol: str) -> Optional[dict[str, Any]]:
    symbol_upper = symbol.upper()

    if load_latest_analysis_artifact is not None:
        try:
            artifact = load_latest_analysis_artifact(symbol_upper, data_dir=_analysis_data_dir())
            if isinstance(artifact, dict):
                return {
                    "last_analysis_at": artifact.get("created_at"),
                    "run_id": artifact.get("run_id"),
                    "model_type": artifact.get("model_type"),
                    "status": "completed",
                    "analysis_id": artifact.get("id"),
                    "summary": artifact.get("summary"),
                }
        except Exception:
            pass

    if list_training_runs is None:
        return None

    try:
        runs = list_training_runs(limit=500)
        for run in runs:
            tickers = run.get("tickers") or []
            # Check if this run includes the symbol
            if symbol_upper in [t.upper() for t in tickers]:
                return {
                    "last_analysis_at": run.get("completed_at") or run.get("created_at"),
                    "run_id": run.get("run_id"),
                    "model_type": run.get("model_type"),
                    "status": run.get("status"),
                }
    except Exception:
        pass

    return None


@app.post("/api/analyze/report")
async def generate_analysis_report_endpoint(
    symbol: str = Query(..., min_length=1, description="Ticker symbol"),
):
    """
    Fast-path report generation without retraining.

    Includes model probabilities when a trained model exists, otherwise returns
    indicator/pattern/news-driven analysis only.
    """
    if generate_analysis_report is None or write_analysis_artifact_store is None:
        raise HTTPException(status_code=503, detail="Analysis pipeline unavailable")

    normalized = _sanitize_symbol(symbol)
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid symbol")

    history_days = int(os.getenv("DPOLARIS_ANALYSIS_HISTORY_DAYS", "730"))
    history_df = await _fetch_analysis_history(normalized, history_days)
    if history_df is None or len(history_df) < 60:
        raise HTTPException(status_code=400, detail=f"Not enough historical data for {normalized}")

    model_signals, model_metadata, model_type, device = await _load_model_context_for_analysis(
        normalized,
        history_df,
    )
    artifact = await _generate_and_store_analysis(
        symbol=normalized,
        history_df=history_df,
        model_signals=model_signals,
        model_metadata=model_metadata,
        model_type=model_type,
        device=device,
        source="api_analyze_report",
        run_id=None,
    )
    if not artifact:
        raise HTTPException(status_code=500, detail="Failed to store analysis artifact")
    return artifact


@app.get("/api/analysis/list")
async def list_analysis_endpoint(limit: int = Query(200, ge=1, le=500)):
    """List analysis artifacts sorted by created_at descending."""
    if list_analysis_artifacts_store is None:
        return []
    return list_analysis_artifacts_store(data_dir=_analysis_data_dir(), limit=limit)


@app.get("/api/analysis/by-symbol/{ticker}")
async def list_analysis_by_symbol_endpoint(
    ticker: str,
    limit: int = Query(50, ge=1, le=500),
):
    """List analysis artifacts for one ticker sorted by created_at descending."""
    symbol = _sanitize_symbol(ticker)
    if not symbol:
        raise HTTPException(status_code=400, detail="Invalid ticker")
    if list_analysis_artifacts_store is None:
        return []
    return list_analysis_artifacts_store(data_dir=_analysis_data_dir(), limit=limit, ticker=symbol)


@app.get("/api/analysis/last")
async def get_analysis_last(symbols: str = Query(..., min_length=1, description="Comma-separated stock symbols")):
    """
    Get the last analysis date for multiple symbols.

    Returns the timestamp of the most recent successful training run artifact
    that included each symbol.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail={"error": "invalid_symbols", "message": "No valid symbols provided"})

    if len(symbol_list) > 100:
        symbol_list = symbol_list[:100]

    results: dict[str, Optional[dict[str, Any]]] = {}

    for sym in symbol_list:
        analysis_info = _find_latest_analysis_for_symbol(sym)
        if analysis_info:
            results[sym] = analysis_info
        else:
            results[sym] = {"last_analysis_at": None, "run_id": None}

    return results


def _build_analysis_artifacts(symbol: str, run_id: str) -> list[dict[str, Any]]:
    """Build a list of analysis artifacts for a symbol from a training run."""
    artifacts: list[dict[str, Any]] = []

    if load_training_artifact is None or list_run_artifact_files is None:
        return artifacts

    try:
        artifact_data = load_training_artifact(run_id)
    except FileNotFoundError:
        return artifacts
    except Exception:
        return artifacts

    # Add run_summary as "dl_training" artifact
    run_summary = artifact_data.get("run_summary", {})
    if run_summary:
        artifacts.append({
            "type": "dl_training",
            "title": f"Deep Learning ({run_summary.get('model_type', 'LSTM').upper()}) Summary",
            "data": {
                "run_id": run_summary.get("run_id"),
                "status": run_summary.get("status"),
                "model_type": run_summary.get("model_type"),
                "target": run_summary.get("target"),
                "horizon": run_summary.get("horizon"),
                "created_at": run_summary.get("created_at"),
                "completed_at": run_summary.get("completed_at"),
                "duration_seconds": run_summary.get("duration_seconds"),
                "tickers": run_summary.get("tickers"),
            },
        })

    # Add metrics_summary as "metrics" artifact
    metrics = artifact_data.get("metrics_summary", {})
    if metrics:
        artifacts.append({
            "type": "metrics",
            "title": "Model Metrics",
            "data": metrics,
        })

    # Add data_summary as "data_quality" artifact
    data_summary = artifact_data.get("data_summary", {})
    if data_summary:
        artifacts.append({
            "type": "data_quality",
            "title": "Data Quality",
            "data": data_summary,
        })

    # Add feature_summary as "features" artifact
    feature_summary = artifact_data.get("feature_summary", {})
    if feature_summary:
        artifacts.append({
            "type": "features",
            "title": "Feature Engineering",
            "data": feature_summary,
        })

    # Add reproducibility_summary if present
    repro = artifact_data.get("reproducibility_summary", {})
    if repro:
        artifacts.append({
            "type": "reproducibility",
            "title": "Reproducibility",
            "data": repro,
        })

    # Add model_summary if present
    model_summary = artifact_data.get("model_summary", {})
    if model_summary:
        artifacts.append({
            "type": "model",
            "title": "Model Architecture",
            "data": model_summary,
        })

    return artifacts


@app.get("/api/analysis/detail/{symbol}")
async def get_analysis_detail(symbol: str):
    """
    Get detailed analysis for a symbol.

    Returns structured JSON with all available artifacts from the most recent
    training run that included this symbol. Works without LLM provider.
    """
    symbol_upper = symbol.strip().upper()
    if not symbol_upper:
        raise HTTPException(status_code=400, detail={"error": "invalid_symbol", "message": "Symbol is required"})

    if load_latest_analysis_artifact is not None:
        latest = load_latest_analysis_artifact(symbol_upper, data_dir=_analysis_data_dir())
        if isinstance(latest, dict):
            return {
                "symbol": symbol_upper,
                "last_analysis_at": latest.get("created_at"),
                "run_id": latest.get("run_id"),
                "analysis_id": latest.get("id"),
                "model_type": latest.get("model_type"),
                "device": latest.get("device"),
                "status": "completed",
                "summary": latest.get("summary"),
                "report_text": latest.get("report_text"),
                "artifacts": [
                    {
                        "type": "analysis_report",
                        "title": "Multi-Section Analysis Report",
                        "data": latest.get("report") or {},
                    }
                ],
                "analysis": latest,
            }

    # Find the latest analysis for this symbol
    analysis_info = _find_latest_analysis_for_symbol(symbol_upper)

    if not analysis_info or not analysis_info.get("run_id"):
        return {
            "symbol": symbol_upper,
            "last_analysis_at": None,
            "run_id": None,
            "artifacts": [],
            "message": "No analysis found for this symbol",
        }

    run_id = analysis_info["run_id"]
    artifacts = _build_analysis_artifacts(symbol_upper, run_id)

    return {
        "symbol": symbol_upper,
        "last_analysis_at": analysis_info.get("last_analysis_at"),
        "run_id": run_id,
        "analysis_id": analysis_info.get("analysis_id"),
        "model_type": analysis_info.get("model_type"),
        "status": analysis_info.get("status"),
        "artifacts": artifacts,
    }


@app.get("/api/analysis/{analysis_id}")
async def get_analysis_artifact_endpoint(analysis_id: str):
    """Get full persisted analysis artifact payload by ID."""
    if load_analysis_artifact_store is None:
        raise HTTPException(status_code=503, detail="Analysis artifact store unavailable")
    try:
        artifact = load_analysis_artifact_store(analysis_id, data_dir=_analysis_data_dir())
        if isinstance(artifact, dict):
            analysis_root = _analysis_data_dir() / "analysis"
            artifact.setdefault("path", str((analysis_root / f"{analysis_id}.json").expanduser()))
            artifact.setdefault("analysis_date", artifact.get("created_at"))
            if artifact.get("ticker") and artifact.get("symbol") is None:
                artifact["symbol"] = artifact.get("ticker")
        return artifact
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Analysis artifact not found: {analysis_id}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
