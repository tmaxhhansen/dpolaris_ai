"""
FastAPI Server for dPolaris

Provides REST API and WebSocket endpoints for the Mac app.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Callable, Optional
from contextlib import asynccontextmanager
import logging
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import Config, get_config
from core.database import Database
from core.memory import DPolarisMemory
from core.ai import DPolarisAI
from tools.market_data import MarketDataService, get_market_overview

logger = logging.getLogger("dpolaris.api")

# Global instances
config: Optional[Config] = None
db: Optional[Database] = None
memory: Optional[DPolarisMemory] = None
ai: Optional[DPolarisAI] = None
market_service: Optional[MarketDataService] = None
training_jobs: dict[str, dict] = {}
training_job_order: list[str] = []
training_job_queue: Optional[asyncio.Queue[str]] = None
training_job_worker_task: Optional[asyncio.Task] = None
server_started_at: Optional[datetime] = None

SUPPORTED_DL_MODELS = {"lstm", "transformer"}
MAX_TRAINING_JOBS = 200
MAX_TRAINING_JOB_LOG_LINES = 500


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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

    return {
        "symbol": symbol,
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
            "accuracy": round(_safe_float(model_accuracy_float, 0.0), 6) if model_accuracy_float is not None else None,
        },
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
        }
        _append_training_job_log(job, "Training completed successfully")

        logger.info("Deep-learning job %s completed for %s", job_id, symbol)

    except Exception as e:
        completed_at = utc_now_iso()
        job["status"] = "failed"
        job["updated_at"] = completed_at
        job["completed_at"] = completed_at
        job["error"] = str(e)
        _append_training_job_log(job, f"Training failed: {e}")
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
    global config, db, memory, ai, market_service
    global training_jobs, training_job_order, training_job_queue, training_job_worker_task
    global server_started_at

    # Startup
    logger.info("Starting dPolaris API...")
    config = get_config()
    db = Database()
    memory = DPolarisMemory(db)
    ai = DPolarisAI(config, db, memory)
    market_service = MarketDataService()
    training_jobs = {}
    training_job_order = []
    training_job_queue = asyncio.Queue()
    training_job_worker_task = asyncio.create_task(_deep_learning_job_worker())
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
    server_started_at = None

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


# ==================== REST Endpoints ====================

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
        tags=trade.tags,
    )
    return {"id": trade_id, "status": "created"}


@app.put("/api/journal/{trade_id}/close")
async def close_trade(trade_id: int, close_data: TradeClose):
    """Close a trade"""
    db.close_trade(
        trade_id=trade_id,
        exit_price=close_data.exit_price,
        outcome_notes=close_data.outcome_notes,
        lessons=close_data.lessons,
    )

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
    scheduler_activity_times: list[datetime] = []

    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler_status = scheduler.get_status()
        daemon_running = bool(scheduler_status.get("running", False))

        for key in ("last_training", "last_news_scan", "last_prediction", "last_sync"):
            parsed = _parse_timestamp(scheduler_status.get(key))
            if parsed is not None:
                scheduler_activity_times.append(parsed)
    except Exception:
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
        "last_activity": last_activity,
        "total_memories": total_memories,
        "total_trades": total_trades,
        "models_available": models_available,
        "uptime": _format_uptime(server_started_at),
        "win_rate": win_rate,
    }


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
    response = await ai.chat(message.message)
    return {"response": response, "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze a symbol"""
    response = await ai.chat(f"@analyze {request.symbol}")
    return {
        "symbol": request.symbol,
        "analysis": response,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/scout")
async def scout():
    """Run opportunity scanner"""
    response = await ai.chat("@scout")
    return {"report": response, "timestamp": datetime.now().isoformat()}


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
    try:
        from ml.deep_learning import get_device, DEVICE
        import torch

        py_major = sys.version_info.major
        py_minor = sys.version_info.minor
        forced_py313 = os.getenv("DPOLARIS_ALLOW_PY313_TORCH") == "1"
        py313_guard_active = (py_major, py_minor) >= (3, 13) and not forced_py313

        return {
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "python_version": f"{py_major}.{py_minor}",
            "deep_learning_enabled": not py313_guard_active,
            "deep_learning_reason": (
                "disabled on Python 3.13 for stability; use Python 3.11/3.12 or set DPOLARIS_ALLOW_PY313_TORCH=1"
                if py313_guard_active
                else "enabled"
            ),
        }
    except Exception as e:
        return {"error": str(e), "device": "unknown"}


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


@app.post("/api/deep-learning/train/{symbol}")
async def train_deep_learning(symbol: str, model_type: str = "lstm", epochs: int = 50):
    """Train deep learning model for a symbol"""
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
        }

    except HTTPException:
        raise
    except Exception as e:
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
        result = trainer.predict(model, scaler, df)

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
        result = trainer.predict(model, scaler, df)

        return {
            "source": "deep_learning",
            "model_name": f"{symbol}_dl",
            "model_type": metadata.get("model_type", "lstm"),
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

    except Exception as e:
        return {"running": False, "error": str(e)}


@app.post("/api/scheduler/start")
async def start_scheduler():
    """Start the scheduler"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler.start()
        return {"status": "started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scheduler/stop")
async def stop_scheduler():
    """Stop the scheduler"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler.stop()
        return {"status": "stopped"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scheduler/run/{job_id}")
async def run_scheduler_job(job_id: str):
    """Manually run a scheduler job"""
    try:
        from ai.scheduler import get_scheduler

        scheduler = get_scheduler()
        await scheduler.run_now(job_id)
        return {"status": "completed", "job": job_id}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
