"""
Isolated deep-learning training worker.

Runs in a subprocess so API server stability is preserved even if PyTorch crashes.
Writes a single JSON result object to stdout on success.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from tools.market_data import MarketDataService


SUPPORTED_MODELS = {"lstm", "transformer"}
logger = logging.getLogger("dpolaris.ml.worker")


async def _fetch_history(symbol: str, days: int):
    market = MarketDataService()
    return await market.get_historical(symbol, days=days)


def _build_training_frame(symbol: str, days: int):
    """
    Build canonical, quality-gated training frame via unified data layer.
    """
    try:
        from data.dataset_builder import DatasetBuildRequest, UnifiedDatasetBuilder

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        builder = UnifiedDatasetBuilder()
        dataset, quality_report, report_path = builder.build(
            DatasetBuildRequest(
                symbol=symbol,
                days=days,
                interval="1d",
                horizon_days=5,
                run_id=run_id,
            )
        )

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Unified dataset missing required columns: {missing_cols}")

        train_df = dataset[required].rename(columns={"timestamp": "date"}).copy()
        train_df["date"] = train_df["date"].astype(str)
        return train_df, quality_report, str(report_path)
    except Exception as exc:
        logger.warning("Unified dataset builder unavailable, falling back to direct history fetch: %s", exc)
        return None, None, None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dPolaris deep-learning worker")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. SPY)")
    parser.add_argument("--model-type", default="lstm", help="lstm or transformer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--history-days",
        type=int,
        default=int(os.getenv("DPOLARIS_DL_HISTORY_DAYS", "3650")),
        help="Historical lookback window in calendar days",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _configure_runtime() -> None:
    """
    Set conservative threading defaults before importing torch/scikit libs.

    This reduces OpenMP contention and crash risk on macOS.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def main() -> int:
    _configure_logging()
    _configure_runtime()
    parser = _build_parser()
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    if not symbol:
        logger.error("Symbol is required")
        return 2

    model_type = args.model_type.strip().lower()
    if model_type not in SUPPORTED_MODELS:
        logger.error(
            f"Unsupported model_type '{model_type}'. Choose one of {sorted(SUPPORTED_MODELS)}",
        )
        return 2

    if args.epochs < 1 or args.epochs > 500:
        logger.error("epochs must be between 1 and 500")
        return 2

    if args.history_days < 365 or args.history_days > 20000:
        logger.error("history-days must be between 365 and 20000")
        return 2

    try:
        if sys.version_info >= (3, 13) and os.getenv("DPOLARIS_ALLOW_PY313_TORCH") != "1":
            logger.error(
                "Deep learning is disabled on Python %s.%s due PyTorch instability on this runtime. "
                "Use Python 3.11/3.12 (recommended) or set DPOLARIS_ALLOW_PY313_TORCH=1 to force.",
                sys.version_info.major,
                sys.version_info.minor,
            )
            return 2

        from ml.deep_learning import DeepLearningTrainer
        import torch

        torch.set_num_threads(int(os.getenv("DPOLARIS_TORCH_THREADS", "1")))
        try:
            torch.set_num_interop_threads(int(os.getenv("DPOLARIS_TORCH_INTEROP_THREADS", "1")))
        except RuntimeError:
            # Can fail if already initialized; continue with current setting.
            pass

        logger.info(
            "Worker started for %s (%s, epochs=%d)",
            symbol,
            model_type.upper(),
            args.epochs,
        )
        logger.info("Fetching historical data (days=%d)", args.history_days)

        df, quality_report, quality_report_path = _build_training_frame(symbol, args.history_days)
        if df is None:
            df = asyncio.run(_fetch_history(symbol, args.history_days))
            quality_report = None
            quality_report_path = None

        if df is None or len(df) < 200:
            logger.error("Not enough historical data")
            return 2

        logger.info("Fetched %d rows", len(df))
        if quality_report_path:
            logger.info("Data quality report: %s", quality_report_path)
            min_history = (quality_report or {}).get("checks", {}).get("minimum_history", {})
            logger.info(
                "Quality minimum history check passed=%s required=%s actual=%s",
                min_history.get("passed"),
                min_history.get("required_rows"),
                min_history.get("actual_rows"),
            )
        logger.info("Starting training pipeline")
        trainer = DeepLearningTrainer()
        result = trainer.train_full_pipeline(
            df=df,
            model_name=symbol,
            model_type=model_type,  # type: ignore[arg-type]
            epochs=args.epochs,
        )
        logger.info(
            "Training finished on %s (epochs_trained=%s)",
            result.get("device", "unknown"),
            result.get("epochs_trained", args.epochs),
        )
        logger.info("Model path: %s", result.get("model_path", "unknown"))

        payload = {
            "symbol": symbol,
            "model_name": result.get("model_name", symbol),
            "model_type": result.get("model_type", model_type),
            "metrics": result.get("metrics"),
            "epochs_trained": result.get("epochs_trained", args.epochs),
            "device": result.get("device", "unknown"),
            "model_path": result.get("model_path"),
            "data_quality_report": quality_report_path,
            "data_quality_summary": quality_report.get("checks") if quality_report else None,
        }
        print(json.dumps(payload), flush=True)
        return 0

    except Exception as exc:
        logger.exception("Worker failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
