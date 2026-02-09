"""
Scheduler for dPolaris AI

Runs automated tasks:
- Deep learning training (nightly)
- News scanning (every 15 minutes)
- Prediction generation (hourly)
- Cloud sync (every 5 minutes)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, List
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger("dpolaris.scheduler")


class DPolarisScheduler:
    """
    Automated task scheduler for dPolaris.

    Designed to run 24/7 on Windows machine with GPU.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        enable_training: bool = True,
        enable_news: bool = True,
        enable_sync: bool = True,
    ):
        self.data_dir = data_dir or Path("~/dpolaris_data").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.enable_training = enable_training
        self.enable_news = enable_news
        self.enable_sync = enable_sync

        self.scheduler = AsyncIOScheduler()
        self._running = False

        # Task state
        self.last_training: Optional[datetime] = None
        self.last_news_scan: Optional[datetime] = None
        self.last_prediction: Optional[datetime] = None
        self.last_sync: Optional[datetime] = None

    def start(self):
        """Start the scheduler"""
        if self._running:
            return

        self._setup_jobs()
        self.scheduler.start()
        self._running = True
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        if not self._running:
            return

        self.scheduler.shutdown()
        self._running = False
        logger.info("Scheduler stopped")

    def _setup_jobs(self):
        """Configure all scheduled jobs"""

        # Deep Learning Training - Every night at 2 AM
        if self.enable_training:
            self.scheduler.add_job(
                self._run_training,
                CronTrigger(hour=2, minute=0),
                id="deep_learning_training",
                name="Deep Learning Training",
                replace_existing=True,
            )
            logger.info("Scheduled: Deep learning training at 2:00 AM daily")

        # News Scanning - Every 15 minutes during market hours
        if self.enable_news:
            self.scheduler.add_job(
                self._run_news_scan,
                IntervalTrigger(minutes=15),
                id="news_scan",
                name="News Sentiment Scan",
                replace_existing=True,
            )
            logger.info("Scheduled: News scanning every 15 minutes")

        # Prediction Generation - Every hour during market hours
        self.scheduler.add_job(
            self._run_predictions,
            CronTrigger(minute=0),  # Top of every hour
            id="predictions",
            name="Generate Predictions",
            replace_existing=True,
        )
        logger.info("Scheduled: Predictions at the top of every hour")

        # Cloud Sync - Every 5 minutes
        if self.enable_sync:
            self.scheduler.add_job(
                self._run_sync,
                IntervalTrigger(minutes=5),
                id="cloud_sync",
                name="Cloud Sync",
                replace_existing=True,
            )
            logger.info("Scheduled: Cloud sync every 5 minutes")

        # Health check - Every minute
        self.scheduler.add_job(
            self._health_check,
            IntervalTrigger(minutes=1),
            id="health_check",
            name="Health Check",
            replace_existing=True,
        )

    async def _run_training(self):
        """Run deep learning training"""
        logger.info("Starting deep learning training...")

        try:
            from ml.deep_learning import DeepLearningTrainer
            from tools.market_data import MarketDataService

            trainer = DeepLearningTrainer(models_dir=self.data_dir / "models")
            market = MarketDataService()

            # Get watchlist symbols
            watchlist = self._get_watchlist()

            for symbol in watchlist:
                try:
                    logger.info(f"Training models for {symbol}...")

                    # Fetch historical data
                    df = await market.get_historical_data(symbol, period="2y")
                    if df is None or len(df) < 200:
                        logger.warning(f"Not enough data for {symbol}")
                        continue

                    # Train LSTM
                    lstm_result = trainer.train_full_pipeline(
                        df=df,
                        model_name=symbol,
                        model_type="lstm",
                        epochs=50,
                    )
                    logger.info(f"LSTM for {symbol}: accuracy={lstm_result['metrics']['accuracy']:.4f}")

                    # Train Transformer
                    transformer_result = trainer.train_full_pipeline(
                        df=df,
                        model_name=symbol,
                        model_type="transformer",
                        epochs=50,
                    )
                    logger.info(f"Transformer for {symbol}: accuracy={transformer_result['metrics']['accuracy']:.4f}")

                    # Push metrics to cloud
                    await self._push_model_metrics(symbol, lstm_result, transformer_result)

                except Exception as e:
                    logger.error(f"Failed to train {symbol}: {e}")
                    continue

            self.last_training = datetime.now()
            logger.info("Deep learning training completed")

        except Exception as e:
            logger.error(f"Training job failed: {e}")

    async def _run_news_scan(self):
        """Run news sentiment analysis"""
        logger.info("Running news scan...")

        try:
            from news import NewsEngine

            engine = NewsEngine(use_finbert=True)

            # Get watchlist symbols
            watchlist = self._get_watchlist()

            # Fetch and analyze news
            sentiment_results = await engine.update(symbols=watchlist)

            # Get market sentiment
            market_sentiment = engine.get_market_sentiment()
            logger.info(f"Market sentiment: {market_sentiment.get('label', 'N/A')}")

            # Push to cloud
            if self.enable_sync:
                await self._push_sentiment_to_cloud(sentiment_results)

            self.last_news_scan = datetime.now()
            logger.info(f"News scan completed: {len(sentiment_results)} symbols analyzed")

            await engine.close()

        except Exception as e:
            logger.error(f"News scan failed: {e}")

    async def _run_predictions(self):
        """Generate predictions for watchlist"""
        logger.info("Generating predictions...")

        try:
            from ml.deep_learning import DeepLearningTrainer
            from tools.market_data import MarketDataService

            trainer = DeepLearningTrainer(models_dir=self.data_dir / "models")
            market = MarketDataService()

            watchlist = self._get_watchlist()
            predictions = []

            for symbol in watchlist:
                try:
                    # Load model
                    try:
                        model, scaler, metadata = trainer.load_model(symbol)
                    except FileNotFoundError:
                        logger.warning(f"No model for {symbol}, skipping")
                        continue

                    # Get recent data
                    df = await market.get_historical_data(symbol, period="3mo")
                    if df is None or len(df) < 60:
                        continue

                    # Make prediction
                    result = trainer.predict(model, scaler, df)

                    predictions.append({
                        "symbol": symbol,
                        "prediction": result["prediction_label"],
                        "confidence": result["confidence"],
                        "model_type": metadata.get("model_type", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                    })

                    logger.info(
                        f"{symbol}: {result['prediction_label']} "
                        f"(confidence: {result['confidence']:.2%})"
                    )

                except Exception as e:
                    logger.error(f"Prediction failed for {symbol}: {e}")

            # Push to cloud
            if self.enable_sync and predictions:
                await self._push_predictions_to_cloud(predictions)

            self.last_prediction = datetime.now()
            logger.info(f"Generated {len(predictions)} predictions")

        except Exception as e:
            logger.error(f"Prediction job failed: {e}")

    async def _run_sync(self):
        """Sync data to cloud"""
        if not self.enable_sync:
            return

        try:
            from cloud import SupabaseSync
            import socket

            sync = SupabaseSync.from_env()
            device_id = socket.gethostname()

            await sync.update_sync_status(device_id, "predictions")
            await sync.update_sync_status(device_id, "sentiment")

            self.last_sync = datetime.now()
            logger.debug("Cloud sync completed")

        except Exception as e:
            logger.error(f"Cloud sync failed: {e}")

    async def _health_check(self):
        """Log health status"""
        status = {
            "running": self._running,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "last_news_scan": self.last_news_scan.isoformat() if self.last_news_scan else None,
            "last_prediction": self.last_prediction.isoformat() if self.last_prediction else None,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
        }
        logger.debug(f"Health check: {status}")

    def _get_watchlist(self) -> List[str]:
        """Get symbols to process"""
        # Default watchlist - can be loaded from config or database
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "AMD", "NFLX", "SPY",
        ]

    async def _push_model_metrics(self, symbol: str, lstm_result: dict, transformer_result: dict):
        """Push model metrics to cloud"""
        if not self.enable_sync:
            return

        try:
            from cloud import SupabaseSync

            sync = SupabaseSync.from_env()

            await sync.push_model_metrics(
                model_name=symbol,
                model_type="lstm",
                metrics=lstm_result["metrics"],
                version=datetime.now().strftime("%Y%m%d"),
            )

            await sync.push_model_metrics(
                model_name=symbol,
                model_type="transformer",
                metrics=transformer_result["metrics"],
                version=datetime.now().strftime("%Y%m%d"),
            )

        except Exception as e:
            logger.error(f"Failed to push model metrics: {e}")

    async def _push_sentiment_to_cloud(self, sentiment_results: dict):
        """Push sentiment data to cloud"""
        try:
            from cloud import SupabaseSync

            sync = SupabaseSync.from_env()

            for symbol, result in sentiment_results.items():
                await sync.push_sentiment(symbol, result.to_dict())

        except Exception as e:
            logger.error(f"Failed to push sentiment: {e}")

    async def _push_predictions_to_cloud(self, predictions: list):
        """Push predictions to cloud"""
        try:
            from cloud import SupabaseSync
            from cloud.supabase_sync import PredictionRecord

            sync = SupabaseSync.from_env()

            records = []
            for p in predictions:
                record = PredictionRecord(
                    id=f"{p['symbol']}_{datetime.now().strftime('%Y%m%d%H%M')}",
                    symbol=p["symbol"],
                    prediction=p["prediction"],
                    confidence=p["confidence"],
                    model_type=p["model_type"],
                    features_used=[],
                    price_at_prediction=0,  # Would need to fetch
                    predicted_at=datetime.now(),
                    horizon_days=5,
                )
                records.append(record)

            await sync.push_predictions(records)

        except Exception as e:
            logger.error(f"Failed to push predictions: {e}")

    def get_status(self) -> dict:
        """Get scheduler status"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })

        return {
            "running": self._running,
            "jobs": jobs,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "last_news_scan": self.last_news_scan.isoformat() if self.last_news_scan else None,
            "last_prediction": self.last_prediction.isoformat() if self.last_prediction else None,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
        }

    async def run_now(self, job_id: str):
        """Manually trigger a job"""
        job_map = {
            "training": self._run_training,
            "news": self._run_news_scan,
            "predictions": self._run_predictions,
            "sync": self._run_sync,
        }

        if job_id in job_map:
            await job_map[job_id]()
        else:
            raise ValueError(f"Unknown job: {job_id}")


# Global scheduler instance
_scheduler: Optional[DPolarisScheduler] = None


def get_scheduler() -> DPolarisScheduler:
    """Get or create scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = DPolarisScheduler()
    return _scheduler


async def start_scheduler():
    """Start the global scheduler"""
    scheduler = get_scheduler()
    scheduler.start()
    return scheduler


async def stop_scheduler():
    """Stop the global scheduler"""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None
