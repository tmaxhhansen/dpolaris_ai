"""
dPolaris Background Daemon

Runs scheduled jobs for:
- Market scanning
- Portfolio updates
- Alert checking
- Daily/weekly summaries
- Model retraining
"""

import asyncio
import signal
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Callable
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.config import Config, get_config
from core.database import Database
from core.memory import DPolarisMemory
from core.ai import DPolarisAI
from tools.market_data import MarketDataService, screen_high_iv_rank, get_market_overview

logger = logging.getLogger("dpolaris.daemon")


class DPolarisDaemon:
    """
    Background daemon for dPolaris.

    Runs scheduled tasks:
    - Market scanning every 5 minutes during market hours
    - Alert checking every minute
    - Portfolio updates every 15 minutes
    - Daily pre-market briefing
    - Post-market summary
    - Weekly performance review
    - Model retraining (weekly)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Database] = None,
    ):
        self.config = config or get_config()
        self.db = db or Database()
        self.memory = DPolarisMemory(self.db)
        self.ai = DPolarisAI(self.config, self.db, self.memory)
        self.market_service = MarketDataService()

        self.scheduler = AsyncIOScheduler()
        self.running = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure daemon logging"""
        log_dir = self.config.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_dir / "daemon.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def setup_jobs(self):
        """Configure all scheduled jobs"""
        schedule = self.config.schedule

        # Market hours jobs (every N minutes)
        self.scheduler.add_job(
            self._job_market_scan,
            IntervalTrigger(minutes=schedule.market_scan_interval_minutes),
            id="market_scan",
            name="Market Scanner",
        )

        self.scheduler.add_job(
            self._job_check_alerts,
            IntervalTrigger(minutes=1),
            id="alert_check",
            name="Alert Checker",
        )

        self.scheduler.add_job(
            self._job_update_portfolio,
            IntervalTrigger(minutes=schedule.portfolio_update_interval_minutes),
            id="portfolio_update",
            name="Portfolio Updater",
        )

        self.scheduler.add_job(
            self._job_track_iv,
            IntervalTrigger(minutes=schedule.iv_tracking_interval_minutes),
            id="iv_tracking",
            name="IV Tracker",
        )

        # Daily jobs
        pre_market_time = schedule.pre_market_briefing_time.split(":")
        self.scheduler.add_job(
            self._job_pre_market_briefing,
            CronTrigger(
                hour=int(pre_market_time[0]),
                minute=int(pre_market_time[1]),
                day_of_week="mon-fri",
            ),
            id="pre_market",
            name="Pre-Market Briefing",
        )

        post_market_time = schedule.post_market_summary_time.split(":")
        self.scheduler.add_job(
            self._job_post_market_summary,
            CronTrigger(
                hour=int(post_market_time[0]),
                minute=int(post_market_time[1]),
                day_of_week="mon-fri",
            ),
            id="post_market",
            name="Post-Market Summary",
        )

        # Backup job (daily)
        backup_time = schedule.backup_time.split(":")
        self.scheduler.add_job(
            self._job_backup,
            CronTrigger(
                hour=int(backup_time[0]),
                minute=int(backup_time[1]),
            ),
            id="backup",
            name="Daily Backup",
        )

        # Weekly jobs
        weekly_time = schedule.weekly_review_time.split(":")
        self.scheduler.add_job(
            self._job_weekly_review,
            CronTrigger(
                day_of_week=schedule.weekly_review_day[:3].lower(),
                hour=int(weekly_time[0]),
                minute=int(weekly_time[1]),
            ),
            id="weekly_review",
            name="Weekly Review",
        )

        # Model retraining (weekly)
        retrain_time = schedule.model_retrain_time.split(":")
        self.scheduler.add_job(
            self._job_retrain_models,
            CronTrigger(
                day_of_week=schedule.model_retrain_day[:3].lower(),
                hour=int(retrain_time[0]),
                minute=int(retrain_time[1]),
            ),
            id="model_retrain",
            name="Model Retraining",
        )

        logger.info("All jobs configured")

    def is_market_hours(self) -> bool:
        """Check if currently in market hours (US Eastern)"""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Simplified market hours check (9:30 AM - 4:00 PM)
        # In production, use proper timezone handling
        market_open = time(9, 30)
        market_close = time(16, 0)

        current_time = now.time()
        return market_open <= current_time <= market_close

    # ==================== Job Functions ====================

    async def _job_market_scan(self):
        """Scan markets for opportunities"""
        if not self.is_market_hours():
            logger.debug("Skipping market scan - outside market hours")
            return

        logger.info("Running market scan...")
        try:
            # Get watchlist symbols
            watchlist = self.db.get_watchlist()
            symbols = [w["symbol"] for w in watchlist] if watchlist else self.config.watchlist

            # Screen for high IV rank (premium selling opportunities)
            high_iv = await screen_high_iv_rank(symbols, min_iv_rank=50)

            if high_iv:
                logger.info(f"Found {len(high_iv)} high IV rank opportunities")
                # Could trigger notification here
                for opp in high_iv[:3]:
                    logger.info(f"  {opp['symbol']}: IV Rank {opp.get('iv_rank', 0):.1f}%")

            # Store market snapshot
            overview = await get_market_overview()
            if overview.get("indices", {}).get("SPY"):
                spy = overview["indices"]["SPY"]
                vix = overview.get("volatility", {}).get("VIX", {})

                self.db.save_market_snapshot(
                    spy_price=spy.get("price", 0),
                    spy_change=((spy.get("price", 0) / spy.get("previous_close", 1)) - 1) * 100,
                    vix=vix.get("price", 0) if vix else 0,
                    regime=self._assess_regime(spy, vix),
                )

        except Exception as e:
            logger.error(f"Market scan error: {e}")

    async def _job_check_alerts(self):
        """Check and trigger alerts"""
        if not self.is_market_hours():
            return

        try:
            alerts = self.db.get_active_alerts()

            for alert in alerts:
                symbol = alert["symbol"]
                quote = await self.market_service.get_quote(symbol)

                if not quote:
                    continue

                price = quote.get("price", 0)
                triggered = False

                if alert["alert_type"] == "price":
                    if alert["condition"] == "above" and price > alert["threshold"]:
                        triggered = True
                    elif alert["condition"] == "below" and price < alert["threshold"]:
                        triggered = True

                if triggered:
                    self.db.trigger_alert(alert["id"])
                    logger.info(f"Alert triggered: {symbol} {alert['condition']} {alert['threshold']}")
                    # Could send notification here

        except Exception as e:
            logger.error(f"Alert check error: {e}")

    async def _job_update_portfolio(self):
        """Update portfolio with current prices"""
        try:
            positions = self.db.get_open_positions()

            for position in positions:
                quote = await self.market_service.get_quote(position["symbol"])
                if quote and quote.get("price"):
                    self.db.update_position_price(position["id"], quote["price"])

            logger.debug(f"Updated {len(positions)} positions")

        except Exception as e:
            logger.error(f"Portfolio update error: {e}")

    async def _job_track_iv(self):
        """Track IV changes for watchlist"""
        if not self.is_market_hours():
            return

        try:
            watchlist = self.db.get_watchlist()
            symbols = [w["symbol"] for w in watchlist] if watchlist else self.config.watchlist[:5]

            for symbol in symbols:
                iv_data = await self.market_service.get_iv_metrics(symbol)
                if iv_data:
                    logger.debug(f"{symbol}: IV Rank {iv_data.get('iv_rank', 0):.1f}%")

        except Exception as e:
            logger.error(f"IV tracking error: {e}")

    async def _job_pre_market_briefing(self):
        """Generate pre-market briefing"""
        logger.info("Generating pre-market briefing...")
        try:
            briefing = await self.ai.chat("""Generate a pre-market briefing:

1. Overnight developments
2. Key levels for SPY/QQQ
3. Today's economic calendar events
4. Earnings reports today
5. Watchlist stocks to focus on

Keep it concise and actionable.""")

            # Save briefing
            briefing_path = self.config.data_dir / "reports" / "daily"
            briefing_path.mkdir(parents=True, exist_ok=True)
            (briefing_path / f"{datetime.now().strftime('%Y-%m-%d')}_premarket.md").write_text(briefing)

            logger.info("Pre-market briefing generated")

        except Exception as e:
            logger.error(f"Pre-market briefing error: {e}")

    async def _job_post_market_summary(self):
        """Generate post-market summary"""
        logger.info("Generating post-market summary...")
        try:
            summary = await self.ai.chat("""Generate a post-market summary:

1. How did the major indices perform?
2. Notable sector moves
3. My portfolio performance today
4. Key takeaways
5. Setup for tomorrow

Be concise.""")

            # Save summary
            summary_path = self.config.data_dir / "reports" / "daily"
            summary_path.mkdir(parents=True, exist_ok=True)
            (summary_path / f"{datetime.now().strftime('%Y-%m-%d')}_postmarket.md").write_text(summary)

            logger.info("Post-market summary generated")

        except Exception as e:
            logger.error(f"Post-market summary error: {e}")

    async def _job_weekly_review(self):
        """Generate weekly performance review"""
        logger.info("Generating weekly review...")
        try:
            review = await self.ai.chat("@performance")

            # Save review
            review_path = self.config.data_dir / "reports" / "weekly"
            review_path.mkdir(parents=True, exist_ok=True)
            (review_path / f"{datetime.now().strftime('%Y-%m-%d')}_weekly.md").write_text(review)

            logger.info("Weekly review generated")

        except Exception as e:
            logger.error(f"Weekly review error: {e}")

    async def _job_retrain_models(self):
        """Retrain ML models with latest data"""
        logger.info("Starting model retraining...")
        try:
            # Retrain for main symbols
            for symbol in ["SPY", "QQQ"]:
                result = await self.ai.chat(f"@train {symbol}")
                logger.info(f"Model retrained for {symbol}")

            logger.info("Model retraining complete")

        except Exception as e:
            logger.error(f"Model retraining error: {e}")

    async def _job_backup(self):
        """Create daily backup"""
        logger.info("Creating backup...")
        try:
            backup_dir = self.config.data_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_path = backup_dir / f"backup_{datetime.now().strftime('%Y%m%d')}.db"
            self.db.backup(backup_path)

            # Also export memory
            self.memory.export(backup_dir / f"memory_{datetime.now().strftime('%Y%m%d')}.json")

            logger.info(f"Backup created: {backup_path}")

        except Exception as e:
            logger.error(f"Backup error: {e}")

    # ==================== Helpers ====================

    def _assess_regime(self, spy_data: dict, vix_data: dict) -> str:
        """Quick regime assessment"""
        vix_level = vix_data.get("price", 20) if vix_data else 20

        if vix_level < 15:
            vol_regime = "low_vol"
        elif vix_level < 20:
            vol_regime = "normal"
        elif vix_level < 30:
            vol_regime = "elevated"
        else:
            vol_regime = "crisis"

        # Simplified trend assessment
        spy_price = spy_data.get("price", 0)
        spy_prev = spy_data.get("previous_close", spy_price)

        if spy_price > spy_prev:
            trend = "bullish"
        else:
            trend = "bearish"

        return f"{trend}_{vol_regime}"

    # ==================== Lifecycle ====================

    async def start(self):
        """Start the daemon"""
        self.running = True
        self.setup_jobs()
        self.scheduler.start()

        logger.info("dPolaris daemon started")
        print("dPolaris daemon running... Press Ctrl+C to stop")

        # Keep running
        while self.running:
            await asyncio.sleep(1)

    def stop(self):
        """Stop the daemon"""
        self.running = False
        self.scheduler.shutdown()
        logger.info("dPolaris daemon stopped")

    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)


async def run_daemon():
    """Run the daemon"""
    daemon = DPolarisDaemon()

    # Setup signal handlers
    signal.signal(signal.SIGINT, daemon.handle_signal)
    signal.signal(signal.SIGTERM, daemon.handle_signal)

    await daemon.start()


def main():
    """Main entry point"""
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
