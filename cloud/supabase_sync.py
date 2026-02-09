"""
Supabase Cloud Sync for dPolaris

Minimal cloud sync - Windows pushes predictions, MacBook pulls them.
All computation happens locally, cloud is just storage.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger("dpolaris.cloud.supabase")


@dataclass
class SyncConfig:
    """Supabase configuration"""
    url: str
    anon_key: str
    service_role_key: Optional[str] = None  # Only needed for admin operations


@dataclass
class PredictionRecord:
    """A prediction record to sync"""
    id: str
    symbol: str
    prediction: str  # UP, DOWN
    confidence: float
    model_type: str  # lstm, transformer, ensemble
    features_used: List[str]
    price_at_prediction: float
    predicted_at: datetime
    horizon_days: int
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    actual_direction: Optional[str] = None  # Filled in later
    was_correct: Optional[bool] = None  # Filled in later

    def to_dict(self) -> dict:
        d = asdict(self)
        d["predicted_at"] = self.predicted_at.isoformat()
        return d


class SupabaseSync:
    """
    Sync predictions and data to Supabase.

    Architecture:
    - Windows machine: Runs training, makes predictions, pushes to cloud
    - MacBook: Pulls predictions from cloud for display
    - Cloud: Just stores JSON data, no computation

    Tables needed in Supabase:
    - predictions: Store all predictions
    - model_metrics: Store model performance over time
    - news_sentiment: Store aggregated sentiment data
    - sync_status: Track last sync times
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config
        self.client = None
        self._initialized = False

        if config:
            self._init_client()

    def _init_client(self):
        """Initialize Supabase client"""
        if not self.config:
            logger.warning("No Supabase config provided")
            return

        try:
            from supabase import create_client, Client

            self.client: Client = create_client(
                self.config.url,
                self.config.anon_key
            )
            self._initialized = True
            logger.info("Supabase client initialized")

        except ImportError:
            logger.warning("supabase-py not installed. Run: pip install supabase")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")

    @classmethod
    def from_env(cls) -> "SupabaseSync":
        """Create from environment variables"""
        import os

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")

        if not url or not key:
            logger.warning("SUPABASE_URL and SUPABASE_ANON_KEY not set")
            return cls()

        return cls(SyncConfig(url=url, anon_key=key))

    @classmethod
    def from_config_file(cls, path: Optional[Path] = None) -> "SupabaseSync":
        """Load config from JSON file"""
        if path is None:
            path = Path("~/.dpolaris/supabase.json").expanduser()

        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)
            return cls(SyncConfig(**data))
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()

    # =========================================================================
    # Predictions
    # =========================================================================

    async def push_prediction(self, prediction: PredictionRecord) -> bool:
        """Push a single prediction to cloud"""
        if not self._initialized:
            logger.warning("Supabase not initialized, saving locally only")
            return False

        try:
            result = self.client.table("predictions").upsert(
                prediction.to_dict()
            ).execute()

            logger.info(f"Pushed prediction for {prediction.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to push prediction: {e}")
            return False

    async def push_predictions(self, predictions: List[PredictionRecord]) -> int:
        """Push multiple predictions"""
        if not self._initialized:
            return 0

        try:
            data = [p.to_dict() for p in predictions]
            result = self.client.table("predictions").upsert(data).execute()

            count = len(predictions)
            logger.info(f"Pushed {count} predictions")
            return count

        except Exception as e:
            logger.error(f"Failed to push predictions: {e}")
            return 0

    async def pull_predictions(
        self,
        symbols: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Pull predictions from cloud"""
        if not self._initialized:
            logger.warning("Supabase not initialized")
            return []

        try:
            query = self.client.table("predictions").select("*")

            if symbols:
                query = query.in_("symbol", symbols)

            if since:
                query = query.gte("predicted_at", since.isoformat())

            query = query.order("predicted_at", desc=True).limit(limit)

            result = query.execute()
            return result.data

        except Exception as e:
            logger.error(f"Failed to pull predictions: {e}")
            return []

    async def get_latest_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction for a symbol"""
        predictions = await self.pull_predictions(symbols=[symbol], limit=1)
        return predictions[0] if predictions else None

    # =========================================================================
    # Model Metrics
    # =========================================================================

    async def push_model_metrics(
        self,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        version: str,
    ) -> bool:
        """Push model performance metrics"""
        if not self._initialized:
            return False

        try:
            data = {
                "id": f"{model_name}_{version}",
                "model_name": model_name,
                "model_type": model_type,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "version": version,
                "recorded_at": datetime.now().isoformat(),
            }

            self.client.table("model_metrics").upsert(data).execute()
            logger.info(f"Pushed metrics for {model_name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
            return False

    async def pull_model_metrics(
        self,
        model_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Pull model metrics history"""
        if not self._initialized:
            return []

        try:
            query = self.client.table("model_metrics").select("*")

            if model_name:
                query = query.eq("model_name", model_name)

            result = query.order("recorded_at", desc=True).limit(limit).execute()
            return result.data

        except Exception as e:
            logger.error(f"Failed to pull metrics: {e}")
            return []

    # =========================================================================
    # News Sentiment
    # =========================================================================

    async def push_sentiment(
        self,
        symbol: str,
        sentiment_data: Dict[str, Any],
    ) -> bool:
        """Push sentiment analysis results"""
        if not self._initialized:
            return False

        try:
            data = {
                "id": f"{symbol}_{datetime.now().strftime('%Y%m%d')}",
                "symbol": symbol,
                "score": sentiment_data.get("score"),
                "label": sentiment_data.get("label"),
                "article_count": sentiment_data.get("article_count"),
                "positive_count": sentiment_data.get("positive_count"),
                "negative_count": sentiment_data.get("negative_count"),
                "top_headlines": json.dumps(sentiment_data.get("top_headlines", [])),
                "recorded_at": datetime.now().isoformat(),
            }

            self.client.table("news_sentiment").upsert(data).execute()
            return True

        except Exception as e:
            logger.error(f"Failed to push sentiment: {e}")
            return False

    async def pull_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Pull sentiment data"""
        if not self._initialized:
            return []

        try:
            query = self.client.table("news_sentiment").select("*")

            if symbols:
                query = query.in_("symbol", symbols)

            result = query.order("recorded_at", desc=True).limit(limit).execute()
            return result.data

        except Exception as e:
            logger.error(f"Failed to pull sentiment: {e}")
            return []

    # =========================================================================
    # Sync Status
    # =========================================================================

    async def update_sync_status(self, device_id: str, sync_type: str) -> bool:
        """Update last sync timestamp for a device"""
        if not self._initialized:
            return False

        try:
            data = {
                "device_id": device_id,
                "sync_type": sync_type,
                "last_sync": datetime.now().isoformat(),
            }

            self.client.table("sync_status").upsert(
                data,
                on_conflict="device_id,sync_type"
            ).execute()
            return True

        except Exception as e:
            logger.error(f"Failed to update sync status: {e}")
            return False

    async def get_sync_status(self, device_id: str) -> Dict[str, datetime]:
        """Get last sync times for a device"""
        if not self._initialized:
            return {}

        try:
            result = self.client.table("sync_status").select("*").eq(
                "device_id", device_id
            ).execute()

            return {
                row["sync_type"]: datetime.fromisoformat(row["last_sync"])
                for row in result.data
            }

        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {}


# =============================================================================
# Convenience Functions
# =============================================================================

async def sync_prediction_to_cloud(
    symbol: str,
    prediction: str,
    confidence: float,
    model_type: str,
    price: float,
    sentiment_score: Optional[float] = None,
) -> bool:
    """
    Quick function to sync a prediction.

    Usage:
        success = await sync_prediction_to_cloud(
            symbol="AAPL",
            prediction="UP",
            confidence=0.75,
            model_type="lstm",
            price=175.50,
        )
    """
    sync = SupabaseSync.from_env()

    record = PredictionRecord(
        id=f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        symbol=symbol,
        prediction=prediction,
        confidence=confidence,
        model_type=model_type,
        features_used=[],
        price_at_prediction=price,
        predicted_at=datetime.now(),
        horizon_days=5,
        sentiment_score=sentiment_score,
    )

    return await sync.push_prediction(record)


# =============================================================================
# Supabase Schema (for reference)
# =============================================================================

SUPABASE_SCHEMA = """
-- Run this in Supabase SQL editor to create tables

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_type TEXT NOT NULL,
    features_used JSONB,
    price_at_prediction REAL,
    predicted_at TIMESTAMPTZ NOT NULL,
    horizon_days INTEGER DEFAULT 5,
    sentiment_score REAL,
    sentiment_label TEXT,
    actual_direction TEXT,
    was_correct BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_symbol ON predictions(symbol);
CREATE INDEX idx_predictions_date ON predictions(predicted_at DESC);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    version TEXT,
    recorded_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_model ON model_metrics(model_name);

-- News sentiment table
CREATE TABLE IF NOT EXISTS news_sentiment (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    score REAL,
    label TEXT,
    article_count INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    top_headlines JSONB,
    recorded_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sentiment_symbol ON news_sentiment(symbol);

-- Sync status table
CREATE TABLE IF NOT EXISTS sync_status (
    device_id TEXT NOT NULL,
    sync_type TEXT NOT NULL,
    last_sync TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (device_id, sync_type)
);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE sync_status ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read/write (adjust as needed)
CREATE POLICY "Allow all for authenticated users" ON predictions
    FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated users" ON model_metrics
    FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated users" ON news_sentiment
    FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated users" ON sync_status
    FOR ALL USING (true);
"""
