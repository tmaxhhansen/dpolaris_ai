"""
Database module for dPolaris AI

Uses SQLite for portability - entire database can be moved with the data folder.
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager
import json
import logging

logger = logging.getLogger("dpolaris.database")


SCHEMA = """
-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cash REAL NOT NULL,
    invested REAL NOT NULL,
    total_value REAL NOT NULL,
    daily_pnl REAL,
    total_pnl REAL,
    goal_progress REAL
);

-- Open positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    position_type TEXT NOT NULL DEFAULT 'stock',
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_date DATETIME NOT NULL,
    current_price REAL,
    unrealized_pnl REAL,
    option_details TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    closed_at DATETIME,
    exit_price REAL,
    realized_pnl REAL,
    is_open INTEGER DEFAULT 1
);

-- Trade journal
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT,
    direction TEXT,
    entry_date DATETIME,
    exit_date DATETIME,
    entry_price REAL,
    exit_price REAL,
    quantity REAL,
    pnl REAL,
    pnl_percent REAL,
    thesis TEXT,
    outcome_notes TEXT,
    lessons TEXT,
    tags TEXT,
    iv_at_entry REAL,
    iv_at_exit REAL,
    market_regime TEXT,
    conviction_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- AI Memory - persistent learning
CREATE TABLE IF NOT EXISTS ai_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    embedding TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME,
    access_count INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1
);

-- Conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens_used INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    thesis TEXT,
    target_entry REAL,
    target_exit REAL,
    stop_loss REAL,
    alerts TEXT,
    ai_notes TEXT,
    priority INTEGER DEFAULT 5,
    added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_analyzed DATETIME
);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    condition TEXT NOT NULL,
    threshold REAL,
    message TEXT,
    is_active INTEGER DEFAULT 1,
    triggered INTEGER DEFAULT 0,
    triggered_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Market snapshots for regime analysis
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    spy_price REAL,
    spy_change REAL,
    vix REAL,
    vix_term_structure TEXT,
    breadth_200dma REAL,
    breadth_50dma REAL,
    sector_data TEXT,
    regime TEXT,
    notes TEXT
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE NOT NULL,
    portfolio_value REAL,
    daily_return REAL,
    cumulative_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    trades_count INTEGER,
    goal_progress REAL
);

-- ML Model tracking
CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    target TEXT NOT NULL,
    features TEXT NOT NULL,
    metrics TEXT,
    file_path TEXT,
    is_active INTEGER DEFAULT 1,
    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_prediction DATETIME,
    prediction_count INTEGER DEFAULT 0
);

-- ML Training data
CREATE TABLE IF NOT EXISTS training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    features TEXT NOT NULL,
    target_values TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- ML Predictions log
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    symbol TEXT NOT NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    target_date DATE,
    predicted_value REAL,
    actual_value REAL,
    confidence REAL,
    features_used TEXT,
    FOREIGN KEY (model_id) REFERENCES ml_models(id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_ai_memory_category ON ai_memory(category);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_timestamp ON market_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
"""


class Database:
    """SQLite database wrapper for dPolaris"""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path("~/dpolaris_data/db/dpolaris.db").expanduser()

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize database with schema"""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== Portfolio Operations ====================

    def save_portfolio_snapshot(
        self,
        cash: float,
        invested: float,
        total_value: float,
        daily_pnl: float = 0,
        total_pnl: float = 0,
        goal_progress: float = 0,
    ) -> int:
        """Save portfolio snapshot"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO portfolio_snapshots
                (cash, invested, total_value, daily_pnl, total_pnl, goal_progress)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cash, invested, total_value, daily_pnl, total_pnl, goal_progress),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_portfolio(self) -> Optional[dict]:
        """Get latest portfolio snapshot"""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM portfolio_snapshots
                ORDER BY timestamp DESC LIMIT 1
                """
            ).fetchone()
            return dict(row) if row else None

    def get_portfolio_history(self, days: int = 30) -> list[dict]:
        """Get portfolio value history"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM portfolio_snapshots
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp ASC
                """,
                (f"-{days} days",),
            ).fetchall()
            return [dict(row) for row in rows]

    # ==================== Position Operations ====================

    def add_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        position_type: str = "stock",
        option_details: Optional[dict] = None,
        notes: str = "",
    ) -> int:
        """Add new position"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO positions
                (symbol, position_type, quantity, entry_price, entry_date, option_details, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    position_type,
                    quantity,
                    entry_price,
                    datetime.now(),
                    json.dumps(option_details) if option_details else None,
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_open_positions(self) -> list[dict]:
        """Get all open positions"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE is_open = 1"
            ).fetchall()
            return [dict(row) for row in rows]

    def update_position_price(self, position_id: int, current_price: float):
        """Update current price and unrealized P&L"""
        with self.get_connection() as conn:
            position = conn.execute(
                "SELECT entry_price, quantity FROM positions WHERE id = ?",
                (position_id,),
            ).fetchone()

            if position:
                pnl = (current_price - position["entry_price"]) * position["quantity"]
                conn.execute(
                    """
                    UPDATE positions
                    SET current_price = ?, unrealized_pnl = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (current_price, pnl, datetime.now(), position_id),
                )
                conn.commit()

    def close_position(self, position_id: int, exit_price: float) -> Optional[dict]:
        """Close a position"""
        with self.get_connection() as conn:
            position = conn.execute(
                "SELECT * FROM positions WHERE id = ?", (position_id,)
            ).fetchone()

            if position:
                pnl = (exit_price - position["entry_price"]) * position["quantity"]
                conn.execute(
                    """
                    UPDATE positions
                    SET is_open = 0, closed_at = ?, exit_price = ?, realized_pnl = ?
                    WHERE id = ?
                    """,
                    (datetime.now(), exit_price, pnl, position_id),
                )
                conn.commit()
                return {"pnl": pnl, "exit_price": exit_price}
            return None

    # ==================== Trade Journal ====================

    def add_trade(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        entry_price: float,
        quantity: float,
        thesis: str = "",
        **kwargs,
    ) -> int:
        """Add trade to journal"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades
                (symbol, strategy, direction, entry_date, entry_price, quantity, thesis,
                 iv_at_entry, market_regime, conviction_score, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    strategy,
                    direction,
                    datetime.now(),
                    entry_price,
                    quantity,
                    thesis,
                    kwargs.get("iv_at_entry"),
                    kwargs.get("market_regime"),
                    kwargs.get("conviction_score"),
                    json.dumps(kwargs.get("tags", [])),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        outcome_notes: str = "",
        lessons: str = "",
    ):
        """Close a trade with exit details"""
        with self.get_connection() as conn:
            trade = conn.execute(
                "SELECT entry_price, quantity FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()

            if trade:
                pnl = (exit_price - trade["entry_price"]) * trade["quantity"]
                pnl_percent = ((exit_price / trade["entry_price"]) - 1) * 100

                conn.execute(
                    """
                    UPDATE trades
                    SET exit_date = ?, exit_price = ?, pnl = ?, pnl_percent = ?,
                        outcome_notes = ?, lessons = ?
                    WHERE id = ?
                    """,
                    (datetime.now(), exit_price, pnl, pnl_percent, outcome_notes, lessons, trade_id),
                )
                conn.commit()

    def get_trades(
        self,
        limit: int = 50,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        """Get trades from journal"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY entry_date DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_trade_stats(self) -> dict:
        """Get aggregate trade statistics"""
        with self.get_connection() as conn:
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl) as avg_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl <= 0 THEN pnl END) as avg_loss,
                    SUM(pnl) as total_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE exit_date IS NOT NULL
                """
            ).fetchone()

            result = dict(stats)
            if result["total_trades"] and result["total_trades"] > 0:
                result["win_rate"] = (result["winning_trades"] / result["total_trades"]) * 100
                if result["avg_loss"] and result["avg_loss"] != 0:
                    result["profit_factor"] = abs(result["avg_win"] / result["avg_loss"])
            return result

    # ==================== AI Memory ====================

    def save_memory(
        self,
        category: str,
        content: str,
        importance: float = 0.5,
        embedding: Optional[list[float]] = None,
    ) -> int:
        """Save AI memory entry"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ai_memory (category, content, importance, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (category, content, importance, json.dumps(embedding) if embedding else None),
            )
            conn.commit()
            return cursor.lastrowid

    def get_memories(
        self,
        category: Optional[str] = None,
        limit: int = 20,
        min_importance: float = 0,
    ) -> list[dict]:
        """Retrieve AI memories"""
        query = "SELECT * FROM ai_memory WHERE is_active = 1 AND importance >= ?"
        params = [min_importance]

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY importance DESC, last_accessed DESC NULLS LAST LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

            # Update access counts
            for row in rows:
                conn.execute(
                    """
                    UPDATE ai_memory
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                    """,
                    (datetime.now(), row["id"]),
                )
            conn.commit()

            return [dict(row) for row in rows]

    # ==================== Watchlist ====================

    def add_to_watchlist(
        self,
        symbol: str,
        thesis: str = "",
        target_entry: Optional[float] = None,
        priority: int = 5,
    ) -> int:
        """Add symbol to watchlist"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO watchlist (symbol, thesis, target_entry, priority)
                VALUES (?, ?, ?, ?)
                """,
                (symbol.upper(), thesis, target_entry, priority),
            )
            conn.commit()
            return cursor.lastrowid

    def get_watchlist(self) -> list[dict]:
        """Get watchlist"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM watchlist ORDER BY priority DESC, added_date DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def remove_from_watchlist(self, symbol: str):
        """Remove from watchlist"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
            conn.commit()

    # ==================== Alerts ====================

    def create_alert(
        self,
        symbol: str,
        alert_type: str,
        condition: str,
        threshold: float,
        message: str = "",
    ) -> int:
        """Create price/IV alert"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO alerts (symbol, alert_type, condition, threshold, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol.upper(), alert_type, condition, threshold, message),
            )
            conn.commit()
            return cursor.lastrowid

    def get_active_alerts(self) -> list[dict]:
        """Get active alerts"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE is_active = 1 AND triggered = 0"
            ).fetchall()
            return [dict(row) for row in rows]

    def trigger_alert(self, alert_id: int):
        """Mark alert as triggered"""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE alerts
                SET triggered = 1, triggered_at = ?
                WHERE id = ?
                """,
                (datetime.now(), alert_id),
            )
            conn.commit()

    # ==================== Market Snapshots ====================

    def save_market_snapshot(
        self,
        spy_price: float,
        spy_change: float,
        vix: float,
        regime: str,
        **kwargs,
    ) -> int:
        """Save market regime snapshot"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO market_snapshots
                (spy_price, spy_change, vix, regime, vix_term_structure,
                 breadth_200dma, breadth_50dma, sector_data, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spy_price,
                    spy_change,
                    vix,
                    regime,
                    kwargs.get("vix_term_structure"),
                    kwargs.get("breadth_200dma"),
                    kwargs.get("breadth_50dma"),
                    json.dumps(kwargs.get("sector_data")) if kwargs.get("sector_data") else None,
                    kwargs.get("notes"),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_market_snapshot(self) -> Optional[dict]:
        """Get latest market snapshot"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    # ==================== ML Models ====================

    def save_model_info(
        self,
        model_name: str,
        model_type: str,
        version: str,
        target: str,
        features: list[str],
        metrics: dict,
        file_path: str,
    ) -> int:
        """Save ML model metadata"""
        with self.get_connection() as conn:
            # Deactivate previous versions
            conn.execute(
                "UPDATE ml_models SET is_active = 0 WHERE model_name = ?",
                (model_name,),
            )

            cursor = conn.execute(
                """
                INSERT INTO ml_models
                (model_name, model_type, version, target, features, metrics, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    model_type,
                    version,
                    target,
                    json.dumps(features),
                    json.dumps(metrics),
                    file_path,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_active_models(self) -> list[dict]:
        """Get all active ML models"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ml_models WHERE is_active = 1"
            ).fetchall()
            return [dict(row) for row in rows]

    def save_prediction(
        self,
        model_id: int,
        symbol: str,
        target_date: date,
        predicted_value: float,
        confidence: float,
        features_used: dict,
    ) -> int:
        """Log a prediction"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions
                (model_id, symbol, target_date, predicted_value, confidence, features_used)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    symbol,
                    target_date,
                    predicted_value,
                    confidence,
                    json.dumps(features_used),
                ),
            )

            # Update model prediction count
            conn.execute(
                """
                UPDATE ml_models
                SET last_prediction = ?, prediction_count = prediction_count + 1
                WHERE id = ?
                """,
                (datetime.now(), model_id),
            )
            conn.commit()
            return cursor.lastrowid

    def update_prediction_actual(self, prediction_id: int, actual_value: float):
        """Update prediction with actual value for model evaluation"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE predictions SET actual_value = ? WHERE id = ?",
                (actual_value, prediction_id),
            )
            conn.commit()

    # ==================== Performance Tracking ====================

    def save_daily_performance(
        self,
        date_val: date,
        portfolio_value: float,
        daily_return: float,
        cumulative_return: float,
        **kwargs,
    ):
        """Save daily performance metrics"""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO performance_metrics
                (date, portfolio_value, daily_return, cumulative_return,
                 sharpe_ratio, max_drawdown, win_rate, trades_count, goal_progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date_val,
                    portfolio_value,
                    daily_return,
                    cumulative_return,
                    kwargs.get("sharpe_ratio"),
                    kwargs.get("max_drawdown"),
                    kwargs.get("win_rate"),
                    kwargs.get("trades_count"),
                    kwargs.get("goal_progress"),
                ),
            )
            conn.commit()

    def get_performance_history(self, days: int = 90) -> list[dict]:
        """Get performance history"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM performance_metrics
                WHERE date >= date('now', ?)
                ORDER BY date ASC
                """,
                (f"-{days} days",),
            ).fetchall()
            return [dict(row) for row in rows]

    # ==================== Conversations ====================

    def save_conversation(self, session_id: str, role: str, content: str, tokens: int = 0):
        """Save conversation turn"""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (session_id, role, content, tokens_used)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, tokens),
            )
            conn.commit()

    def get_recent_conversations(self, session_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get recent conversation history"""
        query = "SELECT * FROM conversations"
        params = []

        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows][::-1]  # Reverse to chronological

    # ==================== Backup ====================

    def backup(self, backup_path: Path):
        """Create database backup"""
        import shutil

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
