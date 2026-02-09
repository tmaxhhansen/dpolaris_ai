"""
Configuration management for dPolaris AI
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class GoalConfig(BaseModel):
    """Goal tracking configuration"""
    target: float = 3_000_000
    start_date: str = "2024-01-01"
    starting_capital: float = 100_000


class RiskConfig(BaseModel):
    """Risk management parameters"""
    max_position_size_percent: float = 5.0
    max_portfolio_risk_percent: float = 2.0
    max_correlated_exposure: float = 15.0
    min_cash_reserve_percent: float = 20.0
    max_drawdown_percent: float = 15.0
    kelly_fraction: float = 0.25


class APIKeysConfig(BaseModel):
    """API keys configuration"""
    anthropic: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    polygon: str = Field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))
    alpha_vantage: str = Field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", ""))


class NotificationConfig(BaseModel):
    """Notification settings"""
    desktop: bool = True
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_from: str = ""
    email_to: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_enabled: bool = False
    discord_webhook_url: str = ""


class AIConfig(BaseModel):
    """AI behavior configuration"""
    model: str = "claude-sonnet-4-20250514"
    opus_model: str = "claude-opus-4-5-20251101"
    temperature: float = 0.3
    max_tokens: int = 8192
    conversation_memory_limit: int = 50
    learning_enabled: bool = True
    use_claude_cli: bool = True  # Enable Claude CLI for deep research


class MLConfig(BaseModel):
    """Machine learning configuration"""
    models_dir: str = "models"
    training_data_days: int = 365 * 2  # 2 years of data
    validation_split: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "auto"  # 'cpu', 'mps' (Apple Silicon), 'cuda', or 'auto'


class ScheduleConfig(BaseModel):
    """Daemon schedule configuration"""
    market_scan_interval_minutes: int = 5
    portfolio_update_interval_minutes: int = 15
    iv_tracking_interval_minutes: int = 30
    pre_market_briefing_time: str = "06:00"
    post_market_summary_time: str = "16:30"
    weekly_review_day: str = "sunday"
    weekly_review_time: str = "18:00"
    model_retrain_day: str = "saturday"
    model_retrain_time: str = "02:00"
    backup_time: str = "23:00"


class Config(BaseSettings):
    """Main configuration class"""

    # Data directory
    data_dir: Path = Path("~/dpolaris_data").expanduser()

    # Sub-configurations
    goal: GoalConfig = Field(default_factory=GoalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)

    # Watchlist
    watchlist: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "VIX", "AAPL", "NVDA"])

    class Config:
        env_prefix = "DPOLARIS_"
        env_nested_delimiter = "__"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path("~/dpolaris_data/config/settings.yaml").expanduser()

        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
                return cls(**data) if data else cls()

        return cls()

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to YAML file"""
        if config_path is None:
            config_path = self.data_dir / "config" / "settings.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key.split(".")
        value = self

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


# Default configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config():
    """Reload configuration from disk"""
    global _config
    _config = Config.load()
    return _config
