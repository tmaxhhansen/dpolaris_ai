"""
Precision configuration for model evaluation and backtesting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import os

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "config" / "prediction_precision.yaml"
)


@dataclass
class TargetConfig:
    """Prediction target/horizon configuration."""

    classification: str = "target_direction"
    regression: str = "target_return"
    horizon_days: int = 5


@dataclass
class ValidationConfig:
    """Validation strategy configuration."""

    method: str = "walk_forward"
    walk_forward_splits: int = 5
    min_samples: int = 500
    enforce_no_lookahead: bool = True
    no_lookahead_sample_count: int = 12


@dataclass
class MetricsConfig:
    """Metric-specific configuration."""

    reliability_bins: int = 10
    regression_hit_rate_thresholds: list[float] = field(
        default_factory=lambda: [0.0, 0.0025, 0.005, 0.01]
    )


@dataclass
class BacktestConfig:
    """Trading backtest assumptions."""

    transaction_cost_bps: float = 2.0
    slippage_bps: float = 3.0
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    annualization_periods: int = 252


@dataclass
class CalibrationConfig:
    """Probability calibration settings."""

    enabled: bool = True
    method: str = "platt"


@dataclass
class PrimaryScoreConfig:
    """Primary optimization objective."""

    objective: str = "maximize_sharpe_with_drawdown_cap"
    metric: str = "sharpe"
    fallback_metric: str = "f1"
    max_drawdown: float = 0.20
    min_trades: int = 25


@dataclass
class PrecisionConfig:
    """Top-level precision/evaluation configuration."""

    random_seed: int = 42
    target: TargetConfig = field(default_factory=TargetConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    primary_score: PrimaryScoreConfig = field(default_factory=PrimaryScoreConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_precision_config(data: dict[str, Any]) -> PrecisionConfig:
    target = TargetConfig(**data.get("target", {}))
    validation = ValidationConfig(**data.get("validation", {}))
    metrics = MetricsConfig(**data.get("metrics", {}))
    backtest = BacktestConfig(**data.get("backtest", {}))
    calibration = CalibrationConfig(**data.get("calibration", {}))
    primary_score = PrimaryScoreConfig(**data.get("primary_score", {}))

    return PrecisionConfig(
        random_seed=int(data.get("random_seed", 42)),
        target=target,
        validation=validation,
        metrics=metrics,
        backtest=backtest,
        calibration=calibration,
        primary_score=primary_score,
    )


def load_precision_config(config_path: Optional[Path | str] = None) -> PrecisionConfig:
    """
    Load precision config from YAML and merge with safe defaults.
    """
    if config_path is None:
        env_path = os.getenv("DPOLARIS_PRECISION_CONFIG")
        config_path = Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path).expanduser()

    defaults = PrecisionConfig().to_dict()

    if not config_path.exists():
        return _build_precision_config(defaults)

    with open(config_path) as f:
        loaded = yaml.safe_load(f) or {}

    merged = _merge_dict(defaults, loaded)
    return _build_precision_config(merged)
