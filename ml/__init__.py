"""
Machine Learning Models for dPolaris AI

Train and use local ML models for:
- Price direction prediction
- Volatility forecasting
- IV rank prediction
- Options strategy selection
- Pattern recognition

Deep Learning (PyTorch):
- LSTM for sequential time series
- Transformer for pattern recognition
- Auto-detects CUDA (Windows GPU) vs CPU (Mac)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .features import FeatureEngine as _FeatureEngine
    from .predictor import Predictor as _Predictor
    from .trainer import ModelTrainer as _ModelTrainer


def __getattr__(name: str) -> Any:
    """
    Lazily resolve heavy ML imports.

    This avoids importing optional OpenMP-backed libraries (xgboost/lightgbm)
    when only deep-learning modules are needed.
    """
    if name == "ModelTrainer":
        from .trainer import ModelTrainer

        return ModelTrainer
    if name == "Predictor":
        from .predictor import Predictor

        return Predictor
    if name == "FeatureEngine":
        from .features import FeatureEngine

        return FeatureEngine
    raise AttributeError(f"module 'ml' has no attribute '{name}'")

# Deep Learning imports (lazy load to avoid slow startup)
def get_deep_learning_trainer():
    """Get DeepLearningTrainer (lazy load)"""
    from .deep_learning import DeepLearningTrainer

    return DeepLearningTrainer


def get_lstm_predictor():
    """Get LSTMPredictor class (lazy load)"""
    from .deep_learning import LSTMPredictor

    return LSTMPredictor


def get_transformer_predictor():
    """Get TransformerPredictor class (lazy load)"""
    from .deep_learning import TransformerPredictor

    return TransformerPredictor


__all__ = [
    "ModelTrainer",
    "Predictor",
    "FeatureEngine",
    "get_deep_learning_trainer",
    "get_lstm_predictor",
    "get_transformer_predictor",
]
