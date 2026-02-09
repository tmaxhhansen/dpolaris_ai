"""
Predictor for dPolaris ML Models

Makes predictions using trained models and tracks prediction performance.
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional
import logging

import numpy as np
import pandas as pd

from .features import FeatureEngine
from .trainer import ModelTrainer

logger = logging.getLogger("dpolaris.ml.predictor")


class Predictor:
    """
    Make predictions using trained models.

    Features:
    - Load and use trained models
    - Generate predictions with confidence scores
    - Track prediction accuracy over time
    - Ensemble multiple models
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path("~/dpolaris_data/models").expanduser()
        self.feature_engine = FeatureEngine()
        self.loaded_models: dict = {}
        self.trainer = ModelTrainer(models_dir=self.models_dir)

    def load_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Load a model into memory"""
        try:
            model, scaler, metadata = self.trainer.load_model(model_name, version)
            self.loaded_models[model_name] = {
                "model": model,
                "scaler": scaler,
                "metadata": metadata,
            }
            logger.info(f"Loaded model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def predict(
        self,
        model_name: str,
        df: pd.DataFrame,
        return_proba: bool = True,
    ) -> dict:
        """
        Make prediction using a loaded model.

        Args:
            model_name: Name of the model to use
            df: DataFrame with OHLCV data (needs enough history for features)
            return_proba: Whether to return probability scores

        Returns:
            Dictionary with prediction and confidence
        """
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                raise ValueError(f"Model {model_name} not found")

        model_data = self.loaded_models[model_name]
        model = model_data["model"]
        scaler = model_data["scaler"]
        metadata = model_data["metadata"]
        feature_names = metadata["feature_names"]

        # Generate features
        df_features = self.feature_engine.generate_features(df, include_targets=False)

        # Get latest row (most recent data)
        if df_features.empty:
            raise ValueError("Not enough data to generate features")

        latest = df_features[feature_names].iloc[-1:].values

        # Scale if scaler exists
        if scaler is not None:
            latest = scaler.transform(latest)

        # Make prediction
        prediction = model.predict(latest)[0]

        result = {
            "model_name": model_name,
            "prediction": int(prediction),
            "prediction_label": "UP" if prediction == 1 else "DOWN",
            "target": metadata["target"],
            "timestamp": datetime.now().isoformat(),
        }

        # Add probability if available
        if return_proba and hasattr(model, "predict_proba"):
            proba = model.predict_proba(latest)[0]
            result["confidence"] = float(max(proba))
            result["probability_up"] = float(proba[1]) if len(proba) > 1 else float(proba[0])
            result["probability_down"] = float(proba[0]) if len(proba) > 1 else 1 - float(proba[0])

        return result

    def predict_multiple_symbols(
        self,
        model_name: str,
        symbol_data: dict[str, pd.DataFrame],
    ) -> dict[str, dict]:
        """
        Make predictions for multiple symbols.

        Args:
            model_name: Model to use
            symbol_data: Dict mapping symbol -> OHLCV DataFrame

        Returns:
            Dict mapping symbol -> prediction result
        """
        results = {}
        for symbol, df in symbol_data.items():
            try:
                results[symbol] = self.predict(model_name, df)
            except Exception as e:
                logger.warning(f"Prediction failed for {symbol}: {e}")
                results[symbol] = {"error": str(e)}

        return results

    def ensemble_predict(
        self,
        model_names: list[str],
        df: pd.DataFrame,
        weights: Optional[list[float]] = None,
    ) -> dict:
        """
        Make ensemble prediction using multiple models.

        Args:
            model_names: List of model names to ensemble
            df: OHLCV DataFrame
            weights: Optional weights for each model (defaults to equal)

        Returns:
            Ensemble prediction with confidence
        """
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)

        predictions = []
        probabilities = []

        for model_name, weight in zip(model_names, weights):
            try:
                pred = self.predict(model_name, df)
                predictions.append(pred["prediction"] * weight)
                if "probability_up" in pred:
                    probabilities.append(pred["probability_up"] * weight)
            except Exception as e:
                logger.warning(f"Ensemble member {model_name} failed: {e}")

        if not predictions:
            raise ValueError("All ensemble models failed")

        # Weighted vote
        weighted_prediction = sum(predictions)
        final_prediction = 1 if weighted_prediction > 0.5 else 0

        result = {
            "prediction": final_prediction,
            "prediction_label": "UP" if final_prediction == 1 else "DOWN",
            "ensemble_score": weighted_prediction,
            "models_used": len(predictions),
            "timestamp": datetime.now().isoformat(),
        }

        if probabilities:
            result["probability_up"] = sum(probabilities)
            result["confidence"] = max(result["probability_up"], 1 - result["probability_up"])

        return result

    def get_signal_strength(self, prediction: dict) -> str:
        """
        Convert prediction to signal strength.

        Returns: "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
        """
        if "probability_up" not in prediction:
            return "NEUTRAL"

        prob_up = prediction["probability_up"]

        if prob_up >= 0.7:
            return "STRONG_BUY"
        elif prob_up >= 0.55:
            return "BUY"
        elif prob_up <= 0.3:
            return "STRONG_SELL"
        elif prob_up <= 0.45:
            return "SELL"
        else:
            return "NEUTRAL"

    def generate_trading_signal(
        self,
        model_name: str,
        df: pd.DataFrame,
        symbol: str,
    ) -> dict:
        """
        Generate a complete trading signal with context.

        Returns:
            Trading signal with direction, strength, and context
        """
        prediction = self.predict(model_name, df)
        signal_strength = self.get_signal_strength(prediction)

        # Get recent performance for context
        df_with_features = self.feature_engine.generate_features(df, include_targets=False)
        latest = df_with_features.iloc[-1]

        signal = {
            "symbol": symbol,
            "signal": signal_strength,
            "direction": prediction["prediction_label"],
            "confidence": prediction.get("confidence", 0.5),
            "probability_up": prediction.get("probability_up", 0.5),
            "context": {
                "rsi": float(latest.get("rsi_14", 50)),
                "trend": "BULLISH" if latest.get("price_sma50_ratio", 1) > 1 else "BEARISH",
                "volatility": "HIGH" if latest.get("hvol_20", 0.2) > 0.3 else "NORMAL",
                "momentum": "POSITIVE" if latest.get("roc_5", 0) > 0 else "NEGATIVE",
            },
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        return signal

    def backtest_predictions(
        self,
        model_name: str,
        df: pd.DataFrame,
        start_idx: int = 200,
    ) -> dict:
        """
        Backtest model predictions on historical data.

        Args:
            model_name: Model to test
            df: Full historical DataFrame
            start_idx: Index to start backtesting from

        Returns:
            Backtest results with accuracy metrics
        """
        if model_name not in self.loaded_models:
            self.load_model(model_name)

        model_data = self.loaded_models[model_name]
        metadata = model_data["metadata"]
        target = metadata["target"]

        # Generate features with targets
        df_full = self.feature_engine.generate_features(df, include_targets=True)

        predictions = []
        actuals = []

        # Walk-forward backtest
        for i in range(start_idx, len(df_full) - 5):  # Leave room for target
            test_df = df_full.iloc[:i+1].copy()
            try:
                pred = self.predict(model_name, test_df, return_proba=True)
                actual = df_full.iloc[i][target]

                if pd.notna(actual):
                    predictions.append(pred["prediction"])
                    actuals.append(int(actual))
            except Exception:
                continue

        if not predictions:
            return {"error": "No valid predictions"}

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        accuracy = (predictions == actuals).mean()
        precision = (predictions[predictions == 1] == actuals[predictions == 1]).mean() if (predictions == 1).any() else 0
        recall = (predictions[actuals == 1] == actuals[actuals == 1]).mean() if (actuals == 1).any() else 0

        return {
            "model_name": model_name,
            "total_predictions": len(predictions),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "predictions_up": int((predictions == 1).sum()),
            "predictions_down": int((predictions == 0).sum()),
            "actual_up": int((actuals == 1).sum()),
            "actual_down": int((actuals == 0).sum()),
        }

    def list_available_models(self) -> list[dict]:
        """List all available models"""
        models = []
        if not self.models_dir.exists():
            return models

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                versions = sorted(model_dir.iterdir(), reverse=True)
                if versions:
                    latest = versions[0]
                    metadata_path = latest / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        models.append({
                            "name": model_dir.name,
                            "latest_version": latest.name,
                            "target": metadata.get("target"),
                            "model_type": metadata.get("model_type"),
                            "metrics": metadata.get("metrics", {}),
                        })

        return models
