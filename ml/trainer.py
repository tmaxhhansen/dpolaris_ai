"""
Model Trainer for dPolaris ML

Trains and saves ML models for price prediction, volatility forecasting, etc.
Supports multiple model types and automatic hyperparameter tuning.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from .features import FeatureEngine

logger = logging.getLogger("dpolaris.ml.trainer")

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


ModelType = Literal["random_forest", "gradient_boosting", "xgboost", "lightgbm", "logistic"]


class ModelTrainer:
    """
    Train ML models for trading predictions.

    Supports:
    - Direction prediction (up/down)
    - Magnitude prediction (strong down to strong up)
    - Volatility regime prediction
    - Custom targets
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: str = "auto",
    ):
        self.models_dir = models_dir or Path("~/dpolaris_data/models").expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.feature_engine = FeatureEngine()
        self.scaler: Optional[StandardScaler] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> tuple:
        """
        Prepare data for training.

        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            test_size: Fraction for test set
            scale_features: Whether to scale features

        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Get feature columns
        feature_cols = self.feature_engine.get_feature_names()
        if not feature_cols:
            # Generate features if not already done
            df = self.feature_engine.generate_features(df)
            feature_cols = self.feature_engine.get_feature_names()

        # Ensure target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Remove rows with NaN target
        df = df.dropna(subset=[target_col])

        X = df[feature_cols].values
        y = df[target_col].values

        # Time-series split (don't shuffle for time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test, feature_cols

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: ModelType = "random_forest",
        tune_hyperparams: bool = True,
        **kwargs,
    ):
        """
        Train a model.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to train
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")

        if model_type == "random_forest":
            model = self._train_random_forest(X_train, y_train, tune_hyperparams, **kwargs)
        elif model_type == "gradient_boosting":
            model = self._train_gradient_boosting(X_train, y_train, tune_hyperparams, **kwargs)
        elif model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            model = self._train_xgboost(X_train, y_train, tune_hyperparams, **kwargs)
        elif model_type == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
            model = self._train_lightgbm(X_train, y_train, tune_hyperparams, **kwargs)
        elif model_type == "logistic":
            model = self._train_logistic(X_train, y_train, tune_hyperparams, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = True,
        **kwargs,
    ):
        """Train Random Forest classifier"""
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            logger.info(f"Best params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 20),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            return model

    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = True,
        **kwargs,
    ):
        """Train Gradient Boosting classifier"""
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
            }
            model = GradientBoostingClassifier(random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            logger.info(f"Best params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            model = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            )
            model.fit(X_train, y_train)
            return model

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = True,
        **kwargs,
    ):
        """Train XGBoost classifier"""
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
            model = xgb.XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            logger.info(f"Best params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)
            return model

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = True,
        **kwargs,
    ):
        """Train LightGBM classifier"""
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 63],
            }
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            logger.info(f"Best params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            model = lgb.LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            return model

    def _train_logistic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = True,
        **kwargs,
    ):
        """Train Logistic Regression"""
        if tune:
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            logger.info(f"Best params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            model = LogisticRegression(
                C=kwargs.get("C", 1.0),
                random_state=42,
                max_iter=1000,
            )
            model.fit(X_train, y_train)
            return model

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Evaluate model performance.

        Returns:
            Dictionary with metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                metrics["roc_auc"] = None

        # Add classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report

        logger.info(f"Model evaluation: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}")
        return metrics

    def get_feature_importance(
        self,
        model,
        feature_names: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Get feature importance from model"""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return []

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        top_features = [
            (feature_names[i], importances[i])
            for i in indices[:top_n]
        ]
        return top_features

    def save_model(
        self,
        model,
        model_name: str,
        model_type: str,
        target: str,
        feature_names: list[str],
        metrics: dict,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save model to disk.

        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model directory
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save scaler
        if self.scaler is not None:
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "version": version,
            "target": target,
            "feature_names": feature_names,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_dir}")
        return model_path

    def load_model(self, model_name: str, version: Optional[str] = None) -> tuple:
        """
        Load model from disk.

        Args:
            model_name: Name of the model
            version: Specific version, or None for latest

        Returns:
            (model, scaler, metadata)
        """
        model_base = self.models_dir / model_name

        if version is None:
            # Get latest version
            versions = sorted(model_base.iterdir(), reverse=True)
            if not versions:
                raise FileNotFoundError(f"No versions found for model {model_name}")
            version_dir = versions[0]
        else:
            version_dir = model_base / version

        # Load model
        model_path = version_dir / "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load scaler
        scaler_path = version_dir / "scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Model loaded from {version_dir}")
        return model, scaler, metadata

    def train_full_pipeline(
        self,
        df: pd.DataFrame,
        model_name: str,
        model_type: ModelType = "xgboost" if HAS_XGBOOST else "random_forest",
        target: str = "target_direction",
        target_horizon: int = 5,
        tune_hyperparams: bool = True,
    ) -> dict:
        """
        Full training pipeline: feature engineering -> training -> evaluation -> save.

        Args:
            df: Raw OHLCV DataFrame
            model_name: Name for the model
            model_type: Type of model to train
            target: Target variable
            target_horizon: Days ahead for prediction
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Dictionary with model info and metrics
        """
        # Generate features
        logger.info("Generating features...")
        df = self.feature_engine.generate_features(
            df,
            include_targets=True,
            target_horizon=target_horizon,
        )

        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            df, target_col=target
        )

        # Train model
        model = self.train_model(
            X_train, y_train,
            model_type=model_type,
            tune_hyperparams=tune_hyperparams,
        )

        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)

        # Get feature importance
        feature_importance = self.get_feature_importance(model, feature_names)
        metrics["feature_importance"] = feature_importance

        # Save model
        model_path = self.save_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            target=target,
            feature_names=feature_names,
            metrics=metrics,
        )

        return {
            "model_name": model_name,
            "model_type": model_type,
            "model_path": str(model_path),
            "target": target,
            "metrics": metrics,
            "feature_importance": feature_importance[:10],
        }


# Use precision-enhanced trainer implementation by default while keeping module path stable.
from .trainer_precision import ModelTrainer as _PrecisionModelTrainer

ModelTrainer = _PrecisionModelTrainer
