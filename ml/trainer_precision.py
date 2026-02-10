"""
Model Trainer for dPolaris ML.

Precision-focused training with:
- Walk-forward validation by default
- Standardized classification/regression/trading metrics
- Probability calibration metadata for confidence quality
- Slippage + transaction-cost-aware backtest scoring
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, Any
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import FeatureEngine
from .evaluation import (
    apply_probability_calibration,
    compute_classification_metrics,
    compute_primary_score,
    compute_regression_metrics,
    compute_trading_metrics,
    fit_platt_calibration,
)
from .precision_config import PrecisionConfig, load_precision_config
from .validation import set_global_seed, validate_no_lookahead_features

logger = logging.getLogger("dpolaris.ml.trainer")

try:
    from registry.model_registry import ModelRegistry
except Exception:  # pragma: no cover - keep trainer functional if registry import fails
    ModelRegistry = None

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

    Precision defaults:
    - Validation method: walk-forward
    - Primary score: Sharpe with drawdown cap
    - Probability confidence: Platt calibration from OOF predictions
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: str = "auto",
        precision_config_path: Optional[Path | str] = None,
    ):
        self.models_dir = models_dir or Path("~/dpolaris_data/models").expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.feature_engine = FeatureEngine()
        self.scaler: Optional[StandardScaler] = None
        self.precision_config: PrecisionConfig = load_precision_config(precision_config_path)
        self.model_registry = None
        if ModelRegistry is not None:
            try:
                self.model_registry = ModelRegistry(root_dir=self.models_dir / "_registry")
            except Exception as exc:
                logger.warning("Model registry init failed: %s", exc)

        set_global_seed(self.precision_config.random_seed)

    def _get_feature_frame(
        self,
        df: pd.DataFrame,
        target_horizon: int,
        include_targets: bool,
    ) -> pd.DataFrame:
        return self.feature_engine.generate_features(
            df,
            include_targets=include_targets,
            target_horizon=target_horizon,
        )

    def _extract_xy(
        self,
        feature_df: pd.DataFrame,
        target_col: str,
        regression_col: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], list[str], pd.DataFrame]:
        feature_cols = self.feature_engine.get_feature_names()
        if not feature_cols:
            raise ValueError("Feature list is empty. Run feature generation before training.")

        if target_col not in feature_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        required_cols = [target_col]
        if regression_col and regression_col in feature_df.columns:
            required_cols.append(regression_col)

        clean_df = feature_df.dropna(subset=required_cols).copy()
        X = clean_df[feature_cols].values
        y = clean_df[target_col].astype(int).values

        y_reg: Optional[np.ndarray] = None
        if regression_col and regression_col in clean_df.columns:
            y_reg = clean_df[regression_col].astype(float).values

        return X, y, y_reg, feature_cols, clean_df

    @staticmethod
    def _build_data_window(clean_df: pd.DataFrame, target_horizon: int) -> dict[str, Any]:
        ts_col = None
        for candidate in ("timestamp", "date", "datetime"):
            if candidate in clean_df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            return {
                "rows": int(len(clean_df)),
                "target_horizon_days": int(target_horizon),
            }

        ts = pd.to_datetime(clean_df[ts_col], utc=True, errors="coerce").dropna()
        if ts.empty:
            return {
                "rows": int(len(clean_df)),
                "target_horizon_days": int(target_horizon),
            }

        return {
            "rows": int(len(clean_df)),
            "start": ts.min().isoformat(),
            "end": ts.max().isoformat(),
            "target_horizon_days": int(target_horizon),
        }

    @staticmethod
    def _predict_probabilities(model, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1].astype(float)
            return proba.ravel().astype(float)

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-scores))

        return np.where(np.asarray(y_pred).astype(int) == 1, 0.75, 0.25).astype(float)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> tuple:
        """
        Prepare holdout split (kept for compatibility).
        """
        feature_cols = self.feature_engine.get_feature_names()
        if not feature_cols:
            df = self.feature_engine.generate_features(df)
            feature_cols = self.feature_engine.get_feature_names()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        df = df.dropna(subset=[target_col])
        X = df[feature_cols].values
        y = df[target_col].values

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info("Data prepared: %d train, %d test samples", len(X_train), len(X_test))
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
        Train one model instance.
        """
        logger.info("Training %s model...", model_type)

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
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            model = RandomForestClassifier(random_state=self.precision_config.random_seed, n_jobs=1)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=1)
            grid.fit(X_train, y_train)
            logger.info("Best params: %s", grid.best_params_)
            return grid.best_estimator_

        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 20),
            random_state=self.precision_config.random_seed,
            n_jobs=1,
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
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
            }
            model = GradientBoostingClassifier(random_state=self.precision_config.random_seed)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=1)
            grid.fit(X_train, y_train)
            logger.info("Best params: %s", grid.best_params_)
            return grid.best_estimator_

        model = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=self.precision_config.random_seed,
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
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
            model = xgb.XGBClassifier(
                random_state=self.precision_config.random_seed,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=1,
            )
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=1)
            grid.fit(X_train, y_train)
            logger.info("Best params: %s", grid.best_params_)
            return grid.best_estimator_

        model = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=self.precision_config.random_seed,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=1,
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
        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 63],
            }
            model = lgb.LGBMClassifier(random_state=self.precision_config.random_seed, verbose=-1, n_jobs=1)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=1)
            grid.fit(X_train, y_train)
            logger.info("Best params: %s", grid.best_params_)
            return grid.best_estimator_

        model = lgb.LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=self.precision_config.random_seed,
            verbose=-1,
            n_jobs=1,
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
        if tune:
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
            }
            model = LogisticRegression(random_state=self.precision_config.random_seed, max_iter=1000)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring="f1", n_jobs=1)
            grid.fit(X_train, y_train)
            logger.info("Best params: %s", grid.best_params_)
            return grid.best_estimator_

        model = LogisticRegression(
            C=kwargs.get("C", 1.0),
            random_state=self.precision_config.random_seed,
            max_iter=1000,
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """
        Standardized evaluation for compatibility path.
        """
        y_pred = model.predict(X_test)
        y_proba = self._predict_probabilities(model, X_test, y_pred)
        metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            reliability_bins=self.precision_config.metrics.reliability_bins,
        )
        logger.info("Model evaluation: accuracy=%.3f, f1=%.3f", metrics["accuracy"], metrics["f1"])
        return metrics

    def get_feature_importance(
        self,
        model,
        feature_names: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return []

        indices = np.argsort(importances)[::-1]
        return [(feature_names[i], float(importances[i])) for i in indices[:top_n]]

    def save_model(
        self,
        model,
        model_name: str,
        model_type: str,
        target: str,
        feature_names: list[str],
        metrics: dict,
        version: Optional[str] = None,
        probability_calibration: Optional[dict[str, float]] = None,
        data_window: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Save model artifacts and metadata.
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        if self.scaler is not None:
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "version": version,
            "target": target,
            "feature_names": feature_names,
            "metrics": metrics,
            "probability_calibration": probability_calibration,
            "precision_config": self.precision_config.to_dict(),
            "data_window": data_window or {},
            "created_at": datetime.now().isoformat(),
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if self.model_registry is not None:
            try:
                self.model_registry.register_model(
                    model_name=model_name,
                    model_path=model_path,
                    config=metadata.get("precision_config"),
                    metrics=metrics,
                    data_window=data_window or {},
                    framework=model_type,
                    notes="Registered by ModelTrainer.save_model",
                    training_trigger="scheduled",
                    version=version,
                    status="active",
                )
            except Exception as exc:
                logger.warning("Model registry write failed: %s", exc)

        logger.info("Model saved to %s", model_dir)
        return model_path

    def load_model(self, model_name: str, version: Optional[str] = None) -> tuple:
        """
        Load model from disk.
        """
        model_base = self.models_dir / model_name

        if version is None:
            versions = sorted(model_base.iterdir(), reverse=True)
            if not versions:
                raise FileNotFoundError(f"No versions found for model {model_name}")
            version_dir = versions[0]
        else:
            version_dir = model_base / version

        model_path = version_dir / "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        scaler_path = version_dir / "scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info("Model loaded from %s", version_dir)
        return model, scaler, metadata

    def _run_walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_reg: Optional[np.ndarray],
        model_type: ModelType,
    ) -> dict[str, Any]:
        cfg = self.precision_config
        n_splits = max(2, int(cfg.validation.walk_forward_splits))

        if len(X) <= n_splits + 1:
            raise ValueError(
                f"Not enough samples ({len(X)}) for walk-forward validation with {n_splits} splits"
            )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        oof_pred = np.full(len(y), np.nan)
        oof_proba = np.full(len(y), np.nan)
        oof_reg = np.full(len(y), np.nan) if y_reg is not None else None

        fold_summaries: list[dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X[train_idx])
            X_test = fold_scaler.transform(X[test_idx])

            # Keep fold models deterministic and stable; no nested tuning inside folds.
            fold_model = self.train_model(
                X_train,
                y[train_idx],
                model_type=model_type,
                tune_hyperparams=False,
            )

            fold_pred = fold_model.predict(X_test)
            fold_proba = self._predict_probabilities(fold_model, X_test, fold_pred)

            oof_pred[test_idx] = fold_pred
            oof_proba[test_idx] = fold_proba
            if oof_reg is not None and y_reg is not None:
                oof_reg[test_idx] = y_reg[test_idx]

            fold_metrics = compute_classification_metrics(
                y_true=y[test_idx],
                y_pred=fold_pred,
                y_proba=fold_proba,
                reliability_bins=cfg.metrics.reliability_bins,
            )
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "classification": {
                        "accuracy": fold_metrics.get("accuracy"),
                        "precision": fold_metrics.get("precision"),
                        "recall": fold_metrics.get("recall"),
                        "f1": fold_metrics.get("f1"),
                        "roc_auc": fold_metrics.get("roc_auc"),
                        "brier_score": fold_metrics.get("brier_score"),
                        "calibration_error": fold_metrics.get("calibration_error"),
                    },
                }
            )

        valid_mask = ~np.isnan(oof_pred)
        if valid_mask.sum() == 0:
            raise ValueError("Walk-forward produced no out-of-fold predictions")

        y_oof = y[valid_mask].astype(int)
        proba_oof_raw = oof_proba[valid_mask].astype(float)

        probability_calibration = None
        if cfg.calibration.enabled:
            probability_calibration = fit_platt_calibration(proba_oof_raw, y_oof)

        proba_oof = apply_probability_calibration(proba_oof_raw, probability_calibration)
        pred_oof_calibrated = (proba_oof >= 0.5).astype(int)

        classification = compute_classification_metrics(
            y_true=y_oof,
            y_pred=pred_oof_calibrated,
            y_proba=proba_oof,
            reliability_bins=cfg.metrics.reliability_bins,
        )

        if y_reg is not None and oof_reg is not None:
            realized_returns = oof_reg[valid_mask].astype(float)
            return_scale = float(np.nanstd(realized_returns))
            if return_scale < 1e-6:
                return_scale = 0.01
            predicted_returns = ((proba_oof - 0.5) * 2.0) * return_scale
            regression = compute_regression_metrics(
                y_true=realized_returns,
                y_pred=predicted_returns,
                thresholds=cfg.metrics.regression_hit_rate_thresholds,
            )
            trading = compute_trading_metrics(
                probability_up=proba_oof,
                realized_returns=realized_returns,
                long_threshold=cfg.backtest.long_threshold,
                short_threshold=cfg.backtest.short_threshold,
                transaction_cost_bps=cfg.backtest.transaction_cost_bps,
                slippage_bps=cfg.backtest.slippage_bps,
                annualization_periods=cfg.backtest.annualization_periods,
            )
        else:
            regression = compute_regression_metrics([], [])
            trading = compute_trading_metrics(
                probability_up=[],
                realized_returns=[],
                long_threshold=cfg.backtest.long_threshold,
                short_threshold=cfg.backtest.short_threshold,
                transaction_cost_bps=cfg.backtest.transaction_cost_bps,
                slippage_bps=cfg.backtest.slippage_bps,
                annualization_periods=cfg.backtest.annualization_periods,
            )

        primary_score = compute_primary_score(
            classification_metrics=classification,
            trading_metrics=trading,
            objective=cfg.primary_score.objective,
            metric=cfg.primary_score.metric,
            fallback_metric=cfg.primary_score.fallback_metric,
            max_drawdown=cfg.primary_score.max_drawdown,
            min_trades=cfg.primary_score.min_trades,
        )

        return {
            "method": "walk_forward",
            "folds": fold_summaries,
            "oof_samples": int(valid_mask.sum()),
            "classification": classification,
            "regression": regression,
            "trading": trading,
            "primary_score": float(primary_score),
            "probability_calibration": probability_calibration,
        }

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
        Full training pipeline with walk-forward validation as default.
        """
        set_global_seed(self.precision_config.random_seed)

        target = target or self.precision_config.target.classification
        target_horizon = int(target_horizon or self.precision_config.target.horizon_days)
        regression_target = self.precision_config.target.regression

        logger.info("Generating features...")

        if self.precision_config.validation.enforce_no_lookahead:
            leak_check = validate_no_lookahead_features(
                feature_engine=self.feature_engine,
                raw_df=df,
                sample_count=self.precision_config.validation.no_lookahead_sample_count,
            )
            if not leak_check["passed"]:
                raise ValueError(
                    "No-lookahead validation failed. "
                    f"Violations: {leak_check['violations'][:3]}"
                )

        feature_df = self._get_feature_frame(
            df,
            target_horizon=target_horizon,
            include_targets=True,
        )
        X, y, y_reg, feature_names, clean_df = self._extract_xy(
            feature_df,
            target_col=target,
            regression_col=regression_target,
        )
        data_window = self._build_data_window(clean_df, target_horizon=target_horizon)

        if len(X) < int(self.precision_config.validation.min_samples):
            raise ValueError(
                f"Need at least {self.precision_config.validation.min_samples} samples after feature engineering, got {len(X)}"
            )

        validation_method = self.precision_config.validation.method.strip().lower()
        if validation_method != "walk_forward":
            logger.warning(
                "Unsupported validation method '%s'. Falling back to walk_forward.",
                validation_method,
            )

        validation_report = self._run_walk_forward_validation(
            X=X,
            y=y,
            y_reg=y_reg,
            model_type=model_type,
        )

        # Fit final production model on full available history.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        model = self.train_model(
            X_train=X_scaled,
            y_train=y,
            model_type=model_type,
            tune_hyperparams=tune_hyperparams,
        )

        feature_importance = self.get_feature_importance(model, feature_names)

        classification = validation_report["classification"]
        regression = validation_report["regression"]
        trading = validation_report["trading"]

        metrics = {
            # Backward-compatible keys
            "accuracy": classification.get("accuracy"),
            "precision": classification.get("precision"),
            "recall": classification.get("recall"),
            "f1": classification.get("f1"),
            "roc_auc": classification.get("roc_auc"),
            # Full metric suites
            "classification": classification,
            "regression": regression,
            "trading": trading,
            "primary_score": validation_report["primary_score"],
            "validation": {
                "method": validation_report["method"],
                "fold_count": len(validation_report["folds"]),
                "oof_samples": validation_report["oof_samples"],
                "folds": validation_report["folds"],
            },
            "feature_importance": feature_importance,
            "backtest_assumptions": self.precision_config.backtest.__dict__,
        }

        model_path = self.save_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            target=target,
            feature_names=feature_names,
            metrics=metrics,
            probability_calibration=validation_report.get("probability_calibration"),
            data_window=data_window,
        )

        logger.info(
            "Training complete: model=%s type=%s primary_score=%.4f",
            model_name,
            model_type,
            metrics["primary_score"],
        )

        return {
            "model_name": model_name,
            "model_type": model_type,
            "model_path": str(model_path),
            "target": target,
            "metrics": metrics,
            "feature_importance": feature_importance[:10],
        }
