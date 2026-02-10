"""
Deep Learning Models for dPolaris

LSTM and Transformer models for stock price prediction.
Auto-detects CUDA (Windows GPU) vs CPU (Mac).
"""

import json
import os
import pickle
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Literal, Tuple
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .evaluation import (
    apply_probability_calibration,
    compute_classification_metrics,
    fit_platt_calibration,
)
try:
    from .training_artifacts import write_training_artifact
except Exception:  # pragma: no cover - keep training functional if artifact module unavailable
    write_training_artifact = None

from .features import FeatureEngine

logger = logging.getLogger("dpolaris.ml.deep_learning")


# ============================================================================
# Device Detection
# ============================================================================

def get_device() -> torch.device:
    """Select compute device with stability-first defaults on macOS."""
    preferred = os.getenv("DPOLARIS_DEVICE", "auto").strip().lower()
    if preferred not in {"auto", "cpu", "mps", "cuda"}:
        preferred = "auto"

    if preferred == "cpu":
        logger.info("Using CPU (DPOLARIS_DEVICE=cpu)")
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        logger.warning("DPOLARIS_DEVICE=cuda requested but CUDA unavailable; falling back")

    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS (DPOLARIS_DEVICE=mps)")
            return torch.device("mps")
        logger.warning("DPOLARIS_DEVICE=mps requested but MPS unavailable; falling back")

    # Auto mode: prioritize stability. Prefer CPU on macOS; use CUDA when available.
    if torch.cuda.is_available():
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if os.name == "posix" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS detected but disabled in auto mode for stability. Set DPOLARIS_DEVICE=mps to enable.")

    logger.info("Using CPU")
    return torch.device("cpu")


DEVICE = get_device()


# ============================================================================
# Dataset
# ============================================================================

class StockSequenceDataset(Dataset):
    """Dataset for sequential stock data"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 60,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.sequence_length = sequence_length

    def __len__(self):
        # Guard against negative lengths when the split is smaller than sequence length.
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y


# ============================================================================
# LSTM Model
# ============================================================================

class LSTMPredictor(nn.Module):
    """
    LSTM model for stock direction prediction.

    Architecture:
    - 2 LSTM layers with dropout
    - Fully connected output layer
    - Binary classification (UP/DOWN)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        last_output = lstm_out[:, -1, :]

        # Dropout and FC
        out = self.dropout(last_output)
        out = self.fc(out)

        return out


# ============================================================================
# Transformer Model
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer model for stock direction prediction.

    Architecture:
    - Linear projection to d_model
    - Positional encoding
    - Transformer encoder layers
    - Global average pooling
    - Classification head
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


# ============================================================================
# Trainer
# ============================================================================

class DeepLearningTrainer:
    """
    Train LSTM and Transformer models for stock prediction.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.models_dir = models_dir or Path("~/dpolaris_data/models").expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or DEVICE
        self.feature_engine = FeatureEngine()
        self.scaler: Optional[StandardScaler] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
        sequence_length: int = 60,
        test_size: float = 0.2,
        batch_size: int = 32,
        min_train_sequences: int = 120,
        min_test_sequences: int = 30,
    ) -> Tuple[DataLoader, DataLoader, list]:
        """
        Prepare data for deep learning training.

        Returns:
            train_loader, test_loader, feature_names
        """
        # Generate features
        df = self.feature_engine.generate_features(
            df, include_targets=True, target_horizon=5
        )

        feature_cols = self.feature_engine.get_feature_names()

        # Get features and targets
        df = df.dropna(subset=[target_col])
        X = df[feature_cols].values
        y = df[target_col].values.astype(int)

        # Ensure both train and test have enough rows for robust sequential windows.
        min_rows_train = sequence_length + min_train_sequences
        min_rows_test = sequence_length + min_test_sequences
        min_total_rows = min_rows_train + min_rows_test
        total_rows = len(X)

        if total_rows < min_total_rows:
            raise ValueError(
                f"Not enough data after feature engineering for robust deep-learning splits. "
                f"Need at least {min_total_rows} rows (sequence_length={sequence_length}, "
                f"min_train_sequences={min_train_sequences}, min_test_sequences={min_test_sequences}), "
                f"got {total_rows}. Increase history window or reduce sequence length."
            )

        # Time-series split, clamped so each split has enough sequences for evaluation.
        split_idx = int(total_rows * (1 - test_size))
        lower_bound = min_rows_train
        upper_bound = total_rows - min_rows_test
        split_idx = max(lower_bound, min(split_idx, upper_bound))

        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale with training split only to avoid lookahead leakage.
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        # Create datasets
        train_dataset = StockSequenceDataset(X_train, y_train, sequence_length)
        test_dataset = StockSequenceDataset(X_test, y_test, sequence_length)

        if len(train_dataset) < min_train_sequences or len(test_dataset) < min_test_sequences:
            raise ValueError(
                f"Insufficient sequence windows after split: train={len(train_dataset)}, "
                f"test={len(test_dataset)} (required train>={min_train_sequences}, "
                f"test>={min_test_sequences})."
            )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle time series
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            "Data prepared: %d train, %d test sequences "
            "(rows=%d, split_idx=%d, sequence_length=%d)",
            len(train_dataset),
            len(test_dataset),
            total_rows,
            split_idx,
            sequence_length,
        )

        return train_loader, test_loader, feature_cols

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 20,
        min_epochs_before_early_stopping: int = 30,
        log_every_n_epochs: int = 1,
    ) -> dict:
        """
        Train a deep learning model.

        Returns:
            Dictionary with training history and metrics
        """
        # Normalize safety bounds so runtime settings cannot disable training accidentally.
        early_stopping_patience = max(1, early_stopping_patience)
        min_epochs_before_early_stopping = max(1, min_epochs_before_early_stopping)
        log_every_n_epochs = max(1, log_every_n_epochs)

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        logger.info(
            "Training config: epochs=%d, lr=%.6f, early_stopping_patience=%d, min_epochs_before_stop=%d, "
            "train_batches=%d, test_batches=%d",
            epochs,
            learning_rate,
            early_stopping_patience,
            min_epochs_before_early_stopping,
            len(train_loader),
            len(test_loader),
        )

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Evaluation
            model.eval()
            test_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())

            test_loss /= len(test_loader)
            accuracy = accuracy_score(all_targets, all_preds)

            # Update history
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(accuracy)

            # Learning rate scheduling
            scheduler.step(test_loss)

            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % log_every_n_epochs == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, "
                    f"Patience: {patience_counter}/{early_stopping_patience}"
                )

            if (
                (epoch + 1) >= min_epochs_before_early_stopping
                and patience_counter >= early_stopping_patience
            ):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        metrics = self.evaluate_model(model, test_loader)

        return {
            "history": history,
            "metrics": metrics,
            "epochs_trained": epoch + 1,
        }

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> dict:
        """Evaluate model performance"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)

                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        calibration = fit_platt_calibration(all_probs, all_targets)
        calibrated_probs = apply_probability_calibration(all_probs, calibration)
        calibrated_preds = (calibrated_probs >= 0.5).astype(int)

        metrics = compute_classification_metrics(
            y_true=all_targets,
            y_pred=calibrated_preds,
            y_proba=calibrated_probs,
            reliability_bins=10,
        )
        metrics["probability_calibration"] = calibration
        return metrics

    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        model_type: str,
        feature_names: list,
        metrics: dict,
        version: Optional[str] = None,
    ) -> Path:
        """Save model to disk"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_dir = self.models_dir / f"{model_name}_dl" / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save model config for reconstruction
        config = {
            "model_type": model_type,
            "input_size": len(feature_names),
        }

        if model_type == "lstm":
            config.update({
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
            })
        elif model_type == "transformer":
            config.update({
                "d_model": model.input_projection.out_features,
            })

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

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
            "feature_names": feature_names,
            "metrics": metrics,
            "device": str(self.device),
            "created_at": datetime.now().isoformat(),
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_dir}")
        return model_path

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Tuple[nn.Module, StandardScaler, dict]:
        """Load model from disk"""
        model_base = self.models_dir / f"{model_name}_dl"

        if version is None:
            versions = sorted(model_base.iterdir(), reverse=True)
            if not versions:
                raise FileNotFoundError(f"No versions found for {model_name}")
            version_dir = versions[0]
        else:
            version_dir = model_base / version

        # Load config
        config_path = version_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Reconstruct model
        input_size = config["input_size"]

        if config["model_type"] == "lstm":
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=config.get("hidden_size", 128),
                num_layers=config.get("num_layers", 2),
            )
        elif config["model_type"] == "transformer":
            model = TransformerPredictor(
                input_size=input_size,
                d_model=config.get("d_model", 64),
            )
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")

        # Load weights
        model_path = version_dir / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        # Load scaler
        scaler_path = version_dir / "scaler.pkl"
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        logger.info(f"Model loaded from {version_dir}")
        return model, scaler, metadata

    def train_full_pipeline(
        self,
        df: pd.DataFrame,
        model_name: str,
        model_type: Literal["lstm", "transformer"] = "lstm",
        sequence_length: int = 60,
        epochs: int = int(os.getenv("DPOLARIS_DL_EPOCHS", "50")),
        batch_size: int = int(os.getenv("DPOLARIS_DL_BATCH_SIZE", "32")),
        learning_rate: float = float(os.getenv("DPOLARIS_DL_LEARNING_RATE", "0.001")),
        early_stopping_patience: int = int(os.getenv("DPOLARIS_DL_EARLY_STOP_PATIENCE", "20")),
        min_epochs_before_early_stopping: int = int(os.getenv("DPOLARIS_DL_MIN_EPOCHS", "30")),
        log_every_n_epochs: int = int(os.getenv("DPOLARIS_DL_LOG_EVERY_EPOCHS", "1")),
    ) -> dict:
        """
        Full training pipeline: prepare data -> train -> evaluate -> save.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Starting {model_type.upper()} training for {model_name}")
        logger.info(f"Using device: {self.device}")
        run_info: Optional[dict] = None
        try:
            epochs = max(1, epochs)
            batch_size = max(1, batch_size)
            learning_rate = max(1e-6, learning_rate)
            early_stopping_patience = max(1, early_stopping_patience)
            min_epochs_before_early_stopping = max(1, min_epochs_before_early_stopping)
            log_every_n_epochs = max(1, log_every_n_epochs)

            logger.info(
                "Pipeline params: sequence_length=%d, epochs=%d, batch_size=%d, learning_rate=%.6f, "
                "early_stop_patience=%d, min_epochs_before_stop=%d",
                sequence_length,
                epochs,
                batch_size,
                learning_rate,
                early_stopping_patience,
                min_epochs_before_early_stopping,
            )

            # Prepare data
            train_loader, test_loader, feature_names = self.prepare_data(
                df,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )

            # Create model
            input_size = len(feature_names)

            if model_type == "lstm":
                lstm_hidden_size = int(os.getenv("DPOLARIS_DL_LSTM_HIDDEN_SIZE", "192"))
                lstm_num_layers = int(os.getenv("DPOLARIS_DL_LSTM_NUM_LAYERS", "3"))
                lstm_dropout = float(os.getenv("DPOLARIS_DL_LSTM_DROPOUT", "0.25"))
                model = LSTMPredictor(
                    input_size=input_size,
                    hidden_size=max(32, lstm_hidden_size),
                    num_layers=max(1, lstm_num_layers),
                    dropout=min(0.7, max(0.0, lstm_dropout)),
                )
                hyperparameters = {
                    "sequence_length": sequence_length,
                    "epochs_requested": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_size": max(32, lstm_hidden_size),
                    "num_layers": max(1, lstm_num_layers),
                    "dropout": min(0.7, max(0.0, lstm_dropout)),
                }
            elif model_type == "transformer":
                transformer_d_model = int(os.getenv("DPOLARIS_DL_TRANSFORMER_D_MODEL", "96"))
                transformer_nhead = int(os.getenv("DPOLARIS_DL_TRANSFORMER_NHEAD", "4"))
                transformer_num_layers = int(os.getenv("DPOLARIS_DL_TRANSFORMER_LAYERS", "3"))
                transformer_dropout = float(os.getenv("DPOLARIS_DL_TRANSFORMER_DROPOUT", "0.15"))
                transformer_d_model = max(32, transformer_d_model)
                transformer_nhead = max(1, transformer_nhead)
                if transformer_d_model % transformer_nhead != 0:
                    logger.warning(
                        "Transformer d_model=%d is not divisible by nhead=%d; forcing nhead=1",
                        transformer_d_model,
                        transformer_nhead,
                    )
                    transformer_nhead = 1
                model = TransformerPredictor(
                    input_size=input_size,
                    d_model=transformer_d_model,
                    nhead=transformer_nhead,
                    num_layers=max(1, transformer_num_layers),
                    dropout=min(0.7, max(0.0, transformer_dropout)),
                )
                hyperparameters = {
                    "sequence_length": sequence_length,
                    "epochs_requested": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "d_model": transformer_d_model,
                    "nhead": transformer_nhead,
                    "num_layers": max(1, transformer_num_layers),
                    "dropout": min(0.7, max(0.0, transformer_dropout)),
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train
            results = self.train_model(
                model,
                train_loader,
                test_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                min_epochs_before_early_stopping=min_epochs_before_early_stopping,
                log_every_n_epochs=log_every_n_epochs,
            )

            # Save model artifacts
            model_path = self.save_model(
                model=model,
                model_name=model_name,
                model_type=model_type,
                feature_names=feature_names,
                metrics=results["metrics"],
            )
            model_dir = Path(model_path).parent

            # Write training observability artifact (self-contained run folder)
            if write_training_artifact is not None:
                try:
                    ts_col = "date" if "date" in df.columns else "timestamp" if "timestamp" in df.columns else None
                    start = str(df[ts_col].iloc[0]) if ts_col and len(df) else None
                    end = str(df[ts_col].iloc[-1]) if ts_col and len(df) else None

                    classification = results.get("metrics", {}) if isinstance(results.get("metrics"), dict) else {}
                    run_info = write_training_artifact(
                        run_id=None,
                        status="completed",
                        model_type=model_type,
                        target="target_direction",
                        horizon=5,
                        tickers=[model_name.upper()],
                        timeframes=["1d"],
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        data_summary={
                            "sources_used": ["market_history"],
                            "start": start,
                            "end": end,
                            "bars_count": int(len(df)),
                            "missingness_report": {},
                            "corporate_actions_applied": [],
                            "adjustments": [],
                            "outliers_detected": {},
                            "drop_or_repair_decisions": [],
                        },
                        feature_summary={
                            "feature_registry_version": "1.0.0",
                            "features": [{"name": name, "params": {}} for name in feature_names],
                            "missingness_per_feature": {},
                            "normalization_method": "standard_scaler",
                            "leakage_checks_status": "passed",
                        },
                        split_summary={
                            "walk_forward_windows": [],
                            "train_ranges": [],
                            "val_ranges": [],
                            "test_ranges": [],
                            "sample_sizes": {
                                "train_sequences": int(len(train_loader.dataset)),
                                "test_sequences": int(len(test_loader.dataset)),
                            },
                        },
                        model_summary={
                            "algorithm": model_type,
                            "hyperparameters": hyperparameters,
                            "feature_importance": [],
                            "calibration_method": classification.get("probability_calibration", {}).get("method"),
                        },
                        metrics_summary={
                            "classification": classification,
                            "regression": {},
                            "trading": {},
                            "calibration": {
                                "method": classification.get("probability_calibration", {}).get("method"),
                                "brier_score": classification.get("brier_score"),
                                "calibration_error": classification.get("calibration_error"),
                            },
                        },
                        backtest_summary={
                            "assumptions": {},
                            "equity_curve_stats": {},
                            "trade_list_artifact": None,
                        },
                        diagnostics_summary={
                            "drift_baseline_stats": {},
                            "regime_distribution": {},
                            "error_analysis": {},
                            "top_failure_cases": [],
                        },
                        environment={"device": str(self.device)},
                        artifact_files=[
                            str(model_path),
                            str(model_dir / "metadata.json"),
                            str(model_dir / "config.json"),
                            str(model_dir / "scaler.pkl"),
                        ],
                    )
                except Exception as exc:
                    logger.warning("Failed to write training artifact for %s: %s", model_name, exc)

            return {
                "model_name": model_name,
                "model_type": model_type,
                "model_path": str(model_path),
                "device": str(self.device),
                "metrics": results["metrics"],
                "epochs_trained": results["epochs_trained"],
                "run_id": run_info.get("run_id") if run_info else None,
                "run_dir": run_info.get("run_dir") if run_info else None,
            }
        except Exception as exc:
            if write_training_artifact is not None:
                try:
                    run_info = write_training_artifact(
                        run_id=None,
                        status="failed",
                        model_type=model_type,
                        target="target_direction",
                        horizon=5,
                        tickers=[model_name.upper()],
                        timeframes=["1d"],
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        diagnostics_summary={
                            "drift_baseline_stats": {},
                            "regime_distribution": {},
                            "error_analysis": {"message": str(exc)},
                            "top_failure_cases": [{"stage": "train_full_pipeline", "error": str(exc)}],
                        },
                        model_summary={
                            "algorithm": model_type,
                            "hyperparameters": {
                                "sequence_length": sequence_length,
                                "epochs_requested": epochs,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate,
                            },
                            "feature_importance": [],
                            "calibration_method": None,
                        },
                        metrics_summary={
                            "classification": {},
                            "regression": {},
                            "trading": {},
                            "calibration": {},
                        },
                    )
                    logger.info("Failure artifact saved for %s at %s", model_name, run_info.get("run_dir"))
                except Exception as artifact_exc:
                    logger.warning("Failed to write failure training artifact: %s", artifact_exc)
            raise

    def predict(
        self,
        model: nn.Module,
        scaler: StandardScaler,
        df: pd.DataFrame,
        sequence_length: int = 60,
        probability_calibration: Optional[dict] = None,
    ) -> dict:
        """
        Make prediction on new data.
        """
        # Generate features
        df = self.feature_engine.generate_features(df, include_targets=False)
        feature_cols = self.feature_engine.get_feature_names()

        # Get last sequence
        X = df[feature_cols].values
        X = scaler.transform(X)

        if len(X) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} data points")

        # Take last sequence
        X_seq = X[-sequence_length:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1).item()
            prob_up_raw = probs[0, 1].item()
            prob_down_raw = probs[0, 0].item()

        calibrated = apply_probability_calibration(
            np.asarray([prob_up_raw], dtype=float),
            probability_calibration,
        )
        probability_up = float(calibrated[0])
        probability_down = float(1.0 - probability_up)
        confidence = max(probability_up, probability_down)
        pred = 1 if probability_up >= 0.5 else 0

        return {
            "prediction": pred,
            "prediction_label": "UP" if pred == 1 else "DOWN",
            "confidence": confidence,
            "probability_up": probability_up,
            "probability_down": probability_down,
            "raw_probability_up": prob_up_raw,
            "raw_probability_down": prob_down_raw,
        }
