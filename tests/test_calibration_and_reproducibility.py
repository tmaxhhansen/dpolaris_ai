from __future__ import annotations

import pytest

from ml.predictor import Predictor
from ml.trainer import ModelTrainer


def test_predictor_returns_calibrated_confidence(synthetic_df, tmp_path):
    models_dir = tmp_path / "models"
    trainer = ModelTrainer(models_dir=models_dir)

    trainer.train_full_pipeline(
        synthetic_df,
        model_name="CAL_TEST",
        model_type="logistic",
        target="target_direction",
        target_horizon=5,
        tune_hyperparams=False,
    )

    predictor = Predictor(models_dir=models_dir)
    prediction = predictor.predict("CAL_TEST", synthetic_df, return_proba=True)

    assert 0.0 <= prediction["probability_up"] <= 1.0
    assert 0.0 <= prediction["probability_down"] <= 1.0
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert "raw_probability_up" in prediction

    model_meta = predictor.loaded_models["CAL_TEST"]["metadata"]
    calibration_meta = model_meta.get("probability_calibration")
    assert calibration_meta is not None
    assert calibration_meta.get("method") == "platt"

    classification = model_meta["metrics"]["classification"]
    assert classification.get("brier_score") is not None
    assert len(classification.get("reliability_curve", [])) > 0


def test_training_is_reproducible_with_fixed_seed(synthetic_df, tmp_path):
    trainer_a = ModelTrainer(models_dir=tmp_path / "models_a")
    trainer_b = ModelTrainer(models_dir=tmp_path / "models_b")

    result_a = trainer_a.train_full_pipeline(
        synthetic_df,
        model_name="REPRO_TEST",
        model_type="logistic",
        target="target_direction",
        target_horizon=5,
        tune_hyperparams=False,
    )
    result_b = trainer_b.train_full_pipeline(
        synthetic_df,
        model_name="REPRO_TEST",
        model_type="logistic",
        target="target_direction",
        target_horizon=5,
        tune_hyperparams=False,
    )

    assert result_a["metrics"]["accuracy"] == pytest.approx(result_b["metrics"]["accuracy"], rel=1e-9)
    assert result_a["metrics"]["f1"] == pytest.approx(result_b["metrics"]["f1"], rel=1e-9)
    assert result_a["metrics"]["primary_score"] == pytest.approx(
        result_b["metrics"]["primary_score"],
        rel=1e-9,
    )
