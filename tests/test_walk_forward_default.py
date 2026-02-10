from __future__ import annotations

from ml.precision_config import load_precision_config
from ml.trainer import ModelTrainer


def test_precision_config_defaults_to_walk_forward():
    cfg = load_precision_config()
    assert cfg.validation.method == "walk_forward"


def test_training_uses_walk_forward_by_default(synthetic_df, tmp_path):
    models_dir = tmp_path / "models"
    trainer = ModelTrainer(models_dir=models_dir)

    result = trainer.train_full_pipeline(
        synthetic_df,
        model_name="WF_TEST",
        model_type="logistic",
        target="target_direction",
        target_horizon=5,
        tune_hyperparams=False,
    )

    metrics = result["metrics"]
    assert metrics["validation"]["method"] == "walk_forward"
    assert metrics["validation"]["fold_count"] >= 2
    assert metrics["validation"]["oof_samples"] > 0
