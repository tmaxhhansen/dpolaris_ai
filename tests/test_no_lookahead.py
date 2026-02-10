from __future__ import annotations

import inspect

from ml.deep_learning import DeepLearningTrainer
from ml.features import FeatureEngine
from ml.validation import validate_no_lookahead_features


def test_feature_generation_no_lookahead(synthetic_df):
    feature_engine = FeatureEngine()
    result = validate_no_lookahead_features(
        feature_engine=feature_engine,
        raw_df=synthetic_df,
        sample_count=12,
    )
    assert result["checked"] > 0
    assert result["passed"], f"lookahead violations: {result['violations']}"


def test_deep_learning_prepare_data_scales_train_split_only():
    source = inspect.getsource(DeepLearningTrainer.prepare_data)
    assert "fit_transform(X_train_raw)" in source
    assert "transform(X_test_raw)" in source
    assert "fit_transform(X)" not in source
