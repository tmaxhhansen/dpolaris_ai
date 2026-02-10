from __future__ import annotations

import numpy as np
import pandas as pd

from features.registry import FeaturePlugin, FeatureRegistry, FeatureSpec
from features.technical import FeatureScaler, TechnicalFeatureLibrary, build_default_registry


def _as_timestamp_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out = out.rename(columns={"date": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _assert_causal_by_truncation(
    library: TechnicalFeatureLibrary,
    base_df: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    benchmark_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    sample_points: int = 10,
) -> None:
    full_features, meta = library.generate(
        base_df,
        specs=specs,
        benchmark_df=benchmark_df,
        sector_df=sector_df,
        include_base_columns=False,
        drop_na=False,
    )
    feature_cols = meta["feature_names"]
    assert len(feature_cols) > 0

    candidate_indices = np.linspace(
        max(40, len(base_df) // 10),
        len(base_df) - 1,
        num=sample_points,
        dtype=int,
    )

    for idx in candidate_indices:
        truncated_base = base_df.iloc[: idx + 1].copy()
        truncated_features, _ = library.generate(
            truncated_base,
            specs=specs,
            benchmark_df=benchmark_df,
            sector_df=sector_df,
            include_base_columns=False,
            drop_na=False,
        )

        full_row = full_features.iloc[idx][feature_cols].to_numpy(dtype=float)
        trunc_row = truncated_features.iloc[-1][feature_cols].to_numpy(dtype=float)
        assert np.allclose(full_row, trunc_row, equal_nan=True, atol=1e-9)


def test_feature_registry_plugin_system():
    registry = FeatureRegistry()

    def custom_plugin(df: pd.DataFrame, params: dict, context: dict | None) -> pd.DataFrame:
        _ = context
        value = float(params.get("value", 1.0))
        return pd.DataFrame({f"custom_const_v{value:g}": [value] * len(df)}, index=df.index)

    registry.register(
        FeaturePlugin(
            name="custom",
            generator=custom_plugin,
            group="custom",
            description="custom constant feature",
        )
    )

    base = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"), "close": [1, 2, 3, 4, 5]})
    out, meta = registry.generate(base, specs=[FeatureSpec(name="custom", params={"value": 7})], context={})

    assert "custom_const_v7" in out.columns
    assert meta["feature_names"] == ["custom_const_v7"]
    assert meta["catalog"][0]["plugin"] == "custom"


def test_technical_features_are_causal_for_price_volume(synthetic_df):
    base = _as_timestamp_frame(synthetic_df)
    library = TechnicalFeatureLibrary(build_default_registry())

    specs = [
        FeatureSpec(name="returns", params={"horizons": [1, 3, 5, 10, 20]}),
        FeatureSpec(name="trend", params={"sma_windows": [10, 20, 50], "ema_windows": [12, 26]}),
        FeatureSpec(name="momentum", params={"rsi_window": 14, "stoch_window": 14}),
        FeatureSpec(name="volatility", params={"atr_window": 14, "rv_windows": [5, 10, 20]}),
        FeatureSpec(name="volume", params={"zscore_windows": [10, 20], "trend_windows": [5, 20]}),
        FeatureSpec(name="structure", params={"lookbacks": [20, 50]}),
        FeatureSpec(name="gaps", params={"fill_windows": [10, 20], "atr_window": 14}),
    ]

    _assert_causal_by_truncation(library, base, specs, sample_points=8)


def test_relative_strength_features_are_causal_with_asof_join(synthetic_df):
    base = _as_timestamp_frame(synthetic_df)

    bench = base[["timestamp", "close"]].copy()
    bench["close"] = np.linspace(100.0, 140.0, len(bench))
    # Publish benchmark series later than base timestamp to force as-of behavior.
    bench["timestamp"] = bench["timestamp"] + pd.Timedelta(hours=8)
    # Inject a large future move; causal join should not leak this into prior rows.
    bench.loc[bench.index[-1], "close"] = 1000.0

    sector = base[["timestamp", "close"]].copy()
    sector["close"] = np.linspace(50.0, 70.0, len(sector))
    sector["timestamp"] = sector["timestamp"] + pd.Timedelta(hours=6)

    library = TechnicalFeatureLibrary(build_default_registry())
    specs = [
        FeatureSpec(
            name="relative_strength",
            params={
                "horizons": [1, 5, 10],
                "beta_window": 40,
                "corr_windows": [20, 40],
            },
        )
    ]

    _assert_causal_by_truncation(
        library,
        base,
        specs,
        benchmark_df=bench,
        sector_df=sector,
        sample_points=8,
    )


def test_feature_scaler_supports_standard_and_robust(synthetic_df):
    base = _as_timestamp_frame(synthetic_df)
    library = TechnicalFeatureLibrary(build_default_registry())
    specs = [FeatureSpec(name="returns"), FeatureSpec(name="volume")]

    frame, meta = library.generate(
        base,
        specs=specs,
        include_base_columns=False,
        drop_na=False,
    )
    feature_names = meta["feature_names"]
    clean = frame[feature_names].fillna(0.0)

    standard_scaled = FeatureScaler(method="standard").fit_transform(clean, feature_names)
    robust_scaled = FeatureScaler(method="robust").fit_transform(clean, feature_names)

    assert set(standard_scaled.columns) == set(clean.columns)
    assert set(robust_scaled.columns) == set(clean.columns)
