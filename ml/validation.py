"""
Validation utilities for data-leakage prevention and reproducibility.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """
    Set reproducible RNG seeds across supported libraries.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional for classic ML path.
        pass


def validate_no_lookahead_features(
    feature_engine,
    raw_df: pd.DataFrame,
    sample_count: int = 12,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Validate that generated feature rows are unchanged when future rows are removed.
    """
    if raw_df is None or raw_df.empty:
        return {"passed": True, "checked": 0, "violations": []}

    sorted_df = raw_df.sort_values("date").reset_index(drop=True)
    full_features = feature_engine.generate_features(sorted_df, include_targets=False)
    if full_features.empty:
        return {"passed": True, "checked": 0, "violations": []}

    feature_cols = feature_engine.get_feature_names()
    full_table = (
        full_features.assign(date=pd.to_datetime(full_features["date"]))
        .set_index("date")[feature_cols]
    )

    check_count = min(max(int(sample_count), 1), len(full_table))
    sample_indices = np.linspace(0, len(full_table) - 1, num=check_count, dtype=int)

    raw_dates = pd.to_datetime(sorted_df["date"])
    violations: list[dict[str, Any]] = []
    checked = 0

    for idx in sample_indices:
        ts = full_table.index[idx]
        truncated_raw = sorted_df.loc[raw_dates <= ts].copy()
        truncated_features = feature_engine.generate_features(
            truncated_raw,
            include_targets=False,
        )
        if truncated_features.empty:
            continue

        truncated_row = (
            truncated_features.assign(date=pd.to_datetime(truncated_features["date"]))
            .set_index("date")[feature_cols]
            .iloc[-1]
        )
        baseline_row = full_table.iloc[idx]

        diff = np.abs(baseline_row.to_numpy() - truncated_row.to_numpy())
        max_abs_diff = float(np.nanmax(diff)) if len(diff) else 0.0
        checked += 1

        if max_abs_diff > tolerance:
            violations.append(
                {
                    "timestamp": str(ts),
                    "max_abs_diff": max_abs_diff,
                }
            )

    return {
        "passed": len(violations) == 0,
        "checked": checked,
        "violations": violations,
    }
