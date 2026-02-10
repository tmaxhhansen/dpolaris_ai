from __future__ import annotations

import numpy as np

from ml.evaluation import compute_trading_metrics


def test_backtest_applies_transaction_cost_and_slippage():
    probability_up = np.array([0.80, 0.78, 0.22, 0.25, 0.62, 0.38, 0.77, 0.20])
    realized_returns = np.array([0.012, -0.004, 0.008, -0.006, 0.010, -0.003, 0.009, -0.002])

    no_cost = compute_trading_metrics(
        probability_up=probability_up,
        realized_returns=realized_returns,
        long_threshold=0.55,
        short_threshold=0.45,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    with_cost = compute_trading_metrics(
        probability_up=probability_up,
        realized_returns=realized_returns,
        long_threshold=0.55,
        short_threshold=0.45,
        transaction_cost_bps=2.0,
        slippage_bps=3.0,
    )

    assert with_cost["cost_total"] > 0.0
    assert with_cost["turnover"] > 0.0
    assert with_cost["net_pnl"] < no_cost["net_pnl"]
