from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, ExecutionConfig, PredictionStrategy, StrategyDecision, StrategyInput


class AllInLimitStrategy:
    def decide(self, signal_input: StrategyInput) -> StrategyDecision:
        _ = signal_input
        return StrategyDecision(
            action="buy",
            target_exposure=1.0,
            order_type="limit",
            limit_price=float(signal_input.close * 1.05),
            stop_loss_pct=None,
            take_profit_pct=None,
            reason="accumulate",
            tag="limit_partial",
        )


def _market_frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 102, 104, 96, 94, 95, 97, 99, 101, 103],
            "high": [103, 105, 106, 98, 96, 97, 99, 101, 103, 105],
            "low": [99, 101, 95, 92, 93, 94, 95, 98, 100, 102],
            "close": [102, 104, 96, 94, 95, 97, 99, 101, 103, 104],
            "volume": [1200, 1100, 1400, 1600, 1500, 1300, 1200, 1100, 1000, 900],
        }
    )


def _signal_frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "prediction": [0.70, 0.72, 0.35, 0.30, 0.62, 0.60, 0.42, 0.40, 0.68, 0.66],
            "confidence": [0.60, 0.65, 0.70, 0.75, 0.55, 0.60, 0.58, 0.52, 0.64, 0.67],
            "regime": ["trend", "trend", "risk_off", "risk_off", "mean_reversion", "trend", "chop", "chop", "trend", "trend"],
        }
    )


def test_backtest_reality_mode_writes_artifacts(tmp_path):
    market = _market_frame()
    signals = _signal_frame()

    cfg = ExecutionConfig(
        initial_cash=100_000.0,
        commission_bps=1.5,
        spread_bps=2.0,
        slippage_bps=3.0,
        allow_partial_fills=False,
        execution_delay_bars=1,
        close_positions_on_end=True,
    )
    engine = BacktestEngine(config=cfg)
    strategy = PredictionStrategy(
        long_threshold=0.55,
        short_threshold=0.45,
        base_size=0.5,
        use_confidence_for_size=True,
        stop_loss_pct=0.04,
        take_profit_pct=0.08,
        allow_short=True,
    )
    result = engine.run(
        market,
        strategy=strategy,
        signals_df=signals,
        run_name="reality_mode_test",
        artifact_root=tmp_path / "reports",
        generate_report=True,
    )

    metrics = result["metrics"]
    assert metrics["cost_total"] > 0.0
    assert metrics["turnover"] > 0.0
    assert metrics["trade_count"] >= 1
    assert metrics["max_drawdown"] >= 0.0

    artifact_dir = tmp_path / "reports"
    run_dirs = sorted(p for p in artifact_dir.iterdir() if p.is_dir())
    assert run_dirs, "expected run artifact directory"
    latest = run_dirs[-1]

    assert (latest / "config_snapshot.json").exists()
    assert (latest / "metrics.json").exists()
    assert (latest / "equity_curve.csv").exists()
    assert (latest / "orders.csv").exists()
    assert (latest / "trades.csv").exists()
    assert (latest / "attribution.csv").exists()
    assert (latest / "report.md").exists()
    assert (latest / "report.html").exists()


def test_backtest_partial_fills_limit_orders(tmp_path):
    market = _market_frame().copy()
    market["volume"] = [200] * len(market)

    cfg = ExecutionConfig(
        initial_cash=100_000.0,
        commission_bps=0.5,
        spread_bps=1.0,
        slippage_bps=1.0,
        allow_partial_fills=True,
        volume_participation=0.01,
        max_fill_fraction=1.0,
        execution_delay_bars=1,
        close_positions_on_end=True,
    )
    engine = BacktestEngine(config=cfg)
    strategy = AllInLimitStrategy()
    result = engine.run(
        market,
        strategy=strategy,
        signals_df=None,
        run_name="partial_fill_test",
        artifact_root=tmp_path / "reports",
        generate_report=False,
    )

    orders = result["orders"]
    assert not orders.empty
    assert (orders["order_type"] == "limit").any()
    assert (orders["was_partial"] == True).any()
    assert (orders["fill_ratio"] < 1.0).any()

