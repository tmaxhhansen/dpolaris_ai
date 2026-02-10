from __future__ import annotations

import pandas as pd

from risk.risk_manager import RiskManager, RiskManagerConfig


def test_volatility_sizing_respects_per_trade_risk_cap():
    manager = RiskManager(
        RiskManagerConfig(
            max_risk_per_trade_pct=0.01,
            max_risk_per_day_pct=0.05,
            max_position_exposure_pct=1.0,
        )
    )

    decision = manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="AAPL",
        side="buy",
        requested_qty=1_000.0,
        price=100.0,
        atr=5.0,
    )

    # $1,000 risk budget / $5 per-share risk = 200 shares.
    assert decision.approved is True
    assert decision.approved_qty == 200.0
    assert 0.009 <= decision.estimated_trade_risk_pct <= 0.011


def test_daily_risk_cap_blocks_new_order_once_budget_is_spent():
    manager = RiskManager(
        RiskManagerConfig(
            max_risk_per_trade_pct=0.02,
            max_risk_per_day_pct=0.02,
            max_position_exposure_pct=1.0,
        )
    )

    first = manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="MSFT",
        side="buy",
        requested_qty=1_000.0,
        price=100.0,
        atr=2.0,
    )
    assert first.approved is True
    assert first.approved_qty == 1_000.0

    second = manager.evaluate_order(
        timestamp="2024-02-01T15:00:00Z",
        equity=100_000.0,
        symbol="NVDA",
        side="buy",
        requested_qty=100.0,
        price=100.0,
        atr=2.0,
    )
    assert second.approved is False
    assert "daily_risk_cap_exceeded" in second.reason_codes


def test_sector_exposure_cap_reduces_size():
    manager = RiskManager(
        RiskManagerConfig(
            max_risk_per_trade_pct=0.50,
            max_risk_per_day_pct=1.0,
            max_position_exposure_pct=1.0,
            max_sector_exposure_pct=0.35,
        )
    )

    holdings = [{"symbol": "MSFT", "qty": 300.0, "price": 100.0, "sector": "tech"}]  # 30% of equity
    decision = manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="AAPL",
        side="buy",
        requested_qty=100.0,
        price=100.0,
        atr=1.0,
        sector="tech",
        holdings=holdings,
    )

    # Tech cap is 35%, so max additional notional is $5,000 -> 50 shares.
    assert decision.approved is True
    assert decision.approved_qty == 50.0
    assert "size_reduced_for_constraints" in decision.reason_codes
    assert decision.sector_exposure_pct <= 0.35 + 1e-9


def test_correlation_cluster_cap_reduces_position_size():
    manager = RiskManager(
        RiskManagerConfig(
            max_risk_per_trade_pct=0.50,
            max_risk_per_day_pct=1.0,
            max_position_exposure_pct=1.0,
            max_sector_exposure_pct=1.0,
            correlation_threshold=0.80,
            max_correlated_exposure_pct=0.40,
        )
    )

    holdings = [{"symbol": "MSFT", "qty": 350.0, "price": 100.0, "sector": "tech"}]  # 35%
    corr = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    decision = manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="AAPL",
        side="buy",
        requested_qty=100.0,
        price=100.0,
        atr=1.0,
        holdings=holdings,
        correlation_matrix=corr,
    )

    assert decision.approved is True
    assert decision.approved_qty == 50.0
    assert decision.correlated_exposure_pct <= 0.40 + 1e-9


def test_gross_exposure_cap_blocks_new_risk_when_already_full():
    manager = RiskManager(
        RiskManagerConfig(
            max_risk_per_trade_pct=0.50,
            max_risk_per_day_pct=1.0,
            max_position_exposure_pct=1.0,
            max_sector_exposure_pct=1.0,
            max_gross_exposure_pct=1.0,
            max_net_exposure_pct=1.0,
        )
    )

    holdings = [{"symbol": "SPY", "qty": 1_000.0, "price": 100.0, "sector": "index"}]  # 100% gross/net
    decision = manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="QQQ",
        side="buy",
        requested_qty=10.0,
        price=100.0,
        atr=1.0,
        holdings=holdings,
    )

    assert decision.approved is False
    assert "gross_exposure_cap" in decision.reason_codes


def test_regime_and_event_throttle_reduce_risk_budget():
    manager = RiskManager(
        RiskManagerConfig(
            max_risk_per_trade_pct=0.02,
            max_risk_per_day_pct=0.10,
            max_position_exposure_pct=1.0,
            high_vol_regime_multiplier=0.50,
            event_risk_multiplier=0.50,
            no_trade_flags=(),
            high_vol_regimes=("high_vol",),
            major_event_flags=("cpi_day",),
        )
    )

    decision = manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="AMZN",
        side="buy",
        requested_qty=1_000.0,
        price=100.0,
        atr=5.0,
        regime="high_vol",
        event_flags={"cpi_day"},
    )

    # 2% * 0.5 * 0.5 = 0.5% risk budget => $500 risk => 100 shares at $5 unit risk.
    assert decision.approved is True
    assert decision.approved_qty == 100.0
    assert "regime_throttle" in decision.reason_codes
    assert "event_throttle" in decision.reason_codes


def test_kill_switch_and_no_trade_windows_are_enforced():
    kill_manager = RiskManager(
        RiskManagerConfig(
            kill_switch_drawdown_pct=0.10,
            max_risk_per_trade_pct=0.50,
            max_risk_per_day_pct=1.0,
            max_position_exposure_pct=1.0,
        )
    )
    kill_manager.update_equity(100_000.0, "2024-02-01T14:00:00Z")
    kill_manager.update_equity(85_000.0, "2024-02-01T15:00:00Z")

    blocked = kill_manager.evaluate_order(
        timestamp="2024-02-01T15:30:00Z",
        equity=85_000.0,
        symbol="TSLA",
        side="buy",
        requested_qty=10.0,
        price=100.0,
        atr=1.0,
    )
    assert blocked.approved is False
    assert "kill_switch_active" in blocked.reason_codes

    window_manager = RiskManager(
        RiskManagerConfig(
            no_trade_flags=("earnings_day",),
            max_risk_per_trade_pct=0.50,
            max_risk_per_day_pct=1.0,
            max_position_exposure_pct=1.0,
        )
    )
    no_trade = window_manager.evaluate_order(
        timestamp="2024-02-01T14:30:00Z",
        equity=100_000.0,
        symbol="META",
        side="buy",
        requested_qty=10.0,
        price=100.0,
        atr=1.0,
        event_flags={"earnings_day"},
    )
    assert no_trade.approved is False
    assert "no_trade_window" in no_trade.reason_codes
