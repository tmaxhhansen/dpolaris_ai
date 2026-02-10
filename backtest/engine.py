"""
Reality-mode backtesting engine with execution frictions and reproducible artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd


_REQUIRED_MARKET_COLS = ["timestamp", "open", "high", "low", "close"]


@dataclass
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 0.0
    commission_per_share: float = 0.0
    commission_min_per_order: float = 0.0
    spread_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_partial_fills: bool = False
    volume_participation: float = 0.10
    max_fill_fraction: float = 1.0
    allow_short: bool = True
    execution_delay_bars: int = 1
    close_positions_on_end: bool = True
    intrabar_priority: str = "worst"  # one of: worst, best
    artifact_root: str = "reports/backtests"


@dataclass
class StrategyInput:
    timestamp: pd.Timestamp
    prediction: float
    confidence: float
    regime: str
    open: float
    high: float
    low: float
    close: float
    position_qty: float
    equity: float
    raw_row: dict[str, Any]


@dataclass
class StrategyDecision:
    action: str = "hold"  # buy, sell, hold
    size: float = 0.0  # fraction of equity in [0, 1]
    target_exposure: Optional[float] = None  # signed fraction in [-1, 1]
    order_type: str = "market"  # market, limit
    limit_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None  # relative to fill price
    take_profit_pct: Optional[float] = None  # relative to fill price
    reason: str = ""
    tag: str = ""


@dataclass
class OrderRequest:
    submitted_timestamp: pd.Timestamp
    execute_index: int
    requested_qty: float  # signed (+buy, -sell)
    order_type: str
    limit_price: Optional[float]
    prediction: float
    confidence: float
    regime: str
    reason: str
    signal: str
    tag: str
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]


@dataclass
class FillResult:
    filled_qty: float
    fill_price: float
    reference_price: float
    commission: float
    slippage_cost: float
    was_partial: bool
    requested_qty: float
    order_type: str


class Strategy(Protocol):
    def decide(self, signal_input: StrategyInput) -> StrategyDecision:
        raise NotImplementedError


class OptionsHook(Protocol):
    def on_bar(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def on_order_fill(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def on_trade_close(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class NoOpOptionsHook:
    def on_bar(self, payload: dict[str, Any]) -> None:
        _ = payload

    def on_order_fill(self, payload: dict[str, Any]) -> None:
        _ = payload

    def on_trade_close(self, payload: dict[str, Any]) -> None:
        _ = payload


class PredictionStrategy:
    """
    Default strategy adapter:
    input predictions/confidence/regime -> directional exposure target.
    """

    def __init__(
        self,
        *,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        base_size: float = 0.5,
        use_confidence_for_size: bool = True,
        min_size: float = 0.10,
        max_size: float = 1.0,
        stop_loss_pct: Optional[float] = 0.03,
        take_profit_pct: Optional[float] = 0.06,
        allow_short: bool = True,
    ):
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.base_size = float(base_size)
        self.use_confidence_for_size = bool(use_confidence_for_size)
        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.allow_short = bool(allow_short)

    def decide(self, signal_input: StrategyInput) -> StrategyDecision:
        prediction = float(signal_input.prediction)
        confidence = float(signal_input.confidence)
        size = self.base_size
        if self.use_confidence_for_size:
            size = np.clip(abs(confidence), self.min_size, self.max_size)

        action = "hold"
        target_exposure: Optional[float] = None
        reason = "no_edge"
        if prediction >= self.long_threshold:
            action = "buy"
            target_exposure = float(size)
            reason = "prob_up"
        elif prediction <= self.short_threshold and self.allow_short:
            action = "sell"
            target_exposure = -float(size)
            reason = "prob_down"

        return StrategyDecision(
            action=action,
            size=float(size),
            target_exposure=target_exposure,
            order_type="market",
            limit_price=None,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            reason=reason,
            tag=f"regime:{signal_input.regime}",
        )


class BacktestEngine:
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        *,
        options_hook: Optional[OptionsHook] = None,
    ):
        self.config = config or ExecutionConfig()
        self.options_hook = options_hook or NoOpOptionsHook()

    @staticmethod
    def _validate_market_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("market_df is empty")
        out = df.copy()
        missing = [c for c in _REQUIRED_MARKET_COLS if c not in out.columns]
        if missing:
            raise ValueError(f"market_df missing required columns: {missing}")

        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "volume" in out.columns:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        else:
            out["volume"] = np.nan

        out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return out

    @staticmethod
    def _prepare_signal_df(signal_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if signal_df is None or signal_df.empty:
            return pd.DataFrame(columns=["timestamp", "prediction", "confidence", "regime"])

        out = signal_df.copy()
        if "timestamp" not in out.columns:
            raise ValueError("signals_df must contain 'timestamp'")
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["prediction"] = pd.to_numeric(out.get("prediction", np.nan), errors="coerce")
        out["confidence"] = pd.to_numeric(out.get("confidence", np.nan), errors="coerce")
        if "regime" not in out.columns:
            out["regime"] = "unknown"
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return out

    def _merge_market_and_signals(
        self,
        market_df: pd.DataFrame,
        signals_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        market = self._validate_market_df(market_df)
        signals = self._prepare_signal_df(signals_df)
        if signals.empty:
            market["prediction"] = np.nan
            market["confidence"] = np.nan
            market["regime"] = "unknown"
            return market

        merged = pd.merge_asof(
            market.sort_values("timestamp"),
            signals.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce")
        merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
        merged["regime"] = merged["regime"].fillna("unknown").astype(str)
        return merged

    @staticmethod
    def _calc_commission(
        qty_abs: float,
        fill_price: float,
        cfg: ExecutionConfig,
    ) -> float:
        notional = abs(qty_abs * fill_price)
        commission = notional * (cfg.commission_bps / 10_000.0) + (qty_abs * cfg.commission_per_share)
        if commission > 0.0:
            commission = max(commission, cfg.commission_min_per_order)
        return float(commission)

    @staticmethod
    def _execution_adjustment_rate(cfg: ExecutionConfig) -> float:
        return (cfg.spread_bps / 20_000.0) + (cfg.slippage_bps / 10_000.0)

    def _simulate_order_fill(
        self,
        order: OrderRequest,
        bar: pd.Series,
    ) -> Optional[FillResult]:
        cfg = self.config
        requested_qty = float(order.requested_qty)
        if abs(requested_qty) < 1e-12:
            return None

        side = float(np.sign(requested_qty))
        open_px = float(bar["open"])
        high_px = float(bar["high"])
        low_px = float(bar["low"])
        volume = float(bar.get("volume", np.nan))
        was_partial = False

        reference_price = open_px
        if order.order_type == "limit":
            limit_px = float(order.limit_price) if order.limit_price is not None else open_px
            if side > 0:  # buy
                if low_px <= limit_px:
                    reference_price = min(limit_px, open_px)
                else:
                    return None
            else:  # sell
                if high_px >= limit_px:
                    reference_price = max(limit_px, open_px)
                else:
                    return None

        fill_abs = abs(requested_qty)
        if cfg.allow_partial_fills:
            cap = fill_abs * max(min(cfg.max_fill_fraction, 1.0), 0.0)
            if np.isfinite(volume) and volume > 0:
                cap = min(cap, volume * max(cfg.volume_participation, 0.0))
            fill_abs = max(0.0, cap)
            was_partial = fill_abs + 1e-12 < abs(requested_qty)

        if fill_abs <= 1e-12:
            return None

        adjustment = self._execution_adjustment_rate(cfg)
        fill_price = reference_price * (1.0 + (side * adjustment))
        filled_qty = side * fill_abs
        commission = self._calc_commission(fill_abs, fill_price, cfg)
        slippage_cost = abs(fill_abs * (fill_price - reference_price))

        return FillResult(
            filled_qty=float(filled_qty),
            fill_price=float(fill_price),
            reference_price=float(reference_price),
            commission=float(commission),
            slippage_cost=float(slippage_cost),
            was_partial=bool(was_partial),
            requested_qty=float(requested_qty),
            order_type=order.order_type,
        )

    def _check_stop_take_trigger(
        self,
        *,
        position_qty: float,
        stop_price: Optional[float],
        take_price: Optional[float],
        bar: pd.Series,
    ) -> Optional[dict[str, Any]]:
        if abs(position_qty) < 1e-12:
            return None
        if stop_price is None and take_price is None:
            return None

        open_px = float(bar["open"])
        high_px = float(bar["high"])
        low_px = float(bar["low"])
        is_long = position_qty > 0

        stop_fill: Optional[float] = None
        take_fill: Optional[float] = None
        if is_long:
            if stop_price is not None:
                if open_px <= stop_price:
                    stop_fill = open_px
                elif low_px <= stop_price:
                    stop_fill = float(stop_price)
            if take_price is not None:
                if open_px >= take_price:
                    take_fill = open_px
                elif high_px >= take_price:
                    take_fill = float(take_price)
        else:
            if stop_price is not None:
                if open_px >= stop_price:
                    stop_fill = open_px
                elif high_px >= stop_price:
                    stop_fill = float(stop_price)
            if take_price is not None:
                if open_px <= take_price:
                    take_fill = open_px
                elif low_px <= take_price:
                    take_fill = float(take_price)

        if stop_fill is None and take_fill is None:
            return None

        if stop_fill is not None and take_fill is not None:
            priority = self.config.intrabar_priority.lower().strip()
            if priority == "best":
                reason = "take_profit"
                price = take_fill
            else:
                reason = "stop_loss"
                price = stop_fill
        elif stop_fill is not None:
            reason = "stop_loss"
            price = stop_fill
        else:
            reason = "take_profit"
            price = take_fill

        return {
            "reason": reason,
            "price": float(price),
        }

    @staticmethod
    def _make_artifact_dir(root: Path | str, run_name: Optional[str]) -> Path:
        base = Path(root).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = run_name.strip().replace(" ", "_") if isinstance(run_name, str) and run_name.strip() else "run"
        out = base / f"{stamp}_{suffix}"
        out.mkdir(parents=True, exist_ok=False)
        return out

    @staticmethod
    def _decision_to_order(
        decision: StrategyDecision,
        *,
        position_qty: float,
        equity: float,
        price_for_sizing: float,
        allow_short: bool,
    ) -> Optional[dict[str, Any]]:
        action = (decision.action or "hold").lower().strip()
        target_exposure = decision.target_exposure
        if target_exposure is None:
            if action == "buy":
                target_exposure = abs(float(decision.size))
            elif action == "sell":
                target_exposure = -abs(float(decision.size))
            else:
                return None

        target_exposure = float(np.clip(target_exposure, -1.0, 1.0))
        if not allow_short:
            target_exposure = max(0.0, target_exposure)

        if not np.isfinite(equity) or equity <= 0:
            return None
        if not np.isfinite(price_for_sizing) or price_for_sizing <= 0:
            return None

        target_qty = (target_exposure * equity) / price_for_sizing
        delta_qty = target_qty - float(position_qty)
        if abs(delta_qty) < 1e-9:
            return None
        return {
            "delta_qty": float(delta_qty),
            "signal": action,
        }

    @staticmethod
    def _finalize_trade(
        open_trade: dict[str, Any],
        *,
        exit_timestamp: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> dict[str, Any]:
        closed = dict(open_trade)
        closed["exit_timestamp"] = exit_timestamp
        closed["exit_price"] = float(exit_price)
        closed["exit_reason"] = exit_reason
        closed["bars_held"] = max(int(closed.get("bars_held", 0)), 1)
        entry_notional = abs(float(closed.get("entry_price", 0.0)) * float(closed.get("entry_qty", 0.0)))
        net_pnl = float(closed.get("realized_pnl", 0.0) - closed.get("costs", 0.0))
        closed["pnl"] = net_pnl
        closed["return"] = (net_pnl / entry_notional) if entry_notional > 1e-12 else 0.0
        return closed

    def run(
        self,
        market_df: pd.DataFrame,
        *,
        strategy: Strategy,
        signals_df: Optional[pd.DataFrame] = None,
        run_name: Optional[str] = None,
        artifact_root: Optional[Path | str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        generate_report: bool = True,
    ) -> dict[str, Any]:
        cfg = self.config
        market = self._merge_market_and_signals(market_df, signals_df)
        artifact_dir = self._make_artifact_dir(
            artifact_root or cfg.artifact_root,
            run_name=run_name,
        )

        cash = float(cfg.initial_cash)
        position_qty = 0.0
        avg_entry_price = 0.0
        stop_price: Optional[float] = None
        take_price: Optional[float] = None

        open_trade: Optional[dict[str, Any]] = None
        pending_orders: dict[int, list[OrderRequest]] = {}
        orders: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []

        turnover_notional = 0.0
        commission_total = 0.0
        slippage_total = 0.0

        for idx, bar in market.iterrows():
            ts = pd.to_datetime(bar["timestamp"], utc=True)
            bar_open = float(bar["open"])
            bar_close = float(bar["close"])

            # 1) Execute queued orders for this bar.
            for order in pending_orders.get(idx, []):
                fill = self._simulate_order_fill(order, bar)
                if fill is None:
                    continue

                before_pos = position_qty
                before_avg = avg_entry_price
                side = float(np.sign(fill.filled_qty))
                fill_total_cost = float(fill.commission + fill.slippage_cost)

                cash -= (fill.filled_qty * fill.fill_price)
                cash -= fill.commission
                commission_total += fill.commission
                slippage_total += fill.slippage_cost
                turnover_notional += abs(fill.filled_qty * fill.fill_price)

                realized_component = 0.0
                closing_ratio = 0.0
                if abs(before_pos) > 1e-12 and np.sign(before_pos) != np.sign(fill.filled_qty):
                    closed_qty = min(abs(before_pos), abs(fill.filled_qty))
                    realized_component = closed_qty * (fill.fill_price - before_avg) * np.sign(before_pos)
                    closing_ratio = closed_qty / abs(fill.filled_qty)

                position_qty = before_pos + fill.filled_qty
                if abs(position_qty) < 1e-12:
                    position_qty = 0.0
                    avg_entry_price = 0.0
                elif abs(before_pos) < 1e-12 or np.sign(before_pos) == np.sign(fill.filled_qty):
                    # Same-direction add/open.
                    if abs(before_pos) < 1e-12:
                        avg_entry_price = fill.fill_price
                    else:
                        avg_entry_price = (
                            (abs(before_pos) * before_avg) + (abs(fill.filled_qty) * fill.fill_price)
                        ) / abs(position_qty)
                elif np.sign(position_qty) == np.sign(before_pos):
                    # Partial close, direction unchanged.
                    avg_entry_price = before_avg
                else:
                    # Reversal: leftover position opens at new fill price.
                    avg_entry_price = fill.fill_price

                # Trade lifecycle bookkeeping.
                closing_cost = fill_total_cost * closing_ratio
                opening_cost = fill_total_cost - closing_cost

                if open_trade is not None:
                    open_trade["bars_held"] = int(open_trade.get("bars_held", 0)) + 1

                if abs(before_pos) < 1e-12 and abs(position_qty) > 1e-12:
                    open_trade = {
                        "entry_timestamp": ts,
                        "entry_price": float(avg_entry_price),
                        "entry_qty": float(abs(position_qty)),
                        "direction": "long" if position_qty > 0 else "short",
                        "entry_signal": order.signal,
                        "entry_prediction": float(order.prediction),
                        "entry_confidence": float(order.confidence),
                        "entry_regime": str(order.regime),
                        "entry_reason": str(order.reason),
                        "entry_tag": str(order.tag),
                        "realized_pnl": 0.0,
                        "costs": float(opening_cost),
                        "bars_held": 0,
                    }
                elif abs(before_pos) > 1e-12 and np.sign(before_pos) == np.sign(position_qty):
                    if open_trade is not None:
                        open_trade["realized_pnl"] = float(open_trade.get("realized_pnl", 0.0) + realized_component)
                        open_trade["costs"] = float(open_trade.get("costs", 0.0) + fill_total_cost)
                        if abs(position_qty) > abs(before_pos):
                            open_trade["entry_qty"] = float(abs(position_qty))
                            open_trade["entry_price"] = float(avg_entry_price)
                elif abs(before_pos) > 1e-12 and abs(position_qty) < 1e-12:
                    if open_trade is not None:
                        open_trade["realized_pnl"] = float(open_trade.get("realized_pnl", 0.0) + realized_component)
                        open_trade["costs"] = float(open_trade.get("costs", 0.0) + closing_cost)
                        closed_trade = self._finalize_trade(
                            open_trade,
                            exit_timestamp=ts,
                            exit_price=float(fill.fill_price),
                            exit_reason="signal",
                        )
                        trades.append(closed_trade)
                        self.options_hook.on_trade_close(closed_trade)
                    open_trade = None
                    stop_price = None
                    take_price = None
                elif abs(before_pos) > 1e-12 and np.sign(before_pos) != np.sign(position_qty):
                    # Reversal: close old, open new.
                    if open_trade is not None:
                        open_trade["realized_pnl"] = float(open_trade.get("realized_pnl", 0.0) + realized_component)
                        open_trade["costs"] = float(open_trade.get("costs", 0.0) + closing_cost)
                        closed_trade = self._finalize_trade(
                            open_trade,
                            exit_timestamp=ts,
                            exit_price=float(fill.fill_price),
                            exit_reason="reversal",
                        )
                        trades.append(closed_trade)
                        self.options_hook.on_trade_close(closed_trade)

                    open_trade = {
                        "entry_timestamp": ts,
                        "entry_price": float(avg_entry_price),
                        "entry_qty": float(abs(position_qty)),
                        "direction": "long" if position_qty > 0 else "short",
                        "entry_signal": order.signal,
                        "entry_prediction": float(order.prediction),
                        "entry_confidence": float(order.confidence),
                        "entry_regime": str(order.regime),
                        "entry_reason": str(order.reason),
                        "entry_tag": str(order.tag),
                        "realized_pnl": 0.0,
                        "costs": float(opening_cost),
                        "bars_held": 0,
                    }

                if abs(position_qty) > 1e-12:
                    if order.stop_loss_pct is not None and order.stop_loss_pct > 0:
                        if position_qty > 0:
                            stop_price = avg_entry_price * (1.0 - float(order.stop_loss_pct))
                        else:
                            stop_price = avg_entry_price * (1.0 + float(order.stop_loss_pct))
                    if order.take_profit_pct is not None and order.take_profit_pct > 0:
                        if position_qty > 0:
                            take_price = avg_entry_price * (1.0 + float(order.take_profit_pct))
                        else:
                            take_price = avg_entry_price * (1.0 - float(order.take_profit_pct))

                order_row = {
                    "timestamp": ts,
                    "order_type": fill.order_type,
                    "signal": order.signal,
                    "requested_qty": float(fill.requested_qty),
                    "filled_qty": float(fill.filled_qty),
                    "fill_ratio": float(abs(fill.filled_qty) / max(abs(fill.requested_qty), 1e-12)),
                    "fill_price": float(fill.fill_price),
                    "reference_price": float(fill.reference_price),
                    "commission": float(fill.commission),
                    "slippage_cost": float(fill.slippage_cost),
                    "was_partial": bool(fill.was_partial),
                    "prediction": float(order.prediction),
                    "confidence": float(order.confidence),
                    "regime": str(order.regime),
                    "reason": str(order.reason),
                    "tag": str(order.tag),
                }
                orders.append(order_row)
                self.options_hook.on_order_fill(order_row)

            # 2) Stop-loss / take-profit checks on active position.
            trigger = self._check_stop_take_trigger(
                position_qty=position_qty,
                stop_price=stop_price,
                take_price=take_price,
                bar=bar,
            )
            if trigger is not None and abs(position_qty) > 1e-12:
                exit_side = -np.sign(position_qty)
                ref = float(trigger["price"])
                fill_price = ref * (1.0 + (exit_side * self._execution_adjustment_rate(cfg)))
                close_qty = -position_qty
                commission = self._calc_commission(abs(close_qty), fill_price, cfg)
                slip_cost = abs(close_qty * (fill_price - ref))

                cash -= (close_qty * fill_price)
                cash -= commission
                commission_total += commission
                slippage_total += slip_cost
                turnover_notional += abs(close_qty * fill_price)

                if open_trade is not None:
                    realized_component = abs(close_qty) * (fill_price - avg_entry_price) * np.sign(position_qty)
                    open_trade["realized_pnl"] = float(open_trade.get("realized_pnl", 0.0) + realized_component)
                    open_trade["costs"] = float(open_trade.get("costs", 0.0) + commission + slip_cost)
                    open_trade["bars_held"] = int(open_trade.get("bars_held", 0)) + 1
                    closed_trade = self._finalize_trade(
                        open_trade,
                        exit_timestamp=ts,
                        exit_price=float(fill_price),
                        exit_reason=str(trigger["reason"]),
                    )
                    trades.append(closed_trade)
                    self.options_hook.on_trade_close(closed_trade)

                orders.append(
                    {
                        "timestamp": ts,
                        "order_type": "stop_take",
                        "signal": "sell" if exit_side < 0 else "buy",
                        "requested_qty": float(close_qty),
                        "filled_qty": float(close_qty),
                        "fill_ratio": 1.0,
                        "fill_price": float(fill_price),
                        "reference_price": float(ref),
                        "commission": float(commission),
                        "slippage_cost": float(slip_cost),
                        "was_partial": False,
                        "prediction": float(bar.get("prediction", np.nan)),
                        "confidence": float(bar.get("confidence", np.nan)),
                        "regime": str(bar.get("regime", "unknown")),
                        "reason": str(trigger["reason"]),
                        "tag": "risk_exit",
                    }
                )

                position_qty = 0.0
                avg_entry_price = 0.0
                stop_price = None
                take_price = None
                open_trade = None

            # 3) Mark portfolio on close.
            position_value = position_qty * bar_close
            equity = cash + position_value
            exposure = abs(position_value) / max(abs(equity), 1e-9)
            self.options_hook.on_bar(
                {
                    "timestamp": ts,
                    "underlying_close": bar_close,
                    "equity": float(equity),
                    "position_qty": float(position_qty),
                    "options_iv": float(bar.get("options_iv", np.nan)),
                    "regime": str(bar.get("regime", "unknown")),
                }
            )
            equity_rows.append(
                {
                    "timestamp": ts,
                    "cash": float(cash),
                    "position_qty": float(position_qty),
                    "position_value": float(position_value),
                    "close": float(bar_close),
                    "equity": float(equity),
                    "exposure": float(exposure),
                }
            )

            if open_trade is not None:
                open_trade["bars_held"] = int(open_trade.get("bars_held", 0)) + 1

            # 4) Generate next decision and queue order for delayed execution.
            signal_input = StrategyInput(
                timestamp=ts,
                prediction=float(bar.get("prediction", np.nan)),
                confidence=float(bar.get("confidence", np.nan)),
                regime=str(bar.get("regime", "unknown")),
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                position_qty=float(position_qty),
                equity=float(equity),
                raw_row={k: bar[k] for k in bar.index},
            )
            decision = strategy.decide(signal_input)
            order_payload = self._decision_to_order(
                decision,
                position_qty=position_qty,
                equity=equity,
                price_for_sizing=bar_close,
                allow_short=cfg.allow_short,
            )
            if order_payload is not None:
                execute_index = int(idx + max(int(cfg.execution_delay_bars), 0))
                if execute_index < len(market):
                    request = OrderRequest(
                        submitted_timestamp=ts,
                        execute_index=execute_index,
                        requested_qty=float(order_payload["delta_qty"]),
                        order_type=(decision.order_type or "market").lower().strip(),
                        limit_price=decision.limit_price,
                        prediction=float(signal_input.prediction) if np.isfinite(signal_input.prediction) else np.nan,
                        confidence=float(signal_input.confidence) if np.isfinite(signal_input.confidence) else np.nan,
                        regime=signal_input.regime,
                        reason=str(decision.reason or ""),
                        signal=(decision.action or "hold").lower().strip(),
                        tag=str(decision.tag or ""),
                        stop_loss_pct=decision.stop_loss_pct,
                        take_profit_pct=decision.take_profit_pct,
                    )
                    pending_orders.setdefault(execute_index, []).append(request)

        # Optional end-of-run liquidation.
        if cfg.close_positions_on_end and abs(position_qty) > 1e-12 and len(market) > 0:
            last = market.iloc[-1]
            ts = pd.to_datetime(last["timestamp"], utc=True)
            ref = float(last["close"])
            side = -np.sign(position_qty)
            close_qty = -position_qty
            fill_price = ref * (1.0 + (side * self._execution_adjustment_rate(cfg)))
            commission = self._calc_commission(abs(close_qty), fill_price, cfg)
            slip_cost = abs(close_qty * (fill_price - ref))
            cash -= (close_qty * fill_price)
            cash -= commission
            commission_total += commission
            slippage_total += slip_cost
            turnover_notional += abs(close_qty * fill_price)

            if open_trade is not None:
                realized_component = abs(close_qty) * (fill_price - avg_entry_price) * np.sign(position_qty)
                open_trade["realized_pnl"] = float(open_trade.get("realized_pnl", 0.0) + realized_component)
                open_trade["costs"] = float(open_trade.get("costs", 0.0) + commission + slip_cost)
                closed_trade = self._finalize_trade(
                    open_trade,
                    exit_timestamp=ts,
                    exit_price=float(fill_price),
                    exit_reason="end_of_test",
                )
                trades.append(closed_trade)
                self.options_hook.on_trade_close(closed_trade)

            orders.append(
                {
                    "timestamp": ts,
                    "order_type": "market",
                    "signal": "sell" if side < 0 else "buy",
                    "requested_qty": float(close_qty),
                    "filled_qty": float(close_qty),
                    "fill_ratio": 1.0,
                    "fill_price": float(fill_price),
                    "reference_price": float(ref),
                    "commission": float(commission),
                    "slippage_cost": float(slip_cost),
                    "was_partial": False,
                    "prediction": float(last.get("prediction", np.nan)),
                    "confidence": float(last.get("confidence", np.nan)),
                    "regime": str(last.get("regime", "unknown")),
                    "reason": "end_of_test",
                    "tag": "liquidation",
                }
            )

            position_qty = 0.0
            avg_entry_price = 0.0

            # refresh final equity row with liquidation.
            if equity_rows:
                equity_rows[-1]["cash"] = float(cash)
                equity_rows[-1]["position_qty"] = 0.0
                equity_rows[-1]["position_value"] = 0.0
                equity_rows[-1]["equity"] = float(cash)
                equity_rows[-1]["exposure"] = 0.0

        equity_df = pd.DataFrame(equity_rows)
        orders_df = pd.DataFrame(orders)
        trades_df = pd.DataFrame(trades)

        if not equity_df.empty:
            equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)
            equity_df["equity_return"] = equity_df["equity"].pct_change().fillna(0.0)
            running_max = equity_df["equity"].cummax()
            equity_df["drawdown"] = (equity_df["equity"] / running_max) - 1.0
        else:
            equity_df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "cash",
                    "position_qty",
                    "position_value",
                    "close",
                    "equity",
                    "exposure",
                    "equity_return",
                    "drawdown",
                ]
            )

        if not trades_df.empty:
            trades_df = trades_df.sort_values("entry_timestamp").reset_index(drop=True)

        metrics = self._compute_metrics(
            equity_df=equity_df,
            trades_df=trades_df,
            turnover_notional=turnover_notional,
            commission_total=commission_total,
            slippage_total=slippage_total,
            initial_cash=cfg.initial_cash,
        )

        attribution_df = self._compute_attribution(trades_df)

        snapshot = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "strategy": {
                "class": strategy.__class__.__name__,
                "module": strategy.__class__.__module__,
                "attrs": getattr(strategy, "__dict__", {}),
            },
            "metadata": extra_metadata or {},
            "input_shapes": {
                "market_rows": int(len(market)),
                "signal_rows": int(len(signals_df)) if signals_df is not None else 0,
            },
        }

        self._write_artifacts(
            artifact_dir=artifact_dir,
            snapshot=snapshot,
            metrics=metrics,
            equity_df=equity_df,
            orders_df=orders_df,
            trades_df=trades_df,
            attribution_df=attribution_df,
        )

        report_paths: dict[str, str] = {}
        if generate_report:
            from .reporting import generate_backtest_report

            report_paths = generate_backtest_report(
                artifact_dir=artifact_dir,
                metrics=metrics,
                config_snapshot=snapshot,
                equity_df=equity_df,
                orders_df=orders_df,
                trades_df=trades_df,
                attribution_df=attribution_df,
            )

        return {
            "artifact_dir": str(artifact_dir),
            "metrics": metrics,
            "equity_curve": equity_df,
            "orders": orders_df,
            "trades": trades_df,
            "attribution": attribution_df,
            "report_paths": report_paths,
            "config_snapshot": snapshot,
        }

    @staticmethod
    def _compute_metrics(
        *,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        turnover_notional: float,
        commission_total: float,
        slippage_total: float,
        initial_cash: float,
        annualization_periods: int = 252,
    ) -> dict[str, float]:
        if equity_df.empty:
            return {
                "initial_equity": float(initial_cash),
                "final_equity": float(initial_cash),
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "exposure": 0.0,
                "turnover": 0.0,
                "trade_count": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "commission_total": float(commission_total),
                "slippage_total": float(slippage_total),
                "cost_total": float(commission_total + slippage_total),
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
            }

        start_equity = float(equity_df["equity"].iloc[0])
        end_equity = float(equity_df["equity"].iloc[-1])
        total_return = (end_equity / max(start_equity, 1e-9)) - 1.0
        periods = len(equity_df)
        years = periods / max(annualization_periods, 1)
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 and (1.0 + total_return) > 0 else 0.0

        rets = pd.to_numeric(equity_df["equity_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mean_ret = float(np.mean(rets)) if len(rets) else 0.0
        std_ret = float(np.std(rets)) if len(rets) else 0.0
        sharpe = float((mean_ret / std_ret) * np.sqrt(annualization_periods)) if std_ret > 1e-12 else 0.0
        downside = rets[rets < 0.0]
        downside_std = float(np.std(downside)) if len(downside) else 0.0
        sortino = float((mean_ret / downside_std) * np.sqrt(annualization_periods)) if downside_std > 1e-12 else 0.0
        max_drawdown = float(abs(float(equity_df["drawdown"].min()))) if "drawdown" in equity_df.columns else 0.0

        avg_equity = float(equity_df["equity"].mean())
        turnover = float(turnover_notional / max(avg_equity, 1e-9))
        exposure = float(pd.to_numeric(equity_df["exposure"], errors="coerce").fillna(0.0).mean())

        gross_pnl = float(end_equity - start_equity + commission_total + slippage_total)
        net_pnl = float(end_equity - start_equity)

        if trades_df is None or trades_df.empty:
            trade_count = 0
            win_rate = 0.0
            profit_factor = 0.0
            average_win = 0.0
            average_loss = 0.0
        else:
            pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
            wins = pnl[pnl > 0.0]
            losses = pnl[pnl < 0.0]
            trade_count = int(len(pnl))
            win_rate = float((pnl > 0.0).mean()) if trade_count else 0.0
            gross_win = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
            if gross_loss > 1e-12:
                profit_factor = float(gross_win / gross_loss)
            elif gross_win > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0
            average_win = float(wins.mean()) if len(wins) else 0.0
            average_loss = float(losses.mean()) if len(losses) else 0.0

        return {
            "initial_equity": float(start_equity),
            "final_equity": float(end_equity),
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "exposure": float(exposure),
            "turnover": float(turnover),
            "trade_count": float(trade_count),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "average_win": float(average_win),
            "average_loss": float(average_loss),
            "commission_total": float(commission_total),
            "slippage_total": float(slippage_total),
            "cost_total": float(commission_total + slippage_total),
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
        }

    @staticmethod
    def _compute_attribution(trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df is None or trades_df.empty:
            return pd.DataFrame(
                columns=["entry_signal", "entry_regime", "entry_reason", "trades", "pnl", "win_rate", "avg_return"]
            )

        grouped = (
            trades_df.groupby(["entry_signal", "entry_regime", "entry_reason"], dropna=False)
            .agg(
                trades=("pnl", "count"),
                pnl=("pnl", "sum"),
                win_rate=("pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
                avg_return=("return", "mean"),
            )
            .reset_index()
            .sort_values("pnl", ascending=False)
            .reset_index(drop=True)
        )
        return grouped

    @staticmethod
    def _write_artifacts(
        *,
        artifact_dir: Path,
        snapshot: dict[str, Any],
        metrics: dict[str, float],
        equity_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        attribution_df: pd.DataFrame,
    ) -> None:
        with open(artifact_dir / "config_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        with open(artifact_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        equity_df.to_csv(artifact_dir / "equity_curve.csv", index=False)
        orders_df.to_csv(artifact_dir / "orders.csv", index=False)
        trades_df.to_csv(artifact_dir / "trades.csv", index=False)
        attribution_df.to_csv(artifact_dir / "attribution.csv", index=False)

