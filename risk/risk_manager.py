"""
Portfolio risk manager with first-class sizing and constraint enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RiskManagerConfig:
    """Configurable limits and throttles."""

    max_risk_per_trade_pct: float = 0.01
    max_risk_per_day_pct: float = 0.03
    max_position_exposure_pct: float = 0.20

    max_sector_exposure_pct: float = 0.35
    max_correlated_exposure_pct: float = 0.45
    correlation_threshold: float = 0.80
    max_gross_exposure_pct: float = 1.50
    max_net_exposure_pct: float = 1.00

    high_vol_regime_multiplier: float = 0.50
    event_risk_multiplier: float = 0.50
    high_vol_regimes: tuple[str, ...] = ("high_vol", "crisis", "risk_off_high_vol")
    major_event_flags: tuple[str, ...] = ("earnings_day", "cpi_day", "fomc_day")
    no_trade_flags: tuple[str, ...] = ("earnings_day", "cpi_day")

    kill_switch_drawdown_pct: float = 0.15
    min_stop_distance_pct: float = 0.0025
    lot_size: float = 1.0


@dataclass
class RiskState:
    """Mutable runtime risk state."""

    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    kill_switch_active: bool = False
    daily_risk_used_pct: float = 0.0
    current_day: Optional[date] = None


@dataclass
class RiskDecision:
    """Decision payload returned by the risk manager."""

    approved: bool
    requested_qty: float
    approved_qty: float
    estimated_trade_risk_pct: float
    daily_risk_used_pct: float
    gross_exposure_pct: float
    net_exposure_pct: float
    sector_exposure_pct: float
    correlated_exposure_pct: float
    reason_codes: list[str] = field(default_factory=list)


class RiskManager:
    """
    First-class portfolio risk manager.

    Responsibilities:
    - volatility/ATR sizing
    - per-trade and per-day risk caps
    - gross/net/sector/correlation exposure constraints
    - regime and event-day risk throttles
    - drawdown kill switch and no-trade windows
    """

    def __init__(
        self,
        config: Optional[RiskManagerConfig] = None,
        state: Optional[RiskState] = None,
    ):
        self.config = config or RiskManagerConfig()
        self.state = state or RiskState()

    def update_equity(self, equity: float, timestamp: Any) -> None:
        """Update drawdown tracking and auto-activate kill switch when needed."""
        ts = _as_timestamp(timestamp)
        self._roll_day(ts)

        eq = float(equity)
        if not np.isfinite(eq) or eq <= 0.0:
            return

        if self.state.peak_equity <= 0.0:
            self.state.peak_equity = eq
            self.state.current_drawdown_pct = 0.0
            return

        self.state.peak_equity = max(self.state.peak_equity, eq)
        self.state.current_drawdown_pct = max(
            0.0,
            1.0 - (eq / max(self.state.peak_equity, 1e-9)),
        )
        if self.state.current_drawdown_pct >= self.config.kill_switch_drawdown_pct:
            self.state.kill_switch_active = True

    def clear_kill_switch(self) -> None:
        self.state.kill_switch_active = False

    def evaluate_order(
        self,
        *,
        timestamp: Any,
        equity: float,
        symbol: str,
        side: str,
        requested_qty: float,
        price: float,
        holdings: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]] = None,
        sector: Optional[str] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        atr: Optional[float] = None,
        realized_vol: Optional[float] = None,
        stop_price: Optional[float] = None,
        regime: Optional[str] = None,
        event_flags: Optional[Iterable[str]] = None,
        commit: bool = True,
    ) -> RiskDecision:
        ts = _as_timestamp(timestamp)
        self._roll_day(ts)

        reason_codes: list[str] = []
        eq = float(equity)
        px = float(price)
        if not np.isfinite(eq) or eq <= 0.0:
            return self._reject(float(requested_qty), "invalid_equity")
        if not np.isfinite(px) or px <= 0.0:
            return self._reject(float(requested_qty), "invalid_price")

        if self.state.kill_switch_active:
            return self._reject(float(requested_qty), "kill_switch_active")

        normalized_events = {str(x).strip().lower() for x in (event_flags or []) if str(x).strip()}
        blocked_flags = normalized_events.intersection({s.lower() for s in self.config.no_trade_flags})
        if blocked_flags:
            return self._reject(float(requested_qty), "no_trade_window")

        signed_side = _side_to_sign(side)
        req_abs = abs(float(requested_qty))
        if req_abs <= 1e-12:
            return self._reject(float(requested_qty), "empty_order")
        signed_request = signed_side * req_abs

        unit_risk = self._resolve_unit_risk(
            price=px,
            stop_price=stop_price,
            atr=atr,
            realized_vol=realized_vol,
        )

        allowed_trade_risk_pct = float(self.config.max_risk_per_trade_pct)
        normalized_regime = (regime or "").strip().lower()
        if normalized_regime and normalized_regime in {s.lower() for s in self.config.high_vol_regimes}:
            allowed_trade_risk_pct *= float(self.config.high_vol_regime_multiplier)
            reason_codes.append("regime_throttle")

        major_flags = normalized_events.intersection({s.lower() for s in self.config.major_event_flags})
        if major_flags:
            allowed_trade_risk_pct *= float(self.config.event_risk_multiplier)
            reason_codes.append("event_throttle")

        daily_remaining = max(0.0, float(self.config.max_risk_per_day_pct) - float(self.state.daily_risk_used_pct))
        if daily_remaining <= 1e-12:
            return self._reject(float(requested_qty), "daily_risk_cap_exceeded")

        risk_budget_pct = min(max(allowed_trade_risk_pct, 0.0), daily_remaining)
        if risk_budget_pct <= 1e-12:
            return self._reject(float(requested_qty), "risk_budget_zero")

        max_qty_risk = (eq * risk_budget_pct) / max(unit_risk, 1e-9)
        max_qty_position = (eq * max(float(self.config.max_position_exposure_pct), 0.0)) / px
        candidate_abs = min(req_abs, max_qty_risk, max_qty_position)
        candidate_abs = self._quantize_qty(candidate_abs)
        if candidate_abs <= 1e-12:
            return self._reject(float(requested_qty), "sized_to_zero")

        candidate_signed = signed_side * candidate_abs
        snapshot, fail_codes = self._validate_portfolio_constraints(
            symbol=symbol,
            sector=sector,
            signed_qty=candidate_signed,
            price=px,
            equity=eq,
            holdings=holdings,
            correlation_matrix=correlation_matrix,
        )

        if fail_codes:
            lot = max(float(self.config.lot_size), 1e-9)
            max_lots = int(np.floor(candidate_abs / lot))
            best_lots = 0
            best_snapshot: Optional[dict[str, float]] = None
            latest_codes = fail_codes

            lo, hi = 1, max_lots
            while lo <= hi:
                mid = (lo + hi) // 2
                mid_abs = float(mid * lot)
                mid_signed = signed_side * mid_abs
                mid_snapshot, mid_failures = self._validate_portfolio_constraints(
                    symbol=symbol,
                    sector=sector,
                    signed_qty=mid_signed,
                    price=px,
                    equity=eq,
                    holdings=holdings,
                    correlation_matrix=correlation_matrix,
                )
                if mid_failures:
                    latest_codes = mid_failures
                    hi = mid - 1
                else:
                    best_lots = mid
                    best_snapshot = mid_snapshot
                    lo = mid + 1

            if best_lots > 0:
                candidate_abs = float(best_lots * lot)
                candidate_signed = signed_side * candidate_abs
                snapshot = best_snapshot or snapshot
                reason_codes.append("size_reduced_for_constraints")
            else:
                return self._reject(
                    float(requested_qty),
                    *latest_codes,
                    snapshot=snapshot,
                )

        trade_risk_pct = (abs(candidate_signed) * unit_risk) / max(eq, 1e-9)
        if trade_risk_pct > daily_remaining + 1e-12:
            max_qty_daily = self._quantize_qty((eq * daily_remaining) / max(unit_risk, 1e-9))
            candidate_signed = signed_side * max_qty_daily
            candidate_abs = abs(candidate_signed)
            if candidate_abs <= 1e-12:
                return self._reject(float(requested_qty), "daily_risk_cap_exceeded", snapshot=snapshot)
            trade_risk_pct = (candidate_abs * unit_risk) / max(eq, 1e-9)

        if commit:
            self.state.daily_risk_used_pct = float(self.state.daily_risk_used_pct + trade_risk_pct)

        return RiskDecision(
            approved=True,
            requested_qty=signed_request,
            approved_qty=float(candidate_signed),
            estimated_trade_risk_pct=float(trade_risk_pct),
            daily_risk_used_pct=float(self.state.daily_risk_used_pct),
            gross_exposure_pct=float(snapshot["gross_exposure_pct"]),
            net_exposure_pct=float(snapshot["net_exposure_pct"]),
            sector_exposure_pct=float(snapshot["sector_exposure_pct"]),
            correlated_exposure_pct=float(snapshot["correlated_exposure_pct"]),
            reason_codes=reason_codes,
        )

    def _roll_day(self, timestamp: pd.Timestamp) -> None:
        day = timestamp.date()
        if self.state.current_day is None:
            self.state.current_day = day
            return
        if day != self.state.current_day:
            self.state.current_day = day
            self.state.daily_risk_used_pct = 0.0

    def _reject(
        self,
        requested_qty: float,
        *reason_codes: str,
        snapshot: Optional[dict[str, float]] = None,
    ) -> RiskDecision:
        snapshot = snapshot or {
            "gross_exposure_pct": 0.0,
            "net_exposure_pct": 0.0,
            "sector_exposure_pct": 0.0,
            "correlated_exposure_pct": 0.0,
        }
        return RiskDecision(
            approved=False,
            requested_qty=float(requested_qty),
            approved_qty=0.0,
            estimated_trade_risk_pct=0.0,
            daily_risk_used_pct=float(self.state.daily_risk_used_pct),
            gross_exposure_pct=float(snapshot["gross_exposure_pct"]),
            net_exposure_pct=float(snapshot["net_exposure_pct"]),
            sector_exposure_pct=float(snapshot["sector_exposure_pct"]),
            correlated_exposure_pct=float(snapshot["correlated_exposure_pct"]),
            reason_codes=[str(r) for r in reason_codes if str(r)],
        )

    def _quantize_qty(self, qty_abs: float) -> float:
        lot = max(float(self.config.lot_size), 1e-9)
        if qty_abs <= 0.0:
            return 0.0
        lots = np.floor(qty_abs / lot)
        return float(lots * lot)

    def _resolve_unit_risk(
        self,
        *,
        price: float,
        stop_price: Optional[float],
        atr: Optional[float],
        realized_vol: Optional[float],
    ) -> float:
        px = float(price)
        if stop_price is not None:
            dist = abs(px - float(stop_price))
            if np.isfinite(dist) and dist > 1e-12:
                return float(dist)

        if atr is not None:
            atr_val = abs(float(atr))
            if np.isfinite(atr_val) and atr_val > 1e-12:
                return float(atr_val)

        if realized_vol is not None:
            rv = abs(float(realized_vol))
            if np.isfinite(rv) and rv > 1e-12:
                return float(max(px * rv, px * self.config.min_stop_distance_pct))

        return float(max(px * self.config.min_stop_distance_pct, 1e-6))

    def _validate_portfolio_constraints(
        self,
        *,
        symbol: str,
        sector: Optional[str],
        signed_qty: float,
        price: float,
        equity: float,
        holdings: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
        correlation_matrix: Optional[pd.DataFrame],
    ) -> tuple[dict[str, float], list[str]]:
        positions = _normalize_holdings(holdings)
        target_symbol = str(symbol).upper().strip()
        target_sector = str(sector or positions.get(target_symbol, {}).get("sector", "unknown")).strip().lower()

        base_values: dict[str, float] = {
            sym: float(pos["qty"] * pos["price"])
            for sym, pos in positions.items()
        }
        base_sectors: dict[str, str] = {sym: str(pos["sector"]).strip().lower() for sym, pos in positions.items()}

        delta_value = float(signed_qty * price)
        base_values[target_symbol] = float(base_values.get(target_symbol, 0.0) + delta_value)
        base_sectors[target_symbol] = target_sector

        gross = sum(abs(v) for v in base_values.values()) / max(equity, 1e-9)
        net = sum(base_values.values()) / max(equity, 1e-9)

        sector_value = sum(
            abs(v)
            for sym, v in base_values.items()
            if base_sectors.get(sym, "unknown") == target_sector
        )
        sector_exposure = sector_value / max(equity, 1e-9)
        correlated_exposure = _compute_correlated_exposure(
            symbol=target_symbol,
            position_values=base_values,
            equity=equity,
            correlation_matrix=correlation_matrix,
            correlation_threshold=self.config.correlation_threshold,
        )

        snapshot = {
            "gross_exposure_pct": float(gross),
            "net_exposure_pct": float(net),
            "sector_exposure_pct": float(sector_exposure),
            "correlated_exposure_pct": float(correlated_exposure),
        }

        failures: list[str] = []
        if gross > float(self.config.max_gross_exposure_pct) + 1e-12:
            failures.append("gross_exposure_cap")
        if abs(net) > float(self.config.max_net_exposure_pct) + 1e-12:
            failures.append("net_exposure_cap")
        if target_sector and target_sector != "unknown":
            if sector_exposure > float(self.config.max_sector_exposure_pct) + 1e-12:
                failures.append("sector_exposure_cap")
        if correlated_exposure > float(self.config.max_correlated_exposure_pct) + 1e-12:
            failures.append("correlated_exposure_cap")

        return snapshot, failures


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _side_to_sign(side: str) -> float:
    token = (side or "").strip().lower()
    if token in {"buy", "long", "cover"}:
        return 1.0
    if token in {"sell", "short"}:
        return -1.0
    return 1.0


def _normalize_holdings(
    holdings: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    if holdings is None:
        return normalized

    if isinstance(holdings, Mapping):
        iterator = []
        for symbol, payload in holdings.items():
            if isinstance(payload, Mapping):
                iterator.append(
                    {
                        "symbol": symbol,
                        "qty": payload.get("qty", payload.get("quantity", 0.0)),
                        "price": payload.get("price", payload.get("close", payload.get("mark_price", 0.0))),
                        "sector": payload.get("sector", "unknown"),
                    }
                )
            else:
                iterator.append({"symbol": symbol, "qty": payload, "price": 0.0, "sector": "unknown"})
    else:
        iterator = list(holdings)

    for row in iterator:
        if not isinstance(row, Mapping):
            continue
        sym = str(row.get("symbol", "")).upper().strip()
        if not sym:
            continue
        qty = float(row.get("qty", row.get("quantity", 0.0)) or 0.0)
        price = float(row.get("price", row.get("close", row.get("mark_price", 0.0))) or 0.0)
        sector = str(row.get("sector", "unknown")).strip().lower()
        normalized[sym] = {"qty": qty, "price": price, "sector": sector}

    return normalized


def _compute_correlated_exposure(
    *,
    symbol: str,
    position_values: Mapping[str, float],
    equity: float,
    correlation_matrix: Optional[pd.DataFrame],
    correlation_threshold: float,
) -> float:
    if correlation_matrix is None or correlation_matrix.empty:
        return 0.0

    sym = str(symbol).upper().strip()
    matrix = correlation_matrix.copy()
    matrix.index = [str(x).upper() for x in matrix.index]
    matrix.columns = [str(x).upper() for x in matrix.columns]

    if sym not in matrix.index:
        return 0.0

    row = pd.to_numeric(matrix.loc[sym], errors="coerce")
    peers = row.index[np.abs(row.values) >= float(correlation_threshold)]
    exposure = 0.0
    for peer in peers:
        exposure += abs(float(position_values.get(str(peer).upper(), 0.0)))

    return float(exposure / max(equity, 1e-9))
