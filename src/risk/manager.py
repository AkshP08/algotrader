"""The risk manager — the single gate every order passes through.

``RiskManager.check_order(signal, portfolio_state)`` returns a
:class:`RiskDecision`. Nothing reaches the broker without an ``approved`` decision.
The checks, in order:

  1. **Kill-switch conditions** (daily loss / max drawdown). A breach returns
     ``kill_switch_triggered=True`` so the orchestration layer can halt trading.
  2. **Trading hours** (09:30-16:00 America/New_York, weekdays) when enabled.
  3. **Opening-order limits** (BUY / SELL-short): per-position size, position
     count, total exposure, and a minimum notional (dust guard).
  4. **Per-trade stop**: a hard 2% stop is enforced, overriding a strategy's
     looser hint (``modified_stop_loss``).

Closing orders (``EXIT``) reduce risk, so they bypass the opening-order limits —
but are still blocked while the kill switch would trigger / outside hours.

All math is pure Python and deterministic; no LLM/AI participates.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

from src.strategies.base import Side, Signal
from src.utils.config import RiskConfig, load_risk_config
from src.utils.logger import get_logger

log = get_logger("risk.manager")

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = _dt.time(9, 30)
_MARKET_CLOSE = _dt.time(16, 0)


@dataclass
class PortfolioState:
    """A snapshot of the account the risk manager reasons about.

    ``positions`` maps symbol -> signed quantity (long > 0, short < 0).
    ``prices`` maps symbol -> last price (used for notional/exposure math).
    """

    equity: float
    starting_equity: float  # start-of-day equity (basis for the daily-loss cap)
    peak_equity: float  # high-water mark (basis for the drawdown cap)
    positions: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    now: Optional[_dt.datetime] = None  # tz-aware; defaults handled at use site

    def open_position_count(self) -> int:
        return sum(1 for q in self.positions.values() if abs(q) > 1e-9)

    def total_exposure(self) -> float:
        return sum(
            abs(q) * self.prices.get(sym, 0.0) for sym, q in self.positions.items()
        )

    def price_for(self, symbol: str) -> Optional[float]:
        px = self.prices.get(symbol)
        return float(px) if px else None


@dataclass
class RiskDecision:
    """Outcome of a risk check. ``modified_quantity`` is set when the risk layer
    sized the order itself (strategy gave no hint); ``modified_stop_loss`` is the
    enforced hard stop when it is tighter than the strategy's hint."""

    approved: bool
    reason: str
    modified_quantity: Optional[float] = None
    kill_switch_triggered: bool = False
    modified_stop_loss: Optional[float] = None


class RiskManager:
    """Stateless gate configured by a :class:`RiskConfig`."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "RiskManager":
        return cls(load_risk_config(path))

    # --- helpers ----------------------------------------------------------

    def _daily_loss_pct(self, state: PortfolioState) -> float:
        if state.starting_equity <= 0:
            return 0.0
        return (state.starting_equity - state.equity) / state.starting_equity

    def _drawdown_pct(self, state: PortfolioState) -> float:
        if state.peak_equity <= 0:
            return 0.0
        return (state.peak_equity - state.equity) / state.peak_equity

    def _within_trading_hours(self, now: _dt.datetime) -> bool:
        et = now.astimezone(_ET) if now.tzinfo else now.replace(tzinfo=_ET)
        if et.weekday() >= 5:  # Sat/Sun
            return False
        return _MARKET_OPEN <= et.timetz().replace(tzinfo=None) <= _MARKET_CLOSE

    def _enforced_stop(self, side: Side, entry: float) -> float:
        pct = self.config.per_trade_stop_loss_pct
        # Long stops sit below entry; short stops sit above.
        return entry * (1 - pct) if side is Side.BUY else entry * (1 + pct)

    def _reconcile_stop(self, side: Side, entry: float, hint: Optional[float]) -> float:
        """Return the stop to use: the hard 2% stop, unless the strategy's hint is
        already tighter (closer to entry)."""
        enforced = self._enforced_stop(side, entry)
        if hint is None:
            return enforced
        if side is Side.BUY:
            return max(enforced, hint)  # higher stop = tighter for a long
        return min(enforced, hint)  # lower stop = tighter for a short

    # --- the gate ---------------------------------------------------------

    def check_order(self, signal: Signal, state: PortfolioState) -> RiskDecision:
        cfg = self.config
        symbol = signal.symbol

        # 1. Kill-switch conditions — checked first, before anything is allowed.
        dl = self._daily_loss_pct(state)
        if dl >= cfg.max_daily_loss_pct:
            reason = f"daily loss {dl:.2%} >= cap {cfg.max_daily_loss_pct:.2%}"
            log.critical("risk_daily_loss_breach", symbol=symbol, daily_loss=round(dl, 4))
            return RiskDecision(False, reason, kill_switch_triggered=True)
        dd = self._drawdown_pct(state)
        if dd >= cfg.max_drawdown_pct:
            reason = f"drawdown {dd:.2%} >= cap {cfg.max_drawdown_pct:.2%}"
            log.critical("risk_drawdown_breach", symbol=symbol, drawdown=round(dd, 4))
            return RiskDecision(False, reason, kill_switch_triggered=True)

        # 2. Trading hours.
        if cfg.trading_hours_only:
            now = state.now or _dt.datetime.now(_dt.timezone.utc)
            if not self._within_trading_hours(now):
                return RiskDecision(
                    False, f"outside trading hours (09:30-16:00 ET): {now.isoformat()}"
                )

        price = state.price_for(symbol)
        if price is None or price <= 0:
            return RiskDecision(False, f"no valid price for {symbol}")

        # --- EXIT: closing reduces risk -> bypass opening-order limits. ---
        if signal.side is Side.EXIT:
            return RiskDecision(True, "exit approved (risk-reducing)")

        # 3. Opening orders (BUY / SELL-short).
        # Size the order if the strategy gave no hint: cap at max_position_size_pct.
        max_pos_notional = cfg.max_position_size_pct * state.equity
        modified_qty: Optional[float] = None
        if signal.quantity_hint is None:
            qty = math.floor(max_pos_notional / price)
            modified_qty = float(qty)
        else:
            qty = float(signal.quantity_hint)
        notional = qty * price

        if qty <= 0:
            return RiskDecision(
                False, f"sized quantity is zero (price {price:.2f} > budget "
                f"{max_pos_notional:.2f})", modified_quantity=modified_qty,
            )

        # min notional (dust guard)
        if notional < cfg.min_order_notional:
            return RiskDecision(
                False,
                f"order notional ${notional:.2f} < min ${cfg.min_order_notional:.2f}",
                modified_quantity=modified_qty,
            )

        # per-position size cap
        if notional > max_pos_notional + 1e-6:
            return RiskDecision(
                False,
                f"position notional ${notional:.2f} > {cfg.max_position_size_pct:.0%} "
                f"of equity (${max_pos_notional:.2f})",
                modified_quantity=modified_qty,
            )

        # max concurrent positions (only new symbols count against the cap)
        is_new_symbol = abs(state.positions.get(symbol, 0.0)) <= 1e-9
        if is_new_symbol and state.open_position_count() >= cfg.max_positions:
            return RiskDecision(
                False,
                f"max positions reached ({cfg.max_positions})",
                modified_quantity=modified_qty,
            )

        # max total exposure
        projected = state.total_exposure() + notional
        max_exposure = cfg.max_total_exposure_pct * state.equity
        if projected > max_exposure + 1e-6:
            return RiskDecision(
                False,
                f"projected exposure ${projected:.2f} > {cfg.max_total_exposure_pct:.0%} "
                f"of equity (${max_exposure:.2f})",
                modified_quantity=modified_qty,
            )

        # 4. Enforce the hard per-trade stop (override looser strategy hints).
        enforced_stop = self._reconcile_stop(signal.side, price, signal.stop_loss)

        log.info(
            "risk_approved",
            symbol=symbol,
            side=signal.side.value,
            qty=qty,
            notional=round(notional, 2),
            stop=round(enforced_stop, 4),
        )
        return RiskDecision(
            True,
            "approved",
            modified_quantity=modified_qty,
            modified_stop_loss=enforced_stop,
        )
