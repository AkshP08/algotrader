"""Moving-average crossover strategy.

Entry : SMA(fast) crosses **above** SMA(slow)  -> BUY
Exit  : SMA(fast) crosses **below** SMA(slow)  -> EXIT

A signal is emitted **only on the bar where the crossover occurs** — not on every
bar the condition holds — by comparing the sign of ``fast - slow`` at bar ``i``
against bar ``i-1``. The decision is made on bar ``i``'s close (``as_of = i``);
the fill belongs to the *next* bar, which the executor / Null backtester places.
No look-ahead: every value used at bar ``i`` is derived from bars ``<= i``.

Trade math is pure Python; no LLM/AI touches this path.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.regime.detector import Regime
from src.strategies.base import Fill, Signal, Strategy
from src.strategies.indicators import atr, sma
from src.utils.logger import get_logger

log = get_logger("strategy.ma_crossover")

ATR_WINDOW = 14
ATR_STOP_MULT = 2.0  # stop-loss hint: 2 * ATR(14) below entry


class MovingAverageCrossover(Strategy):
    """SMA(fast)/SMA(slow) crossover with an ATR-based stop hint."""

    # Trend-following: only run when the market is actually trending.
    allowed_regimes: set[Regime] = {
        Regime.LOW_VOL_TRENDING,
        Regime.HIGH_VOL_TRENDING,
    }

    def __init__(
        self,
        symbol: str = "SPY",
        fast_window: int = 20,
        slow_window: int = 50,
        name: str | None = None,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be < slow_window ({slow_window})"
            )
        super().__init__(
            name=name or f"ma_crossover[{symbol},{fast_window}/{slow_window}]",
            symbols=(symbol,),
            # Need slow_window bars for the SMA plus one prior bar to see the cross.
            lookback_bars=slow_window + 1,
        )
        self.symbol = symbol
        self.fast_window = fast_window
        self.slow_window = slow_window
        self._position_qty = 0.0

    def generate_signals(self, bars: pd.DataFrame) -> list[Signal]:
        self.validate_bars(bars)
        if len(bars) < self.slow_window + 1:
            return []

        close = bars["close"]
        fast = sma(close, self.fast_window)
        slow = sma(close, self.slow_window)
        atr14 = atr(bars, ATR_WINDOW)

        diff = fast - slow
        prev_diff = diff.shift(1)

        # Crossover only: sign flip between the previous bar and this bar.
        both_valid = diff.notna() & prev_diff.notna()
        up_cross = both_valid & (prev_diff <= 0) & (diff > 0)
        down_cross = both_valid & (prev_diff >= 0) & (diff < 0)

        signals: list[Signal] = []
        for ts in bars.index[up_cross | down_cross]:
            px = float(close.loc[ts])
            spread = float(diff.loc[ts])
            is_entry = bool(up_cross.loc[ts])
            a = atr14.loc[ts]
            stop = (
                float(px - ATR_STOP_MULT * a)
                if is_entry and pd.notna(a)
                else None
            )
            # Confidence: crossover magnitude relative to price, clamped to [0, 1].
            conf = min(1.0, abs(spread) / px) if px else 0.0
            signals.append(
                Signal(
                    symbol=self.symbol,
                    side="BUY" if is_entry else "EXIT",
                    quantity_hint=None,  # risk layer sizes the order
                    confidence=conf,
                    entry_reason=(
                        f"SMA{self.fast_window} crossed "
                        f"{'above' if is_entry else 'below'} SMA{self.slow_window} "
                        f"(fast-slow={spread:+.4f})"
                    ),
                    stop_loss=stop,
                    as_of=ts,
                    meta={"strategy": self.name, "atr14": None if pd.isna(a) else float(a)},
                )
            )
        log.info(
            "ma_crossover_signals",
            symbol=self.symbol,
            bars=len(bars),
            signals=len(signals),
        )
        return signals

    # --- live-state hooks (audit only; the scan above is pure) --------------

    def on_fill(self, fill: Fill) -> None:
        delta = fill.filled_qty if fill.side.value == "BUY" else -fill.filled_qty
        self._position_qty += delta

    def on_reject(self, order: Any) -> None:
        log.error("ma_crossover_order_rejected", order=str(order))

    def state_snapshot(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "position_qty": self._position_qty,
        }
