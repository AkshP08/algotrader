"""RSI mean-reversion strategy.

Entry : RSI crosses **below** ``oversold`` -> BUY (betting on reversion up)
Exit  : RSI crosses **above** the 50 midline -> EXIT (or a hard stop, enforced
        downstream by the risk manager's per-trade stop)

RSI is computed with the ``ta`` library (Wilder smoothing). A signal fires only
on the bar where the threshold is crossed, comparing RSI at ``i`` against ``i-1``
— never on every bar the condition holds. Decisions are made on bar ``i``'s close
with no look-ahead. Trade math is pure Python; no LLM/AI touches this path.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from ta.momentum import RSIIndicator

from src.regime.detector import Regime
from src.strategies.base import Fill, Signal, Strategy
from src.strategies.indicators import atr
from src.utils.logger import get_logger

log = get_logger("strategy.rsi_mean_reversion")

ATR_WINDOW = 14
ATR_STOP_MULT = 1.5  # stop-loss hint: 1.5 * ATR(14) below entry
MIDLINE = 50.0


class RSIMeanReversion(Strategy):
    """Buy oversold RSI dips; exit when RSI reverts through the midline."""

    # Mean reversion dies in volatile markets — restrict to calm, rangebound tape.
    allowed_regimes: set[Regime] = {Regime.LOW_VOL_RANGING}

    def __init__(
        self,
        symbol: str = "QQQ",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        name: str | None = None,
    ) -> None:
        if not 0 < oversold < MIDLINE < overbought < 100:
            raise ValueError(
                "require 0 < oversold < 50 < overbought < 100, got "
                f"oversold={oversold}, overbought={overbought}"
            )
        super().__init__(
            name=name or f"rsi_mean_reversion[{symbol},{rsi_period}/{oversold:g}/{overbought:g}]",
            symbols=(symbol,),
            lookback_bars=rsi_period + 1,
        )
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self._position_qty = 0.0

    def compute_rsi(self, close: pd.Series) -> pd.Series:
        """RSI via ``ta`` (Wilder). Isolated so tests can target it directly."""
        return RSIIndicator(close=close, window=self.rsi_period).rsi()

    def generate_signals(self, bars: pd.DataFrame) -> list[Signal]:
        self.validate_bars(bars)
        if len(bars) < self.rsi_period + 1:
            return []

        close = bars["close"]
        rsi = self.compute_rsi(close)
        atr14 = atr(bars, ATR_WINDOW)

        prev = rsi.shift(1)
        valid = rsi.notna() & prev.notna()
        entry = valid & (prev >= self.oversold) & (rsi < self.oversold)
        exit_ = valid & (prev <= MIDLINE) & (rsi > MIDLINE)

        signals: list[Signal] = []
        for ts in bars.index[entry | exit_]:
            px = float(close.loc[ts])
            is_entry = bool(entry.loc[ts])
            a = atr14.loc[ts]
            r = float(rsi.loc[ts])
            stop = (
                float(px - ATR_STOP_MULT * a)
                if is_entry and pd.notna(a)
                else None
            )
            if is_entry:
                # Deeper oversold -> higher confidence (0 at threshold, 1 at RSI 0).
                conf = max(0.0, min(1.0, (self.oversold - r) / self.oversold))
            else:
                conf = 0.0
            signals.append(
                Signal(
                    symbol=self.symbol,
                    side="BUY" if is_entry else "EXIT",
                    quantity_hint=None,
                    confidence=conf,
                    entry_reason=(
                        f"RSI({self.rsi_period}) crossed "
                        f"{'below ' + format(self.oversold, 'g') if is_entry else 'above 50'} "
                        f"(rsi={r:.2f})"
                    ),
                    stop_loss=stop,
                    as_of=ts,
                    meta={"strategy": self.name, "rsi": r},
                )
            )
        log.info(
            "rsi_mean_reversion_signals",
            symbol=self.symbol,
            bars=len(bars),
            signals=len(signals),
        )
        return signals

    def on_fill(self, fill: Fill) -> None:
        delta = fill.filled_qty if fill.side.value == "BUY" else -fill.filled_qty
        self._position_qty += delta

    def on_reject(self, order: Any) -> None:
        log.error("rsi_mean_reversion_order_rejected", order=str(order))

    def state_snapshot(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "position_qty": self._position_qty,
        }
