"""Pairs (statistical-arbitrage) trading strategy.

Trades the mean-reverting spread between two cointegrated symbols:

    spread_t = log(price_a_t) - beta * log(price_b_t)

where ``beta`` is the OLS hedge ratio estimated over a trailing ``lookback``
window. The spread's rolling z-score drives entries and exits:

  * Entry when ``|z| > entry_z``: short the overpriced leg, long the underpriced
    leg (two signals — one BUY, one SELL-SHORT).
  * Exit  when ``|z| < exit_z`` **or** after ``max_hold_bars`` bars.

A **cointegration gate** (Engle-Granger via statsmodels) runs first: if the
ADF-based p-value for the pair exceeds ``adf_max_pvalue`` (0.1) the strategy
refuses to trade at all. Everything is causal — bar ``i`` uses only bars ``<= i``.

Interface note: the base ``generate_signals(bars)`` takes a single OHLCV frame,
but a pair needs two price series. This strategy therefore consumes a frame with
aligned ``a_close`` / ``b_close`` columns; :meth:`build_pair_frame` constructs
one from two OHLCV frames. Trade math is pure Python; no LLM/AI touches it.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.strategies.base import Fill, Signal, Strategy
from src.utils.logger import get_logger

log = get_logger("strategy.pairs_trading")

PAIR_COLUMNS = ("a_close", "b_close")


class PairsTrading(Strategy):
    """Cointegration-gated mean-reversion on a two-symbol log-price spread."""

    def __init__(
        self,
        symbol_a: str = "XLE",
        symbol_b: str = "XOP",
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_hold_bars: int = 20,
        adf_max_pvalue: float = 0.1,
        name: str | None = None,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")
        super().__init__(
            name=name or f"pairs_trading[{symbol_a}/{symbol_b},lb={lookback}]",
            symbols=(symbol_a, symbol_b),
            lookback_bars=lookback,
        )
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_hold_bars = max_hold_bars
        self.adf_max_pvalue = adf_max_pvalue
        self._position = 0  # +1 long-spread, -1 short-spread, 0 flat

    # --- Pair-frame helpers ------------------------------------------------

    @staticmethod
    def build_pair_frame(bars_a: pd.DataFrame, bars_b: pd.DataFrame) -> pd.DataFrame:
        """Align two OHLCV frames into a single ``a_close``/``b_close`` frame.

        Inner-joins on the (ascending) timestamp index so only timestamps present
        in *both* series survive — no forward-filling, no fabricated bars.
        """
        joined = pd.DataFrame(
            {
                "a_close": bars_a["close"],
                "b_close": bars_b["close"],
            }
        ).dropna()
        return joined.sort_index()

    def _validate_pair(self, bars: pd.DataFrame) -> None:
        missing = [c for c in PAIR_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"pair bars missing required columns: {missing}")
        if not bars.index.is_monotonic_increasing:
            raise ValueError("pair bars index must be sorted ascending (no look-ahead)")

    # --- Cointegration gate ------------------------------------------------

    def cointegration_pvalue(self, a_close, b_close) -> float:
        """Engle-Granger cointegration p-value on log prices (lower = tighter)."""
        la = np.log(np.asarray(a_close, dtype="float64"))
        lb = np.log(np.asarray(b_close, dtype="float64"))
        # coint() is the Engle-Granger two-step test; element [1] is the p-value.
        return float(coint(la, lb)[1])

    def is_cointegrated(self, a_close, b_close) -> bool:
        return self.cointegration_pvalue(a_close, b_close) <= self.adf_max_pvalue

    # --- Spread / hedge-ratio math ----------------------------------------

    @staticmethod
    def _ols_beta(log_a: np.ndarray, log_b: np.ndarray) -> float:
        """OLS slope of ``log_a`` on ``log_b`` (with intercept)."""
        X = np.column_stack([np.ones_like(log_b), log_b])
        # Least squares: coef = (X'X)^-1 X'y ; coef[1] is the slope (beta).
        coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
        return float(coef[1])

    # --- Signal generation -------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> list[Signal]:
        self._validate_pair(bars)
        n = len(bars)
        if n < self.lookback:
            return []

        la_all = np.log(bars["a_close"].to_numpy(dtype="float64"))
        lb_all = np.log(bars["b_close"].to_numpy(dtype="float64"))

        # Cointegration gate on the full provided history. Refuse if it fails.
        pval = self.cointegration_pvalue(bars["a_close"], bars["b_close"])
        if pval > self.adf_max_pvalue:
            log.warning(
                "pairs_not_cointegrated",
                symbol_a=self.symbol_a,
                symbol_b=self.symbol_b,
                adf_pvalue=round(pval, 6),
                threshold=self.adf_max_pvalue,
            )
            return []

        idx = bars.index
        position = 0  # local scan state: +1 long-spread, -1 short-spread
        entry_pos = 0  # index of the bar we entered on
        signals: list[Signal] = []

        # Start once a full trailing window exists at bar i (i >= lookback-1).
        for i in range(self.lookback - 1, n):
            lo = i - self.lookback + 1
            win_a = la_all[lo : i + 1]
            win_b = lb_all[lo : i + 1]
            beta = self._ols_beta(win_a, win_b)
            spread_win = win_a - beta * win_b
            mu = spread_win.mean()
            sd = spread_win.std(ddof=1)
            if sd == 0 or not np.isfinite(sd):
                continue
            z = (spread_win[-1] - mu) / sd
            ts = idx[i]

            if position == 0:
                if abs(z) > self.entry_z:
                    position = 1 if z < 0 else -1  # z<0: spread low -> long spread
                    entry_pos = i
                    signals.extend(self._entry_signals(ts, z, beta, bars.iloc[i]))
            else:
                held = i - entry_pos
                if abs(z) < self.exit_z or held >= self.max_hold_bars:
                    reason = (
                        f"|z|={abs(z):.2f} < exit_z={self.exit_z}"
                        if abs(z) < self.exit_z
                        else f"max_hold_bars={self.max_hold_bars} reached"
                    )
                    signals.extend(self._exit_signals(ts, z, reason))
                    position = 0
                    entry_pos = 0

        log.info(
            "pairs_trading_signals",
            symbol_a=self.symbol_a,
            symbol_b=self.symbol_b,
            bars=n,
            adf_pvalue=round(pval, 6),
            signals=len(signals),
        )
        return signals

    def _entry_signals(self, ts, z: float, beta: float, row: pd.Series) -> list[Signal]:
        """Two legs. z>0: a overpriced -> SELL a / BUY b. z<0: BUY a / SELL b."""
        conf = min(1.0, (abs(z) - self.entry_z) / self.entry_z + 0.5)
        a_side = "SELL" if z > 0 else "BUY"
        b_side = "BUY" if z > 0 else "SELL"
        reason = f"spread z={z:+.2f} (|z|>{self.entry_z}); hedge beta={beta:.3f}"
        meta = {"strategy": self.name, "z": float(z), "beta": float(beta), "leg": "pair"}
        return [
            Signal(symbol=self.symbol_a, side=a_side, quantity_hint=None,
                   confidence=conf, entry_reason=f"[A] {reason}", as_of=ts,
                   meta={**meta, "role": "a"}),
            Signal(symbol=self.symbol_b, side=b_side, quantity_hint=None,
                   confidence=conf, entry_reason=f"[B] {reason}", as_of=ts,
                   meta={**meta, "role": "b"}),
        ]

    def _exit_signals(self, ts, z: float, reason: str) -> list[Signal]:
        meta = {"strategy": self.name, "z": float(z), "leg": "pair"}
        return [
            Signal(symbol=self.symbol_a, side="EXIT", confidence=0.0,
                   entry_reason=f"[A] exit: {reason}", as_of=ts, meta={**meta, "role": "a"}),
            Signal(symbol=self.symbol_b, side="EXIT", confidence=0.0,
                   entry_reason=f"[B] exit: {reason}", as_of=ts, meta={**meta, "role": "b"}),
        ]

    def on_fill(self, fill: Fill) -> None:
        log.info("pairs_fill", symbol=fill.symbol, side=fill.side.value, qty=fill.filled_qty)

    def on_reject(self, order: Any) -> None:
        log.error("pairs_trading_order_rejected", order=str(order))

    def state_snapshot(self) -> dict:
        return {
            "name": self.name,
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "lookback": self.lookback,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "max_hold_bars": self.max_hold_bars,
            "position": self._position,
        }
