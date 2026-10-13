"""Market-regime detector.

Computes two independent signals from a benchmark (SPY by default) and combines
them into one of four regimes that gate which strategies may trade:

  * **Volatility** — the annualized standard deviation of the last ``vol_window``
    daily returns. ``HIGH`` if it exceeds ``vol_threshold`` (25%), else ``LOW``.
  * **Trend** — Wilder's ADX(14) on the benchmark's daily bars. ``TRENDING`` if
    ADX exceeds ``adx_threshold`` (25), else ``RANGING``.

The four corners (:class:`Regime`) are the Cartesian product of the two. Each
strategy declares an ``allowed_regimes`` set; :meth:`active_strategies` filters a
set of strategies down to those permitted in the current regime.

All math is pure Python/NumPy and deterministic — no LLM/AI participates in the
gating decision. The computed regime is persisted to the ``regime_state`` table
for audit. There is no look-ahead: every value at the latest bar derives only
from bars at or before it.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, TypeVar

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.storage.db import RegimeState, get_sessionmaker, init_db, make_engine
from src.strategies.indicators import adx
from src.utils.logger import get_logger

log = get_logger("regime.detector")

TRADING_DAYS_PER_YEAR = 252


class VolRegime(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class TrendRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"


class Regime(str, Enum):
    """The four combined regimes a strategy can be gated on."""

    HIGH_VOL_TRENDING = "HIGH_VOL_TRENDING"
    HIGH_VOL_RANGING = "HIGH_VOL_RANGING"
    LOW_VOL_TRENDING = "LOW_VOL_TRENDING"
    LOW_VOL_RANGING = "LOW_VOL_RANGING"

    @classmethod
    def of(cls, vol: VolRegime, trend: TrendRegime) -> "Regime":
        return cls[f"{vol.value}_VOL_{trend.value}"]


@dataclass(frozen=True)
class RegimeResult:
    """The outcome of a regime computation (plus the raw signals behind it)."""

    as_of: pd.Timestamp
    vol_regime: VolRegime
    trend_regime: TrendRegime
    combined: Regime
    realized_vol: float
    adx: float


_S = TypeVar("_S")


class RegimeDetector:
    """Classifies the market regime from a benchmark's daily OHLCV bars."""

    def __init__(
        self,
        benchmark_symbol: str = "SPY",
        vol_window: int = 20,
        vol_threshold: float = 0.25,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        engine: Optional[Engine] = None,
    ) -> None:
        self.benchmark_symbol = benchmark_symbol
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.engine = engine or make_engine()
        init_db(self.engine)
        self._Session = get_sessionmaker(self.engine)

    # --- signals ----------------------------------------------------------

    def realized_vol(self, close: pd.Series) -> float:
        """Annualized stdev of the last ``vol_window`` daily simple returns.

        Uses the sample standard deviation (ddof=1) of the trailing returns,
        scaled by ``sqrt(252)``. Requires ``vol_window + 1`` closes.
        """
        returns = close.astype("float64").pct_change().dropna()
        if len(returns) < self.vol_window:
            raise ValueError(
                f"need >= {self.vol_window} returns for realized_vol, "
                f"got {len(returns)}"
            )
        window = returns.to_numpy()[-self.vol_window :]
        daily = float(np.std(window, ddof=1))
        return daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    def compute_adx(self, bars: pd.DataFrame) -> float:
        """Latest Wilder ADX(``adx_period``) value from daily bars."""
        series = adx(bars, self.adx_period)
        last = series.iloc[-1]
        if pd.isna(last):
            raise ValueError(
                f"not enough bars to compute ADX({self.adx_period}): "
                f"need >= {2 * self.adx_period}, got {len(bars)}"
            )
        return float(last)

    # --- classification ---------------------------------------------------

    def classify(
        self, realized_vol: float, adx_value: float
    ) -> tuple[VolRegime, TrendRegime, Regime]:
        vol = VolRegime.HIGH if realized_vol > self.vol_threshold else VolRegime.LOW
        trend = (
            TrendRegime.TRENDING if adx_value > self.adx_threshold else TrendRegime.RANGING
        )
        return vol, trend, Regime.of(vol, trend)

    def detect(self, bars: pd.DataFrame) -> RegimeResult:
        """Compute the regime from a benchmark OHLCV frame (ascending index)."""
        if not bars.index.is_monotonic_increasing:
            raise ValueError("bars index must be sorted ascending (no look-ahead)")
        rv = self.realized_vol(bars["close"])
        adx_value = self.compute_adx(bars)
        vol, trend, combined = self.classify(rv, adx_value)
        result = RegimeResult(
            as_of=pd.Timestamp(bars.index[-1]),
            vol_regime=vol,
            trend_regime=trend,
            combined=combined,
            realized_vol=rv,
            adx=adx_value,
        )
        log.info(
            "regime_detected",
            symbol=self.benchmark_symbol,
            combined=combined.value,
            realized_vol=round(rv, 4),
            adx=round(adx_value, 2),
            as_of=str(result.as_of),
        )
        return result

    # --- persistence ------------------------------------------------------

    def persist(self, result: RegimeResult, now: Optional[_dt.datetime] = None) -> None:
        """Write the regime to ``regime_state`` for audit."""
        ts = now or _dt.datetime.now(_dt.timezone.utc)
        with self._Session() as session:
            session.add(
                RegimeState(
                    timestamp=ts,
                    vol_regime=result.vol_regime.value,
                    trend_regime=result.trend_regime.value,
                    combined=result.combined.value,
                    realized_vol=result.realized_vol,
                    adx=result.adx,
                )
            )
            session.commit()

    def latest(self) -> Optional[RegimeState]:
        """Return the most recently persisted regime row (or None)."""
        from sqlalchemy import desc, select

        with self._Session() as session:
            return (
                session.execute(
                    select(RegimeState).order_by(desc(RegimeState.id)).limit(1)
                )
                .scalars()
                .first()
            )

    # --- strategy gating --------------------------------------------------

    @staticmethod
    def active_strategies(
        strategies: Mapping[str, _S], regime: Regime
    ) -> dict[str, _S]:
        """Filter ``strategies`` to those whose ``allowed_regimes`` admit ``regime``.

        Works on any object exposing an ``allowed_regimes`` set, so it does not
        import the concrete strategy classes (keeps the dependency one-directional:
        strategies know about regimes, not the reverse).
        """
        active: dict[str, _S] = {}
        for name, strat in strategies.items():
            allowed = getattr(strat, "allowed_regimes", None)
            if allowed and regime in allowed:
                active[name] = strat
        return active
