"""Tests for the RSI mean-reversion strategy.

The headline test compares the strategy's ``ta``-computed RSI against an
INDEPENDENT from-scratch Wilder implementation (no ``ta`` import) — a genuine
hand-computed reference — and requires a machine-precision match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Side
from src.strategies.rsi_mean_reversion import RSIMeanReversion


def _bars_from_closes(closes):
    idx = pd.DatetimeIndex(
        pd.bdate_range("2023-01-02", periods=len(closes)), name="timestamp"
    )
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(closes), 1_000.0),
        },
        index=idx,
    )


def _wilder_rsi_from_scratch(prices, period=14):
    """RSI from first principles (Wilder smoothing, no `ta`). Mirrors the EMA
    recursion ta uses: seed at the first diff, alpha = 1/period, adjust=False.
    """
    p = np.asarray(prices, dtype="float64")
    n = len(p)
    diff = np.empty(n)
    diff[0] = np.nan
    diff[1:] = np.diff(p)
    up = np.where(diff > 0, diff, 0.0)      # NaN>0 is False -> 0.0 (as ta does)
    dn = np.where(diff < 0, -diff, 0.0)
    a = 1.0 / period
    au = np.empty(n)
    ad = np.empty(n)
    au[0], ad[0] = up[0], dn[0]
    for i in range(1, n):
        au[i] = (1 - a) * au[i - 1] + a * up[i]
        ad[i] = (1 - a) * ad[i - 1] + a * dn[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(ad == 0, 100.0, 100 - 100 / (1 + au / ad))
    rsi[: period - 1] = np.nan  # min_periods = window
    return rsi


def test_rsi_matches_hand_computed_reference():
    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.normal(0, 1, 120))
    strat = RSIMeanReversion(symbol="QQQ", rsi_period=14)

    ta_rsi = strat.compute_rsi(pd.Series(prices)).to_numpy()
    ref = _wilder_rsi_from_scratch(prices, 14)

    valid = ~np.isnan(ta_rsi) & ~np.isnan(ref)
    assert valid.sum() > 100
    assert np.max(np.abs(ta_rsi[valid] - ref[valid])) < 1e-9


def test_rsi_all_gains_is_100():
    # Strictly increasing -> no losses -> RSI exactly 100 (trivial hand check).
    strat = RSIMeanReversion(rsi_period=14)
    rsi = strat.compute_rsi(pd.Series(np.arange(1, 40, dtype="float64")))
    assert np.isclose(rsi.iloc[-1], 100.0)


def _dip_then_recover():
    # Rise first (RSI established well above 30), then decline through 30 (a real
    # downward CROSS), then recover through the 50 midline (an upward cross).
    up1 = list(np.linspace(90, 112, 16))
    down = list(np.linspace(112, 86, 22))
    up2 = list(np.linspace(86, 125, 26))
    return up1 + down + up2


def test_buy_on_cross_below_oversold_and_exit_above_midline():
    closes = _dip_then_recover()
    strat = RSIMeanReversion(symbol="QQQ", rsi_period=14, oversold=30, overbought=70)
    signals = strat.generate_signals(_bars_from_closes(closes))

    buys = [s for s in signals if s.side is Side.BUY]
    exits = [s for s in signals if s.side is Side.EXIT]
    assert len(buys) >= 1, "expected an oversold entry"
    assert len(exits) >= 1, "expected a midline exit"
    # The entry precedes the exit in time.
    assert buys[0].as_of < exits[-1].as_of
    # Entry carries a stop (1.5*ATR below); never open-ended.
    assert buys[0].stop_loss is not None
    assert buys[0].stop_loss < float(_bars_from_closes(closes).loc[buys[0].as_of, "close"])


def test_only_one_signal_per_threshold_cross():
    # One dip below 30 then a small recovery: exactly one BUY (on the cross).
    closes = _dip_then_recover()
    strat = RSIMeanReversion(symbol="QQQ", rsi_period=14, oversold=30)
    buys = [s for s in strat.generate_signals(_bars_from_closes(closes))
            if s.side is Side.BUY]
    assert len(buys) == 1
