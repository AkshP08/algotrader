"""Tests for the moving-average crossover strategy.

Properties under test:
  * BUY on up-cross, EXIT on down-cross — and ONLY on the crossover bar.
  * Signal count matches an independent recomputation of the crossovers.
  * BUY signals carry an ATR-based stop (never open-ended); exit logic present.
  * Not enough bars -> no signals; fast >= slow is rejected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Side
from src.strategies.indicators import sma
from src.strategies.ma_crossover import MovingAverageCrossover


def _bars_from_closes(closes):
    idx = pd.DatetimeIndex(
        pd.bdate_range("2023-01-02", periods=len(closes)), name="timestamp"
    )
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(len(closes), 1_000.0),
        },
        index=idx,
    )


def _expected_crossovers(closes, fast_w, slow_w):
    df = _bars_from_closes(closes)
    diff = (sma(df["close"], fast_w) - sma(df["close"], slow_w))
    prev = diff.shift(1)
    valid = diff.notna() & prev.notna()
    up = valid & (prev <= 0) & (diff > 0)
    down = valid & (prev >= 0) & (diff < 0)
    return int(up.sum()), int(down.sum())


def test_rejects_fast_ge_slow():
    with pytest.raises(ValueError):
        MovingAverageCrossover(symbol="SPY", fast_window=50, slow_window=50)


def test_too_few_bars_no_signals():
    strat = MovingAverageCrossover(symbol="SPY", fast_window=3, slow_window=5)
    bars = _bars_from_closes([100, 101, 102])  # < slow_window + 1
    assert strat.generate_signals(bars) == []


def _sine_closes():
    # A smooth oscillation produces several clean crossovers AFTER warmup,
    # avoiding the degenerate "already crossed before the SMA is valid" case.
    t = np.arange(80)
    return 100 + 10 * np.sin(t / 5.0)


def test_buy_on_up_cross_exit_on_down_cross():
    closes = _sine_closes()
    strat = MovingAverageCrossover(symbol="SPY", fast_window=3, slow_window=5)
    signals = strat.generate_signals(_bars_from_closes(closes))

    buys = [s for s in signals if s.side is Side.BUY]
    exits = [s for s in signals if s.side is Side.EXIT]
    assert len(buys) >= 1, "expected at least one golden cross"
    assert len(exits) >= 1, "expected at least one death cross"

    # Count matches an independent crossover recomputation (only on cross bars).
    n_up, n_down = _expected_crossovers(closes, 3, 5)
    assert len(buys) == n_up
    assert len(exits) == n_down


def test_buy_signal_has_atr_stop_and_exit_has_none():
    closes = _sine_closes()
    bars = _bars_from_closes(closes)
    strat = MovingAverageCrossover(symbol="SPY", fast_window=3, slow_window=5)
    signals = strat.generate_signals(bars)
    assert signals
    for s in signals:
        if s.side is Side.BUY:
            assert s.stop_loss is not None
            assert s.stop_loss < float(bars.loc[s.as_of, "close"])
        else:
            assert s.stop_loss is None
        assert s.as_of is not None  # every decision is timestamped


def test_signal_only_on_crossover_bar_not_while_condition_holds():
    # Flat (fast == slow) then a sustained uptrend: fast stays above slow for many
    # bars but a BUY must be emitted on exactly ONE bar (the cross), not after.
    closes = [100.0] * 10 + list(np.arange(100, 140, dtype="float64"))
    strat = MovingAverageCrossover(symbol="SPY", fast_window=3, slow_window=5)
    buys = [s for s in strat.generate_signals(_bars_from_closes(closes))
            if s.side is Side.BUY]
    assert len(buys) == 1


def test_determinism():
    closes = _sine_closes()
    strat = MovingAverageCrossover(symbol="SPY", fast_window=3, slow_window=5)
    bars = _bars_from_closes(closes)
    a = strat.generate_signals(bars)
    b = strat.generate_signals(bars)
    assert [(s.side, s.as_of) for s in a] == [(s.side, s.as_of) for s in b]
