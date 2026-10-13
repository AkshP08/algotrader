"""Tests for the pairs-trading strategy.

Core guarantees:
  * Cointegration gate: a synthetic cointegrated pair PASSES; two independent
    random walks FAIL.
  * A non-cointegrated pair yields zero signals (refusal to trade).
  * A cointegrated pair with an induced divergence yields a paired entry
    (one BUY leg + one SELL leg) and a later paired EXIT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Side
from src.strategies.pairs_trading import PairsTrading


def _pair_frame(a, b):
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=len(a)), name="timestamp")
    return pd.DataFrame({"a_close": np.asarray(a, "float64"),
                         "b_close": np.asarray(b, "float64")}, index=idx)


def test_cointegrated_series_passes_check():
    rng = np.random.default_rng(42)
    b = np.cumsum(rng.normal(0, 1, 250)) + 100.0
    a = 2.0 * b + rng.normal(0, 1.0, 250)  # a tracks b + stationary noise
    strat = PairsTrading(symbol_a="XLE", symbol_b="XOP")
    assert strat.is_cointegrated(a, b) is True
    assert strat.cointegration_pvalue(a, b) <= 0.1


def test_two_random_walks_fail_check():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(0, 1, 250)) + 100.0
    b = np.cumsum(rng.normal(0, 1, 250)) + 100.0  # independent -> not cointegrated
    strat = PairsTrading(symbol_a="XLE", symbol_b="XOP")
    assert strat.is_cointegrated(a, b) is False
    assert strat.cointegration_pvalue(a, b) > 0.1


def test_non_cointegrated_pair_emits_no_signals():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(0, 1, 250)) + 100.0
    b = np.cumsum(rng.normal(0, 1, 250)) + 100.0
    strat = PairsTrading(symbol_a="XLE", symbol_b="XOP", lookback=60)
    assert strat.generate_signals(_pair_frame(a, b)) == []


def test_cointegrated_divergence_emits_paired_entry_and_exit():
    rng = np.random.default_rng(42)
    n = 300
    b = np.cumsum(rng.normal(0, 1, n)) + 100.0
    noise = rng.normal(0, 1.0, n)
    a = 2.0 * b + noise
    # Induce a large, temporary divergence in the middle that then reverts.
    a[150:160] += 12.0
    strat = PairsTrading(symbol_a="XLE", symbol_b="XOP",
                         lookback=60, entry_z=2.0, exit_z=0.5, max_hold_bars=40)
    signals = strat.generate_signals(_pair_frame(a, b))
    assert signals, "expected entry/exit signals on a diverging cointegrated pair"

    # First event is a paired entry: two legs, opposite sides, both symbols.
    entries = [s for s in signals if s.side in (Side.BUY, Side.SELL)]
    exits = [s for s in signals if s.side is Side.EXIT]
    assert len(entries) >= 2 and len(exits) >= 2

    first_two = signals[:2]
    assert {s.symbol for s in first_two} == {"XLE", "XOP"}
    assert {s.side for s in first_two} == {Side.BUY, Side.SELL}
    assert first_two[0].as_of == first_two[1].as_of  # same bar


def test_too_few_bars_no_signals():
    rng = np.random.default_rng(3)
    a = np.cumsum(rng.normal(0, 1, 40)) + 100.0
    b = 2 * a + rng.normal(0, 1, 40)
    strat = PairsTrading(lookback=60)
    assert strat.generate_signals(_pair_frame(a, b)) == []
