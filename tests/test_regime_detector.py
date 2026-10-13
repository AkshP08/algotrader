"""Tests for the market-regime detector.

``realized_vol`` and ADX are checked against independent, clean-room reference
implementations computed over a deterministic 60-bar synthetic series (the series
exists only to exercise the math — it is never fed to the live trading path).
Classification is checked at all four corners, and strategy gating is checked per
regime against the three real strategies.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    Regime,
    RegimeDetector,
    TrendRegime,
    VolRegime,
)
from src.storage.db import RegimeState
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.pairs_trading import PairsTrading
from src.strategies.rsi_mean_reversion import RSIMeanReversion


def _synthetic_ohlcv(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """A deterministic OHLCV frame (seeded RNG) — for indicator math only."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    vol = rng.integers(1_000_000, 2_000_000, n).astype(float)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _reference_realized_vol(close: pd.Series, window: int = 20) -> float:
    """Independent realized-vol: sample-stdev of last `window` simple returns * sqrt(252)."""
    c = close.to_numpy(dtype="float64")
    rets = c[1:] / c[:-1] - 1.0
    w = rets[-window:]
    mean = w.sum() / len(w)
    var = ((w - mean) ** 2).sum() / (len(w) - 1)
    return float(var**0.5 * (252**0.5))


def _reference_adx(high, low, close, period: int = 14) -> float:
    """Clean-room Wilder ADX (classic sum-seeding); returns the final value."""
    n = len(close)
    pdm = [0.0] * n
    mdm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = [None] * n
    pds = [None] * n
    mds = [None] * n
    atr[period] = sum(tr[1 : period + 1])
    pds[period] = sum(pdm[1 : period + 1])
    mds[period] = sum(mdm[1 : period + 1])
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
        pds[i] = pds[i - 1] - pds[i - 1] / period + pdm[i]
        mds[i] = mds[i - 1] - mds[i - 1] / period + mdm[i]

    dx = [None] * n
    for i in range(period, n):
        if atr[i] == 0:
            dx[i] = 0.0
            continue
        pdi = 100.0 * pds[i] / atr[i]
        mdi = 100.0 * mds[i] / atr[i]
        denom = pdi + mdi
        dx[i] = 0.0 if denom == 0 else 100.0 * abs(pdi - mdi) / denom

    first = 2 * period - 1
    adx = [None] * n
    adx[first] = sum(dx[period : first + 1]) / period
    for i in range(first + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return float(adx[-1])


# --- indicator math --------------------------------------------------------

def test_realized_vol_matches_reference(tmp_engine):
    det = RegimeDetector("SPY", engine=tmp_engine)
    bars = _synthetic_ohlcv()
    assert det.realized_vol(bars["close"]) == pytest.approx(
        _reference_realized_vol(bars["close"]), rel=1e-9
    )


def test_adx_matches_reference(tmp_engine):
    det = RegimeDetector("SPY", engine=tmp_engine)
    bars = _synthetic_ohlcv()
    expected = _reference_adx(
        bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(), 14
    )
    assert det.compute_adx(bars) == pytest.approx(expected, rel=1e-9)


def test_realized_vol_needs_enough_returns(tmp_engine):
    det = RegimeDetector("SPY", engine=tmp_engine)
    with pytest.raises(ValueError):
        det.realized_vol(pd.Series([100.0, 101.0, 102.0]))


# --- classification: all four corners --------------------------------------

@pytest.mark.parametrize(
    "rv,adx,expected",
    [
        (0.30, 30.0, Regime.HIGH_VOL_TRENDING),
        (0.30, 10.0, Regime.HIGH_VOL_RANGING),
        (0.10, 30.0, Regime.LOW_VOL_TRENDING),
        (0.10, 10.0, Regime.LOW_VOL_RANGING),
    ],
)
def test_classify_four_corners(tmp_engine, rv, adx, expected):
    det = RegimeDetector("SPY", engine=tmp_engine)
    vol, trend, combined = det.classify(rv, adx)
    assert combined is expected
    assert vol is (VolRegime.HIGH if rv > 0.25 else VolRegime.LOW)
    assert trend is (TrendRegime.TRENDING if adx > 25 else TrendRegime.RANGING)


def test_classify_thresholds_are_strict(tmp_engine):
    det = RegimeDetector("SPY", engine=tmp_engine)
    # Exactly at the threshold is NOT high / NOT trending (strict ">" rule).
    _, _, combined = det.classify(0.25, 25.0)
    assert combined is Regime.LOW_VOL_RANGING


# --- detect + persist ------------------------------------------------------

def test_detect_and_persist_writes_regime_state(tmp_engine):
    det = RegimeDetector("SPY", engine=tmp_engine)
    bars = _synthetic_ohlcv()
    result = det.detect(bars)
    det.persist(result, now=_dt.datetime(2026, 10, 13, 13, 25, tzinfo=_dt.timezone.utc))

    latest = det.latest()
    assert isinstance(latest, RegimeState)
    assert latest.combined == result.combined.value
    assert latest.vol_regime == result.vol_regime.value
    assert latest.trend_regime == result.trend_regime.value
    assert latest.realized_vol == pytest.approx(result.realized_vol)
    assert latest.adx == pytest.approx(result.adx)


# --- strategy gating -------------------------------------------------------

def _strategies():
    return {
        "ma": MovingAverageCrossover(symbol="SPY", fast_window=20, slow_window=50),
        "rsi": RSIMeanReversion(symbol="QQQ"),
        "pairs": PairsTrading(symbol_a="XLE", symbol_b="XOP"),
    }


@pytest.mark.parametrize(
    "regime,expected_active",
    [
        (Regime.HIGH_VOL_TRENDING, {"ma"}),
        (Regime.HIGH_VOL_RANGING, set()),
        (Regime.LOW_VOL_TRENDING, {"ma", "pairs"}),
        (Regime.LOW_VOL_RANGING, {"rsi", "pairs"}),
    ],
)
def test_active_strategies_per_regime(regime, expected_active):
    active = RegimeDetector.active_strategies(_strategies(), regime)
    assert set(active) == expected_active
