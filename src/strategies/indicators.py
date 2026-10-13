"""Pure-Python technical indicators shared by strategies.

Every function here is deterministic and strictly **causal** — the value at bar
``i`` depends only on bars ``<= i``. That is the structural guarantee against
look-ahead bias: slicing the input to ``bars[:i+1]`` and reading the last value
returns exactly what the full-series computation reports at ``i``.

RSI lives in the strategy that uses it (computed via the ``ta`` library per the
spec); SMA and ATR are here because more than one strategy needs them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. NaN until ``window`` observations exist."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return series.rolling(window=window, min_periods=window).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder true range: max(H-L, |H-prevC|, |L-prevC|).

    The first bar has no previous close, so its TR is simply ``H-L`` (pandas'
    row-wise ``max`` skips the resulting NaN columns).
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA, alpha = 1/period).

    Equivalent to an EMA with ``adjust=False`` seeded at the first TR value — the
    standard ATR definition. NaN until ``period`` true-range observations exist.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    tr = true_range(bars["high"], bars["low"], bars["close"])
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder), pure Python.

    ADX measures *trend strength* (not direction) on a 0-100 scale. The classic
    Wilder construction is implemented explicitly here — directional movement
    (+DM/-DM), true range, Wilder running-sum smoothing, directional indicators
    (+DI/-DI), the directional index (DX), then a Wilder average of DX:

        +DM_i = (H_i - H_{i-1})  if that up-move dominates the down-move and > 0
        -DM_i = (L_{i-1} - L_i)  if that down-move dominates the up-move and > 0
        +DI   = 100 * smoothed(+DM) / smoothed(TR)
        -DI   = 100 * smoothed(-DM) / smoothed(TR)
        DX    = 100 * |+DI - -DI| / (+DI + -DI)
        ADX   = Wilder average of DX over `period`

    Strictly causal: the value at bar ``i`` uses only bars ``<= i``. The first
    non-NaN ADX appears at index ``2*period - 1`` (a full DM window plus a full
    DX window), so at least ``2*period`` bars are required; fewer returns all-NaN.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    high = bars["high"].astype("float64").to_numpy()
    low = bars["low"].astype("float64").to_numpy()
    close = bars["close"].astype("float64").to_numpy()
    n = len(bars)
    out = np.full(n, np.nan)
    if n < 2 * period:
        return pd.Series(out, index=bars.index, name="adx")

    # Per-bar directional movement and true range (index 0 has no predecessor).
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder running-sum smoothing, seeded with the first `period` values (i=1..period).
    atr_s = np.full(n, np.nan)
    pdm_s = np.full(n, np.nan)
    mdm_s = np.full(n, np.nan)
    atr_s[period] = tr[1 : period + 1].sum()
    pdm_s[period] = plus_dm[1 : period + 1].sum()
    mdm_s[period] = minus_dm[1 : period + 1].sum()
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]

    # Directional index DX (defined once smoothing exists, i >= period).
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr_s[i] == 0:
            dx[i] = 0.0
            continue
        pdi = 100.0 * pdm_s[i] / atr_s[i]
        mdi = 100.0 * mdm_s[i] / atr_s[i]
        denom = pdi + mdi
        dx[i] = 0.0 if denom == 0 else 100.0 * abs(pdi - mdi) / denom

    # ADX: seed with the mean of the first `period` DX values at index 2*period-1,
    # then Wilder-average the rest.
    first = 2 * period - 1
    out[first] = float(np.mean(dx[period : first + 1]))
    for i in range(first + 1, n):
        out[i] = (out[i - 1] * (period - 1) + dx[i]) / period

    return pd.Series(out, index=bars.index, name="adx")
