"""Pure-Python technical indicators shared by strategies.

Every function here is deterministic and strictly **causal** — the value at bar
``i`` depends only on bars ``<= i``. That is the structural guarantee against
look-ahead bias: slicing the input to ``bars[:i+1]`` and reading the last value
returns exactly what the full-series computation reports at ``i``.

RSI lives in the strategy that uses it (computed via the ``ta`` library per the
spec); SMA and ATR are here because more than one strategy needs them.
"""
from __future__ import annotations

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
