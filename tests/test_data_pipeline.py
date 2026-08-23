"""Tests for the historical data pipeline.

Covers the three correctness properties that matter most:
  1. Look-ahead guard (with a deliberate cheat the guard must beat).
  2. Gap detection (real missing business day flagged; weekends/holidays not).
  3. Buy-and-hold parity: return from stored bars == return from fetched bars
     to machine epsilon (proves the store/read round-trip is lossless).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import HistoricalDataFetcher, buy_and_hold_return


def _daily_df(dates, closes):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="timestamp")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000.0] * n,
        },
        index=idx,
    )


class _StubClient:
    """Minimal broker stand-in exposing get_bars — no network, no synthetic magic."""

    def __init__(self, df):
        self._df = df

    def get_bars(self, symbol, timeframe, start, end):
        return self._df.copy()


# --- 1. Look-ahead guard ---------------------------------------------------

def test_get_bars_as_of_never_returns_future_rows(tmp_engine):
    fetcher = HistoricalDataFetcher(engine=tmp_engine)
    df = _daily_df(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [100.0, 101.0, 102.0, 103.0],
    )
    fetcher.upsert_bars("SPY", df)

    cutoff = "2024-01-03"
    out = fetcher.get_bars_as_of("SPY", cutoff)

    # The guard: nothing after the cutoff may appear.
    assert out.index.max() <= pd.Timestamp(cutoff)
    assert pd.Timestamp("2024-01-04") not in out.index
    assert pd.Timestamp("2024-01-05") not in out.index
    assert len(out) == 2


def test_look_ahead_guard_beats_a_deliberate_cheat(tmp_engine):
    """A naive reader that returns *all* rows would leak the future; ours must not."""
    fetcher = HistoricalDataFetcher(engine=tmp_engine)
    df = _daily_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [100.0, 101.0, 102.0],
    )
    fetcher.upsert_bars("SPY", df)
    cutoff = pd.Timestamp("2024-01-02")

    # The "cheat": read everything, ignoring the cutoff.
    cheat = fetcher.get_all_bars("SPY")
    assert cheat.index.max() > cutoff  # cheat DOES leak the future...

    honest = fetcher.get_bars_as_of("SPY", cutoff)
    assert honest.index.max() <= cutoff  # ...our guard does not.
    assert len(honest) == 1


# --- 2. Gap detection ------------------------------------------------------

def test_gap_detection_flags_missing_business_day(tmp_engine):
    fetcher = HistoricalDataFetcher(engine=tmp_engine)
    # 2024-01-02..05 are Tue-Fri. Drop Wed 2024-01-03 to create a real gap.
    df = _daily_df(
        ["2024-01-02", "2024-01-04", "2024-01-05"],
        [100.0, 101.0, 102.0],
    )
    missing = fetcher.detect_gaps("SPY", df, timeframe="1Day")
    assert pd.Timestamp("2024-01-03").date() in missing


def test_gap_detection_ignores_weekends_and_holidays(tmp_engine):
    fetcher = HistoricalDataFetcher(engine=tmp_engine)
    # Fri 2024-01-05 -> Mon 2024-01-08 spans a weekend only: no gap.
    # 2024-01-15 is MLK Day (holiday) so 12th(Fri)->16th(Tue) also has no gap.
    df = _daily_df(
        ["2024-01-05", "2024-01-08", "2024-01-12", "2024-01-16"],
        [100.0, 101.0, 102.0, 103.0],
    )
    missing = fetcher.detect_gaps("SPY", df, timeframe="1Day")
    # No weekend day and no holiday should be reported as missing.
    assert pd.Timestamp("2024-01-06").date() not in missing  # Saturday
    assert pd.Timestamp("2024-01-07").date() not in missing  # Sunday
    assert pd.Timestamp("2024-01-15").date() not in missing  # MLK Day


# --- 3. Buy-and-hold parity (self-proof) -----------------------------------

def test_buy_and_hold_parity_to_machine_epsilon(tmp_engine):
    df = _daily_df(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [100.0, 103.5, 98.25, 107.125],
    )
    client = _StubClient(df)
    fetcher = HistoricalDataFetcher(client=client, engine=tmp_engine)

    # Fetch (from the "broker") + store, then read back from the DB.
    fetcher.fetch_and_store(["SPY"], "1Day", "2024-01-01", "2024-01-06")
    stored = fetcher.get_all_bars("SPY")

    r_fetched = buy_and_hold_return(df)
    r_stored = buy_and_hold_return(stored)

    assert abs(r_fetched - r_stored) <= np.finfo(np.float64).eps


def test_buy_and_hold_needs_two_bars():
    with pytest.raises(ValueError):
        buy_and_hold_return(_daily_df(["2024-01-02"], [100.0]))


# --- Upsert idempotency ----------------------------------------------------

def test_upsert_is_idempotent(tmp_engine):
    fetcher = HistoricalDataFetcher(engine=tmp_engine)
    df = _daily_df(["2024-01-02", "2024-01-03"], [100.0, 101.0])
    fetcher.upsert_bars("SPY", df)
    fetcher.upsert_bars("SPY", df)  # second time must not duplicate
    assert len(fetcher.get_all_bars("SPY")) == 2
