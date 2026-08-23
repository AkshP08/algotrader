"""Interface-conformance tests for the Strategy base class.

A dummy subclass exercises the full contract, and we assert the interface is
Null-compatible: it consumes the same OHLCV DataFrame shape Null will feed it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.base import (
    OHLCV_COLUMNS,
    Fill,
    Side,
    Signal,
    Strategy,
)


def _bars():
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000.0, 1_100.0, 1_200.0],
        },
        index=idx,
    )


class DummyStrategy(Strategy):
    """A minimal, deterministic strategy with BOTH entry and exit logic."""

    name = "dummy"
    symbols = ("SPY",)
    lookback_bars = 2

    def __init__(self):
        super().__init__()
        self.fills = []
        self.rejects = []

    def generate_signals(self, bars: pd.DataFrame):
        self.validate_bars(bars)
        last = bars.iloc[-1]
        # Entry with an explicit exit (stop + take) -> never open-ended.
        return [
            Signal(
                symbol="SPY",
                side="BUY",
                quantity_hint=1,
                confidence=0.6,
                entry_reason="dummy: last close above open",
                stop_loss=float(last["low"]),
                take_profit=float(last["close"]) * 1.05,
                as_of=bars.index[-1],
            )
        ]

    def on_fill(self, fill):
        self.fills.append(fill)

    def on_reject(self, order):
        self.rejects.append(order)

    def state_snapshot(self):
        return {"fills": len(self.fills), "rejects": len(self.rejects)}


# --- Conformance -----------------------------------------------------------

def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        Strategy()  # abstract methods unimplemented


def test_dummy_conforms_to_interface():
    s = DummyStrategy()
    assert s.name == "dummy"
    assert s.symbols == ("SPY",)
    assert s.lookback_bars == 2
    assert s.has_exit_logic is True

    signals = s.generate_signals(_bars())
    assert isinstance(signals, list)
    assert all(isinstance(sig, Signal) for sig in signals)

    sig = signals[0]
    assert sig.side is Side.BUY
    assert sig.stop_loss is not None and sig.take_profit is not None  # exit intent
    assert 0.0 <= sig.confidence <= 1.0


def test_on_fill_and_snapshot_update_state():
    s = DummyStrategy()
    assert s.state_snapshot() == {"fills": 0, "rejects": 0}
    s.on_fill(
        Fill(
            symbol="SPY",
            side=Side.BUY,
            filled_qty=1,
            filled_avg_price=100.0,
            filled_at=pd.Timestamp("2024-01-04"),
        )
    )
    s.on_reject({"symbol": "SPY", "reason": "insufficient buying power"})
    assert s.state_snapshot() == {"fills": 1, "rejects": 1}


# --- Signal validation -----------------------------------------------------

def test_signal_normalizes_string_side():
    sig = Signal(symbol="SPY", side="exit", confidence=0.5)
    assert sig.side is Side.EXIT


def test_signal_rejects_bad_confidence():
    with pytest.raises(ValueError):
        Signal(symbol="SPY", side="BUY", confidence=1.5)


# --- Null compatibility ----------------------------------------------------

def test_interface_accepts_null_input_shape():
    """validate_bars accepts exactly the OHLCV frame Null will pass."""
    bars = _bars()
    assert list(OHLCV_COLUMNS) == ["open", "high", "low", "close", "volume"]
    # Same object Null feeds strategies must validate cleanly.
    assert Strategy.validate_bars(bars) is bars


def test_validate_bars_rejects_unsorted_index():
    bars = _bars().iloc[::-1]  # descending -> potential look-ahead
    with pytest.raises(ValueError):
        Strategy.validate_bars(bars)


def test_validate_bars_rejects_missing_columns():
    bars = _bars().drop(columns=["volume"])
    with pytest.raises(ValueError):
        Strategy.validate_bars(bars)
