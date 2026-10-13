"""Tests for the risk manager gate.

Covers: approval of a safe order, rejection on each hard limit, and kill-switch
firing on both the daily-loss and drawdown breaches. Trading-hours checks use a
fixed in-hours / out-of-hours timestamp so they are deterministic.
"""
from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import pytest

from src.risk.manager import PortfolioState, RiskManager
from src.strategies.base import Signal
from src.utils.config import RiskConfig

_ET = ZoneInfo("America/New_York")

# A Wednesday at 11:00 ET -> always within trading hours.
IN_HOURS = _dt.datetime(2024, 1, 3, 11, 0, tzinfo=_ET)
# A Wednesday at 20:00 ET -> after the close.
AFTER_HOURS = _dt.datetime(2024, 1, 3, 20, 0, tzinfo=_ET)
# A Saturday -> market closed.
WEEKEND = _dt.datetime(2024, 1, 6, 11, 0, tzinfo=_ET)


def _state(**over):
    base = dict(
        equity=100_000.0,
        starting_equity=100_000.0,
        peak_equity=100_000.0,
        positions={},
        prices={"SPY": 400.0},
        now=IN_HOURS,
    )
    base.update(over)
    return PortfolioState(**base)


def _buy(symbol="SPY", qty=None, stop=None):
    return Signal(symbol=symbol, side="BUY", quantity_hint=qty, confidence=0.5,
                  entry_reason="test", stop_loss=stop)


def test_approves_safe_order_and_sizes_it():
    rm = RiskManager()
    d = rm.check_order(_buy(), _state())
    assert d.approved is True
    # No hint given -> risk sized the order within 10% of equity.
    assert d.modified_quantity is not None
    assert d.modified_quantity * 400.0 <= 0.10 * 100_000 + 1e-6
    # Hard 2% stop attached below entry.
    assert d.modified_stop_loss == pytest.approx(400.0 * 0.98)


def test_rejects_position_size_breach():
    rm = RiskManager()
    # 100 shares * $400 = $40k = 40% of equity > 10% cap.
    d = rm.check_order(_buy(qty=100), _state())
    assert d.approved is False and "position notional" in d.reason


def test_rejects_min_notional_dust():
    rm = RiskManager()
    # 0.1 * 400 = $40 < $100 minimum.
    d = rm.check_order(_buy(qty=0.1), _state())
    assert d.approved is False and "min" in d.reason


def test_rejects_max_positions():
    rm = RiskManager()
    positions = {s: 1.0 for s in ("AAA", "BBB", "CCC", "DDD", "EEE")}
    prices = {s: 10.0 for s in positions}
    prices["SPY"] = 400.0
    d = rm.check_order(_buy(qty=1), _state(positions=positions, prices=prices))
    assert d.approved is False and "max positions" in d.reason


def test_rejects_total_exposure_breach():
    rm = RiskManager()
    # Existing exposure already at 59% of equity; a new ~5% order tips over 60%.
    positions = {"AAA": 1475.0}  # 1475 * 40 = 59,000 = 59%
    prices = {"AAA": 40.0, "SPY": 400.0}
    d = rm.check_order(_buy(qty=25), _state(positions=positions, prices=prices))  # +10,000
    assert d.approved is False and "exposure" in d.reason


def test_rejects_outside_trading_hours():
    rm = RiskManager()
    d = rm.check_order(_buy(qty=1), _state(now=AFTER_HOURS))
    assert d.approved is False and "trading hours" in d.reason


def test_rejects_on_weekend():
    rm = RiskManager()
    d = rm.check_order(_buy(qty=1), _state(now=WEEKEND))
    assert d.approved is False and "trading hours" in d.reason


def test_trading_hours_can_be_disabled():
    rm = RiskManager(RiskConfig(trading_hours_only=False))
    d = rm.check_order(_buy(qty=1), _state(now=AFTER_HOURS))
    assert d.approved is True


def test_kill_switch_fires_on_daily_loss():
    rm = RiskManager()
    # Equity down 3.5% on the day -> over the 3% cap.
    d = rm.check_order(_buy(qty=1), _state(equity=96_500.0))
    assert d.approved is False
    assert d.kill_switch_triggered is True
    assert "daily loss" in d.reason


def test_kill_switch_fires_on_drawdown():
    rm = RiskManager()
    # 11% below the peak -> over the 10% drawdown cap (but same-day, so daily
    # loss basis is the lower starting equity -> only drawdown trips).
    d = rm.check_order(
        _buy(qty=1),
        _state(equity=89_000.0, starting_equity=89_000.0, peak_equity=100_000.0),
    )
    assert d.approved is False
    assert d.kill_switch_triggered is True
    assert "drawdown" in d.reason


def test_exit_bypasses_opening_limits():
    rm = RiskManager()
    # At max positions, a huge-notional BUY would be rejected; an EXIT is allowed.
    positions = {s: 1.0 for s in ("AAA", "BBB", "CCC", "DDD", "EEE")}
    prices = {s: 10.0 for s in positions}
    prices["AAA"] = 400.0
    exit_sig = Signal(symbol="AAA", side="EXIT", confidence=0.0, entry_reason="exit")
    d = rm.check_order(exit_sig, _state(positions=positions, prices=prices))
    assert d.approved is True


def test_enforced_stop_overrides_looser_hint():
    rm = RiskManager()
    # Strategy hint is 5% below entry (looser than the 2% hard stop).
    d = rm.check_order(_buy(qty=10, stop=380.0), _state())
    assert d.approved is True
    assert d.modified_stop_loss == pytest.approx(392.0)  # 2% stop, tighter


def test_tighter_strategy_hint_is_kept():
    rm = RiskManager()
    # Hint 1% below entry is tighter than the 2% hard stop -> keep the hint.
    d = rm.check_order(_buy(qty=10, stop=396.0), _state())
    assert d.approved is True
    assert d.modified_stop_loss == pytest.approx(396.0)
