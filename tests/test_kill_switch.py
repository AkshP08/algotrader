"""Tests for the kill switch.

Verifies that triggering cancels all orders and closes every open position (via a
mocked Alpaca client), that state is SQLite-backed and derived from the latest
event, and that a reset requires a non-empty acknowledgment and never happens
automatically.
"""
from __future__ import annotations

import pytest

from src.risk.kill_switch import KillSwitch


class _Pos:
    def __init__(self, symbol):
        self.symbol = symbol


def _mock_client(mocker, positions=("SPY", "QQQ")):
    client = mocker.MagicMock(name="AlpacaClient")
    client.get_positions.return_value = [_Pos(s) for s in positions]
    client.get_account.return_value = mocker.MagicMock(equity="100000")
    return client


def test_starts_inactive(tmp_engine, mocker):
    ks = KillSwitch(client=_mock_client(mocker), engine=tmp_engine)
    assert ks.is_active() is False


def test_trigger_cancels_orders_and_closes_each_position(tmp_engine, mocker):
    client = _mock_client(mocker, positions=("SPY", "QQQ", "XLE"))
    ks = KillSwitch(client=client, engine=tmp_engine)

    ks.trigger(reason="max_drawdown", triggered_by="risk_manager")

    client.cancel_all_orders.assert_called_once()
    assert client.close_position.call_count == 3
    closed = {c.args[0] for c in client.close_position.call_args_list}
    assert closed == {"SPY", "QQQ", "XLE"}
    assert ks.is_active() is True


def test_trigger_persists_across_instances(tmp_engine, mocker):
    ks = KillSwitch(client=_mock_client(mocker), engine=tmp_engine)
    ks.trigger(reason="daily_loss_cap", triggered_by="risk_manager")
    # A fresh instance on the same DB must see the active state (process-level).
    ks2 = KillSwitch(client=_mock_client(mocker), engine=tmp_engine)
    assert ks2.is_active() is True


def test_reset_clears_and_requires_note(tmp_engine, mocker):
    ks = KillSwitch(client=_mock_client(mocker), engine=tmp_engine)
    ks.trigger(reason="max_drawdown", triggered_by="risk_manager")
    assert ks.is_active() is True

    with pytest.raises(ValueError):
        ks.reset(acknowledged_by="trader", note="")  # empty note rejected
    with pytest.raises(ValueError):
        ks.reset(acknowledged_by="trader", note="   ")  # whitespace rejected
    assert ks.is_active() is True  # still active after failed resets

    ks.reset(acknowledged_by="trader", note="reviewed; cause understood")
    assert ks.is_active() is False


def test_no_automatic_reset(tmp_engine, mocker):
    ks = KillSwitch(client=_mock_client(mocker), engine=tmp_engine)
    ks.trigger(reason="max_drawdown", triggered_by="risk_manager")
    # Re-reading state many times never clears it; only reset() can.
    for _ in range(5):
        assert ks.is_active() is True


def test_retrigger_after_reset(tmp_engine, mocker):
    ks = KillSwitch(client=_mock_client(mocker), engine=tmp_engine)
    ks.trigger(reason="max_drawdown", triggered_by="rm")
    ks.reset(acknowledged_by="trader", note="ok")
    assert ks.is_active() is False
    ks.trigger(reason="daily_loss_cap", triggered_by="rm")
    assert ks.is_active() is True
