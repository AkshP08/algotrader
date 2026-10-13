"""Tests for the order execution layer (all broker calls mocked).

Verifies: client_order_id uniqueness + determinism, that submission is blocked
when the risk decision is not approved or the kill switch is active, order/fill
persistence, reconciliation, and rejection dispatch to the strategy.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from src.execution.executor import OrderExecutor
from src.risk.manager import RiskDecision
from src.storage.db import FillRecord, OrderRecord, get_sessionmaker
from src.strategies.base import Signal


def _approved(qty=10.0):
    return RiskDecision(approved=True, reason="ok", modified_quantity=qty,
                        modified_stop_loss=392.0)


def _signal(symbol="SPY", side="BUY", strategy="ma_crossover_spy", ts="2024-03-01"):
    return Signal(symbol=symbol, side=side, confidence=0.5, entry_reason="t",
                  as_of=pd.Timestamp(ts), meta={"strategy": strategy})


def _mock_client(mocker):
    client = mocker.MagicMock(name="AlpacaClient")
    client.submit_order.return_value = mocker.MagicMock(id="broker-1", status="accepted")
    client.close_position.return_value = mocker.MagicMock(id="broker-close", status="accepted")
    return client


# --- client_order_id --------------------------------------------------------

def test_client_order_id_is_deterministic_and_unique():
    s1 = _signal(symbol="SPY", ts="2024-03-01")
    s2 = _signal(symbol="SPY", ts="2024-03-04")   # different bar -> different id
    s3 = _signal(symbol="QQQ", ts="2024-03-01")   # different symbol -> different id
    s4 = _signal(symbol="SPY", strategy="other", ts="2024-03-01")  # different strat

    ids = {
        OrderExecutor.build_client_order_id(s)
        for s in (s1, s2, s3, s4)
    }
    assert len(ids) == 4  # all unique
    # Deterministic: the same logical signal yields the same id (idempotent retry).
    assert OrderExecutor.build_client_order_id(s1) == OrderExecutor.build_client_order_id(
        _signal(symbol="SPY", ts="2024-03-01")
    )


def test_submit_sets_unique_ids_across_orders(tmp_engine, mocker):
    ex = OrderExecutor(client=_mock_client(mocker), engine=tmp_engine)
    ex.submit(_signal(symbol="SPY", ts="2024-03-01"), _approved())
    ex.submit(_signal(symbol="QQQ", ts="2024-03-01"), _approved())
    with get_sessionmaker(tmp_engine)() as s:
        ids = [r.client_order_id for r in s.execute(select(OrderRecord)).scalars().all()]
    assert len(ids) == 2 and len(set(ids)) == 2


# --- gating -----------------------------------------------------------------

def test_not_submitted_when_risk_rejected(tmp_engine, mocker):
    client = _mock_client(mocker)
    ex = OrderExecutor(client=client, engine=tmp_engine)
    out = ex.submit(_signal(), RiskDecision(approved=False, reason="too big"))
    assert out is None
    client.submit_order.assert_not_called()
    with get_sessionmaker(tmp_engine)() as s:
        assert s.execute(select(OrderRecord)).scalars().all() == []


def test_not_submitted_when_kill_switch_active(tmp_engine, mocker):
    client = _mock_client(mocker)
    ks = mocker.MagicMock()
    ks.is_active.return_value = True
    ex = OrderExecutor(client=client, engine=tmp_engine, kill_switch=ks)
    out = ex.submit(_signal(), _approved())
    assert out is None
    client.submit_order.assert_not_called()


def test_submit_records_order_and_returns_broker_id(tmp_engine, mocker):
    client = _mock_client(mocker)
    ex = OrderExecutor(client=client, engine=tmp_engine)
    out = ex.submit(_signal(), _approved(qty=10.0))
    assert out == "broker-1"
    client.submit_order.assert_called_once()
    _, kwargs = client.submit_order.call_args
    assert kwargs["side"] == "buy"
    assert kwargs["order_type"] == "market"
    assert kwargs["time_in_force"] == "day"
    assert kwargs["extended_hours"] is False
    with get_sessionmaker(tmp_engine)() as s:
        rows = s.execute(select(OrderRecord)).scalars().all()
    assert len(rows) == 1 and rows[0].quantity == 10.0 and rows[0].symbol == "SPY"


def test_exit_routes_to_close_position(tmp_engine, mocker):
    client = _mock_client(mocker)
    ex = OrderExecutor(client=client, engine=tmp_engine)
    out = ex.submit(_signal(side="EXIT"), _approved(qty=10.0))
    client.close_position.assert_called_once_with("SPY")
    client.submit_order.assert_not_called()
    assert out == "broker-close"


# --- reconciliation ---------------------------------------------------------

def test_reconcile_updates_status_and_writes_fill(tmp_engine, mocker):
    client = _mock_client(mocker)
    ex = OrderExecutor(client=client, engine=tmp_engine)
    ex.submit(_signal(), _approved(qty=5.0))  # recorded as 'accepted'

    client.get_order.return_value = mocker.MagicMock(
        status="filled", filled_qty="5", filled_avg_price="401.25",
        filled_at=pd.Timestamp("2024-03-01T15:00:00Z"),
    )
    summary = ex.reconcile()
    assert summary["filled"] == 1
    with get_sessionmaker(tmp_engine)() as s:
        orders = s.execute(select(OrderRecord)).scalars().all()
        fills = s.execute(select(FillRecord)).scalars().all()
    assert orders[0].status == "filled"
    assert len(fills) == 1 and fills[0].filled_qty == 5.0 and fills[0].filled_avg_price == 401.25


def test_reconcile_is_idempotent_no_duplicate_fills(tmp_engine, mocker):
    client = _mock_client(mocker)
    ex = OrderExecutor(client=client, engine=tmp_engine)
    ex.submit(_signal(), _approved(qty=5.0))
    client.get_order.return_value = mocker.MagicMock(
        status="filled", filled_qty="5", filled_avg_price="401.25", filled_at=None,
    )
    ex.reconcile()
    ex.reconcile()  # second pass: order already terminal, no new fill
    with get_sessionmaker(tmp_engine)() as s:
        fills = s.execute(select(FillRecord)).scalars().all()
    assert len(fills) == 1


def test_reconcile_dispatches_on_fill_to_strategy(tmp_engine, mocker):
    client = _mock_client(mocker)
    strat = mocker.MagicMock()
    ex = OrderExecutor(client=client, engine=tmp_engine,
                       strategies={"ma_crossover_spy": strat})
    ex.submit(_signal(strategy="ma_crossover_spy"), _approved(qty=5.0))
    client.get_order.return_value = mocker.MagicMock(
        status="filled", filled_qty="5", filled_avg_price="400", filled_at=None,
    )
    ex.reconcile()
    strat.on_fill.assert_called_once()


# --- rejection --------------------------------------------------------------

def test_handle_rejection_notifies_strategy(tmp_engine, mocker):
    strat = mocker.MagicMock()
    ex = OrderExecutor(client=_mock_client(mocker), engine=tmp_engine,
                       strategies={"ma_crossover_spy": strat})
    order = mocker.MagicMock(symbol="SPY", strategy="ma_crossover_spy", status="rejected")
    ex.handle_rejection(order)
    strat.on_reject.assert_called_once_with(order)
