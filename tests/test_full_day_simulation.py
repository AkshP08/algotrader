"""Full simulated trading day over REAL stored bars with a mock Alpaca client.

Walks the orchestrator's calendar-exact schedule for a regular Monday with an
injected clock, asserting the job sequence, that the regime was computed, that an
EOD summary was written, and that no orders are submitted after a kill switch is
tripped at 13:00 ET.
"""
from __future__ import annotations

import datetime as _dt

from sqlalchemy import select

from src.orchestrator.runner import PRE_CLOSE, PRE_OPEN, TICK, Orchestrator
from src.regime.detector import Regime
from src.storage.db import DailyPnL, OrderRecord
from src.strategies.base import Signal, Strategy
from src.validation.null_wall import NullWall

SIM_DAY = _dt.date(2026, 10, 19)  # a regular trading Monday
EARLY_CUTOFF = _dt.time(13, 0)


class AlwaysBuy(Strategy):
    allowed_regimes = set(Regime)

    def __init__(self):
        super().__init__(name="always_buy", symbols=("SPY",), lookback_bars=1)

    def generate_signals(self, bars):
        self.validate_bars(bars)
        ts = bars.index[-1]
        return [Signal(symbol="SPY", side="BUY", confidence=0.9,
                       entry_reason="stub", as_of=ts, meta={"strategy": "always_buy"})]

    def on_fill(self, fill):  # pragma: no cover
        pass

    def on_reject(self, order):  # pragma: no cover
        pass

    def state_snapshot(self):  # pragma: no cover
        return {}


# --- 1 pre_open + N ticks + 1 pre_close, in order --------------------------

def test_full_day_runs_preopen_ticks_preclose_in_order(seeded_engine, fake_client):
    o = Orchestrator(client=fake_client, engine=seeded_engine)
    o.startup(now=_dt.date(2026, 10, 13))

    results = o.simulate_day(SIM_DAY)
    jobs = [r["job"] for r in results]

    assert jobs[0] == PRE_OPEN
    assert jobs[-1] == PRE_CLOSE
    assert jobs.count(PRE_OPEN) == 1
    assert jobs.count(PRE_CLOSE) == 1
    assert jobs.count(TICK) == 78

    # Regime was computed at pre_open.
    assert results[0]["regime"] in {r.value for r in Regime}

    # Whatever was submitted during the day was recorded as an order (logged).
    with o._Session() as s:
        order_count = s.execute(select(OrderRecord)).scalars().all()
    total_submitted = sum(r.get("submitted", 0) for r in results)
    assert len(order_count) == total_submitted == len(fake_client.submitted)

    # EOD summary written: a daily_pnl row for the day with ending equity set.
    with o._Session() as s:
        row = s.execute(
            select(DailyPnL).where(DailyPnL.date == _dt.datetime(2026, 10, 19))
        ).scalars().first()
    assert row is not None
    assert row.ending_equity is not None
    assert row.starting_equity is not None


def test_signal_submitted_goes_through_full_path(seeded_engine, fake_client):
    """A generated signal is risk-checked, submitted (mock), and recorded."""
    stub = AlwaysBuy()
    o = Orchestrator(
        client=fake_client, engine=seeded_engine,
        strategy_configs=[], strategies={"always_buy": stub},
        null_wall=NullWall(validation_dir="none"),
    )
    results = o.simulate_day(SIM_DAY)
    total = sum(r.get("submitted", 0) for r in results)
    assert total >= 1  # the stub fires on the latest bar

    with o._Session() as s:
        orders = s.execute(select(OrderRecord)).scalars().all()
    assert len(orders) == total
    assert all(ord_.strategy == "always_buy" for ord_ in orders)
    assert fake_client.submitted  # reached the (mock) broker


# --- kill switch at 13:00: nothing submitted afterwards --------------------

def test_no_orders_after_kill_switch_at_1300(seeded_engine, fake_client):
    stub = AlwaysBuy()
    o = Orchestrator(
        client=fake_client, engine=seeded_engine,
        strategy_configs=[], strategies={"always_buy": stub},
        null_wall=NullWall(validation_dir="none"),
    )

    submitted_before = {"n": 0}
    submitted_after = {"n": 0}
    tripped = {"done": False}

    def on_step(now, job):
        if job != TICK:
            return
        # Clear per-bar idempotency so only the kill switch can stop trading,
        # making this a real test of the halt (not of de-duplication).
        o._submitted_ids.clear()
        et = now.timetz().replace(tzinfo=None)
        if et >= EARLY_CUTOFF and not tripped["done"]:
            o.kill_switch.trigger("simulated breach at 13:00", "test")
            tripped["done"] = True

    before = len(fake_client.submitted)

    def counting_tick(now):
        n0 = len(fake_client.submitted)
        res = Orchestrator.tick(o, now)
        delta = len(fake_client.submitted) - n0
        et = now.timetz().replace(tzinfo=None)
        if et < EARLY_CUTOFF:
            submitted_before["n"] += delta
        else:
            submitted_after["n"] += delta
        return res

    o.tick = counting_tick  # type: ignore[assignment]
    o.simulate_day(SIM_DAY, on_step=on_step)

    assert tripped["done"] is True
    assert submitted_before["n"] >= 1  # trading happened pre-breach
    assert submitted_after["n"] == 0  # kill switch blocked everything after 13:00
    assert o.kill_switch.is_active() is True
