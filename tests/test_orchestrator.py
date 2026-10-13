"""Tests for the orchestrator: schedule shape, startup gate, kill switch, ticks."""
from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.orchestrator.runner import PRE_CLOSE, PRE_OPEN, TICK, Orchestrator
from src.regime.detector import Regime
from src.strategies.base import Signal, Strategy
from src.utils.config import OrchestratorConfig
from src.validation.null_wall import NullWall, ValidationError

_ET = ZoneInfo("America/New_York")


class AlwaysBuy(Strategy):
    """A stub that always emits a BUY on the latest bar (any regime)."""

    allowed_regimes = set(Regime)

    def __init__(self):
        super().__init__(name="always_buy", symbols=("SPY",), lookback_bars=1)

    def generate_signals(self, bars):
        self.validate_bars(bars)
        ts = bars.index[-1]
        return [
            Signal(symbol="SPY", side="BUY", confidence=0.9,
                   entry_reason="stub buy", as_of=ts, meta={"strategy": "always_buy"})
        ]

    def on_fill(self, fill):  # pragma: no cover - trivial
        pass

    def on_reject(self, order):  # pragma: no cover - trivial
        pass

    def state_snapshot(self):  # pragma: no cover - trivial
        return {}


def _orch(engine, client, **kw):
    """Orchestrator with an empty strategy-config set (Null wall trivially passes)."""
    return Orchestrator(
        client=client,
        engine=engine,
        strategy_configs=[],
        null_wall=NullWall(validation_dir=str(engine.url.database) + "_none"),
        **kw,
    )


# --- planned schedule ------------------------------------------------------

def test_planned_schedule_regular_day(fake_client, tmp_engine):
    o = _orch(tmp_engine, fake_client)
    plan = o.planned_schedule(_dt.date(2026, 10, 19))  # a regular Monday
    jobs = [j for _, j in plan]
    assert jobs.count(PRE_OPEN) == 1
    assert jobs.count(PRE_CLOSE) == 1
    assert jobs.count(TICK) == 78  # 09:30..15:55 inclusive, every 5 min
    # Order: pre_open first, pre_close last.
    assert jobs[0] == PRE_OPEN
    assert jobs[-1] == PRE_CLOSE
    # pre_open at 09:25, first tick 09:30, pre_close 15:55.
    assert plan[0][0].strftime("%H:%M") == "09:25"
    assert plan[1][0].strftime("%H:%M") == "09:30"
    assert plan[-1][0].strftime("%H:%M") == "15:55"
    # Strictly non-decreasing times.
    times = [t for t, _ in plan]
    assert times == sorted(times)


def test_planned_schedule_skips_holiday(fake_client, tmp_engine):
    o = _orch(tmp_engine, fake_client)
    assert o.planned_schedule(_dt.date(2026, 7, 3)) == []  # July 4 (observed)
    assert o.planned_schedule(_dt.date(2026, 12, 25)) == []  # Christmas
    assert o.planned_schedule(_dt.date(2026, 10, 18)) == []  # Sunday


def test_planned_schedule_early_close(fake_client, tmp_engine):
    o = _orch(tmp_engine, fake_client)
    plan = o.planned_schedule(_dt.date(2026, 11, 27))  # day after Thanksgiving, 13:00 close
    jobs = [j for _, j in plan]
    assert jobs.count(PRE_OPEN) == 1 and jobs.count(PRE_CLOSE) == 1
    # 09:30..12:55 every 5 min -> 42 ticks, and pre_close at 12:55 (not 15:55).
    assert jobs.count(TICK) == 42
    assert plan[-1][0].strftime("%H:%M") == "12:55"


def test_tick_interval_configurable(fake_client, tmp_engine):
    o = Orchestrator(
        client=fake_client, engine=tmp_engine, strategy_configs=[],
        orchestrator_config=OrchestratorConfig(tick_interval_minutes=15),
        null_wall=NullWall(validation_dir="nope"),
    )
    plan = o.planned_schedule(_dt.date(2026, 10, 19))
    ticks = [t for t, j in plan if j == TICK]
    assert (ticks[1] - ticks[0]) == _dt.timedelta(minutes=15)


# --- startup gate ----------------------------------------------------------

def test_null_wall_failure_prevents_startup(fake_client, tmp_engine, tmp_path):
    from src.utils.config import StrategyInstanceConfig

    cfg = StrategyInstanceConfig(
        name="ma_crossover_spy", type="ma_crossover",
        params={"symbol": "SPY", "fast_window": 20, "slow_window": 50},
    )
    o = Orchestrator(
        client=fake_client, engine=tmp_engine,
        strategy_configs=[cfg],
        strategies={},
        null_wall=NullWall(validation_dir=tmp_path),  # empty dir -> missing file
    )
    with pytest.raises(ValidationError):
        o.startup(now=_dt.date(2026, 10, 13))


def test_startup_passes_with_valid_records(seeded_engine, fake_client):
    # Default config + committed validation records -> startup succeeds.
    o = Orchestrator(client=fake_client, engine=seeded_engine)
    o.startup(now=_dt.date(2026, 10, 13))  # should not raise


# --- ticks: risk + execution path ------------------------------------------

def test_tick_submits_approved_signal(seeded_engine, fake_client):
    stub = AlwaysBuy()
    o = _orch(seeded_engine, fake_client, strategies={"always_buy": stub})
    o._active = {"always_buy": stub}

    now = _dt.datetime(2026, 1, 5, 11, 0, tzinfo=_ET)
    res = o.tick(now)
    assert res["halted"] is False
    assert res["submitted"] == 1
    assert len(fake_client.submitted) == 1
    assert fake_client.submitted[0]["symbol"] == "SPY"
    assert fake_client.submitted[0]["side"] == "buy"


def test_tick_is_idempotent_for_same_bar(seeded_engine, fake_client):
    stub = AlwaysBuy()
    o = _orch(seeded_engine, fake_client, strategies={"always_buy": stub})
    o._active = {"always_buy": stub}
    now = _dt.datetime(2026, 1, 5, 11, 0, tzinfo=_ET)
    o.tick(now)
    o.tick(now)  # same latest bar -> no second submission
    assert len(fake_client.submitted) == 1


def test_tick_records_signal(seeded_engine, fake_client):
    from sqlalchemy import select

    from src.storage.db import SignalRecord

    stub = AlwaysBuy()
    o = _orch(seeded_engine, fake_client, strategies={"always_buy": stub})
    o._active = {"always_buy": stub}
    o.tick(_dt.datetime(2026, 1, 5, 11, 0, tzinfo=_ET))

    with o._Session() as s:
        sigs = s.execute(select(SignalRecord)).scalars().all()
    assert len(sigs) == 1 and sigs[0].strategy == "always_buy"


def test_kill_switch_halts_ticks(seeded_engine, fake_client):
    stub = AlwaysBuy()
    o = _orch(seeded_engine, fake_client, strategies={"always_buy": stub})
    o._active = {"always_buy": stub}
    o.kill_switch.trigger("manual test", "tester")

    res = o.tick(_dt.datetime(2026, 1, 5, 11, 0, tzinfo=_ET))
    assert res["halted"] is True
    assert res["submitted"] == 0
    assert len(fake_client.submitted) == 0
    assert o._halted is True


def test_risk_breach_trips_kill_switch(seeded_engine, fake_client):
    # Equity far below start-of-day -> daily-loss breach -> kill switch + halt.
    stub = AlwaysBuy()
    o = _orch(seeded_engine, fake_client, strategies={"always_buy": stub})
    o._active = {"always_buy": stub}
    o._starting_equity = 100_000.0
    o._peak_equity = 100_000.0
    fake_client._equity = 90_000.0  # 10% daily loss, past the 3% cap

    res = o.tick(_dt.datetime(2026, 1, 5, 11, 0, tzinfo=_ET))
    assert res["halted"] is True
    assert o.kill_switch.is_active() is True
    assert len(fake_client.submitted) == 0


# --- scheduler wiring ------------------------------------------------------

def test_build_scheduler_registers_three_jobs(fake_client, tmp_engine):
    o = _orch(tmp_engine, fake_client)
    sched = o.build_scheduler()
    try:
        ids = {j.id for j in sched.get_jobs()}
        assert ids == {PRE_OPEN, TICK, PRE_CLOSE}
    finally:
        # Scheduler was never started; nothing to tear down beyond dropping refs.
        pass
