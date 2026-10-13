"""End-to-end dry run over REAL stored SPY history (no live Alpaca).

Pipeline exercised:
    stored bars -> MovingAverageCrossover -> RiskManager (mock $100k portfolio)
                -> MockOrderExecutor (records, never hits a broker)

Assertions:
  * order ids are unique,
  * open positions never exceed max_positions,
  * no orders are submitted after a simulated kill-switch trigger,
  * the strategy produced >= 2 trades on real SPY data (sanity).

Reads the default market DB. If it has not been seeded with SPY history (e.g. a
fresh clone), the test skips with a clear message rather than fabricating data —
the standing rule forbids synthetic/fabricated market data.
"""
from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import pytest

from src.data.pipeline import HistoricalDataFetcher
from src.risk.manager import PortfolioState, RiskManager
from src.strategies.base import Side
from src.strategies.ma_crossover import MovingAverageCrossover

_ET = ZoneInfo("America/New_York")
# A fixed in-hours weekday timestamp so the risk gate's trading-hours check is
# deterministic regardless of when the suite runs.
IN_HOURS = _dt.datetime(2024, 1, 3, 11, 0, tzinfo=_ET)
START_EQUITY = 100_000.0


class MockOrderExecutor:
    """A broker-free executor: enforces the same two gates as the real one
    (kill switch + risk approval) and records orders in memory."""

    def __init__(self, kill_switch_active=lambda: False):
        self.kill_switch_active = kill_switch_active
        self.orders: list[dict] = []

    def submit(self, signal, risk_decision):
        if self.kill_switch_active():
            return None
        if not risk_decision.approved:
            return None
        qty = risk_decision.modified_quantity or signal.quantity_hint or 0
        ts_ms = int(signal.as_of.timestamp() * 1000)
        strategy = signal.meta.get("strategy", "strategy")
        client_order_id = f"{strategy}-{signal.symbol}-{ts_ms}"
        self.orders.append(
            {
                "client_order_id": client_order_id,
                "symbol": signal.symbol,
                "side": signal.side,
                "qty": qty,
                "as_of": signal.as_of,
            }
        )
        return client_order_id


@pytest.fixture()
def spy_bars():
    bars = HistoricalDataFetcher().get_all_bars("SPY")
    if len(bars) < 60:
        pytest.skip(
            "market.db has no SPY history; seed it first "
            "(python -m src.hello_world or the pipeline). Refusing to fabricate data."
        )
    return bars


def test_end_to_end_dry_run_over_real_spy(spy_bars):
    strat = MovingAverageCrossover(symbol="SPY", fast_window=20, slow_window=50)
    signals = strat.generate_signals(spy_bars)

    # Sanity: a 1y SPY window must cross at least a couple of times.
    assert len(signals) >= 2, f"expected >= 2 trades on real SPY, got {len(signals)}"

    rm = RiskManager()
    executor = MockOrderExecutor()

    # A tiny simulated book: positions tracked by signed share count.
    positions: dict[str, float] = {}
    submitted = []
    for s in signals:
        price = float(spy_bars.loc[s.as_of, "close"])
        state = PortfolioState(
            equity=START_EQUITY,
            starting_equity=START_EQUITY,
            peak_equity=START_EQUITY,
            positions=dict(positions),
            prices={s.symbol: price},
            now=IN_HOURS,
        )
        decision = rm.check_order(s, state)
        order_id = executor.submit(s, decision)
        if order_id is None:
            continue
        submitted.append(order_id)
        # Update the simulated book.
        qty = decision.modified_quantity or s.quantity_hint or 0
        if s.side is Side.BUY:
            positions[s.symbol] = positions.get(s.symbol, 0.0) + qty
        elif s.side is Side.EXIT:
            positions[s.symbol] = 0.0
        # Positions never exceed the configured concurrency cap.
        open_count = sum(1 for q in positions.values() if abs(q) > 1e-9)
        assert open_count <= rm.config.max_positions

    # Order ids are unique.
    assert len(submitted) == len(set(submitted))


def test_no_orders_after_kill_switch_trigger(spy_bars):
    strat = MovingAverageCrossover(symbol="SPY", fast_window=20, slow_window=50)
    signals = strat.generate_signals(spy_bars)
    assert len(signals) >= 2

    rm = RiskManager()
    # Flip the kill switch on after the first approved order.
    tripped = {"active": False}
    executor = MockOrderExecutor(kill_switch_active=lambda: tripped["active"])

    submitted_after_trip = 0
    for i, s in enumerate(signals):
        price = float(spy_bars.loc[s.as_of, "close"])
        state = PortfolioState(
            equity=START_EQUITY, starting_equity=START_EQUITY, peak_equity=START_EQUITY,
            positions={}, prices={s.symbol: price}, now=IN_HOURS,
        )
        decision = rm.check_order(s, state)
        order_id = executor.submit(s, decision)
        if tripped["active"] and order_id is not None:
            submitted_after_trip += 1
        if i == 0:
            tripped["active"] = True  # kill switch fires

    assert submitted_after_trip == 0, "orders must not be submitted after kill switch"


def test_integration_order_ids_unique_and_deterministic(spy_bars):
    strat = MovingAverageCrossover(symbol="SPY", fast_window=20, slow_window=50)
    signals = strat.generate_signals(spy_bars)
    rm = RiskManager()

    def run_once():
        ex = MockOrderExecutor()
        for s in signals:
            price = float(spy_bars.loc[s.as_of, "close"])
            state = PortfolioState(
                equity=START_EQUITY, starting_equity=START_EQUITY, peak_equity=START_EQUITY,
                positions={}, prices={s.symbol: price}, now=IN_HOURS,
            )
            ex.submit(s, rm.check_order(s, state))
        return [o["client_order_id"] for o in ex.orders]

    ids_a = run_once()
    ids_b = run_once()
    assert len(ids_a) == len(set(ids_a))  # unique
    assert ids_a == ids_b  # deterministic across runs
