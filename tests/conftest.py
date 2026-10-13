"""Shared pytest fixtures + a safe, paper-only test environment.

These env vars are set *before* any ``src`` module is imported so the broker
client's import-time paper-URL guard passes with a paper endpoint and never
touches the network. Credentials here are dummy strings — tests mock the SDK.
"""
from __future__ import annotations

import os

# Must be set before importing src.broker.alpaca_client (import-time guard).
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
os.environ.setdefault("ALPACA_API_KEY_PAPER", "test-key")
os.environ.setdefault("ALPACA_SECRET_KEY_PAPER", "test-secret")
os.environ.setdefault("ALPACA_DATA_FEED", "iex")

import datetime as _dt  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.data.pipeline import HistoricalDataFetcher  # noqa: E402
from src.storage.db import init_db, make_engine  # noqa: E402


@pytest.fixture()
def tmp_engine(tmp_path):
    """A throwaway SQLite engine backed by a temp file, tables created."""
    db_path = tmp_path / "test_market.db"
    engine = make_engine(db_path)
    init_db(engine)
    return engine


class FakeAlpacaClient:
    """In-memory paper broker for orchestrator/dashboard tests.

    Records submissions, fills every order immediately (mock), and exposes the
    account/positions/order methods the rest of the system calls. No network.
    """

    def __init__(self, equity: float = 100_000.0, positions=None, fill_price: float = 100.0):
        self.base_url = "https://paper-api.alpaca.markets"
        self._equity = equity
        self._bp = equity * 2
        self._positions = list(positions or [])
        self._fill_price = fill_price
        self.submitted: list[dict] = []
        self.closed: list[str] = []
        self.cancel_calls = 0
        self._orders: dict[str, SimpleNamespace] = {}
        self._seq = 0

    def get_account(self):
        return SimpleNamespace(
            equity=str(self._equity), buying_power=str(self._bp), status="ACTIVE"
        )

    def get_all_positions(self):
        return list(self._positions)

    def get_positions(self):
        return list(self._positions)

    def submit_order(self, symbol, qty, side, order_type="market", time_in_force="day",
                     client_order_id=None, extended_hours=False, **kw):
        self._seq += 1
        oid = f"ord-{self._seq}"
        self.submitted.append(
            {"symbol": symbol, "qty": qty, "side": side, "client_order_id": client_order_id}
        )
        order = SimpleNamespace(
            id=oid, status="filled", filled_qty=str(qty),
            filled_avg_price=str(self._fill_price), filled_at=_dt.datetime(2026, 1, 5, 15, 0),
            symbol=symbol, side=side, client_order_id=client_order_id,
        )
        self._orders[oid] = order
        return order

    def get_order(self, order_id):
        return self._orders[order_id]

    def close_position(self, symbol):
        self._seq += 1
        oid = f"close-{self._seq}"
        self.closed.append(symbol)
        order = SimpleNamespace(
            id=oid, status="filled", filled_qty="0", filled_avg_price="0",
            filled_at=_dt.datetime(2026, 1, 5, 15, 0), symbol=symbol,
        )
        self._orders[oid] = order
        return order

    def cancel_all_orders(self):
        self.cancel_calls += 1
        return []

    def get_bars(self, *a, **k):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture()
def fake_client():
    return FakeAlpacaClient()


# Symbols the orchestrator / full-day tests need real history for.
_SEED_SYMBOLS = ("SPY", "QQQ", "XLE", "XOP")


@pytest.fixture()
def seeded_engine(tmp_engine):
    """A temp engine seeded with REAL stored bars copied from the default market.db.

    Never fabricates data: if the default DB has not been seeded, the test skips.
    """
    source = HistoricalDataFetcher()  # default market.db
    dest = HistoricalDataFetcher(engine=tmp_engine)
    seeded_any = False
    for sym in _SEED_SYMBOLS:
        bars = source.get_all_bars(sym)
        if len(bars) >= 60:
            dest.upsert_bars(sym, bars)
            seeded_any = True
    if not seeded_any:
        pytest.skip(
            "market.db has no seed history; seed it first (python -m src.hello_world). "
            "Refusing to fabricate market data."
        )
    return tmp_engine
