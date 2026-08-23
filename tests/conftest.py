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

import pytest  # noqa: E402

from src.storage.db import init_db, make_engine  # noqa: E402


@pytest.fixture()
def tmp_engine(tmp_path):
    """A throwaway SQLite engine backed by a temp file, tables created."""
    db_path = tmp_path / "test_market.db"
    engine = make_engine(db_path)
    init_db(engine)
    return engine
