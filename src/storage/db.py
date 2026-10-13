"""SQLAlchemy storage layer for Sentinel.

Defines the full persistence schema and an ``init_db()`` that creates any
missing tables. Everything the system does that must be auditable — bars,
signals, orders, fills, daily P&L, strategy state snapshots, and kill-switch
events — has a table here.

The database is SQLite by default (``data/market.db``); the path is
configurable via ``MARKET_DB_PATH`` so tests can point at a temp file.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class Bar(Base):
    """OHLCV bar. Composite PK (symbol, timestamp) makes upserts idempotent."""

    __tablename__ = "bars"

    symbol = Column(String, primary_key=True, nullable=False)
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class SignalRecord(Base):
    """A signal emitted by a strategy (audit trail; not necessarily executed)."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    strategy = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # BUY / SELL / EXIT
    quantity_hint = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)
    entry_reason = Column(Text, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    bar_timestamp = Column(DateTime, nullable=False)  # the bar this decision was made on


class OrderRecord(Base):
    """An order submitted to the broker."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    broker_order_id = Column(String, nullable=True, index=True)
    client_order_id = Column(String, nullable=True, index=True)
    strategy = Column(String, nullable=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    order_type = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    time_in_force = Column(String, nullable=False, default="day")
    status = Column(String, nullable=False, default="new")


class FillRecord(Base):
    """A (partial) fill for an order."""

    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    broker_order_id = Column(String, nullable=True, index=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    filled_qty = Column(Float, nullable=False)
    filled_avg_price = Column(Float, nullable=False)
    filled_at = Column(DateTime, nullable=False)


class DailyPnL(Base):
    """One row per trading day: realized/unrealized P&L and equity marks."""

    __tablename__ = "daily_pnl"
    __table_args__ = (UniqueConstraint("date", name="uq_daily_pnl_date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    starting_equity = Column(Float, nullable=True)
    ending_equity = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=True)


class StrategyState(Base):
    """Serialized ``state_snapshot()`` of a strategy at a point in time."""

    __tablename__ = "strategy_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    strategy = Column(String, nullable=False, index=True)
    snapshot_json = Column(Text, nullable=False)


class RegimeState(Base):
    """Market-regime snapshot recomputed at session open (audit trail).

    ``combined`` is one of the four corners (e.g. ``LOW_VOL_TRENDING``); the raw
    ``realized_vol`` (annualized) and ``adx`` are stored so a later reader can see
    exactly why that classification was made.
    """

    __tablename__ = "regime_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    vol_regime = Column(String, nullable=False)  # HIGH / LOW
    trend_regime = Column(String, nullable=False)  # TRENDING / RANGING
    combined = Column(String, nullable=False)  # one of the four corners
    realized_vol = Column(Float, nullable=False)
    adx = Column(Float, nullable=False)


class KillSwitchEvent(Base):
    """A kill-switch activation — CRITICAL. Recorded for post-mortem."""

    __tablename__ = "kill_switch_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    trigger = Column(String, nullable=False)  # e.g. "max_drawdown", "daily_loss_cap"
    detail = Column(Text, nullable=False)
    equity_at_trigger = Column(Float, nullable=True)


def default_db_path() -> Path:
    return Path(os.getenv("MARKET_DB_PATH", "data/market.db"))


def make_engine(db_path: str | os.PathLike | None = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for the SQLite database at ``db_path``."""
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=echo, future=True)


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables if missing. Returns the engine used."""
    engine = engine or make_engine()
    Base.metadata.create_all(engine)
    return engine


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker:
    engine = engine or make_engine()
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
