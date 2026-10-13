"""Tests for the read-only monitoring dashboard."""
from __future__ import annotations

import datetime as _dt
import os
import time

import pytest
from sqlalchemy import func, select

from src.dashboard.app import create_app
from src.orchestrator.lockfile import HEARTBEAT_MAX_AGE_SECONDS
from src.storage.db import Bar


@pytest.fixture()
def app(tmp_engine):
    # No broker: account shows "unavailable" but the page still renders.
    return create_app(engine=tmp_engine, client_factory=lambda: None)


@pytest.fixture()
def client(app):
    return app.test_client()


# --- routes ----------------------------------------------------------------

def test_index_returns_200_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.mimetype == "text/html"
    assert b"algotrader" in r.data
    assert b"Market Regime" in r.data


def test_api_state_shape(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.get_json()
    expected = {
        "generated_at", "regime", "kill_switch_active", "account",
        "strategies", "orders", "fills", "todays_pnl", "equity_curve",
    }
    assert expected <= set(data)
    assert isinstance(data["strategies"], list)
    assert isinstance(data["orders"], list)
    assert isinstance(data["fills"], list)
    assert isinstance(data["kill_switch_active"], bool)


def test_api_state_lists_configured_strategies(client):
    data = client.get("/api/state").get_json()
    names = {s["name"] for s in data["strategies"]}
    assert {"ma_crossover_spy", "rsi_mean_reversion_qqq", "pairs_trading_xle_xop"} <= names
    for s in data["strategies"]:
        assert s["null_status"] in {"SURVIVED", "SHAKY", "DEAD", "MISSING", "DRIFTED", "INVALID"}


# --- healthz ---------------------------------------------------------------

def test_healthz_missing_lockfile_503(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_LOCKFILE", str(tmp_path / "absent.pid"))
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.get_json()["status"] == "unhealthy"


def test_healthz_fresh_lockfile_200(client, tmp_path, monkeypatch):
    lock = tmp_path / "o.pid"
    lock.write_text("12345")
    monkeypatch.setenv("ORCHESTRATOR_LOCKFILE", str(lock))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_healthz_stale_lockfile_503(client, tmp_path, monkeypatch):
    lock = tmp_path / "o.pid"
    lock.write_text("12345")
    old = time.time() - (HEARTBEAT_MAX_AGE_SECONDS + 60)
    os.utime(lock, (old, old))
    monkeypatch.setenv("ORCHESTRATOR_LOCKFILE", str(lock))
    r = client.get("/healthz")
    assert r.status_code == 503


# --- read-only guarantee ---------------------------------------------------

def test_dashboard_does_not_write_to_db(app, client, tmp_engine):
    from src.storage.db import get_sessionmaker

    reader = get_sessionmaker(tmp_engine)
    with reader() as s:
        before = s.execute(select(func.count()).select_from(Bar)).scalar()

    client.get("/")
    client.get("/api/state")
    client.get("/healthz")

    with reader() as s:
        after = s.execute(select(func.count()).select_from(Bar)).scalar()
    assert before == after


# --- demo mode -------------------------------------------------------------

@pytest.fixture()
def demo_client(tmp_engine):
    return create_app(engine=tmp_engine, client_factory=lambda: None, demo=True).test_client()


def test_demo_api_flags_synthetic(demo_client):
    data = demo_client.get("/api/state").get_json()
    assert data["demo"] is True
    assert len(data["orders"]) == 15
    assert len(data["fills"]) == 10
    assert len(data["equity_curve"]) == 30
    assert data["account"]["status"].endswith("(DEMO)")


def test_demo_html_shows_banner(demo_client):
    html = demo_client.get("/").data.decode()
    assert "DEMO MODE" in html
    assert "synthetic data" in html


def test_demo_kill_switch_variant(demo_client):
    data = demo_client.get("/api/state?kill=1").get_json()
    assert data["demo"] is True
    assert data["kill_switch_active"] is True
    html = demo_client.get("/?kill=1").data.decode()
    assert "ACTIVE" in html


def test_normal_mode_is_not_demo(client):
    assert client.get("/api/state").get_json()["demo"] is False


def test_demo_mode_does_not_touch_db(tmp_engine):
    """Demo mode must not read or write the database at all."""
    from src.storage.db import get_sessionmaker

    app = create_app(engine=tmp_engine, client_factory=lambda: None, demo=True)
    reader = get_sessionmaker(tmp_engine)
    with reader() as s:
        before = s.execute(select(func.count()).select_from(Bar)).scalar()
    c = app.test_client()
    c.get("/")
    c.get("/api/state")
    with reader() as s:
        after = s.execute(select(func.count()).select_from(Bar)).scalar()
    assert before == after == 0


def test_dashboard_session_rejects_writes(app):
    """The app's session has a before_flush guard that blocks any write."""
    Session = app.config["SESSION"]
    with Session() as s:
        s.add(
            Bar(symbol="ZZZ", timestamp=_dt.datetime(2026, 1, 1),
                open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        )
        with pytest.raises(RuntimeError, match="read-only"):
            s.flush()
