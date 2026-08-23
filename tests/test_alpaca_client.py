"""Tests for the paper-only Alpaca client.

Everything here is mocked — no network, no real keys. The critical assertion is
that a live/funded URL is rejected hard.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.broker import alpaca_client as ac
from src.broker.alpaca_client import AlpacaClient, LiveTradingError, assert_paper_trading

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


# --- Paper-only hard fail --------------------------------------------------

def test_assert_paper_trading_accepts_paper():
    assert assert_paper_trading(PAPER_URL) == PAPER_URL
    assert assert_paper_trading("https://paper.alpaca.markets") is not None


@pytest.mark.parametrize("bad", [LIVE_URL, "https://API.ALPACA.MARKETS", "", None])
def test_assert_paper_trading_rejects_live(bad):
    with pytest.raises(LiveTradingError):
        assert_paper_trading(bad)


def test_client_init_rejects_live_url():
    with pytest.raises(LiveTradingError):
        AlpacaClient(api_key="k", secret_key="s", base_url=LIVE_URL)


def test_client_init_requires_keys(monkeypatch):
    # Clear the env fallback so a truly keyless init is exercised.
    monkeypatch.delenv("ALPACA_API_KEY_PAPER", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY_PAPER", raising=False)
    with pytest.raises(RuntimeError):
        AlpacaClient(api_key=None, secret_key=None, base_url=PAPER_URL)


# --- Method surface --------------------------------------------------------

def test_client_has_required_methods():
    required = [
        "get_account",
        "get_positions",
        "get_bars",
        "submit_order",
        "cancel_all_orders",
        "get_order",
        "close_position",
    ]
    for name in required:
        assert callable(getattr(AlpacaClient, name)), f"missing method: {name}"


def _paper_client(mocker):
    """A client with mocked SDK sub-clients injected (no network)."""
    c = AlpacaClient(api_key="k", secret_key="s", base_url=PAPER_URL)
    c._trading = mocker.MagicMock(name="trading")
    c._data = mocker.MagicMock(name="data")
    return c


def test_get_account_delegates(mocker):
    c = _paper_client(mocker)
    c._trading.get_account.return_value = mocker.MagicMock(equity="100000", status="ACTIVE")
    acct = c.get_account()
    c._trading.get_account.assert_called_once()
    assert acct.equity == "100000"


def test_get_positions_delegates(mocker):
    c = _paper_client(mocker)
    c._trading.get_all_positions.return_value = [1, 2, 3]
    assert c.get_positions() == [1, 2, 3]
    c._trading.get_all_positions.assert_called_once()


def test_cancel_all_orders_delegates(mocker):
    c = _paper_client(mocker)
    c._trading.cancel_orders.return_value = []
    c.cancel_all_orders()
    c._trading.cancel_orders.assert_called_once()


def test_get_order_delegates(mocker):
    c = _paper_client(mocker)
    c._trading.get_order_by_id.return_value = mocker.MagicMock(status="filled")
    order = c.get_order("abc123")
    c._trading.get_order_by_id.assert_called_once_with("abc123")
    assert order.status == "filled"


def test_close_position_delegates(mocker):
    c = _paper_client(mocker)
    c.close_position("SPY")
    c._trading.close_position.assert_called_once_with("SPY")


def test_submit_order_builds_market_request(mocker):
    c = _paper_client(mocker)
    c._trading.submit_order.return_value = mocker.MagicMock(id="oid", status="accepted")
    c.submit_order("SPY", qty=1, side="buy", order_type="market")
    c._trading.submit_order.assert_called_once()


def test_get_bars_flattens_multiindex(mocker):
    c = _paper_client(mocker)
    idx = pd.MultiIndex.from_product(
        [["SPY"], pd.to_datetime(["2024-01-02", "2024-01-03"])],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        },
        index=idx,
    )
    c._data.get_stock_bars.return_value = mocker.MagicMock(df=df)
    out = c.get_bars("SPY", "1Day", "2024-01-01", "2024-01-04")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert not isinstance(out.index, pd.MultiIndex)


# --- Retry / backoff -------------------------------------------------------

def test_retry_recovers_from_transient_error(mocker):
    mocker.patch.object(ac.time, "sleep")  # no real delay
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert ac._retry(flaky, op="test") == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_validation_error(mocker):
    sleep = mocker.patch.object(ac.time, "sleep")

    def bad():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        ac._retry(bad, op="test")
    sleep.assert_not_called()


def test_retry_gives_up_after_max(mocker):
    mocker.patch.object(ac.time, "sleep")
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        ac._retry(always_fail, op="test")
    assert calls["n"] == ac.MAX_RETRIES
