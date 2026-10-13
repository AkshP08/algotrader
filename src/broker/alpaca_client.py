"""Paper-only Alpaca broker client.

Safety is the point of this module:

  * **Paper-only hard fail.** On import (when ``ALPACA_BASE_URL`` is set) and again
    in ``AlpacaClient.__init__``, the base URL is checked with
    :func:`assert_paper_trading`. Anything that is not a paper host raises
    ``LiveTradingError`` immediately. Live/funded keys can never be used.
  * **Retry with exponential backoff** (max 3 attempts) on transient network
    errors only — never on validation/auth errors, which should fail fast.
  * **No secrets in logs.** Only non-sensitive request/response metadata is logged.

The heavy Alpaca SDK objects are imported lazily so the module (and the paper-URL
guard) can be imported and unit-tested without the SDK or a network connection.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()  # populate os.environ from .env if present (no-op if absent)

log = get_logger("broker.alpaca")

# Hostnames we accept. The paper endpoint is the only trading host allowed.
_PAPER_MARKERS = ("paper-api.alpaca.markets", "paper.alpaca.markets")

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5


class LiveTradingError(RuntimeError):
    """Raised when a non-paper (live/funded) endpoint is configured."""


def assert_paper_trading(base_url: Optional[str]) -> str:
    """Return ``base_url`` if it is a paper endpoint, else raise ``LiveTradingError``.

    The rule is deliberately strict: the URL must contain ``paper``. A missing or
    live URL (e.g. ``https://api.alpaca.markets``) is rejected.
    """
    if not base_url:
        raise LiveTradingError(
            "ALPACA_BASE_URL is not set. Refusing to run without an explicit "
            "paper endpoint (https://paper-api.alpaca.markets)."
        )
    url = base_url.strip().lower()
    if "paper" not in url:
        raise LiveTradingError(
            f"Refusing to connect: ALPACA_BASE_URL={base_url!r} is not a paper "
            f"endpoint. Sentinel is paper-trading only."
        )
    return base_url


# --- Import-time guard -----------------------------------------------------
# If a base URL is configured at import time, validate it now so a live URL can
# never slip through even before a client is constructed. If it's unset we defer
# to AlpacaClient.__init__ (so tooling can import this module without env).
_ENV_BASE_URL = os.getenv("ALPACA_BASE_URL")
if _ENV_BASE_URL is not None:
    assert_paper_trading(_ENV_BASE_URL)


def _is_retryable(exc: Exception) -> bool:
    """Transient network/timeout errors are retryable; everything else is not."""
    retryable_types = (ConnectionError, TimeoutError)
    if isinstance(exc, retryable_types):
        return True
    # requests/urllib3 style errors, if present, without a hard dependency.
    name = type(exc).__name__.lower()
    return any(tok in name for tok in ("timeout", "connection", "readtimeout"))


def _retry(fn: Callable[[], Any], *, op: str) -> Any:
    """Call ``fn`` with exponential backoff on transient errors (max 3 tries)."""
    last: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise non-retryable below
            last = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                log.error("alpaca_call_failed", op=op, attempt=attempt, error=str(exc))
                raise
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "alpaca_call_retry",
                op=op,
                attempt=attempt,
                next_delay_s=sleep_s,
                error=str(exc),
            )
            time.sleep(sleep_s)
    # Unreachable, but keeps type-checkers happy.
    raise last  # type: ignore[misc]


class AlpacaClient:
    """Thin, paper-only wrapper over the Alpaca trading + market-data SDKs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        data_feed: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY_PAPER")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY_PAPER")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL")
        self.data_feed = (data_feed or os.getenv("ALPACA_DATA_FEED", "iex")).lower()

        # Paper-only guard — enforced again here regardless of import-time state.
        assert_paper_trading(self.base_url)

        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Missing Alpaca paper credentials. Set ALPACA_API_KEY_PAPER and "
                "ALPACA_SECRET_KEY_PAPER in your .env."
            )

        self._trading = None  # lazily constructed TradingClient
        self._data = None  # lazily constructed StockHistoricalDataClient
        log.info(
            "alpaca_client_init",
            base_url=self.base_url,
            data_feed=self.data_feed,
            paper=True,
        )

    # --- Lazy SDK construction -------------------------------------------

    @property
    def trading(self):
        if self._trading is None:
            from alpaca.trading.client import TradingClient

            self._trading = TradingClient(
                self.api_key, self.secret_key, paper=True
            )
        return self._trading

    @property
    def data(self):
        if self._data is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._data = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._data

    # --- Account / positions ---------------------------------------------

    def get_account(self):
        """Return the paper account object (equity, buying power, etc.)."""
        acct = _retry(lambda: self.trading.get_account(), op="get_account")
        log.info(
            "get_account",
            equity=getattr(acct, "equity", None),
            buying_power=getattr(acct, "buying_power", None),
            status=str(getattr(acct, "status", None)),
        )
        return acct

    def get_positions(self):
        """Return all open positions."""
        positions = _retry(lambda: self.trading.get_all_positions(), op="get_positions")
        log.info("get_positions", count=len(positions))
        return positions

    def close_position(self, symbol: str):
        """Liquidate the position in ``symbol``."""
        res = _retry(
            lambda: self.trading.close_position(symbol), op="close_position"
        )
        log.info("close_position", symbol=symbol)
        return res

    # --- Market data ------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, start, end):
        """Fetch OHLCV bars for ``symbol`` as a pandas DataFrame.

        ``timeframe`` is a string like ``"1Day"``, ``"1Hour"``, ``"1Min"``.
        Returns a DataFrame indexed by timestamp with columns
        ``[open, high, low, close, volume]`` (empty DataFrame if none).
        """
        import pandas as pd
        from alpaca.data.requests import StockBarsRequest

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_parse_timeframe(timeframe),
            start=start,
            end=end,
            feed=self.data_feed,
        )
        barset = _retry(lambda: self.data.get_stock_bars(req), op="get_bars")

        df = getattr(barset, "df", None)
        if df is None or len(df) == 0:
            log.warning("get_bars_empty", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Alpaca returns a MultiIndex (symbol, timestamp); flatten to timestamp.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].sort_index()
        log.info("get_bars", symbol=symbol, timeframe=timeframe, rows=len(df))
        return df

    # --- Orders -----------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ):
        """Submit an order to the paper account.

        Supports market/limit/stop orders. ``extended_hours`` defaults to False
        (and must be False for market orders, which Alpaca only fills in regular
        hours). Raises on validation errors (not retried); retries only transient
        network failures.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopOrderRequest,
        )

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce(time_in_force.lower())
        ot = order_type.lower()

        common = dict(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
            client_order_id=client_order_id,
            extended_hours=extended_hours,
        )
        if ot == "market":
            request = MarketOrderRequest(**common)
        elif ot == "limit":
            request = LimitOrderRequest(limit_price=limit_price, **common)
        elif ot == "stop":
            request = StopOrderRequest(stop_price=stop_price, **common)
        else:
            raise ValueError(f"unsupported order_type: {order_type!r}")

        order = _retry(
            lambda: self.trading.submit_order(request), op="submit_order"
        )
        log.info(
            "submit_order",
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=ot,
            broker_order_id=str(getattr(order, "id", None)),
            status=str(getattr(order, "status", None)),
        )
        return order

    def get_order(self, order_id: str):
        """Fetch a single order by broker id."""
        order = _retry(
            lambda: self.trading.get_order_by_id(order_id), op="get_order"
        )
        log.info(
            "get_order",
            broker_order_id=str(order_id),
            status=str(getattr(order, "status", None)),
        )
        return order

    def get_orders(self, status: str = "open"):
        """List orders by status (default ``open``). Used by order reconciliation."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus(status.lower()))
        orders = _retry(lambda: self.trading.get_orders(filter=req), op="get_orders")
        log.info("get_orders", status=status, count=len(orders) if orders is not None else 0)
        return orders

    def cancel_all_orders(self):
        """Cancel every open order (used by the kill switch)."""
        res = _retry(lambda: self.trading.cancel_orders(), op="cancel_all_orders")
        log.warning("cancel_all_orders", cancelled=len(res) if res is not None else 0)
        return res


def _parse_timeframe(timeframe: str):
    """Map a string like ``"1Day"`` / ``"15Min"`` to an Alpaca ``TimeFrame``."""
    import re

    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    s = str(timeframe).strip()
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", s)
    if not m:
        raise ValueError(f"cannot parse timeframe: {timeframe!r}")
    amount = int(m.group(1))
    unit_raw = m.group(2).lower()
    unit_map = {
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    if unit_raw not in unit_map:
        raise ValueError(f"unknown timeframe unit: {unit_raw!r}")
    return TimeFrame(amount, unit_map[unit_raw])
