"""Historical market-data pipeline.

Responsibilities:
  * Pull OHLCV bars for a set of symbols/timeframe from Alpaca and **upsert** them
    into SQLite (``data/market.db``), keyed by (symbol, timestamp) so re-runs are
    idempotent and never duplicate rows.
  * Enforce a hard **look-ahead guard**: :meth:`get_bars_as_of` can never return a
    bar whose timestamp is after the ``as_of`` cutoff. This is the single most
    important correctness property for honest backtesting.
  * **Gap detection**: flag missing trading days (weekends/US market holidays are
    expected and ignored; any other missing business day is warned).

No synthetic data is ever generated — every row originates from the broker feed.
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterable, Optional, Sequence

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from src.storage.db import Bar, init_db, make_engine
from src.utils.logger import get_logger

log = get_logger("data.pipeline")

_OHLCV = ["open", "high", "low", "close", "volume"]


def _to_datetime(value) -> _dt.datetime:
    """Coerce a date/str/Timestamp to a python ``datetime`` (naive UTC wall clock)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


class HistoricalDataFetcher:
    """Fetch bars from a broker client and persist them to SQLite."""

    def __init__(self, client=None, engine: Optional[Engine] = None) -> None:
        """``client`` is any object exposing ``get_bars(symbol, timeframe, start, end)``
        (e.g. :class:`~src.broker.alpaca_client.AlpacaClient`). ``engine`` lets tests
        point at a temp database.
        """
        self.client = client
        self.engine = engine or make_engine()
        init_db(self.engine)

    # --- Fetch + store ----------------------------------------------------

    def fetch_and_store(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start,
        end,
    ) -> dict[str, int]:
        """Fetch bars for each symbol and upsert them. Returns rows-written per symbol."""
        if self.client is None:
            raise RuntimeError("HistoricalDataFetcher has no broker client to fetch with")

        written: dict[str, int] = {}
        for symbol in symbols:
            df = self.client.get_bars(symbol, timeframe, start, end)
            n = self.upsert_bars(symbol, df)
            written[symbol] = n
            self.detect_gaps(symbol, df, timeframe=timeframe)
        log.info("fetch_and_store_complete", symbols=list(symbols), written=written)
        return written

    def upsert_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """Upsert an OHLCV DataFrame for ``symbol`` into the bars table.

        Idempotent: existing (symbol, timestamp) rows are updated in place.
        """
        if df is None or len(df) == 0:
            return 0

        missing = [c for c in _OHLCV if c not in df.columns]
        if missing:
            raise ValueError(f"bars for {symbol} missing columns: {missing}")

        rows = []
        for ts, row in df.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": _to_datetime(ts),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        stmt = sqlite_insert(Bar).values(rows)
        # ON CONFLICT(symbol, timestamp) DO UPDATE — keep the latest values.
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timestamp"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        log.info("upsert_bars", symbol=symbol, rows=len(rows))
        return len(rows)

    # --- Read with look-ahead guard --------------------------------------

    def get_bars_as_of(self, symbol: str, as_of_date) -> pd.DataFrame:
        """Return stored bars for ``symbol`` with ``timestamp <= as_of_date``.

        **Look-ahead guard.** This method must NEVER return a bar timestamped after
        ``as_of_date``. It is the only sanctioned way for a backtest/strategy to read
        history at a simulated point in time.
        """
        cutoff = _to_datetime(as_of_date)
        stmt = (
            select(
                Bar.timestamp, Bar.open, Bar.high, Bar.low, Bar.close, Bar.volume
            )
            .where(Bar.symbol == symbol)
            .where(Bar.timestamp <= cutoff)  # <= the guard
            .order_by(Bar.timestamp.asc())
        )
        with self.engine.begin() as conn:
            result = conn.execute(stmt).all()

        if not result:
            return pd.DataFrame(columns=_OHLCV, index=pd.DatetimeIndex([], name="timestamp"))

        df = pd.DataFrame(result, columns=["timestamp", *_OHLCV])
        df = df.set_index("timestamp").sort_index()

        # Defense in depth: assert the guard actually held before returning.
        if len(df) and df.index.max() > pd.Timestamp(cutoff):
            raise AssertionError(
                "look-ahead violation: get_bars_as_of returned a bar after the cutoff"
            )
        return df

    def get_all_bars(self, symbol: str) -> pd.DataFrame:
        """Return every stored bar for ``symbol`` (no cutoff)."""
        stmt = (
            select(Bar.timestamp, Bar.open, Bar.high, Bar.low, Bar.close, Bar.volume)
            .where(Bar.symbol == symbol)
            .order_by(Bar.timestamp.asc())
        )
        with self.engine.begin() as conn:
            result = conn.execute(stmt).all()
        if not result:
            return pd.DataFrame(columns=_OHLCV, index=pd.DatetimeIndex([], name="timestamp"))
        df = pd.DataFrame(result, columns=["timestamp", *_OHLCV])
        return df.set_index("timestamp").sort_index()

    # --- Gap detection ----------------------------------------------------

    def detect_gaps(
        self, symbol: str, df: pd.DataFrame, timeframe: str = "1Day"
    ) -> list[_dt.date]:
        """Detect missing *trading* days in a daily bar set.

        Weekends and US market holidays are expected and ignored. Any other missing
        business day between the first and last bar is flagged (WARN) and returned.
        Only meaningful for daily timeframes; intraday sets return ``[]``.
        """
        if "day" not in timeframe.lower() or df is None or len(df) < 2:
            return []

        idx = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
        present = set(idx.date)
        start, end = idx.min(), idx.max()

        business_days = pd.bdate_range(start=start, end=end)
        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
        holiday_dates = {h.date() for h in pd.DatetimeIndex(holidays)}

        expected = {d.date() for d in business_days} - holiday_dates
        missing = sorted(expected - present)

        if missing:
            log.warning(
                "data_gap_detected",
                symbol=symbol,
                missing_days=[d.isoformat() for d in missing],
                count=len(missing),
            )
        return missing


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """Simple buy-and-hold return over an OHLCV frame: last/first close - 1.

    Used by the self-proof test: the return computed from stored bars must match
    the return computed directly from the freshly fetched bars to machine epsilon,
    proving the storage/read round-trip is lossless.
    """
    if df is None or len(df) < 2:
        raise ValueError("need at least two bars to compute a buy-and-hold return")
    closes = df["close"].to_numpy(dtype="float64")
    return float(closes[-1] / closes[0] - 1.0)
