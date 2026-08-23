"""End-to-end smoke test for Sentinel's foundation.

Run:  ``python -m src.hello_world``

Steps (each real — nothing is faked):
  1. Connect to the Alpaca **paper** account and print equity.
  2. Fetch the last 30 days of SPY daily bars.
  3. Upsert them into ``data/market.db``.
  4. Print the stored row count for SPY.

Requires valid paper credentials in ``.env``. Exits non-zero on any error.
"""
from __future__ import annotations

import datetime as _dt
import sys

from src.broker.alpaca_client import AlpacaClient
from src.data.pipeline import HistoricalDataFetcher
from src.storage.db import init_db, make_engine
from src.utils.logger import get_logger

log = get_logger("hello_world")

SYMBOL = "SPY"
TIMEFRAME = "1Day"
LOOKBACK_DAYS = 30


def main() -> int:
    log.info("hello_world_start", symbol=SYMBOL)

    # 1. Connect to paper account (AlpacaClient hard-fails on a non-paper URL).
    client = AlpacaClient()
    account = client.get_account()
    equity = getattr(account, "equity", None)
    print(f"[account] status={getattr(account, 'status', '?')} equity={equity}")

    # 2/3. Fetch last 30 days of SPY bars and store them.
    end = _dt.datetime.now(_dt.timezone.utc)
    start = end - _dt.timedelta(days=LOOKBACK_DAYS)

    engine = init_db(make_engine())
    fetcher = HistoricalDataFetcher(client=client, engine=engine)
    written = fetcher.fetch_and_store([SYMBOL], TIMEFRAME, start, end)
    print(f"[fetch] wrote {written.get(SYMBOL, 0)} {SYMBOL} bars "
          f"({start.date()} -> {end.date()})")

    # 4. Report stored row count (read back from the DB, not the fetch result).
    stored = fetcher.get_all_bars(SYMBOL)
    print(f"[store] SPY rows in market.db: {len(stored)}")

    log.info("hello_world_done", symbol=SYMBOL, stored_rows=len(stored))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # surface a clean, non-zero failure
        log.error("hello_world_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
