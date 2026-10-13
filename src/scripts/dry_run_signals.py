"""Dry-run signal generator — NEVER submits to Alpaca.

Reads stored bars from the local market DB, runs the moving-average crossover
strategy, and prints the signals whose decision bar falls within the last
``--days`` (default 60) calendar days.

    python -m src.scripts.dry_run_signals SPY
    python -m src.scripts.dry_run_signals SPY --fast 20 --slow 50 --days 60

The SMAs are warmed up over the *full* stored history (so a recent crossover is
not truncated), then signals are filtered to the recent window for display. This
is read-only: it touches the broker in no way.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys

import pandas as pd

from src.data.pipeline import HistoricalDataFetcher
from src.strategies.ma_crossover import MovingAverageCrossover
from src.utils.logger import get_logger

log = get_logger("scripts.dry_run_signals")


def run(symbol: str, fast: int, slow: int, days: int) -> int:
    fetcher = HistoricalDataFetcher()  # default market.db; no broker client needed
    bars = fetcher.get_all_bars(symbol)
    if bars.empty:
        print(f"[dry-run] no stored bars for {symbol} in market.db — "
              f"run `python -m src.hello_world` or seed data first.")
        return 0

    strat = MovingAverageCrossover(symbol=symbol, fast_window=fast, slow_window=slow)
    signals = strat.generate_signals(bars)

    cutoff = bars.index.max() - pd.Timedelta(days=days)
    recent = [s for s in signals if s.as_of is not None and s.as_of >= cutoff]

    def _fmt(s):
        stop = f"{s.stop_loss:.2f}" if s.stop_loss is not None else "-"
        return (f"  {s.as_of.date()}  {s.side.value:<4}  {s.symbol:<6} "
                f"conf={s.confidence:.3f}  stop={stop}  :: {s.entry_reason}")

    print(f"[dry-run] {symbol} MA crossover {fast}/{slow} "
          f"(DOES NOT SUBMIT TO ALPACA)")
    print(f"[dry-run] stored bars: {len(bars)} "
          f"({bars.index.min().date()} -> {bars.index.max().date()})")
    print(f"[dry-run] signals in last {days} days: {len(recent)}")
    for s in recent:
        print(_fmt(s))
    if not recent:
        print("  (no crossover in the last "
              f"{days} days — crossovers are infrequent; this is expected, not a bug)")

    # Always demonstrate that generation works by showing the full-history total
    # and the most recent crossover on record.
    print(f"[dry-run] signals over full stored history: {len(signals)}")
    if signals:
        print("[dry-run] most recent crossover on record:")
        print(_fmt(signals[-1]))

    log.info("dry_run_signals_done", symbol=symbol, total_signals=len(signals),
             recent=len(recent), days=days)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run MA crossover signals (no submission).")
    parser.add_argument("symbol", nargs="?", default="SPY", help="symbol (default SPY)")
    parser.add_argument("--fast", type=int, default=20, help="fast SMA window")
    parser.add_argument("--slow", type=int, default=50, help="slow SMA window")
    parser.add_argument("--days", type=int, default=60, help="recent window in days")
    args = parser.parse_args(argv)
    return run(args.symbol.upper(), args.fast, args.slow, args.days)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # clean non-zero on failure
        log.error("dry_run_signals_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
