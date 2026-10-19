"""One-shot daily paper-trading cycle — the entry point for scheduled CI runs.

``src.scripts.run_bot`` runs a *blocking* intraday scheduler (pre_open at 09:25,
ticks every few minutes, pre_close at 15:55) and never returns. That model needs
a long-lived host. GitHub Actions runners are ephemeral, so this script runs
exactly **one** cycle and exits:

    python -m src.scripts.run_once

Why once per day is enough: every strategy consumes **daily bars**, so the
signal set only changes when a new daily bar lands — once per session. A single
cycle after the open therefore captures the whole day's decision; the intraday
tick loop exists only to spread fills across a long-running process.

Sequence (mirrors the orchestrator's own pre_open → tick):
  1. Connect to the Alpaca **paper** account (AlpacaClient hard-fails on a
     non-paper URL).
  2. Null-wall startup gate — if any enabled strategy's validation is missing,
     stale, drifted, or not SURVIVED, exit 2 without trading.
  3. Trading-calendar gate — on a weekend/holiday there is no session, so log
     and exit 0 (a clean no-op). GitHub's cron already restricts us to Mon-Fri;
     this catches market holidays and is the authoritative check.
  4. pre_open: refresh bars, recompute + persist the regime, re-enforce the
     wall, pick the regime's active strategies, reset the daily risk counters.
  5. tick: generate signals → risk check → submit approved paper orders.
  6. reconcile: poll the broker so just-submitted orders' fills are recorded.

Exit codes: 0 = ran (or clean no-op on a non-trading day), 2 = Null wall
blocked, 1 = unexpected error. No secret is ever printed.
"""
from __future__ import annotations

import datetime as _dt
import sys

from sqlalchemy import func, select

from src.broker.alpaca_client import AlpacaClient
from src.orchestrator.runner import Orchestrator
from src.storage.db import OrderRecord, SignalRecord
from src.utils.logger import get_logger
from src.validation.null_wall import ValidationError

log = get_logger("scripts.run_once")

# Warmup history. The orchestrator's pre_open only tops up the last ~10 days,
# which is enough for a warm DB but not for a COLD one (the first run, or after
# a cache eviction). The regime detector needs >= 21 daily bars and the enabled
# strategies look back up to 60, so a cold DB must be backfilled first. We target
# ~220 trading days and fetch ~500 calendar days when any symbol is short.
WARMUP_MIN_BARS = 220
WARMUP_LOOKBACK_DAYS = 500


def _max_id(session, model) -> int:
    """Highest primary-key id in a table (0 if empty) — used to count new rows."""
    return int(session.execute(select(func.coalesce(func.max(model.id), 0))).scalar() or 0)


def _ensure_warmup(orch: Orchestrator, now: _dt.datetime) -> None:
    """Backfill deep daily history for any symbol whose stored history is too thin.

    Self-healing: no-op once the cache carries a warm DB forward, but guarantees a
    cold runner can still compute the regime and run every strategy.
    """
    symbols = orch._all_symbols()
    cold = [s for s in symbols if len(orch.fetcher.get_all_bars(s)) < WARMUP_MIN_BARS]
    if not cold:
        return
    end = now.astimezone(_dt.timezone.utc)
    start = end - _dt.timedelta(days=WARMUP_LOOKBACK_DAYS)
    print(
        f"[run_once] cold database — backfilling ~{WARMUP_LOOKBACK_DAYS}d of daily "
        f"bars for {cold} before trading."
    )
    log.info("run_once_warmup_backfill", symbols=cold, lookback_days=WARMUP_LOOKBACK_DAYS)
    orch.fetcher.fetch_and_store(cold, "1Day", start, end)


def run() -> int:
    log.info("run_once_start")

    # 1. Paper broker — refuses any non-paper endpoint at construction time.
    client = AlpacaClient()

    # 2. Orchestrator + Null-wall startup gate.
    orch = Orchestrator(client=client)
    try:
        orch.startup()
    except ValidationError as exc:
        log.critical("run_once_null_wall_blocked", error=str(exc))
        print(f"REFUSING TO TRADE: {exc}", file=sys.stderr)
        return 2

    now = _dt.datetime.now(orch.tz)
    today = now.date()

    # 3. Trading-calendar gate (holidays / weekends / early-close aware).
    bounds = orch.session_bounds(today)
    if bounds is None:
        msg = f"{today.isoformat()} is not an NYSE trading day — nothing to do."
        log.info("run_once_non_trading_day", date=today.isoformat())
        print(f"[run_once] {msg}")
        return 0
    open_et, close_et = bounds
    print(
        f"[run_once] {today.isoformat()} is a trading day "
        f"(session {open_et.strftime('%H:%M')}-{close_et.strftime('%H:%M %Z')}); "
        f"running one cycle at {now.strftime('%H:%M %Z')}."
    )

    # Seed deep history on a cold DB so the regime detector and strategies have
    # enough bars to work with (no-op once the cache carries a warm DB forward).
    _ensure_warmup(orch, now)

    # Snapshot table high-water marks so we can report only *this run's* rows.
    with orch._Session() as session:
        sig_before = _max_id(session, SignalRecord)
        ord_before = _max_id(session, OrderRecord)

    # 4 + 5. One pre_open → tick cycle.
    regime = orch.pre_open(now)
    tick_result = orch.tick(now)

    # 6. Reconcile so fills on just-submitted orders are captured before we exit.
    recon = orch.executor.reconcile()

    # --- Report ----------------------------------------------------------
    with orch._Session() as session:
        new_signals = session.execute(
            select(SignalRecord).where(SignalRecord.id > sig_before)
        ).scalars().all()
        new_orders = session.execute(
            select(OrderRecord).where(OrderRecord.id > ord_before)
        ).scalars().all()

    regime_label = regime.combined.value if regime is not None else "unknown"
    active = sorted(orch._active)
    submitted = tick_result.get("submitted", 0)
    halted = tick_result.get("halted", False)

    print("[run_once] ---- cycle summary ----")
    print(f"[run_once] regime: {regime_label}")
    print(f"[run_once] active strategies this regime: {active or '(none)'}")
    print(f"[run_once] kill switch halted run: {halted}")
    print(f"[run_once] signals generated: {len(new_signals)}")
    for s in new_signals:
        print(f"[run_once]   signal  {s.strategy:<24} {s.symbol:<6} {s.side:<4} "
              f"conf={s.confidence:.3f} :: {s.entry_reason}")
    print(f"[run_once] orders submitted: {submitted}")
    for o in new_orders:
        print(f"[run_once]   order   {o.strategy or '-':<24} {o.symbol:<6} {o.side:<4} "
              f"qty={o.quantity} type={o.order_type} status={o.status}")
    print(f"[run_once] reconcile: {recon['updated']} status update(s), "
          f"{recon['filled']} fill(s) recorded")

    log.info(
        "run_once_done",
        regime=regime_label,
        active=active,
        signals=len(new_signals),
        orders=submitted,
        reconciled=recon,
    )
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # clean non-zero on any failure
        log.error("run_once_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
