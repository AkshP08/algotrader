"""Launch the live (paper) orchestrator.

    python -m src.scripts.run_bot

Startup sequence:
  1. Connect to the Alpaca **paper** account (hard-fails on a non-paper URL).
  2. Build the orchestrator and run the **Null-wall startup gate** — if any
     configured strategy's validation is missing, stale, drifted, or not
     SURVIVED, the process logs CRITICAL and exits non-zero without scheduling.
  3. Write the PID lockfile (the dashboard's heartbeat source).
  4. Register the trading-calendar-aware scheduler and run it (blocking).

SIGTERM / SIGINT are handled gracefully: pending orders are cancelled, the
scheduler is shut down, and the lockfile is removed.
"""
from __future__ import annotations

import signal
import sys

from src.broker.alpaca_client import AlpacaClient
from src.orchestrator.lockfile import remove_lockfile, touch_lockfile, write_lockfile
from src.orchestrator.runner import Orchestrator
from src.utils.logger import get_logger
from src.validation.null_wall import ValidationError

log = get_logger("scripts.run_bot")


def _install_signal_handlers(orchestrator: Orchestrator, scheduler) -> None:
    def _graceful_shutdown(signum, _frame):
        log.warning("run_bot_signal", signal=signum)
        try:
            if orchestrator.client is not None:
                orchestrator.client.cancel_all_orders()
        except Exception as exc:  # never block shutdown on a broker error
            log.error("run_bot_cancel_failed", error=str(exc))
        try:
            if getattr(scheduler, "running", False):
                scheduler.shutdown(wait=False)
        except Exception as exc:
            log.error("run_bot_scheduler_shutdown_failed", error=str(exc))
        remove_lockfile()
        log.warning("run_bot_stopped")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)


def main() -> int:
    log.info("run_bot_start")

    # 1. Paper broker (AlpacaClient refuses any non-paper endpoint).
    client = AlpacaClient()

    # 2. Orchestrator + Null-wall startup gate.
    orchestrator = Orchestrator(client=client)
    try:
        orchestrator.startup()
    except ValidationError as exc:
        log.critical("run_bot_null_wall_blocked", error=str(exc))
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return 2

    # 3. Lockfile heartbeat.
    write_lockfile()
    touch_lockfile()

    # 4. Scheduler (blocking).
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone=orchestrator.tz)
    orchestrator.build_scheduler(scheduler)
    _install_signal_handlers(orchestrator, scheduler)

    log.info("run_bot_scheduler_starting", timezone=str(orchestrator.tz))
    print("[run_bot] orchestrator scheduled. Ctrl-C / SIGTERM to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        remove_lockfile()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # clean non-zero on failure
        log.error("run_bot_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
