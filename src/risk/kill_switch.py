"""The automatic kill switch.

Process-level, SQLite-backed state: both *trigger* and *reset* events are written
to the ``kill_switch_events`` table (from Prompt 1), and :meth:`is_active` is
derived from the most recent event. There is **no automatic reset in code** — the
only way back to inactive is a manual :meth:`reset` with a non-empty acknowledgment.

On :meth:`trigger` the switch:
  1. logs CRITICAL,
  2. cancels all open orders via ``AlpacaClient.cancel_all_orders()``,
  3. closes every open position via ``AlpacaClient.close_position(symbol)``,
  4. writes the event.

Once active, the orchestration layer (Prompt 3) must process no signals and submit
no orders; the executor also guards on :meth:`is_active`.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Optional

from sqlalchemy import desc, select
from sqlalchemy.engine import Engine

from src.storage.db import KillSwitchEvent, get_sessionmaker, init_db, make_engine
from src.utils.logger import get_logger

log = get_logger("risk.kill_switch")

EVENT_TRIGGER = "trigger"
EVENT_RESET = "reset"
RESET_MARKER = "__reset__"  # stored in the non-nullable `trigger` column for resets


class KillSwitch:
    """Automatic trading halt with manual-only reset."""

    def __init__(self, client=None, engine: Optional[Engine] = None) -> None:
        """``client`` is an :class:`AlpacaClient` (needed to flatten on trigger);
        ``engine`` lets tests point at a temp database."""
        self.client = client
        self.engine = engine or make_engine()
        init_db(self.engine)
        self._Session = get_sessionmaker(self.engine)

    # --- state ------------------------------------------------------------

    def _latest_event(self) -> Optional[KillSwitchEvent]:
        with self._Session() as session:
            return session.execute(
                select(KillSwitchEvent).order_by(desc(KillSwitchEvent.id)).limit(1)
            ).scalars().first()

    def is_active(self) -> bool:
        """Active iff the most recent event is a trigger (not a reset)."""
        latest = self._latest_event()
        if latest is None:
            return False
        try:
            event_type = json.loads(latest.detail).get("event")
        except (ValueError, TypeError):
            # Legacy/plain detail rows are treated as triggers (fail safe).
            event_type = EVENT_TRIGGER
        return event_type == EVENT_TRIGGER

    # --- trigger / reset --------------------------------------------------

    def trigger(self, reason: str, triggered_by: str) -> None:
        """Halt trading: cancel orders, flatten positions, and record the event."""
        equity = self._current_equity()
        log.critical(
            "kill_switch_triggered",
            reason=reason,
            triggered_by=triggered_by,
            equity=equity,
        )

        cancelled = closed = 0
        if self.client is not None:
            # 1. Cancel all resting orders.
            self.client.cancel_all_orders()
            cancelled = 1
            # 2. Flatten every open position.
            try:
                positions = self.client.get_positions()
            except Exception as exc:  # defensive: never let flatten-read abort the halt
                log.error("kill_switch_positions_read_failed", error=str(exc))
                positions = []
            for pos in positions or []:
                symbol = getattr(pos, "symbol", None) or (
                    pos.get("symbol") if isinstance(pos, dict) else None
                )
                if symbol:
                    self.client.close_position(symbol)
                    closed += 1
        else:
            log.warning("kill_switch_no_client", note="cannot flatten without a client")

        detail = json.dumps(
            {
                "event": EVENT_TRIGGER,
                "reason": reason,
                "triggered_by": triggered_by,
                "positions_closed": closed,
            }
        )
        self._write_event(trigger=reason, detail=detail, equity=equity)
        log.critical(
            "kill_switch_halted",
            reason=reason,
            orders_cancelled=bool(cancelled),
            positions_closed=closed,
        )

    def reset(self, acknowledged_by: str, note: str) -> None:
        """Manually clear the switch. Requires a non-empty acknowledgment ``note``."""
        if not note or not note.strip():
            raise ValueError("reset requires a non-empty acknowledgment note")
        if not acknowledged_by or not acknowledged_by.strip():
            raise ValueError("reset requires a non-empty acknowledged_by")
        detail = json.dumps(
            {"event": EVENT_RESET, "acknowledged_by": acknowledged_by, "note": note}
        )
        self._write_event(trigger=RESET_MARKER, detail=detail, equity=None)
        log.warning("kill_switch_reset", acknowledged_by=acknowledged_by, note=note)

    # --- internals --------------------------------------------------------

    def _current_equity(self) -> Optional[float]:
        if self.client is None:
            return None
        try:
            acct = self.client.get_account()
            eq = getattr(acct, "equity", None)
            return float(eq) if eq is not None else None
        except Exception as exc:
            log.error("kill_switch_equity_read_failed", error=str(exc))
            return None

    def _write_event(
        self, *, trigger: str, detail: str, equity: Optional[float]
    ) -> None:
        with self._Session() as session:
            session.add(
                KillSwitchEvent(
                    created_at=_dt.datetime.now(_dt.timezone.utc),
                    trigger=trigger,
                    detail=detail,
                    equity_at_trigger=equity,
                )
            )
            session.commit()
