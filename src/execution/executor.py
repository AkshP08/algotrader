"""Order execution layer.

``OrderExecutor`` turns an approved :class:`~src.risk.manager.RiskDecision` into a
paper-market order, keeps the ``orders`` / ``fills`` tables in sync with the
broker, and routes rejections back to the originating strategy.

Two hard safety gates apply to **every** submission:
  * the kill switch — if active, nothing is submitted;
  * the risk decision — if not ``approved``, nothing is submitted.

**Idempotency.** Each order carries a deterministic
``client_order_id = f"{strategy}-{symbol}-{utc_ms}"`` derived from the signal's
``as_of`` bar timestamp. Re-submitting the same logical signal produces the same
id, so a retry can never double-send (Alpaca rejects a duplicate client id).
"""
from __future__ import annotations

import datetime as _dt
from typing import Mapping, Optional

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.risk.manager import RiskDecision
from src.storage.db import (
    FillRecord,
    OrderRecord,
    get_sessionmaker,
    init_db,
    make_engine,
)
from src.strategies.base import Fill, Side, Signal
from src.utils.logger import get_logger

log = get_logger("execution.executor")

# Broker statuses that mean "nothing more will happen" — skipped by reconcile.
TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}
_OPEN_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "pending"}


class OrderExecutor:
    """Submits orders, reconciles their state, and handles rejections."""

    def __init__(
        self,
        client,
        engine: Optional[Engine] = None,
        kill_switch=None,
        strategies: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.client = client
        self.engine = engine or make_engine()
        init_db(self.engine)
        self._Session = get_sessionmaker(self.engine)
        self.kill_switch = kill_switch
        # name -> Strategy, for on_fill / on_reject dispatch.
        self.strategies: dict[str, object] = dict(strategies or {})

    # --- idempotency ------------------------------------------------------

    @staticmethod
    def _utc_ms(signal: Signal) -> int:
        """Milliseconds for the client id — deterministic from the signal's bar."""
        ts = signal.as_of
        if ts is not None:
            return int(ts.timestamp() * 1000)
        return int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)

    @staticmethod
    def build_client_order_id(signal: Signal) -> str:
        strategy = str(signal.meta.get("strategy", "strategy"))
        return f"{strategy}-{signal.symbol}-{OrderExecutor._utc_ms(signal)}"

    # --- submission -------------------------------------------------------

    def submit(self, signal: Signal, risk_decision: RiskDecision) -> Optional[str]:
        """Submit a market order if (and only if) it is permitted.

        Returns the broker order id (or the client id as a fallback), or ``None``
        if the submission was blocked by the kill switch or the risk decision.
        """
        # Gate 1: kill switch.
        if self.kill_switch is not None and self.kill_switch.is_active():
            log.warning(
                "submit_blocked_kill_switch", symbol=signal.symbol, side=signal.side.value
            )
            return None

        # Gate 2: risk approval.
        if not risk_decision.approved:
            log.warning(
                "submit_blocked_not_approved",
                symbol=signal.symbol,
                reason=risk_decision.reason,
            )
            return None

        qty = risk_decision.modified_quantity
        if qty is None:
            qty = signal.quantity_hint
        if qty is None or qty <= 0:
            log.error("submit_no_quantity", symbol=signal.symbol)
            return None

        client_order_id = self.build_client_order_id(signal)
        strategy = str(signal.meta.get("strategy", "strategy"))

        # EXIT closes a position; the broker decides the offsetting side.
        if signal.side is Side.EXIT:
            order = self.client.close_position(signal.symbol)
            broker_id = str(getattr(order, "id", None)) if order is not None else None
            self._record_order(
                client_order_id=client_order_id, broker_order_id=broker_id,
                strategy=strategy, symbol=signal.symbol, side="exit",
                order_type="close", qty=float(qty),
                status=str(getattr(order, "status", "new")),
            )
            log.info("submit_exit", symbol=signal.symbol, client_order_id=client_order_id)
            return broker_id or client_order_id

        broker_side = "buy" if signal.side is Side.BUY else "sell"
        order = self.client.submit_order(
            symbol=signal.symbol,
            qty=float(qty),
            side=broker_side,
            order_type="market",
            time_in_force="day",
            client_order_id=client_order_id,
            extended_hours=False,
        )
        broker_id = str(getattr(order, "id", None)) if order is not None else None
        self._record_order(
            client_order_id=client_order_id, broker_order_id=broker_id,
            strategy=strategy, symbol=signal.symbol, side=broker_side,
            order_type="market", qty=float(qty),
            status=str(getattr(order, "status", "new")),
        )
        log.info(
            "submit_order_recorded",
            symbol=signal.symbol, side=broker_side, qty=float(qty),
            client_order_id=client_order_id, broker_order_id=broker_id,
        )
        return broker_id or client_order_id

    # --- reconciliation ---------------------------------------------------

    def reconcile(self) -> dict:
        """Poll the broker for our open orders; sync statuses and record fills.

        Returns a small summary dict (updated / filled counts) for logging/tests.
        """
        updated = filled = 0
        with self._Session() as session:
            open_orders = session.execute(
                select(OrderRecord).where(OrderRecord.status.notin_(TERMINAL_STATUSES))
            ).scalars().all()

            for rec in open_orders:
                if not rec.broker_order_id:
                    continue
                broker = self.client.get_order(rec.broker_order_id)
                status = str(getattr(broker, "status", "")).lower().replace("orderstatus.", "")
                if not status:
                    continue
                if status != rec.status:
                    rec.status = status
                    updated += 1

                if status in ("filled", "partially_filled"):
                    if self._record_fill_if_new(session, rec, broker):
                        filled += 1
                        self._dispatch_fill(rec, broker)
            session.commit()
        log.info("reconcile_complete", updated=updated, filled=filled)
        return {"updated": updated, "filled": filled}

    def _record_fill_if_new(self, session, rec: OrderRecord, broker) -> bool:
        """Insert a fill row for this broker order id if none exists yet."""
        existing = session.execute(
            select(FillRecord).where(FillRecord.broker_order_id == rec.broker_order_id)
        ).scalars().first()
        if existing is not None:
            return False
        filled_qty = float(getattr(broker, "filled_qty", 0) or 0)
        filled_price = float(getattr(broker, "filled_avg_price", 0) or 0)
        filled_at = getattr(broker, "filled_at", None) or _dt.datetime.now(_dt.timezone.utc)
        session.add(
            FillRecord(
                broker_order_id=rec.broker_order_id,
                symbol=rec.symbol,
                side=rec.side,
                filled_qty=filled_qty,
                filled_avg_price=filled_price,
                filled_at=filled_at,
            )
        )
        return True

    def _dispatch_fill(self, rec: OrderRecord, broker) -> None:
        strat = self.strategies.get(rec.strategy or "")
        if strat is None:
            return
        side = Side.BUY if rec.side == "buy" else (Side.EXIT if rec.side == "exit" else Side.SELL)
        try:
            strat.on_fill(
                Fill(
                    symbol=rec.symbol,
                    side=side,
                    filled_qty=float(getattr(broker, "filled_qty", 0) or 0),
                    filled_avg_price=float(getattr(broker, "filled_avg_price", 0) or 0),
                    filled_at=getattr(broker, "filled_at", None) or _dt.datetime.now(_dt.timezone.utc),
                )
            )
        except Exception as exc:  # a strategy callback must never break reconcile
            log.error("on_fill_dispatch_failed", strategy=rec.strategy, error=str(exc))

    # --- rejection handling ----------------------------------------------

    def handle_rejection(self, order) -> None:
        """Log a rejected order and notify its strategy via ``on_reject``."""
        symbol = getattr(order, "symbol", None)
        strategy_name = getattr(order, "strategy", None)
        log.error("order_rejected", symbol=symbol, strategy=strategy_name, order=str(order))
        strat = self.strategies.get(strategy_name or "")
        if strat is not None:
            strat.on_reject(order)

    # --- internals --------------------------------------------------------

    def _record_order(
        self, *, client_order_id, broker_order_id, strategy, symbol, side,
        order_type, qty, status,
    ) -> None:
        clean_status = status.lower().replace("orderstatus.", "") if status else "new"
        with self._Session() as session:
            session.add(
                OrderRecord(
                    broker_order_id=broker_order_id,
                    client_order_id=client_order_id,
                    strategy=strategy,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=qty,
                    time_in_force="day",
                    status=clean_status,
                )
            )
            session.commit()
