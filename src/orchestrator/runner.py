"""The orchestrator — the thing that runs the bot by itself.

On a trading-calendar-aware schedule (APScheduler, NYSE via
``pandas_market_calendars``) it runs three jobs:

  * **pre_open** (09:25 ET): refresh overnight bars, recompute the market regime,
    re-enforce the Null wall, pick the active strategy set for the regime, and
    reset the day's risk counters.
  * **tick** (09:30-15:55 ET, every ``tick_interval_minutes``): for each active
    strategy, generate signals from the latest bars, push each fresh signal
    through the :class:`RiskManager`, submit approved orders via the
    :class:`OrderExecutor`, and reconcile open orders. Checks the kill switch
    first — if active, all trading halts until a manual reset.
  * **pre_close** (15:55 ET): reconcile all open orders and write the EOD summary.

The schedule itself (:meth:`planned_schedule`) is a pure, calendar-exact function
— holidays return an empty schedule, early-close days shorten the tick grid — so
it can be asserted deterministically in tests. :meth:`simulate_day` walks that
schedule with an injected clock to exercise a whole trading day without real time.

Only orchestration lives here; every trading *decision* is made by the pure-Python
risk/strategy/regime layers. No LLM/AI touches the decision path.
"""
from __future__ import annotations

import datetime as _dt
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from src.data.pipeline import HistoricalDataFetcher
from src.execution.executor import OrderExecutor
from src.regime.detector import RegimeDetector, RegimeResult
from src.risk.kill_switch import KillSwitch
from src.risk.manager import PortfolioState, RiskManager
from src.storage.db import (
    DailyPnL,
    SignalRecord,
    get_sessionmaker,
    init_db,
    make_engine,
)
from src.strategies.base import Strategy
from src.strategies.factory import build_strategies
from src.strategies.pairs_trading import PairsTrading
from src.utils.config import (
    OrchestratorConfig,
    RiskConfig,
    load_orchestrator_config,
    load_risk_config,
    load_strategies_config,
)
from src.utils.logger import get_logger
from src.validation.null_wall import NullWall, ValidationError

log = get_logger("orchestrator.runner")

_NYSE = mcal.get_calendar("NYSE")

PRE_OPEN = "pre_open"
TICK = "tick"
PRE_CLOSE = "pre_close"
# Minutes before the open/close that pre_open / pre_close fire.
PRE_OPEN_OFFSET_MIN = 5
PRE_CLOSE_OFFSET_MIN = 5


class Orchestrator:
    """Drives the full fetch → regime → risk → execute loop on a schedule."""

    def __init__(
        self,
        client,
        engine=None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        strategy_configs=None,
        strategies: Optional[dict[str, Strategy]] = None,
        null_wall: Optional[NullWall] = None,
        regime_detector: Optional[RegimeDetector] = None,
        kill_switch: Optional[KillSwitch] = None,
        executor: Optional[OrderExecutor] = None,
    ) -> None:
        self.config = orchestrator_config or load_orchestrator_config()
        self.tz = ZoneInfo(self.config.timezone)
        self.engine = engine or make_engine()
        init_db(self.engine)
        self._Session = get_sessionmaker(self.engine)

        self.client = client
        self.strategy_configs = (
            list(strategy_configs)
            if strategy_configs is not None
            else load_strategies_config().strategies
        )
        self.strategies = (
            strategies
            if strategies is not None
            else build_strategies(self.strategy_configs)
        )
        self.risk = RiskManager(risk_config or load_risk_config())
        self.kill_switch = kill_switch or KillSwitch(client=client, engine=self.engine)
        self.executor = executor or OrderExecutor(
            client,
            engine=self.engine,
            kill_switch=self.kill_switch,
            strategies=self.strategies,
        )
        self.null_wall = null_wall or NullWall()
        self.regime = regime_detector or RegimeDetector(
            self.config.benchmark_symbol, engine=self.engine
        )
        self.fetcher = HistoricalDataFetcher(client=client, engine=self.engine)
        self.benchmark = self.config.benchmark_symbol

        # Live state.
        self._submitted_ids: set[str] = set()
        self._current_regime: Optional[RegimeResult] = None
        self._active: dict[str, Strategy] = {}
        self._halted = False
        self._day_trades = 0
        self._starting_equity: Optional[float] = None
        self._peak_equity: Optional[float] = None
        self._pre_close_done_date: Optional[_dt.date] = None
        self._scheduler = None

    # --- startup gate -----------------------------------------------------

    def enforce_null_wall(self, now: Optional[_dt.date] = None) -> dict:
        """Run the Null wall; raises ``ValidationError`` if any strategy fails."""
        return self.null_wall.enforce(self.strategy_configs, now=now)

    def startup(self, now: Optional[_dt.date] = None) -> None:
        """Startup gate: refuse to run unless every strategy passes the Null wall."""
        log.info("orchestrator_startup", strategies=sorted(self.strategies))
        self.enforce_null_wall(now=now)  # raises ValidationError -> no start
        log.info("orchestrator_startup_ok")

    # --- calendar / schedule ---------------------------------------------

    def session_bounds(
        self, day: _dt.date
    ) -> Optional[tuple[_dt.datetime, _dt.datetime]]:
        """Return (open, close) in the configured tz for ``day``, or None if closed."""
        sched = _NYSE.schedule(start_date=day.isoformat(), end_date=day.isoformat())
        if sched.empty:
            return None
        row = sched.iloc[0]
        open_et = row["market_open"].to_pydatetime().astimezone(self.tz)
        close_et = row["market_close"].to_pydatetime().astimezone(self.tz)
        return open_et, close_et

    def planned_schedule(self, day: _dt.date) -> list[tuple[_dt.datetime, str]]:
        """Exact, calendar-aware job schedule for ``day`` (empty on non-trading days).

        * one ``pre_open`` at (open - 5 min),
        * ``tick`` every ``tick_interval_minutes`` from the open through
          (close - 5 min) inclusive,
        * one ``pre_close`` at (close - 5 min).
        Early-close days naturally yield a shorter tick grid.
        """
        bounds = self.session_bounds(day)
        if bounds is None:
            return []
        open_et, close_et = bounds
        step = _dt.timedelta(minutes=self.config.tick_interval_minutes)
        last_tick = close_et - _dt.timedelta(minutes=PRE_CLOSE_OFFSET_MIN)

        plan: list[tuple[_dt.datetime, str]] = [
            (open_et - _dt.timedelta(minutes=PRE_OPEN_OFFSET_MIN), PRE_OPEN)
        ]
        t = open_et
        while t <= last_tick:
            plan.append((t, TICK))
            t += step
        plan.append((last_tick, PRE_CLOSE))
        plan.sort(key=lambda it: (it[0], 0 if it[1] == TICK else 1))
        return plan

    # --- data / state helpers --------------------------------------------

    def _all_symbols(self) -> list[str]:
        syms: set[str] = {self.benchmark}
        for strat in self.strategies.values():
            syms.update(strat.symbols)
        return sorted(syms)

    def _refresh_bars(self, now: _dt.datetime) -> None:
        """Fetch recent daily bars for all symbols (no-op if no client / offline)."""
        if self.client is None:
            return
        end = now.astimezone(_dt.timezone.utc)
        start = end - _dt.timedelta(days=10)
        try:
            self.fetcher.fetch_and_store(self._all_symbols(), "1Day", start, end)
        except Exception as exc:  # never let a data hiccup abort the session
            log.error("refresh_bars_failed", error=str(exc))

    def _bars_for(self, strat: Strategy) -> Optional[pd.DataFrame]:
        """Assemble the input frame a strategy expects from stored bars."""
        syms = tuple(strat.symbols)
        if len(syms) == 1:
            bars = self.fetcher.get_all_bars(syms[0])
            return bars if len(bars) else None
        if len(syms) == 2 and isinstance(strat, PairsTrading):
            a = self.fetcher.get_all_bars(syms[0])
            b = self.fetcher.get_all_bars(syms[1])
            if len(a) == 0 or len(b) == 0:
                return None
            frame = PairsTrading.build_pair_frame(a, b)
            return frame if len(frame) else None
        return None

    @staticmethod
    def _latest_prices(strat: Strategy, bars: pd.DataFrame) -> dict[str, float]:
        syms = tuple(strat.symbols)
        if len(syms) == 2 and "a_close" in bars.columns:
            return {
                syms[0]: float(bars["a_close"].iloc[-1]),
                syms[1]: float(bars["b_close"].iloc[-1]),
            }
        return {syms[0]: float(bars["close"].iloc[-1])}

    def _account_equity(self) -> float:
        acct = self.client.get_account()
        return float(getattr(acct, "equity", 0.0) or 0.0)

    def _open_positions(self) -> dict[str, float]:
        getter = getattr(self.client, "get_all_positions", None) or getattr(
            self.client, "get_positions", None
        )
        positions: dict[str, float] = {}
        if getter is None:
            return positions
        try:
            for pos in getter() or []:
                symbol = getattr(pos, "symbol", None) or (
                    pos.get("symbol") if isinstance(pos, dict) else None
                )
                qty = getattr(pos, "qty", None)
                if qty is None and isinstance(pos, dict):
                    qty = pos.get("qty")
                if symbol is not None and qty is not None:
                    positions[symbol] = float(qty)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("open_positions_read_failed", error=str(exc))
        return positions

    def _portfolio_state(
        self, now: _dt.datetime, prices: dict[str, float]
    ) -> PortfolioState:
        equity = self._account_equity()
        if self._starting_equity is None:
            self._starting_equity = equity
        self._peak_equity = max(self._peak_equity or equity, equity)
        return PortfolioState(
            equity=equity,
            starting_equity=self._starting_equity,
            peak_equity=self._peak_equity,
            positions=self._open_positions(),
            prices=prices,
            now=now,
        )

    # --- jobs -------------------------------------------------------------

    def pre_open(self, now: _dt.datetime) -> Optional[RegimeResult]:
        """Refresh data, recompute regime, re-enforce the wall, reset counters."""
        log.info("pre_open_start", now=now.isoformat())
        self._halted = False
        self._day_trades = 0
        self._submitted_ids.clear()

        self._refresh_bars(now)

        bars = self.fetcher.get_all_bars(self.benchmark)
        if len(bars) == 0:
            log.error("pre_open_no_benchmark_bars", symbol=self.benchmark)
            self._active = {}
            return None
        result = self.regime.detect(bars)
        self.regime.persist(result, now=now)
        self._current_regime = result

        # Re-enforce the Null wall daily (catches newly-stale validations).
        try:
            self.enforce_null_wall(now=now.date())
        except ValidationError as exc:
            log.critical("pre_open_null_wall_failed", error=str(exc))
            self._active = {}
            return result

        self._active = RegimeDetector.active_strategies(
            self.strategies, result.combined
        )
        self._reset_daily_counters(now)
        log.info(
            "pre_open_done",
            regime=result.combined.value,
            active_strategies=sorted(self._active),
            realized_vol=round(result.realized_vol, 4),
            adx=round(result.adx, 2),
        )
        return result

    def tick(self, now: _dt.datetime) -> dict:
        """Run one scan/risk/execute cycle across the active strategies."""
        if self.kill_switch.is_active():
            self._halt("kill_switch_active")
            return {"halted": True, "submitted": 0}

        submitted = 0
        for name, strat in self._active.items():
            bars = self._bars_for(strat)
            if bars is None or len(bars) < max(strat.lookback_bars, 1):
                continue
            signals = strat.generate_signals(bars)
            latest_ts = pd.Timestamp(bars.index[-1])
            prices = self._latest_prices(strat, bars)
            for sig in signals:
                if sig.as_of is None or pd.Timestamp(sig.as_of) != latest_ts:
                    continue  # act only on the freshest bar's decision
                cid = OrderExecutor.build_client_order_id(sig)
                if cid in self._submitted_ids:
                    continue  # already acted on this bar's signal
                self._record_signal(name, sig)
                state = self._portfolio_state(now, prices)
                decision = self.risk.check_order(sig, state)
                if decision.kill_switch_triggered:
                    self.kill_switch.trigger(decision.reason, "risk_manager")
                    self._halt("risk_breach")
                    return {"halted": True, "submitted": submitted}
                order_id = self.executor.submit(sig, decision)
                if order_id is not None:
                    self._submitted_ids.add(cid)
                    submitted += 1
                    self._day_trades += 1

        self.executor.reconcile()
        return {"halted": False, "submitted": submitted}

    def pre_close(self, now: _dt.datetime) -> dict:
        """Reconcile open orders and write the EOD summary."""
        recon = self.executor.reconcile()
        summary = self._finalize_daily(now)
        self._pre_close_done_date = now.astimezone(self.tz).date()
        log.info(
            "pre_close_eod_summary",
            now=now.isoformat(),
            trades=self._day_trades,
            regime=self._current_regime.combined.value if self._current_regime else None,
            active_strategies=sorted(self._active),
            reconciled=recon,
            **summary,
        )
        return {"trades": self._day_trades, **summary}

    def _record_signal(self, strategy_name: str, sig) -> None:
        """Persist a signal to the audit trail (``signals`` table)."""
        with self._Session() as session:
            session.add(
                SignalRecord(
                    strategy=strategy_name,
                    symbol=sig.symbol,
                    side=sig.side.value,
                    quantity_hint=sig.quantity_hint,
                    confidence=sig.confidence,
                    entry_reason=sig.entry_reason,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    bar_timestamp=pd.Timestamp(sig.as_of).to_pydatetime(),
                )
            )
            session.commit()

    # --- daily P&L counters ----------------------------------------------

    def _daily_date(self, now: _dt.datetime) -> _dt.datetime:
        et = now.astimezone(self.tz)
        return _dt.datetime(et.year, et.month, et.day)

    def _reset_daily_counters(self, now: _dt.datetime) -> None:
        """Establish start-of-day equity (the basis for the daily-loss cap)."""
        equity = self._account_equity()
        self._starting_equity = equity
        self._peak_equity = max(self._peak_equity or equity, equity)
        day = self._daily_date(now)
        with self._Session() as session:
            from sqlalchemy import select

            row = session.execute(
                select(DailyPnL).where(DailyPnL.date == day)
            ).scalars().first()
            if row is None:
                session.add(DailyPnL(date=day, starting_equity=equity, realized_pnl=0.0))
            else:
                row.starting_equity = equity
            session.commit()
        log.info("daily_counters_reset", date=day.date().isoformat(), starting_equity=equity)

    def _finalize_daily(self, now: _dt.datetime) -> dict:
        equity = self._account_equity()
        day = self._daily_date(now)
        start = self._starting_equity if self._starting_equity is not None else equity
        realized = equity - start
        with self._Session() as session:
            from sqlalchemy import select

            row = session.execute(
                select(DailyPnL).where(DailyPnL.date == day)
            ).scalars().first()
            if row is None:
                row = DailyPnL(date=day, starting_equity=start)
                session.add(row)
            row.ending_equity = equity
            row.realized_pnl = realized
            session.commit()
        return {"ending_equity": equity, "starting_equity": start, "realized_pnl": realized}

    # --- halt -------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        if not self._halted:
            log.critical("orchestrator_halted", reason=reason)
        self._halted = True
        sch = self._scheduler
        if sch is not None and getattr(sch, "running", False):
            try:
                sch.pause()
            except Exception as exc:  # pragma: no cover - defensive
                log.error("scheduler_pause_failed", error=str(exc))

    # --- simulation (deterministic, no real time) -------------------------

    def simulate_day(
        self,
        day: _dt.date,
        on_step: Optional[Callable[[_dt.datetime, str], None]] = None,
    ) -> list[dict]:
        """Walk ``planned_schedule(day)`` with an injected clock.

        ``on_step(now, job)`` (if given) runs *before* each job — tests use it to,
        e.g., trip the kill switch at a chosen time. Returns one result dict per
        executed step. Once halted, remaining ticks are recorded as halted and
        submit nothing (mirroring the live scheduler pausing).
        """
        results: list[dict] = []
        for now, job in self.planned_schedule(day):
            if on_step is not None:
                on_step(now, job)
            if job == PRE_OPEN:
                regime = self.pre_open(now)
                results.append(
                    {"now": now, "job": PRE_OPEN, "regime": regime.combined.value if regime else None}
                )
            elif job == TICK:
                res = self.tick(now)
                results.append({"now": now, "job": TICK, **res})
            elif job == PRE_CLOSE:
                res = self.pre_close(now)
                results.append({"now": now, "job": PRE_CLOSE, **res})
        return results

    # --- live scheduler ---------------------------------------------------

    def _now(self) -> _dt.datetime:
        return _dt.datetime.now(self.tz)

    def _run_if_planned(self, job: str) -> None:
        """Run ``job`` only if the calendar says it is a trading moment."""
        from src.orchestrator.lockfile import touch_lockfile

        now = self._now()
        bounds = self.session_bounds(now.date())
        if bounds is None:
            return  # holiday / weekend
        touch_lockfile()  # heartbeat
        open_et, close_et = bounds
        last_tick = close_et - _dt.timedelta(minutes=PRE_CLOSE_OFFSET_MIN)

        if job == PRE_OPEN:
            self.pre_open(now)
        elif job == TICK:
            if open_et <= now <= last_tick:
                self.tick(now)
                # Fire pre_close from the final tick too (covers early-close days).
                if now >= last_tick and self._pre_close_done_date != now.date():
                    self.pre_close(now)
        elif job == PRE_CLOSE:
            if self._pre_close_done_date != now.date():
                self.pre_close(now)

    def build_scheduler(self, scheduler=None):
        """Register the three cron jobs. Does not start the scheduler."""
        from apscheduler.triggers.cron import CronTrigger

        if scheduler is None:
            from apscheduler.schedulers.background import BackgroundScheduler

            scheduler = BackgroundScheduler(timezone=self.tz)

        n = self.config.tick_interval_minutes
        scheduler.add_job(
            lambda: self._run_if_planned(PRE_OPEN),
            CronTrigger(day_of_week="mon-fri", hour=9, minute=25, timezone=self.tz),
            id=PRE_OPEN,
            replace_existing=True,
        )
        scheduler.add_job(
            lambda: self._run_if_planned(TICK),
            CronTrigger(day_of_week="mon-fri", hour="9-15", minute=f"*/{n}", timezone=self.tz),
            id=TICK,
            replace_existing=True,
        )
        scheduler.add_job(
            lambda: self._run_if_planned(PRE_CLOSE),
            CronTrigger(day_of_week="mon-fri", hour=15, minute=55, timezone=self.tz),
            id=PRE_CLOSE,
            replace_existing=True,
        )
        self._scheduler = scheduler
        return scheduler
