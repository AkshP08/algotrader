"""Read-only monitoring dashboard (Flask).

A single dark page (plus a JSON twin at ``/api/state``) that reads *only* from
SQLite — it never writes. To make that guarantee enforceable and testable, every
request uses a session whose ``before_flush`` raises, so an accidental write would
blow up rather than persist.

Routes:
  * ``GET /``         — the HTML dashboard (auto-refreshes every 60s).
  * ``GET /api/state``— the same data as JSON.
  * ``GET /healthz``  — 200 if the orchestrator lockfile exists and is fresh
                        (< 15 min old), else 503.

Account equity / buying power are pulled live from Alpaca through a 60-second
cache; if the broker is unreachable the page still renders (account shows
"unavailable"). The visual language is intentionally the same family as Null —
dark, monospace numbers — so the two tools read as siblings.
"""
from __future__ import annotations

import datetime as _dt
import html
import time
from typing import Callable, Optional

from flask import Flask, Response, jsonify, request
from sqlalchemy import desc, event, select

from src.orchestrator.lockfile import lockfile_status
from src.risk.kill_switch import KillSwitch
from src.storage.db import (
    DailyPnL,
    FillRecord,
    OrderRecord,
    RegimeState,
    SignalRecord,
    get_sessionmaker,
    make_engine,
)
from src.strategies.factory import build_strategies, params_hash
from src.utils.config import load_strategies_config
from src.utils.logger import get_logger
from src.validation.null_wall import NullWall

log = get_logger("dashboard.app")

ACCOUNT_CACHE_TTL = 60  # seconds


def _default_client_factory():
    """Construct a paper AlpacaClient, or return None if it cannot be built."""
    try:
        from src.broker.alpaca_client import AlpacaClient

        return AlpacaClient()
    except Exception as exc:  # missing creds / non-paper URL / offline
        log.warning("dashboard_no_client", error=str(exc))
        return None


def _readonly_guard(session, flush_context, instances):  # pragma: no cover - defensive
    if session.new or session.dirty or session.deleted:
        raise RuntimeError("dashboard is read-only; refusing to write to the database")


def create_app(
    engine=None,
    client_factory: Optional[Callable[[], object]] = None,
    validation_dir: Optional[str] = None,
    demo: bool = False,
) -> Flask:
    app = Flask(__name__)
    app.config["DEMO"] = demo
    app.config["ENGINE"] = engine or make_engine()
    Session = get_sessionmaker(app.config["ENGINE"])
    # Enforce read-only: any attempted write flush raises.
    event.listen(Session, "before_flush", _readonly_guard)
    app.config["SESSION"] = Session
    app.config["CLIENT_FACTORY"] = client_factory or _default_client_factory
    app.config["NULL_WALL"] = NullWall(validation_dir)
    app.config["ACCOUNT_CACHE"] = {"ts": 0.0, "data": None}

    # ------------------------------------------------------------------
    # Data access (SELECT-only)
    # ------------------------------------------------------------------

    def _cached_account() -> Optional[dict]:
        cache = app.config["ACCOUNT_CACHE"]
        now = time.time()
        if cache["data"] is not None and (now - cache["ts"]) < ACCOUNT_CACHE_TTL:
            return cache["data"]
        data = None
        try:
            client = app.config["CLIENT_FACTORY"]()
            if client is not None:
                acct = client.get_account()
                data = {
                    "equity": float(getattr(acct, "equity", 0.0) or 0.0),
                    "buying_power": float(getattr(acct, "buying_power", 0.0) or 0.0),
                    "status": str(getattr(acct, "status", "?")),
                }
        except Exception as exc:
            log.warning("dashboard_account_unavailable", error=str(exc))
            data = None
        cache["ts"] = now
        cache["data"] = data
        return data

    def _fills_with_pnl(fills: list[FillRecord]) -> list[dict]:
        """Walk fills chronologically, estimating realized P&L (avg-cost, long-only).

        An opening buy has 0 realized P&L; a sell/exit realizes (price - avg_cost)
        on the closed quantity. This is an estimate for display, not accounting.
        """
        book: dict[str, dict] = {}
        rows: list[dict] = []
        for f in sorted(fills, key=lambda r: (r.filled_at or _dt.datetime.min, r.id)):
            pos = book.setdefault(f.symbol, {"qty": 0.0, "avg": 0.0})
            qty = float(f.filled_qty or 0.0)
            price = float(f.filled_avg_price or 0.0)
            pnl = None
            if f.side == "buy":
                total = pos["qty"] + qty
                pos["avg"] = (
                    (pos["qty"] * pos["avg"] + qty * price) / total if total else price
                )
                pos["qty"] = total
                pnl = 0.0
            else:  # sell / exit
                closed = min(qty, pos["qty"]) if pos["qty"] > 0 else qty
                pnl = (price - pos["avg"]) * closed
                pos["qty"] = max(0.0, pos["qty"] - qty)
            rows.append(
                {
                    "filled_at": f.filled_at.isoformat() if f.filled_at else None,
                    "symbol": f.symbol,
                    "side": f.side,
                    "filled_qty": qty,
                    "filled_avg_price": price,
                    "pnl": round(pnl, 2) if pnl is not None else None,
                }
            )
        return rows

    def build_state() -> dict:
        # Demo mode: serve bundled synthetic data, never the DB/broker. Clearly
        # flagged with demo=True; a ?kill=1 query renders the kill-switch view.
        if app.config["DEMO"]:
            from src.dashboard.demo_data import demo_state

            kill = request.args.get("kill") in ("1", "true", "yes")
            return demo_state(kill_switch=kill)

        Session = app.config["SESSION"]
        null_wall: NullWall = app.config["NULL_WALL"]
        cfgs = load_strategies_config().strategies
        strategies = build_strategies(cfgs)

        with Session() as session:
            regime_row = (
                session.execute(select(RegimeState).order_by(desc(RegimeState.id)).limit(1))
                .scalars()
                .first()
            )
            regime = (
                {
                    "combined": regime_row.combined,
                    "vol_regime": regime_row.vol_regime,
                    "trend_regime": regime_row.trend_regime,
                    "realized_vol": regime_row.realized_vol,
                    "adx": regime_row.adx,
                    "as_of": regime_row.timestamp.isoformat() if regime_row.timestamp else None,
                }
                if regime_row
                else None
            )
            current_regime = regime_row.combined if regime_row else None

            orders = [
                {
                    "created_at": o.created_at.isoformat() if o.created_at else None,
                    "strategy": o.strategy,
                    "symbol": o.symbol,
                    "side": o.side,
                    "order_type": o.order_type,
                    "quantity": o.quantity,
                    "status": o.status,
                }
                for o in session.execute(
                    select(OrderRecord).order_by(desc(OrderRecord.id)).limit(20)
                ).scalars().all()
            ]

            fills_recent = (
                session.execute(select(FillRecord).order_by(desc(FillRecord.id)).limit(200))
                .scalars()
                .all()
            )
            fills_pnl = _fills_with_pnl(list(fills_recent))
            fills = list(reversed(fills_pnl))[:20]

            # Strategy table: regime-allowed + Null status + last signal.
            strat_rows = []
            for cfg in cfgs:
                strat = strategies.get(cfg.name)
                allowed = (
                    sorted(r.value for r in strat.allowed_regimes) if strat else []
                )
                regime_allowed = current_regime in allowed if current_regime else None
                try:
                    record = null_wall.load_record(cfg.name)
                    expected = params_hash(cfg.params)
                    null_status = record.verdict
                    if record.params_hash != expected:
                        null_status = "DRIFTED"
                except FileNotFoundError:
                    null_status = "MISSING"
                except Exception:
                    null_status = "INVALID"
                last_sig = (
                    session.execute(
                        select(SignalRecord)
                        .where(SignalRecord.strategy == cfg.name)
                        .order_by(desc(SignalRecord.id))
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )
                strat_rows.append(
                    {
                        "name": cfg.name,
                        "type": cfg.type,
                        "enabled": cfg.enabled,
                        "allowed_regimes": allowed,
                        "regime_allowed": regime_allowed,
                        "null_status": null_status,
                        "last_signal": (
                            {
                                "side": last_sig.side,
                                "symbol": last_sig.symbol,
                                "at": last_sig.created_at.isoformat()
                                if last_sig.created_at
                                else None,
                            }
                            if last_sig
                            else None
                        ),
                    }
                )

            # 60-day equity curve from daily_pnl.
            pnl_rows = (
                session.execute(select(DailyPnL).order_by(desc(DailyPnL.date)).limit(60))
                .scalars()
                .all()
            )
            equity_curve = [
                {
                    "date": r.date.date().isoformat() if r.date else None,
                    "ending_equity": r.ending_equity,
                    "starting_equity": r.starting_equity,
                    "realized_pnl": r.realized_pnl,
                }
                for r in reversed(pnl_rows)
            ]

        # Today's running P&L: realized from today's fills (local date).
        today = _dt.date.today().isoformat()
        todays_pnl = round(
            sum(
                f["pnl"] or 0.0
                for f in fills_pnl
                if f["filled_at"] and f["filled_at"][:10] == today
            ),
            2,
        )

        kill_active = KillSwitch(engine=app.config["ENGINE"]).is_active()

        return {
            "demo": False,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "regime": regime,
            "kill_switch_active": kill_active,
            "account": _cached_account(),
            "strategies": strat_rows,
            "orders": orders,
            "fills": fills,
            "todays_pnl": todays_pnl,
            "equity_curve": equity_curve,
        }

    app.config["BUILD_STATE"] = build_state

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/healthz")
    def healthz():
        status = lockfile_status()
        code = 200 if status["fresh"] else 503
        body = {
            "status": "ok" if status["fresh"] else "unhealthy",
            "lockfile": status,
        }
        return jsonify(body), code

    @app.route("/api/state")
    def api_state():
        return jsonify(build_state())

    @app.route("/")
    def index():
        return Response(_render(build_state()), mimetype="text/html")

    return app


# ----------------------------------------------------------------------
# HTML rendering (inline CSS, dark theme, mono numbers, responsive @390px)
# ----------------------------------------------------------------------

_CSS = """
:root{--bg:#0b0e14;--panel:#121722;--line:#1f2633;--txt:#d7dde8;--muted:#7c879b;
--green:#3fb950;--red:#f85149;--amber:#d29922;--accent:#58a6ff;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--txt);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
font-size:14px;line-height:1.5;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-variant-numeric:tabular-nums;}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px 48px;}
header{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px;
border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:20px;}
h1{font-size:16px;margin:0;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.gen{color:var(--muted);font-size:12px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:20px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;}
.card .label{color:var(--muted);font-size:11px;letter-spacing:.1em;text-transform:uppercase;}
.card .big{font-size:30px;margin-top:6px;overflow-wrap:anywhere;}
.card .big.regime{font-size:21px;letter-spacing:0;line-height:1.2;}
.card .sub{color:var(--muted);font-size:12px;margin-top:4px;}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;}
.pill.green{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.4);}
.pill.red{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.4);}
.pill.amber{background:rgba(210,153,34,.15);color:var(--amber);border:1px solid rgba(210,153,34,.4);}
.pill.muted{background:var(--line);color:var(--muted);}
section{margin-bottom:26px;}
section h2{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
border-bottom:1px solid var(--line);padding-bottom:8px;margin:0 0 10px;}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:520px;}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.06em;text-transform:uppercase;}
td.num{text-align:right;}
.pos{color:var(--green);} .neg{color:var(--red);}
.empty{color:var(--muted);padding:12px 2px;font-style:italic;}
.demo-banner{background:#d29922;color:#111;font-weight:700;text-align:center;
padding:10px 14px;border-radius:8px;margin-bottom:18px;letter-spacing:.02em;
border:1px solid #b5801a;}
.spark{display:flex;align-items:flex-end;gap:2px;height:60px;}
.spark div{flex:1;background:var(--accent);min-width:2px;border-radius:1px 1px 0 0;opacity:.8;}
@media(max-width:390px){.card .big{font-size:24px;}.wrap{padding:14px 10px 36px;}}
"""


def _pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{html.escape(str(text))}</span>'


def _fmt_money(v) -> str:
    if v is None:
        return "—"
    return f"${v:,.2f}"


def _render(state: dict) -> str:
    e = html.escape

    regime = state.get("regime")
    if regime:
        regime_big = e(regime["combined"])
        regime_sub = (
            f'vol {regime["realized_vol"]*100:.1f}% · adx {regime["adx"]:.1f} · '
            f'as of {e(str(regime["as_of"]))}'
        )
    else:
        regime_big = "UNKNOWN"
        regime_sub = "no regime computed yet"

    ks = state.get("kill_switch_active")
    ks_pill = _pill("ACTIVE", "red") if ks else _pill("NORMAL", "green")

    acct = state.get("account")
    if acct:
        equity_big = _fmt_money(acct["equity"])
        bp_sub = f'buying power {_fmt_money(acct["buying_power"])} · {e(acct["status"])}'
    else:
        equity_big = "—"
        bp_sub = "account unavailable"

    tp = state.get("todays_pnl") or 0.0
    tp_cls = "pos" if tp >= 0 else "neg"

    # Strategy rows
    strat_html = []
    for s in state["strategies"]:
        ra = s["regime_allowed"]
        ra_pill = (
            _pill("YES", "green") if ra else (_pill("NO", "muted") if ra is not None else _pill("—", "muted"))
        )
        ns = s["null_status"]
        ns_kind = "green" if ns == "SURVIVED" else ("amber" if ns == "SHAKY" else "red")
        last = s["last_signal"]
        last_txt = f'{e(last["side"])} {e(last["symbol"])}' if last else "—"
        strat_html.append(
            f"<tr><td>{e(s['name'])}</td><td class='mono'>{e(', '.join(s['allowed_regimes']))}</td>"
            f"<td>{ra_pill}</td><td>{_pill(ns, ns_kind)}</td><td>{last_txt}</td></tr>"
        )
    if not strat_html:
        strat_html.append('<tr><td colspan="5" class="empty">no strategies configured</td></tr>')

    # Orders
    order_html = []
    for o in state["orders"]:
        order_html.append(
            f"<tr><td class='mono'>{e(str(o['created_at']))}</td><td>{e(str(o['strategy']))}</td>"
            f"<td>{e(o['symbol'])}</td><td>{e(o['side'])}</td>"
            f"<td class='num mono'>{o['quantity']:g}</td><td>{_pill(o['status'], 'muted')}</td></tr>"
        )
    if not order_html:
        order_html.append('<tr><td colspan="6" class="empty">no orders yet</td></tr>')

    # Fills
    fill_html = []
    for f in state["fills"]:
        pnl = f["pnl"]
        pnl_txt = "—" if pnl is None else f"{pnl:+,.2f}"
        pnl_cls = "" if pnl is None else ("pos" if pnl >= 0 else "neg")
        fill_html.append(
            f"<tr><td class='mono'>{e(str(f['filled_at']))}</td><td>{e(f['symbol'])}</td>"
            f"<td>{e(f['side'])}</td><td class='num mono'>{f['filled_qty']:g}</td>"
            f"<td class='num mono'>{f['filled_avg_price']:,.2f}</td>"
            f"<td class='num mono {pnl_cls}'>{pnl_txt}</td></tr>"
        )
    if not fill_html:
        fill_html.append('<tr><td colspan="6" class="empty">no fills yet</td></tr>')

    # Equity sparkline
    curve = [p for p in state["equity_curve"] if p["ending_equity"] is not None]
    if curve:
        vals = [p["ending_equity"] for p in curve]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        bars = "".join(
            f'<div style="height:{8 + 52*(v-lo)/span:.0f}px" title="{e(p["date"])}: {_fmt_money(v)}"></div>'
            for p, v in zip(curve, vals)
        )
        spark = f'<div class="spark">{bars}</div>'
        spark_sub = f'{len(curve)} day(s) · {_fmt_money(lo)} – {_fmt_money(hi)}'
    else:
        spark = '<div class="empty">no daily P&L history yet</div>'
        spark_sub = ""

    demo_banner = (
        '<div class="demo-banner">⚠ DEMO MODE — synthetic data, not live. '
        "Equity figures are illustrative.</div>"
        if state.get("demo")
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Sentinel — Monitor</title>
<style>{_CSS}</style>
</head><body><div class="wrap">
{demo_banner}
<header>
  <h1>Sentinel · Live Monitor</h1>
  <span class="gen mono">updated {e(state['generated_at'])}</span>
</header>

<div class="grid">
  <div class="card"><div class="label">Market Regime</div>
    <div class="big mono regime">{regime_big}</div><div class="sub">{regime_sub}</div></div>
  <div class="card"><div class="label">Kill Switch</div>
    <div class="big">{ks_pill}</div><div class="sub">auto-halt on risk breach</div></div>
  <div class="card"><div class="label">Account Equity</div>
    <div class="big mono">{equity_big}</div><div class="sub">{bp_sub}</div></div>
  <div class="card"><div class="label">Today's P&amp;L</div>
    <div class="big mono {tp_cls}">{tp:+,.2f}</div><div class="sub">realized from today's fills</div></div>
</div>

<section><h2>Strategies</h2><div class="scroll"><table>
<thead><tr><th>Name</th><th>Regime-allowed in</th><th>Active now</th><th>Null</th><th>Last signal</th></tr></thead>
<tbody>{''.join(strat_html)}</tbody></table></div></section>

<section><h2>Equity Curve (60d)</h2>{spark}<div class="sub mono">{spark_sub}</div></section>

<section><h2>Recent Orders</h2><div class="scroll"><table>
<thead><tr><th>Created</th><th>Strategy</th><th>Symbol</th><th>Side</th><th class="num">Qty</th><th>Status</th></tr></thead>
<tbody>{''.join(order_html)}</tbody></table></div></section>

<section><h2>Recent Fills</h2><div class="scroll"><table>
<thead><tr><th>Filled</th><th>Symbol</th><th>Side</th><th class="num">Qty</th><th class="num">Price</th><th class="num">Est. P&amp;L</th></tr></thead>
<tbody>{''.join(fill_html)}</tbody></table></div></section>

</div></body></html>"""
