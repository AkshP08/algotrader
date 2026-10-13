"""Bundled synthetic dataset for the dashboard's ``--demo`` mode.

This is the **one** place in Sentinel where synthetic data is allowed, and only
because it is *advertising, not analysis*: it lets a reader click through the UI
without Alpaca keys or a seeded database. Every value here is fabricated and
clearly labeled — every payload that uses it carries ``"demo": true`` and the
page shows a loud banner. None of this is ever presented as real, and none of it
touches the trading decision path.

The data is fully deterministic (hard-coded literals / closed-form curves, no RNG
or wall-clock) so screenshots are reproducible across runs.
"""
from __future__ import annotations

# A fixed "as of" so demo output never drifts between runs.
DEMO_AS_OF = "2026-10-12T13:25:00+00:00"
DEMO_GENERATED_AT = "2026-10-12T20:00:00+00:00"

_START_EQUITY = 100_000.0


def _equity_curve(days: int = 30) -> list[dict]:
    """A gentle, deterministic 30-day equity curve (closed-form, no randomness)."""
    curve: list[dict] = []
    equity = _START_EQUITY
    for i in range(days):
        # Small deterministic daily move: a slow drift plus a shallow oscillation.
        drift = 45.0
        wobble = 180.0 * (((i * 7) % 11) - 5) / 5.0  # ranges ~[-180, +180]
        start = equity
        equity = round(equity + drift + wobble, 2)
        day = 10 + i  # 2026-09-10 .. 2026-10-09 range (illustrative)
        month = 9 if day <= 30 else 10
        d = day if day <= 30 else day - 30
        curve.append(
            {
                "date": f"2026-{month:02d}-{d:02d}",
                "starting_equity": round(start, 2),
                "ending_equity": equity,
                "realized_pnl": round(equity - start, 2),
            }
        )
    return curve


_ORDERS = [
    ("2026-10-12T13:30:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "buy", "market", 18, "filled"),
    ("2026-10-12T13:30:00+00:00", "pairs_trading_xle_xop", "XLE", "sell", "market", 55, "filled"),
    ("2026-10-12T13:30:00+00:00", "pairs_trading_xle_xop", "XOP", "buy", "market", 90, "filled"),
    ("2026-10-12T14:05:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "exit", "close", 18, "filled"),
    ("2026-10-12T14:35:00+00:00", "pairs_trading_xle_xop", "XLE", "exit", "close", 55, "filled"),
    ("2026-10-12T14:35:00+00:00", "pairs_trading_xle_xop", "XOP", "exit", "close", 90, "filled"),
    ("2026-10-12T15:00:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "buy", "market", 17, "filled"),
    ("2026-10-12T15:40:00+00:00", "pairs_trading_xle_xop", "XLE", "sell", "market", 52, "filled"),
    ("2026-10-12T15:40:00+00:00", "pairs_trading_xle_xop", "XOP", "buy", "market", 88, "filled"),
    ("2026-10-12T16:10:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "exit", "close", 17, "filled"),
    ("2026-10-12T17:15:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "buy", "market", 16, "filled"),
    ("2026-10-12T18:00:00+00:00", "pairs_trading_xle_xop", "XLE", "exit", "close", 52, "filled"),
    ("2026-10-12T18:00:00+00:00", "pairs_trading_xle_xop", "XOP", "exit", "close", 88, "filled"),
    ("2026-10-12T18:45:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "exit", "close", 16, "filled"),
    ("2026-10-12T19:30:00+00:00", "rsi_mean_reversion_qqq", "QQQ", "buy", "market", 15, "new"),
]

_FILLS = [
    ("2026-10-12T13:30:02+00:00", "QQQ", "buy", 18, 548.20, 0.0),
    ("2026-10-12T13:30:02+00:00", "XLE", "sell", 55, 92.10, 0.0),
    ("2026-10-12T13:30:02+00:00", "XOP", "buy", 90, 141.65, 0.0),
    ("2026-10-12T14:05:04+00:00", "QQQ", "exit", 18, 551.90, 66.60),
    ("2026-10-12T14:35:03+00:00", "XLE", "exit", 55, 91.40, 38.50),
    ("2026-10-12T14:35:03+00:00", "XOP", "exit", 90, 142.30, 58.50),
    ("2026-10-12T15:00:05+00:00", "QQQ", "buy", 17, 549.75, 0.0),
    ("2026-10-12T16:10:02+00:00", "QQQ", "exit", 17, 548.10, -28.05),
    ("2026-10-12T15:40:04+00:00", "XLE", "sell", 52, 92.55, 0.0),
    ("2026-10-12T18:00:03+00:00", "XOP", "exit", 88, 141.10, -22.00),
]


def _strategies(current_regime: str) -> list[dict]:
    rows = [
        {
            "name": "ma_crossover_spy",
            "type": "ma_crossover",
            "allowed": ["HIGH_VOL_TRENDING", "LOW_VOL_TRENDING"],
            "last": None,
        },
        {
            "name": "rsi_mean_reversion_qqq",
            "type": "rsi_mean_reversion",
            "allowed": ["LOW_VOL_RANGING"],
            "last": {"side": "BUY", "symbol": "QQQ", "at": "2026-10-12T19:30:00+00:00"},
        },
        {
            "name": "pairs_trading_xle_xop",
            "type": "pairs_trading",
            "allowed": ["LOW_VOL_RANGING", "LOW_VOL_TRENDING"],
            "last": {"side": "EXIT", "symbol": "XOP", "at": "2026-10-12T18:00:00+00:00"},
        },
    ]
    return [
        {
            "name": r["name"],
            "type": r["type"],
            "enabled": True,
            "allowed_regimes": r["allowed"],
            "regime_allowed": current_regime in r["allowed"],
            "null_status": "SURVIVED",
            "last_signal": r["last"],
        }
        for r in rows
    ]


def demo_state(kill_switch: bool = False) -> dict:
    """Return a complete, clearly-synthetic dashboard state (``demo: True``)."""
    current_regime = "LOW_VOL_RANGING"
    curve = _equity_curve()
    todays_pnl = round(sum(f[5] for f in _FILLS), 2)
    return {
        "demo": True,
        "generated_at": DEMO_GENERATED_AT,
        "regime": {
            "combined": current_regime,
            "vol_regime": "LOW",
            "trend_regime": "RANGING",
            "realized_vol": 0.112,
            "adx": 17.4,
            "as_of": DEMO_AS_OF,
        },
        "kill_switch_active": bool(kill_switch),
        "account": {
            "equity": curve[-1]["ending_equity"],
            "buying_power": round(curve[-1]["ending_equity"] * 2, 2),
            "status": "ACTIVE (DEMO)",
        },
        "strategies": _strategies(current_regime),
        "orders": [
            {
                "created_at": o[0],
                "strategy": o[1],
                "symbol": o[2],
                "side": o[3],
                "order_type": o[4],
                "quantity": o[5],
                "status": o[6],
            }
            for o in reversed(_ORDERS)
        ],
        "fills": [
            {
                "filled_at": f[0],
                "symbol": f[1],
                "side": f[2],
                "filled_qty": f[3],
                "filled_avg_price": f[4],
                "pnl": f[5],
            }
            for f in reversed(_FILLS)
        ],
        "todays_pnl": todays_pnl,
        "equity_curve": curve,
    }
