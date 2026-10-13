"""Launch the read-only monitoring dashboard.

    python -m src.scripts.run_dashboard
    python -m src.scripts.run_dashboard --host 0.0.0.0 --port 5000

Serves the Flask app from :mod:`src.dashboard.app` at ``localhost:5000`` by
default. It only reads SQLite and (cached) Alpaca account data — it never writes
and it submits no orders, so it is safe to run alongside (or without) the bot.
"""
from __future__ import annotations

import argparse
import sys

from src.dashboard.app import create_app
from src.utils.logger import get_logger

log = get_logger("scripts.run_dashboard")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Sentinel monitoring dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="bind port (default 5000)")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args(argv)

    app = create_app()
    log.info("dashboard_start", host=args.host, port=args.port)
    print(f"[dashboard] serving on http://{args.host}:{args.port} (read-only)")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # clean non-zero on failure
        log.error("run_dashboard_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
