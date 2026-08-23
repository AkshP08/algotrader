"""Structured logging for Sentinel.

structlog is configured to emit:
  * JSON lines to ``logs/{YYYY-MM-DD}.log`` (machine-readable audit trail), and
  * colored, human-readable output to the console.

Log-level convention used across the project:
  INFO     normal operations (orders, fills, data fetches)
  WARNING  risk events (limit approached/breached, data gaps)
  ERROR    failures (API errors, rejects)
  CRITICAL kill-switch activations

Secrets are never logged. Callers must pass metadata only; the Alpaca client
scrubs keys before they ever reach a log call.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
from pathlib import Path

import structlog

_CONFIGURED = False


def _log_dir() -> Path:
    return Path(os.getenv("LOG_DIR", "logs"))


def _log_file_path() -> Path:
    # Date-stamped file. Uses UTC so the audit trail is unambiguous.
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    return _log_dir() / f"{today}.log"


def configure_logging(level: str | None = None, force: bool = False) -> None:
    """Configure structlog + stdlib logging once (idempotent).

    JSON records are written to the date-stamped log file; a colored,
    developer-friendly renderer goes to the console.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_no = getattr(logging, level_name, logging.INFO)

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Shared processors run before the final per-handler renderer.
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Console: colored key/value. File: JSON lines.
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(_log_file_path(), encoding="utf-8")
    file_handler.setFormatter(json_formatter)

    root = logging.getLogger()
    # Clear existing handlers so re-configuration (tests) doesn't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(level_no)

    _CONFIGURED = True


def get_logger(name: str = "sentinel") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
