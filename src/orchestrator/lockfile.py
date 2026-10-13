"""PID lockfile used as the orchestrator's liveness heartbeat.

``run_bot`` writes its PID here on startup and *touches* the file on every
scheduled job (a heartbeat). The dashboard's ``/healthz`` reads it: a missing
file, or one whose mtime is older than :data:`HEARTBEAT_MAX_AGE_SECONDS`, means
the bot is not running (or has hung) and health is reported unhealthy.

Kept in its own tiny module (no apscheduler / pandas imports) so the dashboard
can depend on it without pulling in the whole runner.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

# 15 minutes: longer than the tick cadence, short enough to notice a dead bot.
HEARTBEAT_MAX_AGE_SECONDS = 15 * 60


def lockfile_path() -> Path:
    return Path(os.getenv("ORCHESTRATOR_LOCKFILE", "data/orchestrator.pid"))


def write_lockfile(pid: Optional[int] = None) -> Path:
    """Write the current (or given) PID to the lockfile, creating parents."""
    path = lockfile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid if pid is not None else os.getpid()), encoding="utf-8")
    return path


def touch_lockfile() -> None:
    """Update the lockfile mtime (heartbeat). No-op if it does not exist."""
    path = lockfile_path()
    if path.exists():
        now = time.time()
        os.utime(path, (now, now))


def remove_lockfile() -> None:
    """Remove the lockfile if present (graceful shutdown)."""
    path = lockfile_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def read_pid() -> Optional[int]:
    path = lockfile_path()
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def lockfile_status(now: Optional[float] = None) -> dict:
    """Return liveness info: ``{exists, pid, age_seconds, fresh}``."""
    path = lockfile_path()
    if not path.exists():
        return {"exists": False, "pid": None, "age_seconds": None, "fresh": False}
    ref = now if now is not None else time.time()
    age = ref - path.stat().st_mtime
    return {
        "exists": True,
        "pid": read_pid(),
        "age_seconds": age,
        "fresh": age < HEARTBEAT_MAX_AGE_SECONDS,
    }


def is_fresh(now: Optional[float] = None) -> bool:
    return lockfile_status(now)["fresh"]
