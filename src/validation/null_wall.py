"""The Null validation wall — enforced at orchestrator startup.

**No strategy trades in the bot until it has passed Null validation.** Each
configured strategy must have a signed record at ``validation/<name>.json`` that
proves it was run through Null (the external validation tool) and survived. At
startup :meth:`NullWall.enforce` loads every configured strategy's record and
FAILS the whole startup (raising :class:`ValidationError`) if any record is:

  * **missing** — the strategy was never validated;
  * **not SURVIVED** — Null's verdict was ``SHAKY`` or ``DEAD``;
  * **stale** — ``validated_at`` is more than ``max_age_days`` (90) old;
  * **drifted** — the stored ``params_hash`` no longer matches the live config,
    i.e. the strategy's parameters changed since it was validated.

This is a *mechanism*, not pre-approval: the committed records are placeholders
the operator replaces with real Null runs before go-live. A passing strategy is
logged at INFO; every failure is logged at CRITICAL before the raise.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.strategies.factory import params_hash
from src.utils.config import StrategyInstanceConfig, load_strategies_config
from src.utils.logger import get_logger

log = get_logger("validation.null_wall")

# Repo-root-relative default location for the validation records.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_DIR = _REPO_ROOT / "validation"

MAX_AGE_DAYS = 90
VERDICT_SURVIVED = "SURVIVED"
VALID_VERDICTS = {"SURVIVED", "SHAKY", "DEAD"}

REQUIRED_FIELDS = (
    "strategy_name",
    "null_url",
    "validated_at",
    "verdict",
    "params_hash",
    "signed_by",
    "notes",
)


class ValidationError(RuntimeError):
    """Raised when one or more strategies fail the Null wall at startup."""


@dataclass(frozen=True)
class ValidationRecord:
    """A parsed ``validation/<name>.json`` record."""

    strategy_name: str
    null_url: str
    validated_at: _dt.date
    verdict: str
    params_hash: str
    signed_by: str
    notes: str

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationRecord":
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"validation record missing fields: {missing}")
        verdict = str(data["verdict"]).upper()
        if verdict not in VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}"
            )
        try:
            validated_at = _dt.date.fromisoformat(str(data["validated_at"]))
        except ValueError as exc:
            raise ValueError(
                f"validated_at must be an ISO date (YYYY-MM-DD), "
                f"got {data['validated_at']!r}"
            ) from exc
        return cls(
            strategy_name=str(data["strategy_name"]),
            null_url=str(data["null_url"]),
            validated_at=validated_at,
            verdict=verdict,
            params_hash=str(data["params_hash"]),
            signed_by=str(data["signed_by"]),
            notes=str(data["notes"]),
        )


class NullWall:
    """Loads and enforces per-strategy Null validation records."""

    def __init__(
        self,
        validation_dir: Optional[str | Path] = None,
        max_age_days: int = MAX_AGE_DAYS,
    ) -> None:
        self.validation_dir = Path(validation_dir or DEFAULT_VALIDATION_DIR)
        self.max_age_days = max_age_days

    def record_path(self, strategy_name: str) -> Path:
        return self.validation_dir / f"{strategy_name}.json"

    def load_record(self, strategy_name: str) -> ValidationRecord:
        """Load and structurally validate a single record (raises on problems)."""
        path = self.record_path(strategy_name)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return ValidationRecord.from_dict(data)

    def _check_one(
        self, cfg: StrategyInstanceConfig, now: _dt.date
    ) -> Optional[str]:
        """Return a failure reason for one strategy, or ``None`` if it passes."""
        try:
            record = self.load_record(cfg.name)
        except FileNotFoundError:
            return (
                f"no Null validation file at {self.record_path(cfg.name)} "
                f"(run it through Null and sign the record first)"
            )
        except (ValueError, json.JSONDecodeError) as exc:
            return f"validation file is malformed: {exc}"

        if record.verdict != VERDICT_SURVIVED:
            return f"Null verdict is {record.verdict}, not {VERDICT_SURVIVED}"

        age_days = (now - record.validated_at).days
        if age_days > self.max_age_days:
            return (
                f"validation is stale: {age_days} days old "
                f"(> {self.max_age_days}); re-validate through Null"
            )
        if age_days < 0:
            return f"validated_at {record.validated_at.isoformat()} is in the future"

        expected = params_hash(cfg.params)
        if record.params_hash != expected:
            return (
                "params_hash mismatch: config parameters changed since validation "
                f"(expected {expected[:12]}…, record has {record.params_hash[:12]}…)"
            )
        return None

    def enforce(
        self,
        strategy_configs: Iterable[StrategyInstanceConfig] | None = None,
        now: Optional[_dt.date] = None,
    ) -> dict[str, ValidationRecord]:
        """Enforce the wall across all configured strategies.

        Returns the map of passing ``name -> ValidationRecord`` on success; raises
        :class:`ValidationError` (after logging each failure CRITICAL) if any
        configured strategy fails.
        """
        if strategy_configs is None:
            strategy_configs = load_strategies_config().strategies
        configs = [c for c in strategy_configs if c.enabled]
        reference = now or _dt.date.today()

        passed: dict[str, ValidationRecord] = {}
        failures: dict[str, str] = {}
        for cfg in configs:
            reason = self._check_one(cfg, reference)
            if reason is None:
                record = self.load_record(cfg.name)
                passed[cfg.name] = record
                log.info(
                    "null_wall_pass",
                    strategy=cfg.name,
                    verdict=record.verdict,
                    validated_at=record.validated_at.isoformat(),
                    signed_by=record.signed_by,
                )
            else:
                failures[cfg.name] = reason
                log.critical("null_wall_fail", strategy=cfg.name, reason=reason)

        if failures:
            detail = "; ".join(f"{name}: {why}" for name, why in failures.items())
            raise ValidationError(
                f"Null wall blocked startup — {len(failures)} strategy(ies) failed "
                f"validation: {detail}"
            )
        log.info("null_wall_enforced", passed=sorted(passed))
        return passed


def template_for(cfg: StrategyInstanceConfig) -> dict:
    """Build an unsigned validation template for ``cfg`` (params_hash prefilled)."""
    return {
        "strategy_name": cfg.name,
        "null_url": "",
        "validated_at": "",
        "verdict": "",
        "params_hash": params_hash(cfg.params),
        "signed_by": "",
        "notes": "",
    }
