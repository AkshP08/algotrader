"""Typed, validated configuration loading.

Risk limits and strategy instances live in ``config/*.yaml`` and are parsed into
**pydantic** models so a malformed or unsafe config fails loudly at load time
rather than silently at trade time. Every limit has a safe default, so the system
is fully configured even with no YAML present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Repo-root-relative default locations.
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_RISK_PATH = _CONFIG_DIR / "risk.yaml"
DEFAULT_STRATEGIES_PATH = _CONFIG_DIR / "strategies.yaml"
DEFAULT_ORCHESTRATOR_PATH = _CONFIG_DIR / "orchestrator.yaml"


class RiskConfig(BaseModel):
    """Hard risk limits. All percentages are fractions in (0, 1]."""

    model_config = ConfigDict(extra="forbid")

    max_position_size_pct: float = Field(0.10, gt=0, le=1)
    max_total_exposure_pct: float = Field(0.60, gt=0, le=1)
    max_positions: int = Field(5, ge=1)
    max_daily_loss_pct: float = Field(0.03, gt=0, le=1)
    max_drawdown_pct: float = Field(0.10, gt=0, le=1)
    per_trade_stop_loss_pct: float = Field(0.02, gt=0, le=1)
    min_order_notional: float = Field(100.0, ge=0)
    trading_hours_only: bool = True

    @model_validator(mode="after")
    def _exposure_ge_position(self) -> "RiskConfig":
        if self.max_total_exposure_pct < self.max_position_size_pct:
            raise ValueError(
                "max_total_exposure_pct must be >= max_position_size_pct"
            )
        return self


class StrategyInstanceConfig(BaseModel):
    """One configured strategy instance (type + params)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["ma_crossover", "rsi_mean_reversion", "pairs_trading"]
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("strategy name must be non-empty")
        return v


class StrategiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: list[StrategyInstanceConfig] = Field(default_factory=list)


class OrchestratorConfig(BaseModel):
    """Scheduler / runner settings for the live orchestration loop."""

    model_config = ConfigDict(extra="forbid")

    tick_interval_minutes: int = Field(5, ge=1, le=60)
    benchmark_symbol: str = "SPY"
    timezone: str = "America/New_York"

    @field_validator("benchmark_symbol")
    @classmethod
    def _symbol_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("benchmark_symbol must be non-empty")
        return v.strip().upper()

    @field_validator("timezone")
    @classmethod
    def _tz_valid(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"invalid timezone: {v!r}") from exc
        return v


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_risk_config(path: Optional[str | Path] = None) -> RiskConfig:
    """Load risk limits. Missing file -> safe defaults."""
    p = Path(path) if path is not None else DEFAULT_RISK_PATH
    if not p.exists():
        return RiskConfig()
    return RiskConfig(**_read_yaml(p))


def load_strategies_config(path: Optional[str | Path] = None) -> StrategiesConfig:
    """Load strategy instances. Missing file -> empty list."""
    p = Path(path) if path is not None else DEFAULT_STRATEGIES_PATH
    if not p.exists():
        return StrategiesConfig()
    return StrategiesConfig(**_read_yaml(p))


def load_orchestrator_config(path: Optional[str | Path] = None) -> OrchestratorConfig:
    """Load orchestrator settings. Missing file -> safe defaults."""
    p = Path(path) if path is not None else DEFAULT_ORCHESTRATOR_PATH
    if not p.exists():
        return OrchestratorConfig()
    return OrchestratorConfig(**_read_yaml(p))
