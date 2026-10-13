"""Tests for typed config loading (pydantic validation)."""
from __future__ import annotations

import pytest

from src.utils.config import (
    DEFAULT_RISK_PATH,
    DEFAULT_STRATEGIES_PATH,
    RiskConfig,
    load_risk_config,
    load_strategies_config,
)


def test_risk_defaults_are_safe():
    cfg = RiskConfig()
    assert cfg.max_position_size_pct == 0.10
    assert cfg.max_total_exposure_pct == 0.60
    assert cfg.max_positions == 5
    assert cfg.max_daily_loss_pct == 0.03
    assert cfg.max_drawdown_pct == 0.10
    assert cfg.per_trade_stop_loss_pct == 0.02
    assert cfg.min_order_notional == 100.0
    assert cfg.trading_hours_only is True


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_risk_config(tmp_path / "nope.yaml")
    assert cfg == RiskConfig()


def test_shipped_risk_yaml_loads():
    assert DEFAULT_RISK_PATH.exists()
    cfg = load_risk_config()
    assert isinstance(cfg, RiskConfig)
    assert 0 < cfg.max_position_size_pct <= cfg.max_total_exposure_pct <= 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_position_size_pct", 0),      # must be > 0
        ("max_position_size_pct", 1.5),    # must be <= 1
        ("max_positions", 0),              # must be >= 1
        ("max_daily_loss_pct", -0.1),      # must be > 0
    ],
)
def test_invalid_risk_values_rejected(field, value):
    with pytest.raises(Exception):
        RiskConfig(**{field: value})


def test_exposure_must_be_ge_position_size():
    with pytest.raises(Exception):
        RiskConfig(max_position_size_pct=0.5, max_total_exposure_pct=0.3)


def test_unknown_risk_key_rejected():
    with pytest.raises(Exception):
        RiskConfig(bogus_key=1)


def test_shipped_strategies_yaml_loads():
    assert DEFAULT_STRATEGIES_PATH.exists()
    cfg = load_strategies_config()
    names = {s.name for s in cfg.strategies}
    types = {s.type for s in cfg.strategies}
    assert {"ma_crossover_spy", "rsi_mean_reversion_qqq", "pairs_trading_xle_xop"} <= names
    assert types <= {"ma_crossover", "rsi_mean_reversion", "pairs_trading"}


def test_unknown_strategy_type_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("strategies:\n  - name: x\n    type: not_a_strategy\n")
    with pytest.raises(Exception):
        load_strategies_config(p)
