"""Tests for the Null validation wall.

Covers the four failure modes (missing / not-SURVIVED / stale / params drift), the
passing case, the template generator, and that the three committed records are
structurally valid for the current config.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from src.strategies.factory import params_hash
from src.utils.config import StrategyInstanceConfig, load_strategies_config
from src.validation.null_wall import (
    DEFAULT_VALIDATION_DIR,
    NullWall,
    ValidationError,
)

TODAY = _dt.date(2026, 10, 13)


def _cfg(name="ma_crossover_spy", params=None):
    return StrategyInstanceConfig(
        name=name,
        type="ma_crossover",
        enabled=True,
        params=params or {"symbol": "SPY", "fast_window": 20, "slow_window": 50},
    )


def _write(dirpath: Path, cfg: StrategyInstanceConfig, **overrides) -> Path:
    record = {
        "strategy_name": cfg.name,
        "null_url": "https://null-aksh.netlify.app",
        "validated_at": TODAY.isoformat(),
        "verdict": "SURVIVED",
        "params_hash": params_hash(cfg.params),
        "signed_by": "Aksh Patel",
        "notes": "test record",
    }
    record.update(overrides)
    path = dirpath / f"{cfg.name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


# --- pass / fail paths -----------------------------------------------------

def test_valid_record_passes(tmp_path):
    cfg = _cfg()
    _write(tmp_path, cfg)
    wall = NullWall(validation_dir=tmp_path)
    passed = wall.enforce([cfg], now=TODAY)
    assert set(passed) == {cfg.name}
    assert passed[cfg.name].verdict == "SURVIVED"


def test_missing_file_fails_startup(tmp_path):
    cfg = _cfg()
    wall = NullWall(validation_dir=tmp_path)  # nothing written
    with pytest.raises(ValidationError, match="no Null validation file"):
        wall.enforce([cfg], now=TODAY)


def test_expired_validation_fails(tmp_path):
    cfg = _cfg()
    old = (TODAY - _dt.timedelta(days=91)).isoformat()
    _write(tmp_path, cfg, validated_at=old)
    wall = NullWall(validation_dir=tmp_path)
    with pytest.raises(ValidationError, match="stale"):
        wall.enforce([cfg], now=TODAY)


def test_validation_at_90_days_still_passes(tmp_path):
    cfg = _cfg()
    edge = (TODAY - _dt.timedelta(days=90)).isoformat()
    _write(tmp_path, cfg, validated_at=edge)
    wall = NullWall(validation_dir=tmp_path)
    assert set(wall.enforce([cfg], now=TODAY)) == {cfg.name}


def test_wrong_params_hash_fails(tmp_path):
    cfg = _cfg()
    _write(tmp_path, cfg, params_hash="deadbeef" * 8)
    wall = NullWall(validation_dir=tmp_path)
    with pytest.raises(ValidationError, match="params_hash mismatch"):
        wall.enforce([cfg], now=TODAY)


def test_changed_params_invalidate_old_validation(tmp_path):
    cfg = _cfg()
    _write(tmp_path, cfg)  # hash for 20/50
    changed = _cfg(params={"symbol": "SPY", "fast_window": 10, "slow_window": 50})
    wall = NullWall(validation_dir=tmp_path)
    with pytest.raises(ValidationError, match="params_hash mismatch"):
        wall.enforce([changed], now=TODAY)


def test_non_survived_verdict_fails(tmp_path):
    cfg = _cfg()
    _write(tmp_path, cfg, verdict="SHAKY")
    wall = NullWall(validation_dir=tmp_path)
    with pytest.raises(ValidationError, match="SHAKY"):
        wall.enforce([cfg], now=TODAY)


def test_future_validation_fails(tmp_path):
    cfg = _cfg()
    future = (TODAY + _dt.timedelta(days=5)).isoformat()
    _write(tmp_path, cfg, validated_at=future)
    wall = NullWall(validation_dir=tmp_path)
    with pytest.raises(ValidationError, match="future"):
        wall.enforce([cfg], now=TODAY)


def test_disabled_strategies_are_skipped(tmp_path):
    cfg = StrategyInstanceConfig(
        name="ma_crossover_spy", type="ma_crossover", enabled=False,
        params={"symbol": "SPY", "fast_window": 20, "slow_window": 50},
    )
    wall = NullWall(validation_dir=tmp_path)  # no file, but disabled -> ignored
    assert wall.enforce([cfg], now=TODAY) == {}


def test_multiple_failures_all_reported(tmp_path):
    a = _cfg(name="a")
    b = _cfg(name="b")
    _write(tmp_path, a, verdict="DEAD")
    # b has no file at all
    wall = NullWall(validation_dir=tmp_path)
    with pytest.raises(ValidationError) as exc:
        wall.enforce([a, b], now=TODAY)
    assert "a:" in str(exc.value) and "b:" in str(exc.value)


# --- template generator ----------------------------------------------------

def test_template_generator_produces_loadable_json(tmp_path, monkeypatch):
    import src.scripts.generate_validation_template as gen

    monkeypatch.setattr(gen, "DEFAULT_VALIDATION_DIR", tmp_path)
    rc = gen.run("ma_crossover_spy")
    assert rc == 0

    out = tmp_path / "ma_crossover_spy.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    # params_hash is prefilled; signed fields are blank for the operator.
    assert len(data["params_hash"]) == 64
    assert data["strategy_name"] == "ma_crossover_spy"
    assert data["verdict"] == "" and data["signed_by"] == "" and data["validated_at"] == ""


def test_template_generator_refuses_overwrite(tmp_path, monkeypatch):
    import src.scripts.generate_validation_template as gen

    monkeypatch.setattr(gen, "DEFAULT_VALIDATION_DIR", tmp_path)
    assert gen.run("ma_crossover_spy") == 0
    assert gen.run("ma_crossover_spy") == 1  # refuses without --force
    assert gen.run("ma_crossover_spy", force=True) == 0


def test_template_generator_unknown_strategy(tmp_path, monkeypatch):
    import src.scripts.generate_validation_template as gen

    monkeypatch.setattr(gen, "DEFAULT_VALIDATION_DIR", tmp_path)
    assert gen.run("does_not_exist") == 2


# --- committed records ------------------------------------------------------

def test_committed_validation_records_are_valid_for_current_config():
    """The three shipped records exist, are SURVIVED, and match current params."""
    cfgs = {c.name: c for c in load_strategies_config().strategies}
    wall = NullWall(validation_dir=DEFAULT_VALIDATION_DIR)
    for name, cfg in cfgs.items():
        record = wall.load_record(name)
        assert record.verdict == "SURVIVED"
        assert record.params_hash == params_hash(cfg.params)
        assert record.signed_by
