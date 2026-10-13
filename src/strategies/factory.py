"""Build strategy instances from validated config, and hash their params.

One place maps a :class:`~src.utils.config.StrategyInstanceConfig` to a concrete
:class:`~src.strategies.base.Strategy`, so the orchestrator, the dashboard, and
the Null wall all agree on how a configured strategy becomes an object.

``params_hash`` produces a stable SHA-256 over a strategy's config params. The
Null wall stores this hash at validation time and refuses startup if the live
config no longer matches — you cannot silently change a strategy's parameters and
keep an old validation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

from src.strategies.base import Strategy
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.pairs_trading import PairsTrading
from src.strategies.rsi_mean_reversion import RSIMeanReversion
from src.utils.config import StrategyInstanceConfig, load_strategies_config

# Maps the ``type`` discriminator in config to a concrete strategy class.
STRATEGY_TYPES: dict[str, type[Strategy]] = {
    "ma_crossover": MovingAverageCrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "pairs_trading": PairsTrading,
}


def params_hash(params: Mapping) -> str:
    """Stable SHA-256 of a params mapping (key order independent)."""
    canonical = json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_strategy(cfg: StrategyInstanceConfig) -> Strategy:
    """Instantiate the strategy described by ``cfg``."""
    try:
        cls = STRATEGY_TYPES[cfg.type]
    except KeyError as exc:  # pragma: no cover - config is Literal-validated upstream
        raise ValueError(f"unknown strategy type: {cfg.type!r}") from exc
    return cls(name=cfg.name, **cfg.params)


def build_strategies(
    configs: Iterable[StrategyInstanceConfig] | None = None,
    include_disabled: bool = False,
) -> dict[str, Strategy]:
    """Build a ``name -> Strategy`` map from config (enabled instances only).

    With no ``configs`` the default ``config/strategies.yaml`` is loaded.
    """
    if configs is None:
        configs = load_strategies_config().strategies
    out: dict[str, Strategy] = {}
    for cfg in configs:
        if cfg.enabled or include_disabled:
            out[cfg.name] = build_strategy(cfg)
    return out
