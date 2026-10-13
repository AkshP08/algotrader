"""Generate an unsigned Null-validation template for a configured strategy.

    python -m src.scripts.generate_validation_template ma_crossover_spy

Writes ``validation/<strategy_name>.json`` with the ``params_hash`` prefilled
from the current config and every signed field left blank for the operator to
complete *after* running the strategy through Null:

    {
      "strategy_name": "...",   # prefilled
      "null_url": "",            # <- paste the Null run URL
      "validated_at": "",        # <- ISO date (YYYY-MM-DD) of the run
      "verdict": "",             # <- SURVIVED | SHAKY | DEAD
      "params_hash": "...",      # prefilled from live config
      "signed_by": "",           # <- your name
      "notes": ""                # <- free text
    }

Refuses to overwrite an existing record unless ``--force`` is given, so a signed
validation is never clobbered by accident. This only writes a template; it does
not and cannot mark a strategy as validated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import load_strategies_config
from src.utils.logger import get_logger
from src.validation.null_wall import DEFAULT_VALIDATION_DIR, template_for

log = get_logger("scripts.generate_validation_template")


def run(strategy_name: str, force: bool = False) -> int:
    cfgs = {c.name: c for c in load_strategies_config().strategies}
    if strategy_name not in cfgs:
        known = ", ".join(sorted(cfgs)) or "(none configured)"
        print(
            f"ERROR: unknown strategy {strategy_name!r}. "
            f"Configured strategies: {known}",
            file=sys.stderr,
        )
        return 2

    template = template_for(cfgs[strategy_name])

    DEFAULT_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = DEFAULT_VALIDATION_DIR / f"{strategy_name}.json"
    if out_path.exists() and not force:
        print(
            f"ERROR: {out_path} already exists. Refusing to overwrite a validation "
            f"record; pass --force to replace it.",
            file=sys.stderr,
        )
        return 1

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"[validation] wrote template: {out_path}")
    print("[validation] fill in null_url, validated_at, verdict, signed_by, notes "
          "after running through Null.")
    log.info("validation_template_written", strategy=strategy_name, path=str(out_path))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an unsigned Null-validation template for a strategy."
    )
    parser.add_argument("strategy_name", help="configured strategy name")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing record"
    )
    args = parser.parse_args(argv)
    return run(args.strategy_name, force=args.force)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # clean non-zero on failure
        log.error("generate_validation_template_failed", error=str(exc))
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
