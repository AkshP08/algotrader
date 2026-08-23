# Sentinel

**Sentinel** is a regime-aware, multi-strategy paper-trading system with a peer-review
validation gate: no strategy is allowed to trade in the live paper account until it passes
independent validation via the **Null** backtester. Trade math is pure, deterministic Python;
any LLM/AI is walled off from the decision path and used only for orchestration, monitoring,
and reporting. Risk limits (per-trade stop, per-day loss cap, max drawdown, max positions,
max exposure) are enforced at every level, and an automatic kill switch halts trading on any
breach — with no manual override in code.

> **Paper trading only.** Sentinel refuses to start against a live/funded Alpaca endpoint:
> the broker client hard-fails unless `ALPACA_BASE_URL` points at a paper host.

## Status

Prompt 1 — **Foundation**: repo scaffold, paper-only Alpaca client, historical data pipeline
with a look-ahead guard, the `Strategy` interface (Null-compatible from day one), SQLAlchemy
storage models, structured logging, and a mocked test suite.

## Standing rules

- Paper trading only. Live/funded keys are never accepted; hard-fail if the base URL is not paper.
- No look-ahead bias, no synthetic/fabricated data, no hardcoded results or metrics.
- Every trading decision is deterministic and logged with timestamp + rationale.
- LLM/AI is walled off from the trading decision path.
- Risk limits enforced at every level; kill switch triggers automatically on breach.
- Every strategy has both entry **and** exit logic. No open-ended positions.
- Secrets via `.env` only (git-ignored). Tests must pass before a task is complete.
- Null wall: no strategy trades until it passes Null validation.

## Layout

```
src/
  broker/       Alpaca paper-trading client (hard paper-only guard, retry/backoff)
  data/         Historical data pipeline: fetch -> SQLite, look-ahead guard, gap detection
  strategies/   Strategy abstract base + Signal dataclass (Null-compatible interface)
  storage/      SQLAlchemy models + init_db()
  utils/        structlog JSON+console logging
  hello_world.py  End-to-end smoke: connect, print equity, fetch/store SPY bars
tests/          Mocked unit tests (no live network required)
data/           SQLite market DB (git-ignored)
logs/           JSON logs by date (git-ignored)
docs/           Design notes
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Alpaca PAPER keys
pytest -v                     # all tests are mocked; no network needed
python -m src.hello_world     # live paper smoke test (requires real paper keys)
```
