# Sentinel — Interview Cheat Sheet

One page. Memorize the shape, not the words. Everything here is defensible from the
code.

---

## The one-sentence summary

> Sentinel is a regime-aware, multi-strategy **paper** trading bot whose defining
> feature is that **no strategy is allowed to trade until it survives adversarial
> validation in a separate tool (Null)** — distrust of my own backtests, enforced
> in code.

---

## 3 interesting decisions I made

1. **Idempotent order IDs derived from the signal's bar timestamp.**
   `client_order_id = f"{strategy}-{symbol}-{bar_ms}"`. Because the ID is a pure
   function of *which bar* produced the decision, re-running a tick on the same bar
   — or retrying after a network blip — produces the *same* ID, and Alpaca rejects
   the duplicate. Idempotency falls out of determinism instead of needing a
   dedupe table. *(Why it matters: exactly-once semantics with no extra state.)*

2. **Early-close handling without a dynamic scheduler.**
   APScheduler cron can't vary per day, so the fixed `pre_close` cron is backed by
   a fallback: the *final tick* of the session also fires `pre_close`, de-duplicated
   by date. On a 1:00pm early-close day the last tick triggers the close summary;
   on a normal day the 3:55pm cron does. The calendar-exact `planned_schedule` /
   `simulate_day` functions are what the tests assert, so correctness is pinned
   even though the live cron is approximate. *(Why it matters: a pure function is
   testable; a live scheduler isn't.)*

3. **Fail-safe behavior when a validation goes stale mid-run.**
   The Null wall hard-fails at *startup*. But if a validation crosses its 90-day
   staleness line while the bot is already running, `pre_open` doesn't crash the
   scheduler — it **clears the active strategy set**, so the bot keeps running but
   trades nothing until the validation is refreshed. Fail-safe (stop trading), not
   fail-loud (take down the process). *(Why it matters: the safe default under
   uncertainty is "don't trade.")*

---

## One hard tradeoff I considered and rejected

> **FastAPI + async for the dashboard.** I chose **Flask** instead. The dashboard
> is strictly read-only and serves one user; there's no async I/O to exploit and no
> request fan-out. FastAPI's async stack (plus an ASGI server) would add moving
> parts without a measurable benefit at this scale. Right-sizing the tool *is* the
> engineering decision.

(Same muscle elsewhere: **SQLite over Postgres** — single-node, no concurrent
writers; **APScheduler over Celery** — a few cron jobs in one process, no message
broker needed.)

---

## Why the Null wall is the interesting part

Most trading projects *say* "validate your strategies." Sentinel makes that a
**system-level dependency the program refuses to boot without.** `NullWall.enforce`
blocks startup unless every strategy has a signed record that is `SURVIVED`, under
90 days old, and whose `params_hash` still matches the live config. Change a
parameter and the old validation is automatically invalidated by the hash. The
discipline "I don't trust my own backtests" is compiled in, not commented in — and
Null is a genuinely separate project that tries to break the strategies.

---

## What Sentinel does NOT do (say this before you're asked)

- **No machine learning, no reinforcement learning.** All trade math is explicit,
  interpretable, pure Python.
- **No news/NLP parsing** in the decision path. Any LLM is confined to
  orchestration/monitoring/reporting.
- **No real money.** Paper-only by construction — the client hard-fails on a
  non-paper endpoint.
- **No GAAP P&L.** The dashboard's realized figure is an average-cost estimate for
  display.

---

## Phase 2 in one sentence

> Swap the 2×2 regime heuristic for a Hidden Markov / Markov-switching model and
> replace file-based validation records with live Null-API calls, so regime
> inference is probabilistic and validation is checked against the source of truth.
