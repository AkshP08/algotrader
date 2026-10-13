# README badges — reference

The badges at the top of [`README.md`](../README.md) are **static** (shields.io
flat badges). There is no CI pipeline yet, so none of them are wired to a live
check — they document facts verified manually before each release commit. This
file explains what each one asserts and how to re-verify it.

| Badge | Asserts | How to re-verify |
|---|---|---|
| `tests — 146 passing` | The full suite passes. | `.venv/bin/pytest -q` → `146 passed` |
| `mode — paper trading (Alpaca)` | The bot is paper-only by construction. | `src/broker/alpaca_client.py` → `assert_paper_trading` hard-fails on a non-paper URL at import time and in the constructor. |
| `python — 3.11+` | Minimum supported Python. | Uses `zoneinfo`, modern typing; developed and tested on 3.11+. |
| `license — MIT` | Licensing. | See [`LICENSE`](../LICENSE). |

## Updating the test-count badge

If the test count changes, update **both** the badge URL in `README.md`
(`tests-<N>%20passing`) and the count referenced in the README "Testing" section.
Keep them in sync with the actual `pytest` output.

## If CI is added later

Replace the static `tests` badge with a GitHub Actions workflow-status badge, e.g.:

```
![Tests](https://github.com/<user>/<repo>/actions/workflows/tests.yml/badge.svg)
```

and keep this file as the record of what the badges mean.
