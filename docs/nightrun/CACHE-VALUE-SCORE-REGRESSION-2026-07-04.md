# Cache-value score regression diagnosis — 2026-07-04 (#2613)

`fak nightrun score --json` is red: realized KV-prefix reuse **0.6918** is below the
**0.751** floor (dogfooded at 75.1% in-kernel serve, #1114). This note is the scrubbed
witness for #2613 — it names the responsible session and files the exact implementation
child (#3031) needed to restore the score honestly, because the code that would restore it
lives outside the `nightrun` lane (see "Why not fixed in this lane" below).

## Before witness — `fak nightrun score --json`

```
total_sessions: 11   multi_turn_sessions: 6   single_turn_sessions: 5
total_turns: 71      frozen_turns: 0          partial_turns: 47   cold_turns: 24
gate_prompt_tokens: 5863   gate_reused_tokens: 4056
realized_reuse_ratio: 0.6917960088691796   floor: 0.751
verdict: regression detected  (exit 1)
```

## The gate corpus (multi-turn sessions, turns >= 2) — `docs/nightrun/cache-value.jsonl`

| date | type | ctx | turns | prompt | reused | cold | partial | frozen | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-28 | run | smollm2 | 5 | 413 | 260 | 1 | 4 | 0 | 0.6295 |
| 2026-06-30 | run | smollm2 | 6 | 433 | 305 | 1 | 5 | 0 | 0.7044 |
| 2026-06-30 | run | smollm2 | 10 | 1356 | 1096 | 1 | 9 | 0 | 0.8083 |
| 2026-06-30 | run | smollm2 | 9 | 1293 | 1022 | 1 | 8 | 0 | 0.7904 |
| 2026-06-30 | run | smollm2 | 8 | 998 | 758 | 1 | 7 | 0 | 0.7595 |
| **2026-07-03** | **guard** | **python (pid 45396)** | **28** | **1370** | **615** | **14** | **14** | **0** | **0.4489** |

## Responsible session

The regression is **one** session, not a fleet-wide drop: the 2026-07-03 `guard`/python
session (28 turns, ratio **0.4489**). The five `run`/smollm2 sessions are linear and each
have exactly **1** cold-start turn; the guard session has **14** cold turns out of 28.

Arithmetic (pooled reused/prompt over the multi-turn corpus):

- with the guard session: `4056 / 5863` = **0.6918**  → below floor
- without the guard session: `3441 / 4493` = **0.7659**  → above floor

So the gate is one flagship guard session pulling five clean runs under the floor.

## Diagnosis: unresolved cold-start vs. real drop (not gameable from the ledger)

`internal/cacheobs` buckets a turn `cold` at `reuse < 0.10`, which lumps two different
turns together: a **cold-start** (a new independent prefix with no prior turn to reuse
from — the gate ALREADY excludes these at *session* granularity as "manufacturing a false
regression") vs. a **real mid-session cache-preservation drop** (the regression the gate
should catch). The guard session's 14 cold turns are unresolved between these, and the
committed ledger row carries only the aggregate `cold_turns` — so the classification
**cannot be settled from the ledger alone**; it needs per-turn KV detail.

Excluding the guard session (or lowering the floor) without that per-turn witness would be
gaming an evidence-backed gate — explicitly out-of-scope for #2613. So this is diagnosed,
not silently "fixed."

## Why not fixed in this lane

The `nightrun` lane tree is `internal/nightrun/**` only. The code that would restore the
score honestly — `internal/cacheobs` (emit a cold-start count), `internal/cachevalueledger`
(record it + exclude at turn granularity), `cmd/fak/guard_child.go` (per-turn snapshot) — is
out-of-lane. #2614 is the open prerequisite that extends lane coverage to those packages.

## Restoration path (filed)

- **#3031** — implementation child: capture per-turn KV detail for a guard session, classify
  the 14 cold turns as cold-STARTS (→ exclude at turn granularity, the missing-classification
  fix) vs. real DROPS (→ fix guard prefix preservation), then re-score.
- **#2614** — lane-coverage prerequisite blocking #3031.

## Acceptance gate (unchanged)

`fak nightrun score --json` returns success, WITH the per-turn rows proving each excluded
cold turn is a genuine cold-start (not a masked drop).
