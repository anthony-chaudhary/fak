---
title: "fak guard-hop RSI loop (issue #733)"
description: "How fak plans an RSI loop to drive guard-hop overhead toward 0, with a keep/revert rung deferred on a live measured baseline."
---

# Guard-hop RSI loop — driving kernel-in-the-loop overhead toward 0 (issue #733)

Audience: maintainers tuning the guarded dogfood fleet after the always-on
gateway path is installed. Prerequisite: start with
[always-on-dogfood-server.md](always-on-dogfood-server.md) and the Tier-1/Tier-2
activation runbooks. You will be able to run the guard-hop RSI planner, keep it
in plan mode until real measurements exist, and verify that it cannot fabricate
a keep-bit.

The dogfood default fronts every worker with `fak guard`, so each tool call crosses the
kernel. Issue [#734](https://github.com/anthony-chaudhary/fak/issues/734) *measures* that
hop's overhead; this loop *minimises* it — a recursive-self-improvement (RSI) loop fed by
live dogfood-fleet telemetry that keeps an optimisation only when a re-measured number and
an external witness agree it helped.

**Scaffold:** `tools/guard_hop_rsi.py` (+ hermetic `tools/guard_hop_rsi_test.py`).
**Status:** loop scaffold + candidate search space land now; the keep/revert rung is
**deferred** on #734's live MEASURED baseline (hardware-gated).

## The loop

```
            ┌──────────────────────────────────────────────────────────────┐
            │  baseline = guard_hop_bench measure (live gateway, #734)       │
            │  telemetry = .dispatch-runs/guard-audit/*.jsonl  (fleet, #729) │
            └───────────────┬──────────────────────────────────────────────┘
                            ▼
   pick worst-overhead candidate ──▶ apply ONE change ──▶ re-measure overhead
                            ▲                                      │
                            │                          ┌───────────┴───────────┐
                            │                          ▼                       ▼
                            │              overhead strictly lower      else / regressed
                            │              AND witness green?                  │
                            │                          │                       │
                            └──────── KEEP ◀───────────┘         REVERT ───────┘
```

The keep/revert decision is **grounded**, not self-asserted: a candidate is kept only when
(1) the re-measured guard-hop overhead is strictly lower AND (2) a witness the loop did not
author — `go test ./...` green — confirms no regression. This mirrors the DOS
enforcement-tuning loop's discipline: a non-forgeable keep-bit, not the loop's say-so.

## Candidate search space (worst-overhead-first)

| id | lever | hypothesis |
|---|---|---|
| `verdict-memoize` | memoize the `Decide` verdict for an identical `(tool, args, worldVer)` | agent loops re-propose the same call; 2nd..Nth Decide becomes O(1) |
| `journal-batch-fsync` | group-commit the decision journal instead of per-row fsync | per-row durability fsync dominates the hop on a busy worker |
| `argpredicate-precompile` | compile ArgPredicates once per policy load | the 362 ns → 605 ns Decide growth under 2000 predicates (committed bench) is hoistable |
| `inproc-colocate` | sentinel: assert adjudication stays in-process | the ~2,849× in-process-vs-spawned tax (committed) is the biggest lever; guard against regressing off it |

These are the *search space*, not claims — none is "done" until measured. The honesty gate
(`guard_hop_rsi.py --check`) refuses any plan that marks a candidate `kept` without a witness
or without a strictly-negative measured delta, so the scaffold cannot fabricate a win.

## Why deferred

The keep/revert rung needs a MEASURED baseline (`guard_hop_bench measure` against a live
`fak serve` gateway) plus live telemetry from a guarded dogfood fleet
([#729](https://github.com/anthony-chaudhary/fak/issues/729)) — both hardware-gated. Until
then the loop runs in **plan mode**: it loads the PROJECTED baseline, enumerates the
candidates as `PENDING_MEASUREMENT`, and emits the iteration plan. When #734's measurement
and #729's fleet are live, the same loop runs the keep/revert rung against real numbers.

```bash
python tools/guard_hop_rsi.py plan                       # plan mode (PROJECTED baseline)
python tools/guard_hop_rsi.py plan --measured row.json   # against a MEASURED #734 baseline
python tools/guard_hop_rsi.py --check plan.json          # honesty gate
```

## Refs

- `tools/guard_hop_bench.py` — the #734 measurement source the loop reads
- [`docs/benchmarks/GUARD-HOP-OVERHEAD-PENDING.md`](../benchmarks/GUARD-HOP-OVERHEAD-PENDING.md) — the overhead row this loop drives down
- `tools/com.fak.dogfood-fleet.plist` — the #729 fleet that produces the live telemetry
