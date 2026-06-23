---
title: "fak note: the O(1) current turn — a planned view over the lossless history (2026-06-23)"
description: "A baseline spine for treating the current turn as an O(1) materialized view over a lossless history store, re-planned each turn by a cost-based, forecast-driven planner — the middle ground between an unbounded linear transcript and lossy compaction. Postgres-planner correspondence and a closed-form scaling law across the 50→1M turn horizon."
---

# The O(1) current turn — a planned view over the lossless history

> Date: 2026-06-23.
> Scope: the design + a shipped baseline spine (`internal/ctxplan`, `cmd/ctxplandemo`).
> This is the unbuilt "context-layout compiler" rung of
> [`ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md`](ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md)
> (Step 4), and a per-turn-horizon refinement of
> [`SCALING-LAWS-OF-AGENTS-2026-06-19.md`](SCALING-LAWS-OF-AGENTS-2026-06-19.md).

## 0. The idea in one paragraph

A long agent session does not have to choose between keeping the whole transcript
resident (exact but O(N) tokens, blows the window) and compacting it (O(1) tokens but
lossy and irreversible). Treat the linear history as a **core dump** — the term the repo
already uses for a finished session (`recall`'s manifest+CAS, `cdb`'s debugger view) —
and treat the **current turn's context as one rendered VIEW over it**. Each turn, a
cost-based planner predicts what the next turns will reference, and OPTIMIZES which spans
are resident under a fixed token budget. The resident view stays O(1); the rest of the
history stays one demand-page away, lossless. A session that would have been 50, 100,
1,000, 10,000, or 1,000,000 linear turns becomes **1 current turn + a flexible history**
the planner re-derives on demand.

The prompt is no longer the memory. The prompt is one render of a queryable memory image
— and now that render is **planned**, not just filtered.

## 1. The three regimes (and why the planned one is the middle ground)

```text
linear      keep the whole transcript resident      -> O(N) tokens, EXACT recall, unbounded window
compaction  summarize at a cap, drop the originals  -> O(1) tokens, LOSSY recall, irreversible
planned     an O(1) resident VIEW + the lossless    -> O(1) tokens, EXACT recall, paying only a
            store behind it, re-planned each turn       bounded forecast-MISS rate (a page fault)
```

The two extremes each sacrifice one of {bounded resident tokens, exact recall}. The
planned view keeps **both**. The price it pays is not a lost fact — it is a *page fault*:
when the forecast misses, the needed span pages back in on demand (cheap, O(query)). A
bad forecast degrades efficiency, never correctness, because the store is lossless.

This is the same move virtual memory makes against "keep everything in RAM" vs "throw the
cold pages away": a bounded resident set over a backing store, with a pager that predicts
faults. `ctxmmu` is literally the context **memory-management unit**; `ctxplan` is the
**pager's prediction + replacement policy** for the live turn.

## 2. The Postgres-planner correspondence

The design leans on cost-based query planning. The mapping is one-to-one:

| relational analogue              | ctxplan                                                            |
|----------------------------------|-------------------------------------------------------------------|
| table / relation                 | the history store (`recall` manifest / `memq` cells)              |
| query                            | the `Forecast` (predicted reference for the next horizon) + `Budget` |
| `pg_statistic` (row stats)        | per-cell benefit signals: relevance, learned utility, durability, recency |
| planner cost constants            | `Forecast.Weights`                                                |
| the planner / optimizer           | `Optimize` — a budgeted 0/1 knapsack: maximize benefit s.t. tokens ≤ W |
| the chosen plan / access path     | `Plan.Selected` (which spans are resident, in render order)       |
| `EXPLAIN` / `EXPLAIN ANALYZE`     | `Plan.Explain` (estimated cost+benefit+density per included/elided span) |
| a materialized view               | the rendered O(1) fresh history (`Materialize`)                    |
| the buffer pool / working set     | the resident view; the backing store is the CAS swap device       |
| a page fault                      | a forecast MISS → demand-page the missing span back in (exact, cheap) |
| a prepared statement / plan cache | reusing a `Forecast` plan across turns (`cachemeta.PlanTemplate`)  |

The planner is the genuinely new layer. `memq` already gives a memory **query language**
(scan/filter/rank/limit/budget/render with `EXPLAIN`); `cdb`/`contextq` already
demand-page one query's working set and materialize it with verdicts. What none of them
do is **optimize the selection under a cost model driven by a prediction** — choose the
best O(1) view, not execute an authored pipeline. That is `Optimize`.

## 3. The baseline spine (shipped)

`internal/ctxplan` (foundation tier, stdlib-only, off the request path, registers
nothing) is a self-contained planner over its own `Span` (SAFE metadata) and `Store`
(spans + a trust-gated `Materialize`) types. A `memq`/`cdb`/`recall` adapter that lowers
their cells/pages into `Span`s is a thin, higher-tier follow-on, so the core stays
dependency-free and builds standalone:

- **`Forecast` + `Benefit`** (`forecast.go`) — the "imagine what I'll need" prediction:
  predicted intents, a horizon, pins, and the weighted benefit model (relevance ·
  learned utility · durability prior · recency). A **sealed or tombstoned cell scores
  exactly 0** — it is never a candidate, mirroring the recall/cdb/memq invariant that
  poison and suppressed spans never enter context.
- **`Optimize`** (`plan.go`) — the budgeted knapsack. Pins are forced resident and
  charged first (a pin naming a sealed span is *refused*, not forced — a pin cannot
  launder poison). The remainder is filled greedy-by-density (the deterministic
  production planner) or by an exact 0/1-DP oracle (`ObjExact`, used to bound the greedy
  gap on small inputs). `Plan.Explain` is the `EXPLAIN ANALYZE`.
- **`Audit` + `CompactionView`** (`faithful.go`) — the honesty gate. A plan is
  **Faithful** iff its resident and elided sets partition every candidate AND every
  elided span carries a page-back-in handle. `CompactionView` strips those handles to
  model lossy compaction, so the contrast is *checkable*: same residency, opposite
  recoverability. This is what lets the design claim "O(1) resident AND exact recall."
- **`Model` + `Compare`** (`scaling.go`) — the closed-form scaling instrument (no wall
  clock, fully reproducible): resident tokens, lossless store, exact-recall fidelity, and
  the forecast-miss fault count per regime across an arbitrary turn schedule.
- **`Materialize`** (`materialize.go`) — scan the store, plan the O(1) view, render the
  selection's bytes into the fresh history **through the trust gate**
  (`Store.Materialize`), in step order. A selected span the gate refuses is reported,
  never rendered. The in-memory `MemStore` is the zero-setup reference `Store` for the
  demo and tests.

Witness:

```bash
go test ./internal/ctxplan        # the invariants below (run under WSL on a native-Windows host)
go run ./cmd/ctxplandemo -selfcheck   # plan an O(1) view, prove the invariants, print the scaling table
```

The demo proves, on a synthetic store that includes a SEALED poison result: the resident
view stays within budget; the poison is elided (sealed) and **never rendered even though
the forecast would want it**; the plan is Faithful while its compaction twin is not; and
the scaling curve bends.

## 4. The scaling law (computed, not asserted)

`scaling.go` makes the curve a number. With a representative `b = 700` tokens/turn,
working set `W = 8,000` tokens, forecast hit rate `p_hit = 0.9`, and compaction retain
`ρ = 0.7`, `cmd/ctxplandemo` prints:

```text
turns     | linear-resident recall | compact-resident recall | planned-resident recall  faults
----------+------------------------+-------------------------+-------------------------+-------
50        | 35.0K            1.000  | 8.0K             0.240   | 8.0K             1.000    4
100       | 70.0K            1.000  | 8.0K             0.058   | 8.0K             1.000    9
1.0K      | 700.0K           1.000  | 8.0K             0.000   | 8.0K             1.000    99
10.0K     | 7.0M             1.000  | 8.0K             0.000   | 8.0K             1.000    999
100.0K    | 70.0M            1.000  | 8.0K             0.000   | 8.0K             1.000    10.0K
1.0M      | 700.0M           1.000  | 8.0K             0.000   | 8.0K             1.000    100.0K
```

The asymptotics:

| regime     | resident tokens C(N) | exact recall          | cumulative prefill tax |
|------------|----------------------|-----------------------|------------------------|
| linear     | Θ(N)                 | 1.0                   | Θ(N²) (Σ b·i)          |
| compaction | Θ(1) (capped at W)   | ρ^Θ(N) → 0            | Θ(W·N)                 |
| planned    | Θ(1) (capped at W)   | 1.0                   | Θ(W·N) + (1−p_hit)·N cheap faults |

Read the 1M-turn row: linear demands a **700-million-token** resident context (intractable);
compaction holds 8K but its exact recall has collapsed to **0** (an early fact has
survived ~87 compactions at ρ=0.7); the planned view holds the same 8K with **exact
recall 1.0**, paying ~100K cheap demand-page faults over the whole million turns
(≈ one fault per ten turns at p_hit=0.9). That is the bend: O(1) resident *and* exact
recall, against a bounded, cheap fault rate — the option that did not exist between the
two extremes.

This refines the house scaling thesis (`SCALING-LAWS-OF-AGENTS`): that note counts
*shared-setup payments* across agents×turns; this one isolates the **per-turn resident
set** as a function of the turn horizon and shows the cap is free of the compaction tax
on recall.

## 5. How it composes (the build-path position)

`ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md` lays out a five-step build path. Steps 1–3 are
shipped: `cdb.WorkingSet` (the query surface), `cachemeta.FromMemoryView` (views as cache
artifacts), `contextq.MaterializationVerdict` (HIT/FAULT/REFUSE/ABSTAIN). **`ctxplan` is
Step 4** — the context-layout compiler that renders the same state under a budget by
*optimizing*, with the faithfulness witness and the scaling instrument. Step 5 (non-prefix
KV reuse) stays a deferred, audited experiment, unchanged.

The trust boundary does not move: `ctxplan` plans over SAFE cell metadata and renders
through the same gated page-in every other consumer uses. A sealed/tombstoned span is
never resident; a pin cannot launder poison.

## 6. Honest fences (what is NOT done)

- **Not yet on the live loop, and not yet wired to a real store.** The spine plans +
  materializes over its own `Store` (the in-memory `MemStore`). The adapters that lower a
  `memq` backend, a `cdb`/`recall` core image, or a `contextq` view into a `ctxplan.Span`
  are named follow-ons (so the leaf stays dependency-free today), and the gateway does not
  yet call the planner each turn to replace compaction. Both are filed as issues.
- **The forecast is authored, not learned.** Intents/pins are supplied; predicting them
  from the trajectory (the real "preemptive planning") is a follow-on. The benefit
  Weights are a sensible seed, not tuned.
- **The scaling model is closed-form, not measured.** The curve is exact arithmetic from
  the regime definitions, deterministic and testable — not a wall-clock run over real
  transcripts. An empirical pass over recorded `cdb` core images (resident tokens, fault
  rate, answer quality vs a compaction baseline) is the named measurement.
- **Greedy is the production planner.** The exact DP is a small-input oracle; the greedy
  optimality gap is measured, not assumed to be zero.

See the open issues filed alongside this note for each follow-on.
