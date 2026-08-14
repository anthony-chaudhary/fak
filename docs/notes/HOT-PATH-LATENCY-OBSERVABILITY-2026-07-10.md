---
title: "Hot-path wall-clock observability + the 5× lifecycle-boundary plan — 2026-07-10"
description: "fak's observability plane grades ~97/100, but the wall-clock of its two most-felt lifecycle boundaries — first launch (guard session start) and concurrency (dispatch tick/spawn) — is barely observable: first-launch is timed-but-truncated-and-ephemeral, dispatch is timed-but-never-read. This note scores that gap and files a maturity-ladder plan to make both paths observable end-to-end and drive a 5× cut on the most-called stage."
---

# Hot-path wall-clock observability + the 5× plan

> **Status:** OPEN · roll-up tracking note (proposed epic — GitHub epic **not yet filed**, see §9).
> **Lanes:** spans `cmd` (guard, dispatch), `gateway`/`metrics`, `turntaxmeter`, `usagelog`, `docs`.
> **Anchor standard:** [`net-true-value`](../standards/net-true-value.md) (WITNESSED/OBSERVED/MODELED; a budget is an envelope, not zero-cost).
> **Sibling epics (different axes — cross-linked, never blended):**
> - [`track-b-performance-parity #306`](track-b-performance-parity-tracking-306.md) — fak-vs-llama.cpp **raw inference** parity.
> - [`self-tax plane #1147`](self-tax-performance-assurance-tracking-1147.md) — fak's **per-call mediation** overhead (Submit/Reap/decode).
> - **This note = the third axis:** the **lifecycle-boundary wall-clock** — first launch and dispatch spawn — which neither sibling covers.

## 1. The gap in one sentence

fak has a 241-family observability plane that grades **96.8/100 (grade A)**
([`OBSERVABILITY-SCORECARD.md`](../OBSERVABILITY-SCORECARD.md)), but that score measures whether
the *served-gateway* metric families exist and correlate — **not** the wall-clock a user actually
feels at the two lifecycle boundaries the loop lives on:

- **First launch** — how long `fak manage` takes before the wrapped `claude` is usable.
- **Concurrency** — how long each dispatch tick takes to evaluate and admit/spawn a worker.

Both boundaries *have* a stopwatch. Neither boundary's wall-clock is **persisted, folded to
percentiles, or gated**. So today "did a good launch get slower than its budget?" — the perf-floor
dual of #1147's question, at the lifecycle boundary — has no witness that can fire.

## 2. The scorecard (honest, per hot path)

Scored on four dimensions an SRE would ask of any latency SLO: **Emitted** (is the wall-clock
measured at all?), **Complete** (does the span cover the whole felt path?), **Persisted** (does it
survive the process, per-invocation?), **Gated** (is there a budget a regression trips?). 0–5 each.

| Hot path | Emitted | Complete | Persisted | Gated | Notes |
|---|---:|---:|---:|---:|---|
| **First launch** (guard session start) | 4 | 2 | 1 | 0 | 8 pre-ready phases timed; **truncated at `MarkReady`** — the post-ready installer tail + the `claude` child cold-start are uninstrumented; metric is pull-only on an ephemeral port nothing scrapes in an attended session; guard sessions never even hit the usagelog (os.Exit gap). |
| **Concurrency** (dispatch tick/spawn) | 3 | 2 | 2 | 0 | One `timings_ms` stopwatch, 8 coarse buckets, **written to the loop ledger but never read**; the fat stages (2–3 process-table walks, a `dos loop` subprocess, lease arbitration) are hidden inside single buckets; `TICK_PHASE_REGRESSION` exists only as a comment; `spawn_admission_rate` aggregates nowhere. |

**Headline:** the served plane grades 96.8/100 (though only 22% of its 241 families are surfaced in
any doc/dashboard/alert); the **self** plane at these two boundaries is ~**1.5/5**. First launch is *timed-but-truncated-and-ephemeral*; concurrency is
*timed-but-never-read*. This is the exact dual gap #1147 names for the per-call tax, now on the two
lifecycle boundaries #1147 explicitly does not cover.

## 3. What already exists (the substrate this plan wires — not greenfield)

Every row is a real seam; the plan extends, folds, and gates them.

### 3.1 First-launch path
| Seam | What it gives us | What's missing |
|---|---|---|
| `cmd/fak/guard.go:192` `t0`; `:938-956` 8 `StartupPhase`s (flag-parse, policy-load, local-detect, remote-preflight, upstream-resolve, path-lookup, model-load, tokenizer-load, listener-bind) | the pre-ready boot timeline | timeline **ends at `MarkReady` (`guard.go:1084`)** |
| `internal/gateway/startup.go:303-304` `fak_gateway_time_to_ready_seconds`; `:317-330` per-phase; `:339-344` `unaccounted` residual | live histograms + `fak info --startup` report | pull-only on the guard's ephemeral port; **no scraper** on an attended run; **no budget** |
| `cmd/fak/guard.go:1092-1406` post-ready region: 7 hook/config installers (`:1240` PreCompact, `:1266` Stop, `:1293` toolproc, `:1318` SessionStart, `:1327` Codex, `:1343` Pi, `:1358` MCP) + startup-report render, then child spawn `:1418/1421` | the setup between "ready" and child exec | **entirely untimed**; each installer does `os.MkdirTemp` + JSON marshal/write, serially |
| `internal/usagelog` `Row.DurationMS` (`usagelog.go:90`) | one wall-clock row per top-level verb | **guard never records** — `os.Exit` bypasses `recordUsage` (documented gap `cmd/fak/usagelog_record.go:17-27`) |

### 3.2 Concurrency path
| Seam | What it gives us | What's missing |
|---|---|---|
| `cmd/fak/dispatch_tick.go:516-522` `dispatchStampMs`; `:549` `timings` map; `:623` `total`; `:1701-1713` `recordDispatchTickLoop` fold to loop-ledger `*_ms` | the sole stopwatch, 8 phases | the fold is **write-only** — grep for the `*_ms` names hits only the write site (`:1709`) |
| `cmd/fak/dispatch_tick.go:1704` | the comment naming `TICK_PHASE_REGRESSION` as the budget this *would* gate | the gate **does not exist** — no consumer, no p50/p99 fold |
| `cmd/fak/dispatch_tick_preflight.go:36-97` `dispatchPreflight` → process-table walks (`:1003-1051` `Get-Process`/`:1054-1086` `ps`, then a **second** `:1101-1124`) + `dispatchPreflightKernel:553-563` (`dos loop` subprocess) | the real per-tick cost | 2–3 full walks + a subprocess spawn, **all collapsed into one `preflight` bucket** |
| `acquireDispatchLaneLease` (`dispatch_tick.go` ~`986-1030`, via `internal/regionadmit`+`internal/leaseref`) | in-process lane arbitration | folded into the `spawn` bucket, **unattributed**; the collision-scan region `:603-715` (`dispatch_tick_livescan.go:92-133`) is **untimed entirely** |
| `AGENTIC-LOOP-KPIS-2026-06-25.md:164` | `spawn_admission_rate` = "NONE (per-decision verdicts exist; nothing aggregates) — proposed" | the admit/refuse-under-cap rate is **not aggregated** |

### 3.3 Reusable folds to build the plan on (already green)
| Seam | Reuse |
|---|---|
| `internal/turntaxmeter/hooklat.go` `FoldHookLatency` → p50/p90/p99 + `DefaultHookP99BudgetMS=250` + closed token `GateLatencyRegression` | **point it at the dispatch `timings_ms` stream and the usagelog `DurationMS` stream** — the percentile+budget machinery already exists |
| `internal/turntaxmeter/overheadbudget.go` `Span.ElapsedNS` + `Budget.MaxNS` + closed token `OverheadBudgetExceeded` | the declared-envelope shape for a first-launch / per-phase budget |
| `internal/gateway/metrics_render.go` / `metrics_http.go` (`withMetrics`) | the `/metrics` histogram + timing-middleware template for a new `fak_hotpath_*` family |
| `internal/bench` `MeasureSpawnedBaseline` (`bench.go:136`) — spawns + wall-clocks the real binary | the harness to wall-clock a real `fak manage`/`fak dispatch` invocation for the 5× before/after |
| `internal/usagelog/fold.go` (`FoldRows`, p50 only at ms res) | enrich to p90/p99 — the `DurationMS` data is already on disk |

## 4. The maturity ladder (the plan)

Each rung is a cluster of tickets; a rung is "done" when it has emission, a fold, and (where it
gates) a witness. Mirrors #1147's L0→Ln shape so this slots into the epic family.

- **L0 — Emit the wall-clock, end to end.** *(T1–T3)*
  - **T1 · First-launch: extend the timeline past `MarkReady`.** Add `StartupPhase`s for the 7
    post-ready installers + MCP registration + a child `spawn→first-usable` span; close the usagelog
    `os.Exit` gap so a guard run records one `DurationMS` row. *Witness:* a test asserts non-zero,
    correctly-attributed spans for the installer tail **and** the child-handshake span; a guard run
    appears in `fak usage`.
  - **T2 · Dispatch: sub-attribute the fat buckets.** Split `preflight` into per-process-table-walk
    + `dos loop` + backpressure sub-spans; add a bucket for the untimed collision-scan region
    (`:603-715`); split the lease-arbitration acquire out of `spawn`. *Witness:* the sub-buckets sum
    to their parent within tolerance on a fixture tick.
  - **T3 · Aggregate `spawn_admission_rate`.** Fold the already-present per-tick `preflight_live` /
    `preflight_cap` verdict into an admit-vs-refuse-under-cap rate. *Witness:* a fixture stream reads
    back the rate under injected cap pressure.
- **L1 — Fold to percentiles.** *(T4)* Build the **consumer** the dispatch fold never had: point
  `hooklat.FoldHookLatency` at the loop-ledger `*_ms` stream (p50/p90/p99 per phase); enrich the
  usagelog fold to p90/p99. *Witness:* golden fixture → golden percentile table.
- **L2 — Budget + ratchet (the SLO).** *(T5)* Declare a `time_to_ready` envelope and a per-dispatch-
  phase envelope (reusing `overheadbudget`); wire the already-named `TICK_PHASE_REGRESSION` and a
  first-launch breach token. *Witness:* a synthetic over-budget span reds; a noise-only run does not.
- **L3 — One read-out.** *(T6)* A `fak perf --hot-paths` fold + a `fak_hotpath_*` `/metrics` family
  over first-launch + dispatch phases (the lifecycle-boundary twin of #1147 T11's `fak perf`).
  *Witness:* verb output golden-tested; family reads back.
- **L4 — Drive the 5× (with the meter now honest).** *(T7–T8)* Attack the dominating stages L0–L1
  expose — §5. *Witness:* a `MeasureSpawnedBaseline` before/after shows ≥5× on the named stage, net,
  WITNESSED, with a committed reproduce command.
- **X — Honesty fences.** *(T9)* Provenance-label every number; the doc stating the 5× scope fence
  (§6). *Witness:* the doc + the label test ship.

## 5. The 5× targets (concrete levers, most-called first)

The meter (L0–L1) exists to make these honest; the levers are already visible from the seams.

**Concurrency — the highest-call-count stage (runs every tick, every dispatcher):**
1. **Collapse the 2–3 per-tick process-table walks into one shared snapshot.** `dispatchPreflight`
   walks the full table via `Get-Process`/`ps` for the host check, **again** for RAM/threads, and
   again for worker-count (`dispatch_tick_preflight.go:1003-1124`, `:1433-1495`). One snapshot per
   tick, read by all probes. This is the single fattest, most-repeated cost — the plausible 5× on
   its own.
2. **Cache the `dos loop` kernel probe across ticks** (`dispatchPreflightKernel:553-563`) — kernel
   state changes slowly; a per-tick subprocess spawn is pure tax.
3. **Make the collision-scan incremental** (`dispatch_tick_livescan.go:92-133` re-walks the runs
   dir every tick).
   → Faster ticks cut the fixed per-spawn tax that competes with useful work, letting the cap rise
   *safely* — the frame [`SAFE-CONCURRENCY-HEADROOM`](SAFE-CONCURRENCY-HEADROOM-2026-07-01.md) sets
   (on this box concurrency is seat-bound, not hardware-bound, so the win is *cheaper spawns*, not
   more threads).

**First launch — runs once per session, but per-worker across the fleet:**
1. **Parallelize / batch the 7 post-ready installers** (`guard.go:1240-1358`) — each is a serial
   `os.MkdirTemp` + JSON marshal/write; batch into one temp root + one write pass, or run them
   concurrently.
2. **Defer/lazy MCP registration** (`guard.go:1358`) off the critical path to child-exec.
3. **Cache the capability-floor policy digest** (`guard_startup.go:168-208`) and cut the
   `upstream-resolve` disk-token reads (`guard.go:625-749`) to a single pass.

## 6. Honest fences (the scope of "5×")

- **5× is on fak's own controllable wall-clock** — the installer tail, the process-table walks, the
  kernel-probe caching — measured **net** and **WITNESSED** via `MeasureSpawnedBaseline`.
- **It is NOT** a claim about the `claude` child's Node cold-start + MCP handshake (external to fak
  — we *instrument* it as **OBSERVED** so we can prove where the wall-clock goes, but we do not own
  its speed), nor about raw fleet concurrency (seat/provider-bound, per §5's frame).
- **A budget is an envelope with a stated scope, not zero cost.** A tick that costs 8% and saves a
  spawn slot is a net win; the gate reds on a *persistent* over-budget breach, never on the
  existence of overhead (the #1147 §6 fence, inherited).
- Every emitted number carries WITNESSED (fak's own spans) / OBSERVED (child/provider-relayed) /
  MODELED (projected) — the two are never summed.

## 7. Definition of Done (each item WITNESSED, no self-report)

1. First-launch timeline extends through the post-ready installer tail **and** a child
   `spawn→first-usable` span; a guard run records a usagelog `DurationMS` row. *(T1)*
2. Dispatch `preflight`/`spawn` are sub-attributed; the sub-buckets sum to their parent on a
   fixture tick; the collision-scan region is timed. *(T2)*
3. `spawn_admission_rate` reads back under injected cap pressure. *(T3)*
4. The dispatch `timings_ms` stream has a **consumer** producing p50/p90/p99 per phase; usagelog
   folds p90/p99. *(T4)*
5. `time_to_ready` and per-dispatch-phase budgets are declared; a synthetic over-budget reds and a
   noise-only run does not (no false red). *(T5)*
6. `fak perf --hot-paths` is golden-tested and the `fak_hotpath_*` family reads back. *(T6)*
7. A before/after `MeasureSpawnedBaseline` shows **≥5×** on the shared-process-table-snapshot stage
   (the most-called), net, WITNESSED, with a committed reproduce command. *(T7)*
8. Every number is provenance-labeled; the scope-fence doc (§6) ships. *(T9)*

## 8. Sequencing

T1/T2/T3 (emit) unlock everything — no wall-clock ⇒ no fold ⇒ no gate. T4 (fold) then T5 (budget)
are the first "is a launch slower than budget?" witness. T7/T8 (the 5×) can begin the moment T1/T2
expose the dominating stage — the process-table-snapshot collapse needs only T2's sub-attribution to
prove its before/after. The honest minimum viable slice is **T1 + T2 + T4 + T7**: both boundaries
emitted, folded to percentiles, and one witnessed 5× on the most-called stage.

## 9. Filing state / next action

This note is the docs-lane increment: it **pins the score, the substrate, the ladder, and the 5×
scope fence** so the build is wiring against a fixed contract. Lanes for the build: the meters + verb
are `cmd`/`gateway`/`metrics`/`turntaxmeter`; only this contract is `docs`.

**Filed 2026-07-10:**

| Issue | Ticket |
|---|---|
| [#4254](https://github.com/anthony-chaudhary/fak/issues/4254) | epic — the hot-path wall-clock plane |
| [#4255](https://github.com/anthony-chaudhary/fak/issues/4255) | T1 · L0 first-launch emit (past `MarkReady` + child span + usagelog gap) |
| [#4256](https://github.com/anthony-chaudhary/fak/issues/4256) | T2 · L0 dispatch sub-attribution |
| [#4257](https://github.com/anthony-chaudhary/fak/issues/4257) | T4 · L1 fold to p50/p90/p99 |
| [#4258](https://github.com/anthony-chaudhary/fak/issues/4258) | T7 · L4 the 5× (one shared process-table snapshot per tick) |

The MVP slice (T1+T2+T4+T7) is filed; the remaining ladder rungs (T3, T5, T6, T8, T9) are enumerated
in the epic checklist and filed as the plane advances.
