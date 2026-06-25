---
title: "Internal benchmark KPIs, from the agent's point of view"
description: "An inventory of the KPIs fak measures today, organized by the five-layer loop ladder, plus ~30 new agentic lifecycle-latency KPIs (time to spawn a turn, reap a turn, query the planner, checkpoint a session, spawn a fan-out wave) — each grounded against a real code seam with a status tag and an honesty fence."
---

# Internal benchmark KPIs, from the agent's point of view

_Generated 2026-06-25. Inventory of `internal/metrics`, `internal/bench`, `internal/turnbench`, `cmd/{radixbench,fanbench,fleetbench,sessionbench,ctxdemo,ctxplanbench}`, cross-checked against the loop-ladder primitives in [`engineering-is-building-loops`](../explainers/engineering-is-building-loops.md). The new KPIs are **proposals with measurement recipes**, not claimed numbers — every one names the real seam it would read and the fence that keeps it honest._

## Why this note

fak measures two things well, and one thing barely at all.

It measures **work eliminated** — turns saved, vDSO hit rate, cross-agent uplift, cache hit rate, tokens-per-task, the session value stack. And it measures **the cost of one decision** — the in-process `Decide` p50 against a spawned-hook baseline (the ~2,849× boundary-tax sentinel). Both are mature, both are honest about measured-vs-modeled, both trace to `BENCHMARK-AUTHORITY.md`.

What it almost never measures is the **lifecycle latency of the loop primitives themselves** — the numbers an agent author actually budgets an unattended run against. How long to *spawn* a turn (assemble the context, plan the view)? How long to *reap* one (gate the tool results, fold them back)? How long does the *planner* take to choose the resident window? How long to *checkpoint* a session, or *spawn* a fan-out wave of N sub-agents? These are the agent's-eye-view questions, and today the catalog is nearly silent on them.

This note inventories the current KPIs by loop-ladder layer, then proposes a new set that fills the lifecycle-latency gap. The new set is ~30 KPIs against a current headline set of ~14 — it more than doubles the catalog, which is the "2× more" ask read literally.

## The ladder, and the status legend

Every KPI is tagged by the loop-ladder layer it lives at (`engineering-is-building-loops.md`): **syscall** (one tool call) → **turn** (one agent step) → **session** (many turns) → **fleet** (many sessions) → **rsi** (the loop that improves the loop), plus an orthogonal **streaming** axis (perceived latency) that cuts across the call.

A status tag, used throughout:

| status | meaning |
|---|---|
| **standing** | wired and emitted on a live or replay run today |
| **modeled** | a transparent, knobbed cost model (the *price* of a turn) — never measured |
| **guarded** | the seam exists, but off-by-default / replay-only / not aggregated into a rate |
| **frozen-sync** | the seam is frozen in the ABI but runs synchronously today |
| **proposed** | nothing records it yet; the measure-point exists, but no counter aggregates it |

Four traps run through the whole exercise, and they are the reason most of the new KPIs are *not* "standing":

1. **Measured ≠ modeled.** Turn-tax dollars, fan-out fold cost, and break-even are transparent cost models with knobs, not wall-clocks. Keep them on their own axis.
2. **Latency ≠ work-count.** The planner reports candidate-scoring *op counts* (`Θ(N²)` vs `Θ(c·N)`), never a wall-clock. Ops are not nanoseconds.
3. **Live-serving ≠ replay/synthetic.** The context planner (`FAK_CTXPLAN_SEAM`) and the in-kernel KV gate (`kvmmu`) are off the live HTTP loop; their numbers come from transcript replay or a synthetic model.
4. **The async seam is frozen-but-sync.** `Reap` blocks inline on `engine.Complete` (no goroutine). A naive reap timer is dominated by the engine and tells you nothing about the kernel. The kernel slice must be isolated (a `Mock` engine, or the existing admit-chain bench), or the KPI just re-measures the model.

---

## Part 1 — what fak measures today

The headline current KPIs, by layer. This is the operationally meaningful set, not every JSON field (the full per-surface field list runs to ~80 counters across `metrics`, `turnbench`, `radixbench`, `fanbench`, `sessionbench`).

### syscall

| KPI | measures | units | seam | status |
|---|---|---|---|---|
| `tool_call_p50_ns` / `p99_ns` | in-process adjudication (`Decide` fold) latency | ns | `metrics.KPIs.ToolCallP50Ns` ← `bench.RunArm` | standing |
| `vdso_hit_rate` | calls served from the local content-cache (`VDSOHits/Calls`) | ratio | `metrics.KPIs.VDSOHitRate` | standing |
| `gate_primary` | sentinel: in-process p50 must beat the spawned-hook baseline (boundary tax) | bool | `metrics.Report.ComputeGate` | guarded |
| `spawned_hook_baseline_p50_ns` | process-per-decide round-trip floor (this machine) | ns | `metrics.Baseline.P50Ns` ← `bench.MeasureSpawnedBaseline` | standing |
| `local_serve_ns` | the µs cost of the "1-shot" `Decide` (turn-tax harness) | ns | `turnbench.Report.LocalServeNs` | standing |
| `denies` / `transforms` / `engine_calls` | kernel verdict tallies on a replayed trace | count | `kernel.Counters` ← `turnbench.replay` | standing |

**Gap:** only `Decide` is timed. `Submit` (the full call boundary the agent waits on) and the kernel slice of `Reap` (admission) have no standing latency. And `preflight_catch_rate` is **declared in the `KPIs` struct but never populated by `bench.Run`** — a wired-but-dead field worth either computing or deleting.

### turn

| KPI | measures | units | seam | status |
|---|---|---|---|---|
| `turns_saved` (forced / elision) | model round-trips deleted, split by kind | count | `turnbench.ClassBreakdown.turnsSaved` | standing |
| `context_pollution_rate` | results quarantined by the ctx-MMU (`Quarantines/Calls`) | ratio | `metrics.KPIs.ContextPollutionRate` | standing |
| `safety_floor.*` | poison admitted / destructive executed, baseline vs fak | count | `turnbench.SafetyFloor` | standing |
| `tokens_saved` / `dollars_saved` / `latency_saved_ms` | `turns_saved` priced through the cost model | tok / $ / ms | `turnbench.netFor` | modeled |
| `candidate_scorings` (planner-cpu) | per-turn planner work, `Θ(N²)` vs index-bounded `Θ(c·N)` | op count | `ctxplan.cumPlannerCompute` | guarded |
| `vdso_off_net` | the real ON/OFF path-swap ablation | count | `turnbench.Run` (vDSO off) | standing |

**Gap:** the planner reports *work* (op counts), never *time*. There is no `turn_spawn`, no `turn_reap`, no `planner_optimize` latency, no demand-page miss rate.

### session

| KPI | measures | units | seam | status |
|---|---|---|---|---|
| `net_value_add_vs_naive` (A/C) | fak vs cold re-prefill (worst-case reference) | ratio | `sessionbench cell.NetVsNaive` | standing (live arm) |
| `net_value_add_vs_tuned` (B/C) | fak vs warm per-agent KV (the honest serving baseline) | ratio | `sessionbench cell.NetVsTuned` | standing (live arm) |
| `prefill_tokens.a_over_c` / `b_over_c` | exact, contention-free prefill work-elimination | ratio | `sessionbench.prefillTokens` | standing |
| `turn_tax_A_over_B` | KV-persistence value (re-prefill vs warm) | ratio | `sessionbench cell.TurnTax` | standing |
| `provider_cache_hits` / `read_tokens` | remote provider prefix-cache wins, kept distinct from local | count | `metrics.Arm.ProviderCacheHits` (#112) | guarded |
| `cache_hit_rate` (radix) | prompt tokens served from the radix cache | % | `radixbench analyze` ← `radixkv.Tree.MatchLen` | standing |

**Gap:** the portable session image (`internal/sessionimage`, shipped 2026-06-24) has **no dump/restore latency**, no resume-warmth, no promotion-rate, no integrity-verify timing.

### fleet

| KPI | measures | units | seam | status |
|---|---|---|---|---|
| `cross_uplift` | turns only the shared fleet buys vs isolated worlds | count | `turnbench.FleetCell.CrossUplift` | standing |
| `prefix_tokens_saved` | exact `(N−1)·prefix` prefill geometry the clone skips | tok | `turnbench.FanoutCell.PrefixTokensSaved` | standing |
| `tax_clawed_back_frac` | fraction of the multi-agent token tax the prefix lever removes | ratio | `fanout.project` | modeled |
| `parallel_speedup` / `fold_turns` | depth-latency vs total-work, plus the coordination tax | ratio | `fanout.project` | modeled |
| `model_calls_spent` | replay coverage at zero model cost (always 0) | count | `fleetcounterfactual.ModelCallsSpent` | standing |

**Gap:** the dispatch loop's `closure_rate` is computed by `issue_closure_audit.py` but is not surfaced as a standing fleet KPI. No spawn latency, no preflight latency, no arbitration latency.

### rsi

Nothing standing. The loop is deliberately wall-clock-free and RNG-free so it reproduces bit-for-bit, so it captures *no* timing and exposes no kept-rate as a metric — only per-cycle `Row.Kept` bits in the journal. This is the emptiest layer, and the cheapest to fill (the bits already exist; nothing aggregates them).

---

## Part 2 — the new agentic KPIs (≈2× more)

Today's KPIs answer *"how much work did the kernel delete, and how cheap is one decision?"* The new set answers the agent author's question: *"how long do the loop's own primitives take, how often do they hit their fast path, and how do they fail?"* Every row below is grounded against a real seam; the fence is the one-line reason it can't overclaim.

### syscall — the call boundary, decomposed

| KPI | agentic question | seam | status | fence |
|---|---|---|---|---|
| **submit_latency** | time at the boundary for a verdict **+** dispatchable handle (full `Submit`, not just the fold) | `kernel.Submit` | live | superset of `Decide`, but the extra (mutex seq-bump + empty FastPath sweep) is sub-µs — report as a decomposition, not a new headline |
| **reap_latency_kernel** | of the result-return time, how much is kernel admission vs the engine round-trip | `kernel.Reap` | frozen-sync | `Reap` blocks inline on `engine.Complete`; measure with a `Mock` no-op engine or the residual is pure engine, not kernel |
| **resolver_put_get_ns** | cost of one content-addressed `Put`+`Resolve` when args/results page through the CAS | `abi.Resolver` ← `internal/blob/store.go` | live | paid only on TRANSFORM / page-out / quarantine, **not** the inline fast path — a per-rewrite cost, not a per-call tax |
| **transform_repair_ns** | cost to repair a misrouted call (synonym arg → canonical) in-syscall instead of burning a turn | `grammar.Rung.Adjudicate` | live | one rank-5 rung folded inside `Decide` — already inside `submit_latency`; dominated by the `putJSON` re-store |
| **admit_chain_ns** (benign / secret / injection) | what the result-side ctx-MMU gate costs per result, by payload class | `ctxmmu.BenchmarkAdmitChain*` (gate: `MMU.Admit`) | live | the 29–87 µs figure is one M3 Pro run; making it "standing" means actually wiring the CI ratchet + adding the secret arm |
| **vdso_hit_latency** | how much faster the fast-path hit is than full adjudicate+dispatch, and its rate | `kernel.Submit` (vDSO branch) + `Counters` | live | rate is free (`VDSOHits/Submits`); latency needs the existing on/off ablation; both are workload-specific (name the trace) |
| **verdict_reason_distribution** | the refusal fingerprint: which of the 12 closed reason codes fired, how often | NONE (Reason field exists; no per-reason counter) | proposed | buildable today via an `EventSubscriber` on `EvDeny`/`EvQuarantine`; out-of-tree reasons need a `REASON_<n>` fallback bucket |

### turn — spawn, plan, reap (the user's headline layer)

| KPI | agentic question | seam | status | fence |
|---|---|---|---|---|
| **turn_spawn_ns** | time to spawn one turn: author the forecast, plan the O(1) view, render the next-turn history | `agent.CtxViewPlanner.RenderTurn` | guarded | seam is OFF by default (`FAK_CTXPLAN_SEAM`) — a transcript-replay number, not a live HTTP turn |
| **planner_optimize_ns** | "query/planner time" — the per-turn bounded knapsack that picks the resident view | `ctxplan.Optimize` | guarded | today only op-*counts* exist, never a wall-clock; objective changes the curve (greedy `n log n` / coverage `n²` / DP `n·budget`) |
| **candidate_scorings_per_turn** | is planning in the `Θ(c·N)` index-bounded regime or unbounded `Θ(N²)` | `ctxplan.cumPlannerCompute` | live | a modeled closed-form `N(N+1)/2`, not an instrumented runtime counter; the cost "O(1) resident" explicitly does not bound |
| **demand_page_miss_rate** | how often the forecast guesses wrong and must page an elided span back in | `ctxplan.DemandPage` (`Fault.Status`) | guarded | `DemandPage` returns a per-call `Fault`; nothing keeps a running rate — the caller aggregates over a replay |
| **page_fault_latency_ns** | when the forecast misses, the cost to page that span back into the resident view | `ctxplan.DemandPage` | guarded | the *token* cost (`Fault.Tokens`) is reported; the wall-clock is not, and is store-backend-specific |
| **turn_reap_ns** | time to reap a turn's tool results: write-time admit gate + fold survivors into the next window | `ctxmmu.MMU.Admit` (+ `kvmmu.AdmitResult`) | live (text) | `ctxmmu.Admit` is live; `kvmmu` is synthetic-only, not on the live HTTP loop — body-size dependent (regex scan) |
| **context_view_compression** | how much smaller the resident window is than the lossless history it can still recover | `ctxplan.Witness.ResidentTokens` / `cumLinear` | guarded | lossless-recoverable compression (page-back-in handle) — not comparable to lossy summarization ratios |
| **kv_evict_ns** | cost to evict a quarantined span's K/V and re-RoPE the survivors | `kvmmu.Context.evict` | guarded | bit-exact on a synthetic model only, not the live loop; cost scales with survivor positions re-RoPEd (early eviction costs more) |
| **screen_quarantine_rate** | fraction of reaped results the gate holds out (poison/secret) vs admits | `ctxmmu.MMU.PollutionRate` | live | live counter, but it measures the *decision* rate, not detection *quality* — the detector has a known evadable ceiling |

### streaming — perceived latency (orthogonal axis, newly unblocked)

Issue #47 ([`TOKEN-STREAMING-TTFT`](TOKEN-STREAMING-TTFT-2026-06-25.md)) changed the streaming contract so safe prose streams while the turn is still decoding, with tool-call bytes held behind `k.Decide`. That note's closing line is the opening for three KPIs: *"making TTFT/TPOT/ITL measurable for the parts of a turn that are safe to expose incrementally."*

| KPI | agentic question | seam | status | fence |
|---|---|---|---|---|
| **TTFT** (time-to-first-token) | how long until the agent sees the first safe content fragment | `agent.StreamingPlanner` (OpenAI `HTTPPlanner` / Anthropic passthrough) | proposed | only the *prose* prefix streams early; a pure tool-call turn has no safe prose, so its first output is still gated — report per turn-shape, not one number |
| **TPOT** (time-per-output-token) | the steady-state inter-content-delta cadence | same | proposed | measured on prose deltas only; tool bytes are held, so TPOT is a prose-stream metric, not a whole-turn one |
| **ITL** (inter-token latency) | jitter between streamed fragments (tail behavior) | same | proposed | non-streaming planners (mock, non-`StreamingPlanner` engines) fall back to buffered — undefined there, not zero |

### session — checkpoint, fork, resume

| KPI | agentic question | seam | status | fence |
|---|---|---|---|---|
| **session_dump_ns / restore_ns** | time to checkpoint a session to a portable image, and fork/resume it back | `sessionimage.DumpDir` / `LoadDir` + `Rehydrate` | guarded | an I/O+hash benchmark, not model latency; excludes the KV cache (`KVIncluded=false`) so resume re-prefills — logical-state restore, not warm-inference-ready |
| **session_resume_warmth** | on reload, what fraction of the working set is reusable without recompute | `recall.LoadOrAttachIndex` (warm vs rebuild) | guarded | two distinct notions: ctxplan-index warmth is real (saves a *build*); KV/prefill warmth is structurally **zero** by design — don't conflate |
| **promotion_rate** | fraction of benign results that earned the durable core image vs default-expire | `recall.RefusedPromotions` + `Page.Durability` | guarded | denominator must be counted at record-time (Enforce drops refused pages from the manifest, so a manifest-only rate reads as a false 100%); default posture is audit-only |
| **quarantine_survives_boundary** | does a sealed slice stay sealed after dump → ship → reload under a new model | `recall.Session.Resolve` + `reScreen` | guarded | a witness/invariant, not a scalar; re-screen runs on page-**in**, lazily — claim "re-screened on page-in", not "on reload" |
| **core_image_size / pageout_ratio** | the durable footprint, and how much lives paged-out in the CAS swap | `recall.Stats.CASBytes` + `sessionimage.Meta.Parts` | guarded | pageout_ratio is near-1 by construction (the page table holds no heavy bytes) — an architecture invariant, not a tunable win |
| **image_integrity_verify_ns / tamper_fail_closed** | time to prove a received image is whole, and does a tampered one fail closed | `sessionimage.verifyParts` + `recall.Load` + `snapshot.Parse` | guarded | scales linearly with CAS bytes, paid on every load; proves byte-integrity, **not** semantic trust (an intact poisoned image is still poisoned) |

### fleet — spawn a wave, gate a spawn, close on evidence

| KPI | agentic question | seam | status | fence |
|---|---|---|---|---|
| **fanout_spawn_ns** | time to spawn N sub-agents off one master goal (per-agent clone cost) | `model.NewBatchFromPrefixReserve`; timed at `sessionbench.liveC` (`cloneMS`) | live | the in-kernel KV-clone cost, **not** launching a real `claude -p` OS process — fak's cross-agent-reuse spawn, bit-identical to a self-prefill |
| **fold_join_ns** | time for the orchestrator to collect and merge N sub-results | `turnbench.fanout` (`foldTurns`) | modeled | a cost-model term (token/turn arithmetic × knobbed latency), the modeled half of fanbench — no live timer around a real fold |
| **arbitration_ns** | the fleet "decide" gate: file-tree collision check + lease admission before a spawn | `dos_arbitrate` (DOS kernel) + `turnbench.toposearch` (collision count) | live | the arbiter lives in the DOS kernel, not fak; the decision is fast — the meaningful KPI is the *collision count* it produces, not its ns |
| **dispatch_preflight_ns** | how long the spawn-gate takes to answer "safe to launch another worker now?" | `dispatch_preflight.py:evaluate` | live | I/O-bound (3 child processes + a process-table walk), paid once per spawn not per call; a failing sub-check both inflates latency and refuses by design |
| **closure_rate** | fraction of resolved work independently witnessed-closed vs merely claimed | `issue_closure_audit.py` (+ `issue_resolve_witnessed.py` per-SHA re-verify) | live | a git fact re-asked at close time, never a worker self-report; requires origin/main reachability or a shared-tree race over-counts |
| **spawn_admission_rate** | under cap pressure, fraction of attempted spawns actually admitted | NONE (per-decision verdicts exist; nothing aggregates) | proposed | a low rate is healthy back-pressure, not failure — read it alongside the refuse-reason histogram |
| **prefix_tokens_saved** | prefill positions saving N siblings vs a no-share engine | `turnbench.FanoutCell.PrefixTokensSaved` | live | exact geometry over the shipped clone, credited only up to the witnessed frontier width — refuses to extrapolate past a recorded N |

### rsi — the loop that improves the loop

| KPI | agentic question | seam | status | fence |
|---|---|---|---|---|
| **keep_rate** | of proposed improvements, what fraction the non-forgeable keep-bit actually kept | `rsiloop.Result.Kept / Cycles` | live | already accumulated — just needs surfacing; a low rate can be the gate working (reverting a non-improver), so it's only meaningful vs a fixed proposer + real headroom |
| **suite_green_rate / truth_clean_rate** | how often the two non-metric witnesses passed (correct & well-formed, apart from faster) | `rsiloop.worktree.runSuite` / `treeChangedOnly` | live | on Windows the suite is `go build`+`go vet`, a compile proxy not `go test`; measure-error candidates are force-recorded red |
| **escalate_after_k** | after how many consecutive non-keeps the loop hands off to a human | `shipgate.Gate` (K, `ConsecutiveNonKeeps`) | live | K is a configured knob; the KPI is the *counter* (how often the breaker trips, at what run-length), not K |
| **vs_main_freshness** | was the kept gain measured against the latest main, or a stale local baseline | `rsiloop.Row.BaselineRef` + `RefName` | live | the SHA pin guarantees one run's internal consistency, not that the operator re-ran against newest main |
| **rsi_cycle_ns** | propose→verify→keep wall-clock, to budget attempts per unattended window | NONE (loop captures no clock, by design) | proposed | wall-clock and platform-dependent — the opposite class from the bit-reproducible keep-bit; report as telemetry, **never** fold into a keep decision |
| **regression_catch_latency** | from a regressing change landing to a RED gate catching it | NONE (detection exists; no timestamp in the journal) | proposed | the detection mechanism is CI-gated per-push (`rsiloop -mode track`), but no clock is journaled, and it only guards the loop's own LRU KPI today |

---

## Part 3 — the shortlist (cheapest to wire, highest value)

Not all ~30 cost the same. Six are almost free — a live counter exists and only needs surfacing or a CI ratchet:

1. **keep_rate** — `Result.Kept / Result.Cycles` is already accumulated in `RunObserved`. Surface it.
2. **closure_rate** — `issue_closure_audit.py` already computes it. Sample it each loop as a standing fleet KPI.
3. **screen_quarantine_rate** — `ctxmmu.PollutionRate()` is a live atomic counter. Read it after a run.
4. **candidate_scorings_per_turn** — `cumPlannerCompute` is already in `Comparison.Table()`. Promote the column to a KPI.
5. **admit_chain_ns** — the bench exists (`BenchmarkAdmitChain{Benign,Poison}`); wire the CI ratchet against the committed M3 baseline and add the secret arm.
6. **prefix_tokens_saved** — exact geometry, already witnessed bit-identical. Already standing; just index it in the authority.

Three are the high-value lifecycle latencies that need real wiring — and they are exactly the user's examples:

- **planner_optimize_ns / turn_spawn_ns** — a new `ctxplan` latency bench over the heaviest transcripts (the `ctxplanbench` corpus already exists; it measures *work*, add *time*). This is the "query/planner time" and "time to spawn a turn" the ask names.
- **reap_latency_kernel** — a `Mock`-engine `Reap` bench to isolate the admission slice from the engine round-trip (without it, the frozen-sync seam just re-measures the model).
- **TTFT / TPOT / ITL** — gateway-side stream timers on `agent.StreamingPlanner`, now that #47 made the prose prefix safe to expose incrementally.

And one cleanup that falls out of the inventory: `preflight_catch_rate` is a `KPIs` field that `bench.Run` never populates. Compute it (it has a natural source: deny-before-dispatch over total) or delete it, so the struct stops advertising a number it doesn't produce.

## Honest fences, in one place

- **Measured vs modeled stays split.** The cost-model KPIs (`tokens_saved`, `fold_join_ns`, break-even, fan-out economics) are knobbed projections, never wall-clocks. A latency KPI that quietly inherits a modeled term is a category error.
- **A latency is not an op-count.** The planner's `Θ(N²)→Θ(c·N)` story is real and measured *as work*; turning it into nanoseconds is a new benchmark, not a relabel.
- **Most turn/session/KV latencies are replay or synthetic, not live-serving.** `FAK_CTXPLAN_SEAM` is off by default and `kvmmu` is not on the HTTP loop, so those figures must be labeled replay/synthetic — the same discipline the `ctxplan` notes already use.
- **The async seam is frozen-but-sync.** Any reap/dispatch latency must isolate the kernel slice (Mock engine) or it measures the engine. Do not claim a non-blocking reap.
- **Rates need their reason histograms.** `spawn_admission_rate`, `screen_quarantine_rate`, and `keep_rate` all read as "broken" when low but are often the system correctly throttling, gating, or reverting. Pair each with its refuse/deny breakdown.

## Where the seams live

```
syscall   kernel.Submit / Reap / Decide        internal/kernel/kernel.go
          adjudicator chain                    internal/adjudicator, internal/grammar
          result admission gate                internal/ctxmmu/mmu.go (Admit, PollutionRate)
          content-addressed resolver           internal/abi (Resolver) / internal/blob
turn      context planner (knapsack)           internal/ctxplan (Optimize, DemandPage, scaling.go)
          live planner seam (guarded)          internal/agent/ctxplan_seam.go (FAK_CTXPLAN_SEAM)
          in-kernel KV gate (synthetic)        internal/kvmmu/kvmmu.go
stream    streaming planner                    internal/agent (StreamingPlanner, HTTPPlanner)
session   portable image dump/restore          internal/sessionimage, internal/snapshot
          durable core image                   internal/recall/recall.go (Durability, RefusedPromotions)
fleet     KV clone / fan-out spawn             internal/model (NewBatchFromPrefixReserve)
          fan-out cost model                   internal/turnbench/fanout.go
          spawn gate / witnessed close         tools/dispatch_preflight.py, tools/issue_resolve_witnessed.py
          arbitration (DOS kernel)             dos_arbitrate ; mirror at internal/turnbench/toposearch.go
rsi       keep-bit / breaker                   internal/shipgate/shipgate.go (Evaluate, Gate)
          the loop                             internal/rsiloop (rsiloop.go, worktree.go)
```

## Reproduce / read next

- The current numbers: [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) (the single source of truth) and the five-demo tour in [`fleet-benchmarks`](../explainers/fleet-benchmarks.md).
- The ladder these layers come from: [`engineering-is-building-loops`](../explainers/engineering-is-building-loops.md).
- The two siblings this note extends: [`CTXPLAN-PLANNING-COST-FLATTEN`](CTXPLAN-PLANNING-COST-FLATTEN-2026-06-23.md) (planner work, not time) and [`TOKEN-STREAMING-TTFT`](TOKEN-STREAMING-TTFT-2026-06-25.md) (the streaming seam that unblocks TTFT/TPOT/ITL).
