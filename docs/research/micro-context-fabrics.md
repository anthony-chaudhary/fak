---
title: "Micro-context fabrics for 100–10,000 parallel agents"
description: "Dedicated research focus for splitting one cached agent base into many bounded logical contexts, with controlled-kernel and API-only paths."
status: active
last_reviewed: 2026-08-12
---

# Micro-context fabrics: one cached base, 10,000 useful agent contexts

## Verdict and first witness

This is a new, high-priority research focus for **micro contexts**: treat the reusable,
cacheable setup as an immutable agent **base**, then run 100, 1,000, or 10,000 small,
independently scheduled delta contexts over it. The target is not “spawn 10,000 OS
processes.” It is to make 10,000 logical context windows cheap enough to schedule,
observe, pause, resume, and fold while physical model slots remain bounded.

The first spine is runnable now:

```bash
go run ./cmd/microcontextdemo -selfcheck -contexts 10000 -workers 64
```

It proves 10,000 isolated logical contexts retire through 64 bounded workers, with one
shared base installed at the gateway and one delta planner call per context. Its JSON
explicitly says `synthetic planner`: it is a harness/concurrency witness, **not** a model
throughput or cache-hit claim. The next model-backed spine must preserve that distinction.

## End-state dimensions

The program succeeds when it can flex along either or both dimensions:

1. **Usable → highly performant:** a model already capable of interactive single-stream
   work gains substantially higher aggregate useful tokens/sec by amortizing prefix
   prefill, batching compatible deltas, and keeping model slots saturated.
2. **Not usable alone → usable as a fabric:** a slow single stream (including an
   offloaded or otherwise latency-bound model) becomes useful for a user, team, or service
   because many independent tasks make aggregate progress concurrently. Per-stream latency
   is reported honestly; aggregate throughput must not disguise an unusable critical path.

At scale, the unit is a logical context descriptor, not a full harness replica:

```text
base_id + task_delta + capability_set + budget + continuation + output_contract
```

The kernel owns admission, scheduling, shared-prefix identity, tool policy, journaling,
and result folding. Model serving owns prefill/decode capacity. A full agent harness is
optional and should be paid for only when a task needs its UI/session semantics.

### General large-input operator

The same substrate can be more than a high-agent-count runtime. The proposed
[large-input operator contract](micro-context-large-input-operators.md) partitions a large
artifact into stable records, runs deterministic filters before semantic work, uses bounded
micro-contexts to select or execute filters and tool calls, emits typed cacheable facts, and
folds them hierarchically with provenance and safe cancellation. That is the general-purpose
claim to test: not "parallelize every prompt," but make micro-context execution one selectable
backend for decomposable large-input work alongside SQL/search, retrieval, compression, coarse
chunks, and tuned long context. The operator contract and this execution fabric are separate
layers; neither is sufficient alone.

## Minimal-spine ladder

Each rung must run end to end before the next is treated as real.

| Rung | Working component | Required witness |
|---|---|---|
| S0 | Synthetic 10k logical contexts over bounded workers (current) | exact completion count, one base install, peak concurrency ≤ worker cap |
| S1 | **Observed:** one real OpenAI-compatible endpoint, 100 delta contexts, no tools | [100/100 captured witness](micro-context-s1-real-endpoint.md): wall time, TTFT, usage-derived token rates, failures |
| S2 | **Observed:** shared-prefix A/B, unique full prompts vs one base + deltas | [No cache benefit on first endpoint](micro-context-s2-prefix-ab.md); scoped concurrency gain retained separately |
| S2b | Shared-prefix A/B on a cache-observable controlled kernel (#5817) | kernel cache hit/miss counters, prefill tokens saved, tuned baselines; supersedes the S2 cache verdict |
| S3 | **Observed:** 1k resumable contexts with bounded scheduler RAM and backpressure | [1,000-context hibernation witness](micro-context-s3-hibernation-restart.md): queue age, resident/hibernated counts, runtime reconstruction |
| S4a | **Observed:** versioned lightweight descriptor through existing Host/Gateway | [Harness inventory and 1,000-context adapter](micro-context-s4-lightweight-descriptor.md) |
| S4b | **Observed:** compatibility-class planner | [Mixed workload](micro-context-s4-compatibility-scheduler.md): isolation, aging, cancellation, padding/fill telemetry |
| S4 | **Observed fixture:** tool-capable microagents through capability/resource/idempotency/readback seams | [Parallel effect-safety witness](micro-context-s4-effect-safety.md) |
| S5a | 1,000 real-model-turn contexts on one controlled node (#5820) | ledger: TTFT/tail, prefill/decode tokens/sec, RAM/KV roofline, useful-result rate |
| S5 | 10k contexts under a controlled kernel | useful-result throughput, tail latency, memory/KV roofline, overload behavior |
| S6 | API-only adapter | provider-supported cache controls or measured natural prefix reuse; no kernel-only claims |
| S7 | Multi-user/fairness mode | tenant isolation, weighted fairness, cancellation, spend and rate-limit envelopes |
| S8 | **Observed fixture:** 1,000-record general large-input operator (#6029) | [partition/filter/map/cache/fold/oracle witness](micro-context-s8-large-input-operator.md) |
| S8a | **Observed fixture:** adaptive filter-stage selector (#6030) | [confusion/cost/cache witness](micro-context-s8a-filter-selector.md) |
| S8b | **Observed fixture:** bounded read-only tool enrichment (#6031) | [request/receipt/restart witness](micro-context-s8b-tool-enrichment.md) |
| S8c | **Observed fixture:** provenance-preserving hierarchical fold (#6032) | [fold-tree/property/invalidation witness](micro-context-s8c-provenance-fold.md) |
| S8d | **Simulated fixture:** tuned-baseline falsification spine (#6100; parent #6033 remains open) | [five-pipeline decision-boundary harness](micro-context-s8d-falsification-spine.md) |
| S8e | **Observed controlled fixture:** effectful stages bound to witnessed receipts (#6034) | [effect journal/read-back state machine](micro-context-s8e-effect-receipts.md) |

Promotion requires a captured artifact at every rung. A synthetic rate is never promoted
as inference throughput. An inference token rate is never promoted as useful agent work.

## Architecture: split the context, not the safety boundary

### Immutable base

The base contains stable system instructions, tool schemas, repository orientation, model
configuration, and other prefix-stable material. It is content-addressed and versioned.
Workers send only a task delta and continuation identity. In a controlled kernel, fak can
install and route this base directly. In API-only mode, the adapter uses provider prompt
caching when exposed, otherwise byte-identical prefixes and measured cache telemetry.

### Micro-context state

Each logical context owns only mutable state: goal/delta, short transcript or summary,
tool/effect journal, budget, priority/deadline, and a continuation token. Cold contexts
hibernate to durable state. Warm contexts consume scarce model/KV slots. The existing
`internal/microagent` host, scheduler, hibernation, warm-band, session gateway, and tool
execution seams are the starting substrate; this program must integrate them rather than
invent a second fleet runtime.

### Ultracode relationship

Ultracode supplies the workflow pattern: decompose work into independently checkable,
disjoint packets and fold only witnessed results. Micro-contexts move that pattern below
full harness instances. Ultracode remains useful for issue/worktree coordination; the
micro-context fabric handles model-turn scheduling and compact context lifetimes. They
compose, but their speedups are measured separately from inference batching/cache gains.

## Constraints, existing footholds, and explicit next proofs

These are not undifferentiated reasons the program might fail. Each constraint is paired
with the fak capability that already reduces it and the remaining experiment or mitigation.
That distinction prevents shipped substrate from being rediscovered as a blocker and keeps
an external limitation beside the route around it.

- **Harness assumptions — start with our harness, then expose micro-agents:** third-party
  harnesses often bind one process, terminal, transcript, cwd, credential set, and approval
  channel to one agent. fak already has a compact `microagent.Descriptor`, bounded host,
  session gateway, and continuation budgets, so the first integration target is fak's own
  harness: keep one top-level operator/harness agent and expose the bounded contexts beneath
  it as micro-agents. Measure that path's bring-up cost and effect boundary first; only then
  map the same descriptor onto Codex, Claude, OpenAI, or other provider/harness adapters
  through `docs/integrations/`, rather than requiring each external harness to become the
  fleet runtime. #5789 shipped the descriptor inventory; the next proof is this
  parent-agent/micro-agent integration shape, not another descriptor design pass.
- **Context segmentation and addressability — extend, do not re-invent:** fak already
  separates immutable base from per-context delta, gives each logical context an ID and
  durable continuation state, and derives in-kernel shared-prefix cache identity from the
  request context (`prefixCacheIdentityFromContext`). The remaining risk is identity drift
  from timestamps, reordered tools, provider wrappers, sampling regimes, or an incomplete
  namespace key. Canonicalize those inputs, make base/context/prefix IDs visible at the
  gateway, and require every miss to carry a reason. S2b (#5817) already proved that this
  addressability reaches real KV reuse on the controlled kernel; the open work is preserving
  and explaining that identity across each new provider wrapper and decode regime.
- **KV/cache capacity — logical scale is not resident scale:** `internal/microagent` already
  has warm-band, warm-reserve, parked, and hibernated state, so 10k context IDs need not mean
  10k live KV allocations. S3 (#5788) and the controlled 10k soak (#5792) exercised the
  bounded lifecycle; each new backend must still report restore latency and bytes against
  recompute under a fixed resident-slot budget so admission can choose warm, parked, or
  hibernated placement from evidence.
- **Batch incompatibility — classify before coalescing:** the compatibility scheduler already
  keys work by model, sampling configuration, tools, prefix identity, phase, and length
  bucket, with incompatible work retaining a singleton path. The remaining experiment is a
  shipped in #5790/#5819. Preserve its proof obligation on every backend: a real-model
  mixed-workload run must report achieved batch size, padding waste, queue-delay tax, and
  singleton fallback rate instead of treating scheduler presence as a throughput gain.
- **Head-of-line blocking — model turns and effects are separate resources:** the bounded
  scheduler and tool-execution seam already prevent a tool call from being the model slot
  itself. Add observable cancellation/preemption and prove with one long tool wait plus
  short decodes that the short contexts continue retiring within their deadlines.
- **Provider/API opacity — integrate at the top-level boundary and narrow the claim:** the
  API-only adapter shipped in #5793 and can carry the same base/delta descriptor to an
  external provider while a top-level fak agent owns decomposition, policy, journals, and
  result folding. For closed
  providers, expose micro-agents through that adapter and claim only observable billed
  cached tokens, latency, errors, and controlled A/B outcomes; reserve KV residency and
  batch-formation claims for fak's controlled-kernel path. Provider-specific integrations
  are follow-ons after the own-harness spine, not prerequisites for proving the fabric.
- **Rate limits and quotas — provider capacity is an admission input:** API mode remains
  bounded by RPM, TPM, and concurrency quotas even when logical contexts are cheap. Feed
  #5793/#5795 feed those limits into the budget/fair scheduler with token buckets, retry
  budgets, and provider-aware fairness. Every provider integration must report admitted,
  delayed, retried, and rejected work by provider rather than calling queued fan-out
  concurrency.
- **Result quality — fold witnessed work, not turn count:** the verifier and quality-ledger
  seams already exist, so use fixed task corpora and verifiers not authored by workers; the
  acceptance metric is pass rate times completed work per time, with duplicated or failed
  outputs charged as cost. #5794 shipped that ledger; new workloads must populate it rather
  than invent a throughput-only score.
- **Tool/effect conflicts — read-only first, leased effects second:** capabilities, the tool
  execution floor, journals, and independent read-back are existing control points. Keep the
  first scale runs read-only; #5791 shipped the effect-safe seam, and effectful micro-agents
  must still enter through its lane/resource leases and idempotency keys while reporting
  denied, conflicted, and independently confirmed effects.
- **Fault amplification — version and canary the shared base:** bounded retries and journal
  quarantine already limit some local failures, but one bad immutable base can still poison
  a cohort. Roll each base version through a small canary cohort, trip a circuit breaker on
  shared failure signatures, quarantine retries, and retain the prior base for rollback
  before raising admission.
- **Observability cardinality — aggregate without losing addressability:** context IDs and
  journals provide the correlation key; full per-turn spans for 10k contexts do not scale.
  Keep aggregate queue/cache/quality metrics for every cohort, sampled spans for normal work,
  and unsampled error/effect journals addressable by context ID.
- **Economics — compare complete systems:** the relevant alternative is a tuned top-level
  agent or harness running the same task corpus, not 10k naive processes. Report net value
  after duplicated output, scheduler and adapter overhead, idle GPU time, cache storage,
  provider charges, and failed work; without that witness the outcome remains `not yet`.

## Measurement contract

Every experiment records: model/provider and hardware provenance; base and delta token
counts; logical contexts and physical slots; prefill/decode/total tokens per second; TTFT
and p50/p95 completion latency; cache hit/miss evidence; queue delay; peak host RAM and
KV/cache bytes; error/retry/cancel counts; verifier pass rate; and useful completed tasks
per wall-clock minute. Required comparisons are tuned sequential, tuned provider-native
batching, and the fak micro-context path. The two headline dimensions are reported
separately—single critical-path usability and aggregate useful throughput.

## Issue map

Epic: [#5785](https://github.com/anthony-chaudhary/fak/issues/5785) (P0, G0).

- [#5786](https://github.com/anthony-chaudhary/fak/issues/5786) — S1 real-endpoint 100-context spine.
- [#5787](https://github.com/anthony-chaudhary/fak/issues/5787) — S2 shared-prefix A/B against tuned batching.
- [#5788](https://github.com/anthony-chaudhary/fak/issues/5788) — S3 1,000-context hibernation and restart.
- [#5789](https://github.com/anthony-chaudhary/fak/issues/5789) — harness-assumption inventory and lightweight descriptor.
- [#5790](https://github.com/anthony-chaudhary/fak/issues/5790) — compatibility-class scheduler.
- [#5791](https://github.com/anthony-chaudhary/fak/issues/5791) — tool/effect safety.
- [#5792](https://github.com/anthony-chaudhary/fak/issues/5792) — S5 controlled-kernel 10k soak.
- [#5793](https://github.com/anthony-chaudhary/fak/issues/5793) — S6 API-only adapter.
- [#5794](https://github.com/anthony-chaudhary/fak/issues/5794) — quality and observability ledger.
- [#5795](https://github.com/anthony-chaudhary/fak/issues/5795) — S7 multi-user fairness and economics.
- [#5817](https://github.com/anthony-chaudhary/fak/issues/5817) — S2b kernel cache-observable shared-prefix A/B.
- [#5818](https://github.com/anthony-chaudhary/fak/issues/5818) — multi-turn descriptor continuation budgets.
- [#5819](https://github.com/anthony-chaudhary/fak/issues/5819) — in-kernel compatibility-batch execution.
- [#5820](https://github.com/anthony-chaudhary/fak/issues/5820) — S5a 1,000 real-model-turn contexts on one controlled node.
- [#5821](https://github.com/anthony-chaudhary/fak/issues/5821) — cache-value Track-1 fold of witnessed shared-base reuse.
- [#5830](https://github.com/anthony-chaudhary/fak/issues/5830)–[#5844](https://github.com/anthony-chaudhary/fak/issues/5844) — hardening fan-out off spine `28846558d0` (qa, dogfood, product, observability, integration, docs, release).
- [#6029](https://github.com/anthony-chaudhary/fak/issues/6029)–[#6034](https://github.com/anthony-chaudhary/fak/issues/6034) — large-input operator cohort: runnable 1,000-issue spine, adaptive filter selection, read-only tool enrichment, provenance-preserving fold, tuned-baseline falsification, then effectful tools; see [the operator note](micro-context-large-input-operators.md).

2026-08-06: #5792–#5795 repaired to dispatch-ready contract bodies (research completion
standard, routed); the `research` label was removed from the open leaves so the
dispatcher can route them (it is a triage-hold label, kept on the epic only).

## Prior art to reuse

- `internal/microagent`: bounded host, context, scheduler, warm reserve/band, hibernation,
  session gateway, retries, and tool execution.
- [`docs/explainers/ultracode-multi-agent-dogfood.md`](../explainers/ultracode-multi-agent-dogfood.md):
  disjoint work packets and honest orchestration concurrency.
- [`docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md`](../notes/AGENTIC-CACHING-SOTA-2026-06-19.md):
  agentic caching layers and open gaps.
- [`docs/notes/MOE-SSD-MULTI-AGENT-NET-TOKS-2026-07-18.md`](../notes/MOE-SSD-MULTI-AGENT-NET-TOKS-2026-07-18.md):
  inference-throughput roofline; do not blend it with orchestration metrics.
