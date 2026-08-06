---
title: "Micro-context fabrics for 100–10,000 parallel agents"
description: "Dedicated research focus for splitting one cached agent base into many bounded logical contexts, with controlled-kernel and API-only paths."
status: active
last_reviewed: 2026-08-06
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

## Minimal-spine ladder

Each rung must run end to end before the next is treated as real.

| Rung | Working component | Required witness |
|---|---|---|
| S0 | Synthetic 10k logical contexts over bounded workers (current) | exact completion count, one base install, peak concurrency ≤ worker cap |
| S1 | **Observed:** one real OpenAI-compatible endpoint, 100 delta contexts, no tools | [100/100 captured witness](micro-context-s1-real-endpoint.md): wall time, TTFT, usage-derived token rates, failures |
| S2 | **Observed:** shared-prefix A/B, unique full prompts vs one base + deltas | [No cache benefit on first endpoint](micro-context-s2-prefix-ab.md); scoped concurrency gain retained separately |
| S3 | **Observed:** 1k resumable contexts with bounded scheduler RAM and backpressure | [1,000-context hibernation witness](micro-context-s3-hibernation-restart.md): queue age, resident/hibernated counts, runtime reconstruction |
| S4a | **Observed:** versioned lightweight descriptor through existing Host/Gateway | [Harness inventory and 1,000-context adapter](micro-context-s4-lightweight-descriptor.md) |
| S4b | **Observed:** compatibility-class planner | [Mixed workload](micro-context-s4-compatibility-scheduler.md): isolation, aging, cancellation, padding/fill telemetry |
| S4 | **Observed fixture:** tool-capable microagents through capability/resource/idempotency/readback seams | [Parallel effect-safety witness](micro-context-s4-effect-safety.md) |
| S5 | 10k contexts under a controlled kernel | useful-result throughput, tail latency, memory/KV roofline, overload behavior |
| S6 | API-only adapter | provider-supported cache controls or measured natural prefix reuse; no kernel-only claims |
| S7 | Multi-user/fairness mode | tenant isolation, weighted fairness, cancellation, spend and rate-limit envelopes |

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

## Known blockers and explicit experiments

- **Harness assumptions:** many harnesses assume one process, terminal, transcript, cwd,
  credential set, and human approval channel per agent. Measure the fixed bring-up cost,
  then define a headless descriptor that omits those surfaces.
- **Prefix identity drift:** timestamps, reordered tools, per-agent IDs, provider wrappers,
  and sampling metadata can destroy cache sharing. Canonicalize the base and expose a
  fingerprint plus miss reason.
- **KV/cache capacity:** 10k logical contexts cannot imply 10k resident KV allocations.
  Build warm/parked/hibernated tiers and measure restore cost versus recompute.
- **Batch incompatibility:** different models, sampling parameters, tools, deadlines, and
  sequence lengths fragment batches. Schedule by compatibility class and report padding
  and queue-delay tax.
- **Head-of-line blocking:** long tool calls and long decodes must not pin model slots.
  Separate model turns from tool waits and make cancellation/preemption observable.
- **Provider/API opacity:** closed APIs may not reveal KV identity, cache eviction, or batch
  formation. Restrict claims to billed cached tokens, latency, and controlled A/B results;
  never infer internal reuse from hope.
- **Rate limits and quotas:** API-only fan-out is bounded by RPM/TPM/concurrency limits.
  Admission needs token buckets, retry budgets, and provider-aware fairness.
- **Result quality:** throughput without useful work is noise. Use fixed task corpora,
  verifiers not authored by the workers, and report pass rate × completed work/time.
- **Tool/effect conflicts:** parallel contexts can collide on files, DB rows, messages, or
  side effects. Capabilities, lane/resource leases, idempotency keys, and independent
  read-back are mandatory before tool-capable scale.
- **Fault amplification:** one bad base or retry policy can fail 10k contexts at once. Add
  canaries, circuit breakers, bounded retries, quarantine, and base-version rollback.
- **Observability cardinality:** per-context events can overwhelm telemetry. Preserve
  traceability with sampled spans plus aggregate queue/cache/quality metrics.
- **Economics:** report net value against a tuned alternative, including duplicated output,
  scheduler overhead, idle GPU time, cache storage, and failed work.

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

## Prior art to reuse

- `internal/microagent`: bounded host, context, scheduler, warm reserve/band, hibernation,
  session gateway, retries, and tool execution.
- [`docs/explainers/ultracode-multi-agent-dogfood.md`](../explainers/ultracode-multi-agent-dogfood.md):
  disjoint work packets and honest orchestration concurrency.
- [`docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md`](../notes/AGENTIC-CACHING-SOTA-2026-06-19.md):
  agentic caching layers and open gaps.
- [`docs/notes/MOE-SSD-MULTI-AGENT-NET-TOKS-2026-07-18.md`](../notes/MOE-SSD-MULTI-AGENT-NET-TOKS-2026-07-18.md):
  inference-throughput roofline; do not blend it with orchestration metrics.
