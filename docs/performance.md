---
title: "Performance outcomes and proof routes"
description: "fak reduces repeated work at the managed-agent boundary; which mechanism applies depends on your run path. Pick the outcome, then reproduce its named witness."
---

# Performance outcomes and proof routes

**Audience:** technical evaluators deciding which `fak` performance mechanism applies to their run path and which current witness to inspect.

**Lifecycle:** current
**Generation:** gen/now
**Owner:** documentation
**Authority:** this page routes performance questions; [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) owns current measured results and baselines.

`fak` reduces repeated work at the managed-agent boundary. The applicable mechanism depends on whether the agent uses a provider, a proxied endpoint, or inference owned by `fak`; no one number applies to every mode.

**Next action:** choose the outcome and run path in the table, then open its proof route and reproduce that witness before quoting a result.

## Native inference target

The local product and performance path is fak-native, not an automatically substituted external
engine. The [canonical inference doctrine](native-inference-goal.md) sets the target: fak-native
is intended to beat llama.cpp in matched, quality-constrained envelopes because fak must retain
control of kernels, memory, scheduling, cache, adaptation, and operations for those gains to
compose.

This direction does not mint a current win. Each comparison still names the executing engine,
the matched envelope, and the authority row. llama.cpp can be selected explicitly as a
benchmark, parity/reference, migration/interoperability, or borrowing aid; it never silently
turns a failed or unsupported native run into a passing external result.

### Current native-performance work

[`benchmarks/NATIVE-PERFORMANCE-CURRENT.md`](benchmarks/NATIVE-PERFORMANCE-CURRENT.md) is the
operational source of truth for the current constraint portfolio: measured drivers, evidence
and authority owners, review dates, dependency-ready arms, collisions, and explicit exit
conditions. `fak native-performance --current` emits the typed JSON snapshot and
`fak native-performance --current-md` regenerates the readable projection.

An active constraint names what limits a result now; its type and horizon say whether it is an
evidence, correctness, dependency, capacity, or coordination limit and whether time alone may
clear it. This status route does not mint benchmark results: immutable receipts and
[`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) remain the measurement authority.

## Choose the outcome you need

| Outcome | Applicable run path | Mechanism | Proof route |
|---|---|---|---|
| Reduce repeated provider input work | A managed agent using a provider through `fak manage` or a compatible gateway path | Keep shared prompt prefixes byte-stable so the provider cache can remain reusable; compact stale history before sending it again. Provider cache rebates and `fak`-authored compaction are separate effects. | Start with [what managed cache means](explainers/what-is-managed-cache.md), then inspect the scoped current result and tuned alternative in [Benchmark Authority](../BENCHMARK-AUTHORITY.md#evaluator-route-result-first-method-second). |
| Keep a long session inside its context budget | Managed sessions whose history grows across turns | Shed or summarize older turns while preserving the working set needed for the next turn. | Inspect the [context-shedding contract and example](explainers/context-shedding.md), then use the linked tests and demo as the behavioral witness. |
| Reuse model state directly | `fak serve` when `fak` owns a supported local inference path | Reuse owned KV or prefix state rather than rebuilding the same model state for each compatible request. This is different from a provider prompt-cache rebate. | Match the workload to the [current primary-number table](../BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers), then run the exact reproduction command named by that result. |
| Measure the kernel's own execution-path cost | In-process kernel operations, separate from model and network latency | Measure the bounded kernel operation directly so provider, model, and transport time do not hide its cost. | Use the scoped [pure-kernel latency witness](../BENCHMARK-AUTHORITY.md#pure-kernel-latency-apple-m3-pro-2026-06-20) and retain its hardware and methodology fences. |

## Interpret a result without overclaiming

1. **Match the mode.** A provider prompt-cache result does not prove direct KV reuse, and a local-inference result does not describe a hosted provider path.
2. **Match workload and hardware.** Model, quantization, turn shape, concurrency, device, and warm-state assumptions stay attached to the result.
3. **Use the tuned alternative.** Quote the tuned baseline as the headline comparison; a naive stateless arm is context, not the real alternative. For native inference work, apply the [matched-envelope rule](native-inference-goal.md#the-matched-envelope-rule).
4. **Separate mechanism from outcome.** Cache reuse, context shedding, and low kernel overhead are mechanisms or component measurements. End-to-end latency, throughput, token work, and session longevity are outcomes with different witnesses.
5. **Stop when evidence is absent.** A result without a reproducible current witness is `not yet`, not a projected gain. Grade a proposed claim with `fak claim-check` under the [net-true-value standard](standards/net-true-value.md).

## Authority boundary

This route explains how to choose evidence; it does not mint performance numbers. Current measured values, tuned baselines, honesty fences, tombstones, and reproduction commands live in [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md). Tagged capability status lives in [`CLAIMS.md`](../CLAIMS.md). Historical experiment pages remain evidence for their recorded generation, not an automatic current product promise.

A completed evaluation names the run path, outcome, tuned alternative, artifact or reproduction command, and evidence status. If any one is missing, report `not yet`.
