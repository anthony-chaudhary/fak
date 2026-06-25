---
title: "fak Batching Configuration Guide"
description: "How fak batches multi-request inference: the decode and prefill batching regimes, the dynamic padding-aware batch-composition policy (batch-size selection + padding minimization, padding overhead bounded ≤10%), the SIMD/GEMM tuning knobs, and the honest status of each B-008 acceptance bar."
---

# Batching Configuration Guide

This guide covers how `fak` batches **multiple concurrent requests** through one weight
stream, and the knobs that control it. It is the operator-facing companion to the model/
compute env reference ([`docs/model-engine-env.md`](../model-engine-env.md)) and the serve
flags reference ([`server-config.md`](server-config.md)).

It is the configuration deliverable of issue
[#272 / B-008 (Dynamic Batching Optimization)](https://github.com/anthony-chaudhary/fak/issues/272).
The honest status of each B-008 acceptance bar is in [§5](#5-status-what-is-shipped-vs-gated).

---

## 1. Why batching exists (the lever)

Batch-1 decode is **memory-bandwidth-bound**: to generate one token the kernel re-streams
every weight byte (~537 MB f32 / ~150 MB Q8 for the reference model), almost independent of
how much arithmetic those bytes drive. Serving one user that way wastes the machine — the
weights are streamed to compute a single token's worth of work, then discarded.

Multi-request batching fixes exactly that. Stack one decode token from each of `B`
independent users into a `[B, *]` panel and run each weight matmul once over the whole panel.
Each weight row is read once and reused across all `B` users, so the bottleneck byte-stream
is amortised `B`-fold and **aggregate throughput scales toward linear in `B`** until the GEMM
becomes compute-bound. This is continuous batching — the single biggest throughput multiplier
in LLM serving — done in-kernel over kernel-owned per-user KV caches. The kernel primitive is
`model.StepBatch` (decode) / `model.PrefillEach` (prefill); the multi-user math is bit-for-bit
identical to running each user serially (`TestBatchedDecodeMatchesSerial`).

---

## 2. The two regimes — and where padding comes from

Batching behaves differently in the two phases of a request, and only one of them pays a
padding cost.

| Regime | What is shared | Padding risk |
|---|---|---|
| **Decode** (`StepBatch` / ragged `StepBatchActive`) | one weight stream per layer across all live lanes | **None from idle lanes.** `StepBatchActive` compacts to only the active lanes, so a lane that is blocked, finished, or idle this step does **zero** work — witnessed exactly by `model.LastStepMACs()` (#520). |
| **Prefill** (`PrefillEach` → rectangular fast path) | one weight stream over a `[B, P]` token panel | **Yes.** The batched rectangular prefill only fires when **every prompt in the panel is the same length `P`** (`model.rectangularPrefillLen`). Mixed-length prompts either fall back to per-request prefill (no batch win) or must be **padded to the longest prompt**, wasting `(maxLen·B − Σlenᵢ)` token-rows. |

So the padding question is a **prefill admission** question: *which* pending requests should
be grouped into a rectangular panel together so the batch wins throughput without paying
unbounded padding?

---

## 3. Dynamic, padding-aware batch composition

`internal/gateway/batchsched.go` is the deterministic policy that answers that question. It
is a pure function of the pending request **prompt lengths** — it moves no KV and runs no
model — so it is byte-deterministic across machines.

### Knobs (`gateway.BatchPolicy`)

| Field | Default | Meaning |
|---|---|---|
| `MaxBatch` | `32` | Upper bound on requests per rectangular batch (the static ceiling the dynamic size selection clamps against). |
| `MaxPadOverhead` | `0.10` | Per-batch padding-overhead ceiling. A batch is never extended past this — which is what bounds the **aggregate** overhead (see below). `0.10` is the B-008 acceptance bar. |
| `MaxPromptLen` | `512` | Longest prompt eligible for batched rectangular prefill (mirrors the in-kernel ceiling `model.batchRectPrefillMaxTokens`). Longer prompts are scheduled as serial singletons — never dragged into a panel. |

Build the shipping defaults with `gateway.DefaultBatchPolicy()`.

### What it does

- **Dynamic batch-size selection** — `DynamicBatchSize(pending)` adapts the admitted size to
  the live queue depth, clamped to `[1, MaxBatch]`: a shallow queue admits a small batch (no
  waiting for a full panel that will not arrive); a deep queue fills to the cap.
- **Padding minimization** — `ComposeBatches(lengths)` sorts the pending requests by prompt
  length and greedily packs each rectangular batch only while its padding overhead stays
  ≤ `MaxPadOverhead`. Similar-length requests batch together; a length jump that would blow
  the ceiling starts a new batch instead.
- **One-call entry** — `PlanBatches(lengths)` returns the batch index-groups plus
  `BatchPlanStats` (`AggregatePadOverhead`, `WorstPadOverhead`, `BatchedFraction`).

### The guarantee (witnessed, not measured)

Because every emitted batch individually satisfies `padᵦ ≤ MaxPadOverhead · rowsᵦ`, the
**aggregate** padding overhead `Σpadᵦ / Σrowsᵦ` is ≤ `MaxPadOverhead` for **any** input
distribution. This is the B-008 "padding overhead ≤ 10%" acceptance criterion as a
closed-form invariant rather than a wall-clock number — it reproduces byte-for-byte on any
host, the same discipline as the kernel's bit-exact witnesses.

Run the witnesses:

```bash
go test ./internal/gateway/ -run 'Batch|Compose|DynamicBatchSize|Padding' -count=1
```

- `TestBatchPaddingOverheadInvariant` — aggregate and worst-case overhead stay under the
  ceiling across the length spread.
- `TestComposeBatchesSplitsWhenPaddingWouldExceedPolicy` — a length jump splits the batch
  rather than paying the padding.
- `TestComposeBatchesCoversEveryRequestOnce` — the composition partitions the queue exactly.
- `TestComposeBatchesSerializesInvalidAndOversizedPrompts` — over-ceiling prompts go serial.
- `TestDynamicBatchSizeClampsToPendingAndMax` — the size selection adapts and clamps.

---

## 4. Decode/prefill tuning knobs (env)

These `FAK_*` variables tune the batched kernels themselves. Full table with `file:line`
sources in [`docs/model-engine-env.md`](../model-engine-env.md); the batching-relevant subset:

| Env var | Default | Effect |
|---|---|---|
| `FAK_QATTN_GQA` | on | Fused GQA attention path. Turn off (`0`/`off`) to A/B. |
| `FAK_FDOT3_SIMD` | on | SIMD `fdot3` (3-row score dot) in batched attention. |
| `FAK_FDOT3_SIMD_MINB` | `64` | Minimum batch size at which SIMD `fdot3` engages. |
| `FAK_SAXPY3_SIMD_MINB` | `1` | Minimum batch size for SIMD `saxpy3` (V accumulate). |
| `FAK_SAXPY3_SIMD_MINPOS` | `1` | Minimum context positions at which SIMD `saxpy3` engages. |
| `FAK_Q_FAST_SWIGLU` | on | Fast quantized SwiGLU. Turn off to A/B against the reference. |
| `FAK_QGEMM_GROUP_MAXP` | `1024` | Max prompt-panel batch width that still groups. |
| `FAK_QPROFILE` | off | Print coarse phase timing (quantize / GEMM / attention) for batched-Q and Metal prefill. |

All are off-by-tuning safe: they change kernel *scheduling*, never the numerics (the batched
paths stay argmax-exact / bit-identical to the serial reference).

---

## 5. Status: what is shipped vs gated

B-008's acceptance has four bars. Their honest on-disk status (see the Track-B tracker
[`docs/notes/track-b-performance-parity-tracking-306.md`](../notes/track-b-performance-parity-tracking-306.md)):

| Acceptance bar | Status |
|---|---|
| **Padding overhead ≤ 10%** | ✅ **Shipped + witnessed** — the composition's closed-form invariant (§3), `TestBatchPaddingOverheadInvariant`. |
| **Configuration guide** | ✅ **This document.** |
| **Near-linear throughput scaling (1.8× on 2× requests)** | 🟡 **Deferred** — a wall-clock measurement that needs the production continuous-batching scheduler (paged KV, preemption, admission) wired into the live serve path on real serving hardware. The decode primitive that makes it *possible* is shipped and bit-exact (`StepBatch`); the scheduler is the deferred sibling serving work. |
| **Latency p99 within 1.5× of single-request** | 🟡 **Deferred** — same gate: needs the SLA-aware admission scheduler + a wall-clock bench. |

The batching **kernel** (decode + ragged idle-lane elimination) and the **composition
policy** (dynamic size + padding minimization) are host-tractable and shipped. The two
wall-clock bars depend on the production scheduler that `internal/modelengine/nativesched.go`
is explicitly **not** (it is a shape proof: no paged KV, no preemption, no fairness/admission)
— tracked by the sibling serving issues, not by this issue.

---

## See also

- [`docs/model-engine-env.md`](../model-engine-env.md) — every `FAK_*` model/compute knob with `file:line`.
- [`server-config.md`](server-config.md) — `fak serve` flags and HTTP/metrics surface.
- [`docs/notes/track-b-performance-parity-tracking-306.md`](../notes/track-b-performance-parity-tracking-306.md) — the Track-B performance-parity status tracker (B-001…B-008).
- `internal/model/batch.go` · `internal/model/batch_step.go` — the in-kernel batched decode primitive.
- `internal/gateway/batchsched.go` — the dynamic padding-aware composition policy described here.
