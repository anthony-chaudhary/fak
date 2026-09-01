---
title: "Triage: KV paging + context-budget tuning for concurrent GLM-5.2 streams (GPU server 2 · Lane B · #3080)"
description: "Generation-horizon classification and the COMPUTED KV-budget half of the Accept table for issue #3080 — the KV-paging / context-budget companion to the L2 concurrency sweep. Derives KV GiB/stream and max-streams-that-fit for GLM-5.2 DSA/MLA at ctx {4k,8k,16k} from the repo's own MLA cache shape, then names the WITNESSED gate (paged-vs-contiguous A/B + aggregate tok/s) a GPU-less worker cannot reach. Every KV cell is MODELED; the model inputs are ESTIMATED (DeepSeek-lineage) pending Lane F GGUF-header truth. NO served number appears here."
---

# Triage — #3080: KV paging + context-budget tuning for concurrent GLM-5.2 streams (GPU server 2 · Lane B)

> **What this is.** A *triage / classification* record for issue
> [#3080](https://github.com/anthony-chaudhary/fak/issues/3080), a child of epic
> [#3073](https://github.com/anthony-chaudhary/fak/issues/3073) and the direct companion
> to the L2 concurrency sweep ([#3079](https://github.com/anthony-chaudhary/fak/issues/3079)).
> It (1) classifies the work's generation horizon, and (2) delivers the **computable half**
> of the issue's Accept table — the `{ctx, KV GiB/stream, max streams that fit}` columns —
> derived analytically from the repo's own GLM-5.2 MLA cache shape.
>
> **What this is NOT.** It is **not** the WITNESSED benchmark artifact. The **`aggregate
> tok/s at the max-concurrency point`** column and the **paged-KV vs contiguous A/B** both
> require a live resident serve on GPU server 2 / Lane B and stay **open** (§4). Every KV
> number here is **MODELED** (a from-first-principles cache-size calc, not a measurement),
> and its inputs are **ESTIMATED** (DeepSeek-lineage MLA/DSA dims) pending the Lane F
> GGUF-header pin. No served throughput or residency number appears here.

## 1. The ask (verbatim intent)

The ~206 GiB VRAM left after the 433.82 GiB weights bounds concurrency via KV. Compute KV
bytes/token/stream for GLM-5.2 DSA/MLA at ctx {4k,8k,16k}; find the max concurrent streams
that fit; A/B paged-KV vs contiguous; feed the max-concurrency point back into the L2
sweep (#3079). **Accept** = a `{ctx, KV GiB/stream, max streams that fit, aggregate tok/s at
that point}` table, recorded under `experiments/benchmark/runs` labelled WITNESSED/OBSERVED.

## 2. Generation classification — `gen/now`

Horizon: **`gen/now`** (milestone *Generation G0 — Now / Immediate*). The issue carried no
generation label or milestone at intake; this record closes that classification gap (the
label/milestone binding is applied on the issue itself). Basis, against the
[`docs/generation.md`](../generation.md) `gen/now` test — "improves the current product,
operator loop, or trunk hygiene with a clear witness and no dependency on a future
architecture bet":

- **Current-product improvement.** Right-sizing `--ctx-size` / paged-KV keeps the *current*
  resident serve from being KV-starved before it is compute-bound — it tunes the serve as
  run today on the *current* 8-GPU datacenter-server (sm_80) iron. No new architecture.
- **Existing engine lever.** Paged/unified KV and `--ctx-size` are already-shipped
  `llama.cpp` capabilities; this is a serve-config + measurement stand-up, not a research bet.
- **Concrete witness.** The witness is the recorded `experiments/benchmark/runs` table
  (§4), not a self-report — exactly what `gen/now` requires.

This is not priority laundering: `gen/now` reflects the horizon, independent of the issue's
B-track priority.

## 3. The COMPUTED KV budget (the analytical half of the Accept table)

Unlike its siblings, #3080's first ask — "compute KV bytes/token/stream" — is a **closed-form
calculation**, so this note delivers it. Every cell below is **MODELED** from the cache shape,
not measured; the inputs are **ESTIMATED** (labeled in §3.2) until Lane F pins them.

### 3.1 The cache shape (from the repo's own GLM-DSA layout)

GLM-5.2 attention is **MLA + DeepSeek-Sparse-Attention (DSA)**. The KV-resident state per
token per layer is *not* a full per-head K/V — MLA caches a compressed latent. From the
repo's GLM-DSA tensor layout (`internal/model/synthetic.go`,
[`GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md`](GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md)):

- `kv_a_proj_with_mqa` → `[KVLoraRank + QKRopeHeadDim]` — the **compressed KV latent**
  (`KVLoraRank`, post `kv_a_layernorm`) **+ the decoupled rotary key** (`QKRopeHeadDim`,
  MQA-shared, single). This is the whole MLA KV-cache footprint: the up-projection
  (`kv_b_proj`) is recomputed from the latent at attention time, not cached.
- DSA adds a **separate indexer key cache** (`indexer.wk` → `[IndexHeadDim, H]`, one key per
  token per full-indexer layer) — the "GLM-DSA separate index/state cache" that
  `internal/model/paged_hal.go` flags as not-yet-paged on the fak path.

So, per token:

```
KV_elems/token = n_layers × ( KVLoraRank + QKRopeHeadDim )        # MLA latent + rope key
               + idx_layers × IndexHeadDim                        # DSA indexer key (≤ n_layers)
KV_bytes/token = KV_elems/token × bytes_per_elem                  # F16 KV ⇒ 2
```

### 3.2 The inputs (ESTIMATED — DeepSeek-lineage, Lane F pins them)

| Input | Value | Provenance / label |
|---|--:|---|
| `n_layers` | **92** | ESTIMATE — ceiling doc "~92-layer" (~89 MoE); GGUF pins it |
| `KVLoraRank` | **512** | ESTIMATE — DeepSeek-MLA lineage default |
| `QKRopeHeadDim` | **64** | ESTIMATE — DeepSeek-MLA lineage default |
| `IndexHeadDim` | **128** | ESTIMATE — DeepSeek-V3.2-DSA lineage default |
| `bytes_per_elem` | **2** | F16 KV (llama.cpp default; KV-quant is a lever, §3.4) |

MLA latent+rope = `92 × (512+64) × 2` = **103.5 KiB/token**. DSA index (upper bound, every
layer indexes) = `92 × 128 × 2` = **23.0 KiB/token**. Combined ≈ **126.5 KiB/token**
(≈ 0.1236 MiB/token). MLA alone is the dominant, best-characterized term; the index term is
an upper bound (some layers share the indexer — `glmDsaIndexerIsShared` — so fewer than 92
carry their own key).

### 3.3 The table (MODELED)

`KV GiB/stream = ctx × KV_bytes/token ÷ 1024³`. Two fit columns: **raw** (all 206 GiB to KV)
and **usable** (÷ a 0.8 headroom factor reserving ~20% of the 206 GiB for activations,
CUDA-graph pools, the batch working set, and paging fragmentation — itself a MODELED
assumption, tightened by the §4 witness).

| ctx | KV GiB/stream (MLA+idx) | KV GiB/stream (MLA only) | max streams @206 GiB raw | max streams @~165 GiB usable |
|--:|--:|--:|--:|--:|
| 4096 (4k) | **0.494** | 0.404 | **~417** | ~334 |
| 8192 (8k) | **0.988** | 0.809 | **~208** | ~167 |
| 16384 (16k) | **1.977** | 1.617 | **~104** | ~83 |

**Read.** MLA is *why* hundreds of streams fit at all: caching a 576-wide latent instead of
a materialized per-head K/V (which for a DeepSeek-lineage head count would be ~20–70× wider)
is the difference between ~200 streams and a dozen at 8k. So at 8k ctx the ~206 GiB free VRAM
admits **~200 concurrent streams before KV starvation** (MODELED) — well past the L2 sweep's
top point (`--parallel 128`), meaning **at 8k the L2 concurrency sweep should reach its knee
compute-bound, not KV-bound.** At 16k the raw fit (~104) drops under 128, so **16k is the ctx
where KV starvation can bite the top of the L2 sweep first** — that is the context-budget knob
#3080 exists to tune.

### 3.4 Levers the table exposes (for the operator run)

- **KV-quant** (`--cache-type-k/v q8_0`/`q4`): halves/quarters `bytes_per_elem` → ~2×/4× the
  fit at a quality/accuracy cost to A/B.
- **ctx right-sizing**: dropping 16k→8k roughly doubles max streams; the pick is a
  latency-vs-concurrency trade the L2 curve prices.
- **paged vs contiguous**: contiguous per-stream allocation sized to *max* ctx wastes the
  unused tail of every short stream; paged/block allocation packs on demand → strictly ≥ the
  contiguous fit when real streams run shorter than ctx. The *size* of that win is the A/B
  the witness must measure (§4).

## 4. Blocker — the WITNESSED columns this host cannot produce

**Accept** requires the full `{ctx, KV GiB/stream, max streams that fit, aggregate tok/s at
that point}` table recorded under `experiments/benchmark/runs`, `claim_class: WITNESSED`.
This note fills the first three columns **MODELED**; the fourth — and the **paged-KV vs
contiguous A/B** — need a **live resident serve** on GPU server 2 / Lane B (sm_80, 433.82 GiB
resident). This dispatch ran on the Windows dev box, which:

- has **no GPU** and cannot host the resident serve;
- reaches the lab iron only through the private control bridge
  ([`private-comms-channel.md`](../private-comms-channel.md) → `../fak-private`), and a
  resident serve + a paged/contiguous concurrency A/B is a major live operation, **outside
  this dispatch's declared "triage only" risk envelope**;
- must not fabricate the numbers — a self-authored throughput cell is not a witness.

So the MODELED columns are landed; the WITNESSED columns are **owed by a GPU-node worker**.
Honest state: `not yet`.

**Durable fact worth keeping (host/lane gotcha).** On the **pure-fak** path, paged-KV for
GLM-5.2 is currently a *no-op-refusal*: `internal/model/paged_hal.go` panics
`FAK_PAGED_KV does not yet support GLM-DSA's separate index/state cache`. So the paged-vs-
contiguous A/B is a **llama.cpp-engine** measurement today (engine-honest baseline, per the
epic rule); a pure-fak paged-KV A/B is blocked until GLM-DSA's index/state cache learns
paging — a real follow-on, not a same-day witness.

## 5. Smallest next step + the artifact contract

Execute on **GPU server 2 / Lane B** (a worker resident on the node, or an operator driving the
private bridge), then record the artifact under:

```
experiments/benchmark/runs/by-machine/gpu-server-2/<UTCstamp>-glm52-kv-budget-contextfit/
  manifest.json     # benchmark/run-manifest.v1 (machine_id: gpu-server-2, claim_class: WITNESSED, git rev/dirty)
  result.json       # one row per (ctx, kv_mode) point
```

Each `result.json` row:

| field | meaning |
|---|---|
| `ctx` | context size (`--ctx-size`), one of {4096, 8192, 16384} |
| `kv_mode` | `contiguous` or `paged` (the A/B) |
| `kv_gib_per_stream` | measured resident KV per stream (validates the §3.3 MODELED cell) |
| `max_streams_fit` | max concurrent streams the serve admits before KV OOM at that ctx |
| `aggregate_decode_toks` | summed decode tok/s at the max-concurrency point |

**Headline to report on the issue:** the `(ctx, max streams that fit, aggregate tok/s)` peak
point, and the paged-vs-contiguous delta. **Feed that max-concurrency point back into the L2
sweep (#3079)** so its `--parallel` top is set at the compute-bound knee, not a KV-starved
one. Keep the llama.cpp engine-honest baseline separate from any pure-fak number (epic rule).

## 6. Closing evidence (generation contract)

- **Promotion evidence.** Parent epic #3073 is an active day-scale drive on current hardware;
  the ctx/paged-KV knobs are already-shipped `llama.cpp` capabilities; the acceptance witness
  is a concrete recorded artifact — all three point *toward* now, so the item is promoted
  from unclassified to `gen/now`. This note additionally *retires the analytical blocker* for
  the first three Accept columns by computing them.
- **Demotion / retirement evidence.** None found. No sibling has recorded a GPU server 2
  Lane-B KV-fit artifact (`experiments/benchmark/runs/by-machine/` has no `gpu-server-2` node
  dir as of this note), and the epic is OPEN. If a peer lands the WITNESSED table first →
  retire as duplicate. If Lane F's GGUF-header truth shows the resident engine caches a
  *decompressed* per-head KV (not the compressed latent modeled here), the §3.3 fit numbers
  are void and the table is re-derived — the *method* stands, the *numbers* move.
- **Invalidating assumption (the one this table exists to test).** That the resident
  `llama.cpp` GLM-5.2 serve actually caches the **MLA-compressed latent** (`KVLoraRank +
  QKRopeHeadDim` per token per layer), at F16, with the DeepSeek-lineage dims in §3.2. If any
  of those is wrong on the real GGUF — a larger `KVLoraRank`, a decompressed cache, a
  different layer count, or a KV precision other than F16 — the KV GiB/stream and max-streams
  cells move proportionally. The §4 `kv_gib_per_stream` measurement is exactly the cheap
  re-witness that confirms or voids the MODELED column.

## 7. Links

- Issue: [#3080](https://github.com/anthony-chaudhary/fak/issues/3080) · Companion L2:
  [#3079](https://github.com/anthony-chaudhary/fak/issues/3079) · Epic:
  [#3073](https://github.com/anthony-chaudhary/fak/issues/3073)
- Ceiling + lever map:
  [`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md)
  (§1 — ~206 GiB free VRAM; Ceiling B — aggregate ~11–14k tok/s COMPUTED)
- Sibling triages: [`GLM52-GPU-SERVER-LANEB-L2-CONTBATCH-TRIAGE-2026-07-06.md`](GLM52-GPU-SERVER-LANEB-L2-CONTBATCH-TRIAGE-2026-07-06.md)
  · [`GLM52-L5-QUANT-SWEEP-TRIAGE-2026-07-06.md`](GLM52-L5-QUANT-SWEEP-TRIAGE-2026-07-06.md)
- MLA cache shape: [`GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md`](GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md)
  · `internal/model/synthetic.go` · `internal/model/paged_hal.go`
- Reaching the hardware: [`private-comms-channel.md`](../private-comms-channel.md)
</content>
</invoke>
