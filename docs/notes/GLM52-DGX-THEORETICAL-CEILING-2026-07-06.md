---
title: "GLM-5.2 on the GPU server 2 / GPU server 3 lab nodes: the theoretical throughput ceiling and the day-scale 80% drive (2026-07-06)"
description: "A roofline upper bound for GLM-5.2 UD-Q4_K_M served on the two identical 8-GPU datacenter server (sm_80) lab nodes — single-stream decode, batched aggregate, and prefill — against the current WITNESSED 23.2 tok/s, with the estimate inputs labelled and the lever map that seeds the concurrent-ticket fleet aimed at 80% of the practical ceiling in a day."
---

# GLM-5.2 on GPU server 2 / GPU server 3: the theoretical ceiling and the 80% drive

> **What this is.** A *roofline* upper bound — hardware limits, not a measurement — for
> GLM-5.2 UD-Q4_K_M on the lab's two identical **8-GPU datacenter server (sm_80)** nodes (GPU server 2 and
> GPU server 3), and the lever decomposition that turns the gap between the current **23.2 tok/s
> WITNESSED** single-stream serve and that ceiling into a fleet of concurrent tickets.
>
> **What this is NOT.** No cell here is a served measurement except the ones explicitly
> tagged **WITNESSED** (all trace to `docs/notes/GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md`
> and the cited companions). The ceiling rests on two **ESTIMATED** model inputs
> (active-params/token, active-bytes/token); the ceiling scales *inversely* with both, and
> the first ticket in the fleet (Lane F) replaces the estimate with the GGUF-header truth.

## 1. The node shape (both GPU server 2 and GPU server 3 — identical)

Reconstructed from `experiments/glm-gpu-witness/gpu-server-node-state-2026-06-25.json` and
the hardware matrix. Both boxes are the **same** shape, which is the point: **two identical
nodes = two parallel experiment lanes** (see §6).

| Quantity | Per GPU | ×8 aggregate | Note |
|---|---|---|---|
| GPUs | datacenter GPU, sm_80 (compute 8.0) | 8 | **below the sm_90 DSA kernel floor** → stock SGLang/vLLM refuse GLM-5.2 (`BLOCKED_ARCH`, vLLM #35021); llama.cpp MLA is the resident path |
| HBM2e bandwidth | ~2.0 TB/s | **~16.0 TB/s** | NVLink-class part assumed; a PCIe part would be ~1.94 TB/s/GPU |
| BF16 tensor | 312 TFLOP/s | **~2.50 PFLOP/s** | Ampere has **no FP8**; BF16/FP16 is the tensor ceiling |
| INT8 tensor | 624 TOP/s | **~4.99 POP/s** | 2× BF16 — the aggregate compute lever if expert GEMMs run int8 tensor |
| NVLink 3.0 | 600 GB/s bidir | — | the TP/EP all-reduce fabric |
| VRAM | 80 GiB | **640 GiB** | model 433.82 GiB resident → **~206 GiB free** for KV + activation + batch |
| Host | — | ~2 TB RAM, 256 cores | NVMe-first staging (`/mnt/sglang_dv3/`) |

## 2. The model (GLM-5.2 UD-Q4_K_M)

| Input | Value | Provenance |
|---|---|---|
| Total params | **753.86 B** | WITNESSED (llama-bench, 2026-06-28) |
| On-disk / resident | **433.82 GiB** | WITNESSED |
| Architecture | `glm_moe_dsa` (MoE + DeepSeek-Sparse-Attention), ~89–92 layers, H≈5120, K≈8 experts/tok + shared expert | repo config estimates (`GLM52-DECODE-PATH-TO-10-TOKS`) |
| **Active params / token** | **~32 B (ESTIMATE)** | GLM-4.5-class active budget; the ceiling scales `1/active` — **Lane F pins this** |
| **Active bytes / token @ UD-Q4_K_M** | **~13 GiB (ESTIMATE, range 10–16)** | ~10 GiB expert stream (repo) + ~3 GiB attn/dense/shared/router |

## 3. The three ceilings (roofline, COMPUTED)

### Ceiling A — single-stream decode (memory-bandwidth bound)

Decode at batch 1 streams the active weights once per token; `tok/s = effective_BW ÷ active_bytes`.

| Regime | tok/s | Assumption |
|---|---|---|
| **Raw roofline** | **~1,230** | MBU 100 %, perfect 8-way TP, zero collective/launch latency — absolute, unachievable |
| **Practical ceiling** | **~150–200** | MBU ~65 % + ~92-layer TP collective+launch latency floor (~3.6 ms/tok fixed + ~1.2 ms bandwidth) |
| **Current WITNESSED** | **23.2** | llama.cpp layer-split → **one GPU active per token** (7 idle); ~12–15 % of practical |
| **80 % target** | **~120–160** | **~5–7× current** — the Lane A drive |

The gap is structural: llama.cpp `-sm layer` pipelines one GPU at a time, so single-stream
decode sees ~1 GPU of bandwidth. Spreading each token across all 8 (tensor/expert parallel)
is the dominant lever.

### Ceiling B — aggregate throughput at concurrency (compute bound)

At high concurrency the weight read amortizes and the expert GEMMs become compute-bound;
`tok/s = tensor_FLOP/s ÷ (2 × active_params)`, with 2×32 GFLOP/token ≈ 64 GFLOP/token.

| Regime | aggregate tok/s | Assumption |
|---|---|---|
| **Raw roofline (BF16)** | **~39,000** | MFU 100 % |
| **Raw roofline (INT8 tensor)** | **~78,000** | int8 expert GEMM on sm_80 tensor cores |
| **Practical ceiling (BF16)** | **~11,000–14,000** | MFU ~30–35 % at concurrency ~64–128 |
| **Current** | **not yet measured** | single-stream only today |
| **80 % target** | **~9,000–11,000** | the Lane B drive (fleet-value throughput) |

### Ceiling C — prefill (compute bound, same law as B)

Prefill is compute-bound at ~64 GFLOP/token → same ~11–14 k tok/s practical ceiling. Current
prefill is **unmeasured** (the 46 tok/s in the 07-01 note is an 11-token prompt, not prefill).

## 4. Current vs ceiling — the gap the fleet closes

| Metric | Current | Practical ceiling | 80 % target | Gap |
|---|--:|--:|--:|--:|
| Single-stream decode | **23.2** (WITNESSED) | ~150–200 | ~120–160 | **~5–7×** |
| Aggregate decode @ conc. | not yet | ~11–14 k | ~9–11 k | **stand up** |
| Prefill | not yet | ~11–14 k | ~9–11 k | **stand up** |

## 5. The lever map (seeds the tickets)

Ranked by expected single-stream multiplier, then by fleet-value:

| Lever | Expected | Bound it moves | Current art |
|---|---|---|---|
| **L1** 8-GPU tensor/row split (kill `-sm layer`) | ~3–6× | A: 1→8 GPU bandwidth | llama.cpp `-sm row`/RPC; SGLang TP (H-only) |
| **L2** Continuous batching (`--parallel`, cont-batch) | 10–40× **aggregate** | B: unlock concurrency | llama.cpp cont-batching, SGLang |
| **L3** Speculative decoding (draft / EAGLE / n-gram) | 1.5–2.5× | A: decode | EAGLE-2/3, prompt-lookup, Medusa |
| **L4** Flash-attention + CUDA graphs | 1.2–1.8× | A: launch/attention | llama.cpp `-fa`, graph capture |
| **L5** Quant sweep for the CUDA resident-Q4_K fast path | 1.1–1.5× | A: bytes/token | UD-Q4_K_S, Q4_K pure |
| **L6** INT8 tensor-core expert GEMM | up to 2× | B: compute | sm_80 int8 tensor (624 TOP/s) |
| **L7** True DSA sparsity (not full MLA) | ctx-dependent | A/C: long-ctx attn | GLM DSA indexer |
| **L8** Warm-start + persisted compile caches | iteration cycle | cold tax ~500s→once | #3051, #3052 |
| **L9** Real prefill path (chunked + FA) | stand up C | prefill | chunked prefill |
| **L10** Native fak resident-EP device-NCCL | pure-kernel witness | A/B on fak's own kernel | #1482 |

## 6. The concurrent drive — two nodes, six lanes, one day

Both nodes are identical, so the fleet runs **two boxes in parallel**. Each ticket names a
node and a lane so workers do not collide; the acceptance of every ticket is a recorded
`experiments/benchmark/runs` artifact, never a self-report.

- **GPU server 3** — Lane A (single-stream: L1/L4/L5), Lane D (prefill: L9), Lane B-cache (L8 RadixAttention A/B).
- **GPU server 2** — Lane B (batched aggregate: L2), Lane A-spec (L3), Lane E (L6/L7/L10), Lane F (ceiling ground-truth).
- Fast-iteration harness (Lane C) so every lever is measurable in <10 min per node.

The epic and its children live in GitHub; this note is the ceiling they aim at. When Lane F
replaces the two ESTIMATED inputs with GGUF-header truth, every ceiling here is re-derived
from the measured active set — the number moves, the method does not.

*Companions:* [8-GPU full-resident serve (23.2 tok/s WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) ·
[decode path to 10 tok/s (levers)](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md) ·
[expert-parallel multi-GPU](GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md) ·
[cold-start vs caching ablation](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md).
</content>
</invoke>
