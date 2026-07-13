---
title: "GLM-5.2 on the GPU server 2 / GPU server 3 lab nodes: the theoretical throughput ceiling and the day-scale 80% drive (2026-07-06)"
description: "A roofline upper bound for GLM-5.2 UD-Q4_K_M on the lab's two sm_80 nodes — GPU server 3 (8×80 GiB/card datacenter GPU, the resident node) and GPU server 2 (8×40 GiB/card, cpu-offload-only for full-size GLM-5.2) — single-stream decode, batched aggregate, and prefill, against the current WITNESSED 23.2 tok/s, with the estimate inputs labelled and the lever map that seeds the concurrent-ticket fleet aimed at 80% of the practical ceiling in a day."
---

# GLM-5.2 on GPU server 2 / GPU server 3: the theoretical ceiling and the 80% drive

> **What this is.** A *roofline* upper bound — hardware limits, not a measurement — for
> GLM-5.2 UD-Q4_K_M on the lab's two sm_80 nodes, and the lever decomposition that turns
> the gap between the current **23.2 tok/s WITNESSED** single-stream serve and that ceiling
> into a fleet of concurrent tickets.
>
> **What this is NOT.** No cell here is a served measurement except the ones explicitly
> tagged **WITNESSED** (all trace to `docs/notes/GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md`
> and the cited companions). The ceiling rests on two **ESTIMATED** model inputs
> (active-params/token, active-bytes/token); the ceiling scales *inversely* with both, and
> the first ticket in the fleet (Lane F) replaces the estimate with the GGUF-header truth.

> ### Correction (2026-07-06): the two nodes are NOT identical
>
> An earlier draft of this note asserted the two boxes were "two identical 8-GPU datacenter
> nodes with 80 GiB/card," reconstructed by *assuming* the second box matched the single
> node-state capture. That assumption is wrong. **GPU server 2 has 40 GiB-per-card GPUs**
> (operator-confirmed; consistent with the 2026-06-25 cross-engine run-log's "40 GB node" that
> ran stock SGLang + llama.cpp cpu-offload). The consequence is load-bearing: **GLM-5.2
> UD-Q4_K_M (433.82 GiB) is resident only on GPU server 3** (640 GiB aggregate), and **cannot**
> be held on GPU server 2 (320 GiB) — not even 8-way expert-parallel, whose 54 GiB/rank band
> exceeds a 40 GiB card. So the resident-serve roofline (§3.1–3.3) and every resident lever
> (§5, L1–L10) are **GPU server 3 work**; GPU server 2's GLM-5.2 role is the cpu-offload path
> (§3.4) or a smaller-quant fit (L12). The §6 lane plan is corrected to match. Per-box
> baselines + next steps: `GLM52-DGX-PER-BOX-BASELINE-AND-NEXT-STEPS-2026-07-06.md`.

## 1. The two node shapes (GPU server 3 ≠ GPU server 2)

Both are sm_80 (compute 8.0), 8-GPU datacenter servers, ~2 TB host RAM, NVMe-first staging
(`/mnt/sglang_dv3/`), below the sm_90 DSA kernel floor → stock SGLang/vLLM refuse GLM-5.2
(`BLOCKED_ARCH`, vLLM #35021); llama.cpp MLA is the resident path. They differ in the one
quantity that decides everything for a 434 GiB model — **per-GPU VRAM**.

| Quantity | Per GPU | **GPU server 3** (×8) | **GPU server 2** (×8) | Note |
|---|---|---|---|---|
| datacenter GPU, sm_80 | — | **80 GiB/card** | **40 GiB/card** | operator-confirmed; GPU server 2 = the 2026-06-25 "40 GB node" |
| VRAM aggregate | — | **640 GiB** | **320 GiB** | server 3: model 433.82 GiB resident → **~206 GiB free**. server 2: **model does not fit** (320 < 434) |
| HBM2e bandwidth | ~2.0 TB/s | ~16.0 TB/s | ~16.0 TB/s | NVLink-class part assumed; same per-GPU BW, so the *resident* roofline law is identical — it is the FIT that differs |
| BF16 tensor | 312 TFLOP/s | ~2.50 PFLOP/s | ~2.50 PFLOP/s | Ampere has **no FP8**; BF16/FP16 is the tensor ceiling |
| INT8 tensor | 624 TOP/s | ~4.99 POP/s | ~4.99 POP/s | 2× BF16 — the compute lever if expert GEMMs run int8 tensor |
| NVLink 3.0 | 600 GB/s bidir | — | — | the TP/EP all-reduce fabric |

The point that was wrong before is now the point that matters: **GPU server 3 is the
GLM-5.2-resident node; GPU server 2 is not.** Two boxes, two *different* experiment lanes
(§6), not two copies of one.

## 2. The model (GLM-5.2 UD-Q4_K_M)

| Input | Value | Provenance |
|---|---|---|
| Total params | **753.86 B** | WITNESSED (llama-bench, 2026-06-28) |
| On-disk / resident | **433.82 GiB** | WITNESSED |
| Architecture | `glm_moe_dsa` (MoE + DeepSeek-Sparse-Attention), **79 layers (3 dense + 76 MoE), H=6144, 256 experts, K=8 experts/tok, 1 shared expert, expert_ffn=2048** | **WITNESSED GGUF header, measured on GPU server 2026-07-13** (`glm52-gguf-header-active-set-2026-07-13.json`, #3074 Lane F closed; corrects the earlier ~89–92/H≈5120 estimate) |
| **Active params / token** | **~30–33 B** (compute-active FFN **27.86 B** [routed 22.95 = K8×2.869 + shared 2.87 + dense 2.04] + MLA attn ~4–6 B) | DERIVED from measured K=8; per-expert **2.869 B** cross-checks total **753.86 B** (bytes÷bpw). Confirms the prior ~32 B estimate. (`RoutedExpertActiveSet.ActiveParamsPerToken` reports a **42.4 B** upper bound — it adds the full embd/output/MTP tables the FLOP divisor omits) |
| **Active bytes / token @ UD-Q4_K_M** | **~32 GiB (DERIVED header, upper bound)**: routed stream **12.95 GiB** (K=8 × **1.619 GiB/expert**) + non-expert band **19.31 GiB** | **Corrects the 07‑13 witness's 1.87 GiB routed figure** — it divided the EP‑7 *one‑rank* 59.9 GiB band by 256 instead of 37 (→ 0.234 GiB/expert). True per‑expert = 414.5 GiB routed band ÷ 256 = **1.619 GiB** (monolith q4_k 256.5 + q5_k 150.6 GiB are all routed; total‑params 753.86 B checks). Routed **~40 %** / non‑expert **~60 %** — same order, **NOT** expert‑tiny. Header‑only via `RoutedExpertActiveSet.ActiveBytesPerToken`; the box‑side per‑op trace remains the exact‑stream witness |
| 8-way EP band / rank | **~54 GiB** | 433.82 / 8 — **> 40 GiB**, so EP-resident does not fit GPU server 2 either |

## 3. The ceilings

### 3.1 Ceiling A — single-stream decode, GPU server 3 resident (memory-bandwidth bound)

Decode at batch 1 streams the active weights once per token; `tok/s = effective_BW ÷ active_bytes`.
**This ceiling is GPU server 3 only** — it assumes the model is VRAM-resident.
> **Divisor correction (2026-07-14):** the DERIVED active-bytes/token is **~32 GiB** (routed
> **12.95 GiB** + non-expert **19.31 GiB**, header arithmetic) — **~2.5× the old ~13 GiB estimate**.
> The estimate got the routed stream about right (~10 vs 12.95 GiB) but under-counted the replicated
> high-precision attn/shared/output band (~3 assumed vs **19.31 GiB** measured). The 07-13 witness's
> ~1.87 GiB *routed* figure was an EP-7-band÷256 slip (see §2, now corrected). **Consequence:** these
> raw-roofline tok/s divide by ~13 GiB and are therefore ~2.5× HIGH — dividing by ~32 GiB drops the
> **raw** single-stream roofline from ~1,230 to **~500 tok/s** and the practical ceiling by ~0.4×
> (to ~60–80). The exact per-token stream (which weights actually move each decode step; token-embd
> is a gather, not a sweep, so ~32 GiB is an upper bound) still awaits the box-side per-op trace.

| Regime | tok/s | Assumption |
|---|---|---|
| **Raw roofline** | **~1,230** | MBU 100 %, perfect 8-way TP, zero collective/launch latency — absolute, unachievable |
| **Practical ceiling** | **~150–200** | MBU ~65 % + 79-layer TP collective+launch latency floor (~3.6 ms/tok fixed + ~1.2 ms bandwidth) |
| **Current WITNESSED (server 3)** | **23.2** | llama.cpp layer-split → **one GPU active per token** (7 idle); ~12–15 % of practical |
| **80 % target** | **~120–160** | **~5–7× current** — the Lane A drive |

The gap is structural: llama.cpp `-sm layer` pipelines one GPU at a time, so single-stream
decode sees ~1 GPU of bandwidth. Spreading each token across all 8 (tensor/expert parallel)
is the dominant lever.

### 3.2 Ceiling B — aggregate throughput at concurrency, GPU server 3 resident (compute bound)

At high concurrency the weight read amortizes and the expert GEMMs become compute-bound;
`tok/s = tensor_FLOP/s ÷ (2 × active_params)`, with 2×32 GFLOP/token ≈ 64 GFLOP/token.
**GPU server 3 only** (resident).

| Regime | aggregate tok/s | Assumption |
|---|---|---|
| **Raw roofline (BF16)** | **~39,000** | MFU 100 % |
| **Raw roofline (INT8 tensor)** | **~78,000** | int8 expert GEMM on sm_80 tensor cores |
| **Practical ceiling (BF16)** | **~11,000–14,000** | MFU ~30–35 % at concurrency ~64–128 |
| **Current** | **not yet measured** | single-stream only today |
| **80 % target** | **~9,000–11,000** | the Lane B drive (fleet-value throughput) |

### 3.3 Ceiling C — prefill, GPU server 3 resident (compute bound, same law as B)

Prefill is compute-bound at ~64 GFLOP/token → same ~11–14 k tok/s practical ceiling. Current
prefill is **unmeasured** (the 46 tok/s in the 07-01 note is an 11-token prompt, not prefill).

### 3.4 GPU server 2's ceiling is a DIFFERENT law — the cpu-offload wall (host-bound)

Because GLM-5.2 does not fit GPU server 2's 320 GiB, that node serves it **cpu-offloaded**
(`--n-cpu-moe` / `--cpu-offload-experts`): the **~13 GiB/token** routed-expert stream (K=8 ×
1.619 GiB) is read from host RAM and the expert GEMM runs off the GPU. The ceiling is then **host memory-bandwidth + host
GEMM**, not the GPU roofline above — an order of magnitude lower, and it *does not batch*
(concurrency measured at **0.27×**, i.e. worse — a shared host-resource bottleneck).

| Regime | tok/s | Source |
|---|---|---|
| fak cpu-offload steady-state TPOT | **0.2324** | WITNESSED (fak kernel, 2026-06-28) |
| llama.cpp CPU mmap baseline (same class) | **0.89** decode / 3.34 prefill | OBSERVED (llama.cpp, 2026-06-28) |
| llama.cpp GPU + `--n-cpu-moe` | **2.62** single / 4.84@2 | OBSERVED (2026-06-25) |

GPU server 2's two honest GLM-5.2 levers are therefore (i) **move the expert GEMM off the host**
(the #971 direction — but resident experts need 80 GiB cards, so on 40 GiB cards the realizable
form is a PCIe-resident *partial* expert set / int8 host GEMM), or (ii) **fit a smaller quant
resident** — the largest GLM-5.2 quant that fits 320 GiB (≈ ≤2.9 bits/weight avg, Q2_K/IQ2/Q3_K_S
territory), which then earns its own §3.1-style roofline at lower quality. That is **L12 / Lane G-fit**.

## 4. Current vs ceiling — the gap the fleet closes

| Metric | Current | Practical ceiling | 80 % target | Gap |
|---|--:|--:|--:|--:|
| Single-stream decode (server 3) | **23.2** (WITNESSED) | ~150–200 | ~120–160 | **~5–7×** |
| Aggregate decode @ conc. (server 3) | not yet | ~11–14 k | ~9–11 k | **stand up** |
| Prefill (server 3) | not yet | ~11–14 k | ~9–11 k | **stand up** |
| GLM-5.2 tok/s (server 2, cpu-offload) | **0.23–2.62** | host-bound (§3.4) | move experts off host / smaller-quant fit | **different law** |

## 5. The lever map (seeds the tickets)

Ranked by expected single-stream multiplier, then by fleet-value. Levers **L1–L10 are
GPU server 3 (resident)**; **L11–L12 are GPU server 2 (its own law)**.

| Lever | Node | Expected | Bound it moves | Current art |
|---|---|---|---|---|
| **L1** 8-GPU tensor/row split (kill `-sm layer`) | server 3 | ~3–6× | A: 1→8 GPU bandwidth | llama.cpp `-sm row`/RPC; SGLang TP (H-only) |
| **L2** Continuous batching (`--parallel`, cont-batch) | server 3 | 10–40× **aggregate** | B: unlock concurrency | llama.cpp cont-batching, SGLang |
| **L3** Speculative decoding (draft / EAGLE / n-gram) | server 3 | 1.5–2.5× | A: decode | EAGLE-2/3, prompt-lookup, Medusa |
| **L4** Flash-attention + CUDA graphs | server 3 | 1.2–1.8× | A: launch/attention | llama.cpp `-fa`, graph capture |
| **L5** Quant sweep for the CUDA resident-Q4_K fast path | server 3 | 1.1–1.5× | A: bytes/token | UD-Q4_K_S, Q4_K pure |
| **L6** INT8 tensor-core expert GEMM | server 3 | up to 2× | B: compute | sm_80 int8 tensor (624 TOP/s) |
| **L7** True DSA sparsity (not full MLA) | server 3 | ctx-dependent | A/C: long-ctx attn | GLM DSA indexer |
| **L8** Warm-start + persisted compile caches | both | iteration cycle | cold tax ~500s→once | #3051, #3052 |
| **L9** Real prefill path (chunked + FA) | server 3 | stand up C | prefill | chunked prefill |
| **L10** Native fak resident-EP device-NCCL | server 3 | pure-kernel witness | A/B on fak's own kernel | #1482 |
| **L11** server 2 cpu-offload expert path off host (PCIe-resident partial / int8 host GEMM) | server 2 | escape 0.23→? | §3.4 host wall | #971 direction |
| **L12** Largest GLM-5.2 quant resident on 320 GiB (Q2_K/IQ2/Q3_K_S) + quality/throughput tradeoff | server 2 | fit → own roofline | §3.4 → resident on server 2 | quant sweep, #2336 (q3 serve) |

## 6. The concurrent drive — two nodes, DIFFERENT lanes, one day

The two nodes are **not** interchangeable, so lanes are assigned by what each box can physically
run. Each ticket names a node + lane so workers do not collide; acceptance is a recorded
`experiments/benchmark/runs` artifact, never a self-report.

- **GPU server 3 (80 GiB/card — the GLM-5.2-resident node)** carries **all resident levers**:
  - Lane A — single-stream decode (L1 tensor split, L4 FA+graphs, L5 quant, L3 spec-decode)
  - Lane B — batched aggregate throughput (L2 continuous batching, L6 int8 expert GEMM)
  - Lane D — real prefill path (L9)
  - Lane E — architecture/kernel levers (L7 true DSA, L10 native fak resident-EP #1482)
  - Lane B-cache — L8 RadixAttention warm-cache A/B
- **GPU server 2 (40 GiB/card — GLM-5.2 does not fit resident)** carries its **own-law lanes**:
  - Lane G-offload — L11 cpu-offload expert path off host (its GLM-5.2 baseline is 0.23–2.62 tok/s)
  - Lane G-fit — L12 largest-quant-that-fits-320GiB resident serve + quality tradeoff
  - Lane H-stock — the stock-engine (SGLang/vLLM) baseline for models that DO fit 40 GiB
    (Qwen3.6-27B: 93.05 single / 607.7@8 already recorded) — the capability floor of the box
  - Lane F — ceiling ground-truth (measure active-params/bytes from the GGUF header): **arch-only,
    box-agnostic**, so it can run on GPU server 2 without a resident serve
- **Lane C (both)** — fast-iteration serve+bench harness + warm-start (L8, #3051/#3052) so every
  lever is measurable in <10 min per node.

The epic (#3073) and its children live in GitHub; this note is the ceiling they aim at. When
Lane F replaces the two ESTIMATED inputs with GGUF-header truth, every ceiling here is re-derived
from the measured active set — the number moves, the method does not.

*Companions:* [per-box baseline + next steps](GLM52-DGX-PER-BOX-BASELINE-AND-NEXT-STEPS-2026-07-06.md) ·
[8-GPU full-resident serve (23.2 tok/s WITNESSED, GPU server 3)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) ·
[decode path to 10 tok/s (levers)](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md) ·
[expert-parallel multi-GPU](GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md) ·
[cold-start vs caching ablation](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md).
