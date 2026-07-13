---
title: "GLM-5.2 per-box baseline + next steps: GPU server 2 (40 GiB/card) vs GPU server 3 (80 GiB/card), 2026-07-06"
description: "The current WITNESSED GLM-5.2 serving baseline for each of the two sm_80 lab nodes and the next-steps plan for each — kept honest by the one fact that splits them: the 433.82 GiB model is resident only on GPU server 3 (640 GiB), never on GPU server 2 (320 GiB). Corrects the earlier 'two identical 80 GiB nodes' assumption and re-homes the mis-assigned drive tickets."
---

# GLM-5.2 per-box baseline + next steps (GPU server 2 vs GPU server 3)

> **The one fact that splits the two boxes.** GLM-5.2 UD-Q4_K_M is **433.82 GiB**. GPU server 3
> has **80 GiB/card → 640 GiB** and holds it **resident**; GPU server 2 has **40 GiB/card →
> 320 GiB** and **cannot** — not even 8-way expert-parallel (54 GiB/rank band > 40 GiB card).
> So the two nodes are not two copies of one lane: **server 3 is the resident node, server 2 is
> the cpu-offload / stock-engine / smaller-quant node.** Everything below follows from that.

This note pairs with the roofline in
[`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md)
(the ceilings + lever map) and the WITNESSED serve in
[`GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md`](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md).
Both boxes are sm_80 (compute 8.0), below the sm_90 DSA kernel floor → **stock SGLang/vLLM refuse
GLM-5.2** (`BLOCKED_ARCH`, vLLM #35021); llama.cpp MLA is the resident path, and fak fronts /
adjudicates either engine.

## GPU server 3 — 8×80 GiB/card (640 GiB) — the GLM-5.2-resident node

### Baseline (WITNESSED, 2026-07-01)

| Metric | Value | Label | Source |
|---|--:|---|---|
| Single-stream decode | **23.2 tok/s** (43.05 ms/tok) | WITNESSED (llama.cpp sm_80 resident) | 07-01 note, two runs 23.23/23.22 |
| End-to-end wall (256 tok) | 22.9 tok/s (11.16 s) | WITNESSED | timed curl, warm |
| Resident footprint | ~434.6 GiB across all 8 GPUs | WITNESSED | `nvidia-smi` per-GPU used |
| Load time (NVMe-first) | ≤6 min to a passed smoke completion | WITNESSED | launch→READY8 |
| vs cpu-offload wall | **~8.9×** the 2.62 tok/s `--n-cpu-moe` baseline | derived | — |
| Cold first-turn tax | ~500 s one-time backend warmup (not a KV miss); warm turns ~0.6–1.8 s | OBSERVED/WITNESSED | cold-start ablation (#3051/#3052/#3053) |

**Engine-honesty fence:** the 23.2 is the **llama.cpp baseline engine, not the pure-fak kernel.**
fak's resident-EP multi-GPU path is code-complete + opt-in on trunk, but a live multi-GPU fak
tok/s is still `not yet` — gated on an on-box `-tags cuda,nccl` build + a live device-NCCL EP run.
Not-yet: aggregate throughput at concurrency, and prefill at depth (the 07-01 note's 46 tok/s
"prompt eval" is an 11-token prompt, not prefill).

### Next steps (drive 23.2 → ~120–160, the 80 % of the ~150–200 practical ceiling)

Ranked by expected single-stream multiplier (see ceiling §5):

1. **L1 — 8-GPU tensor/row split (kill `-sm layer`), ~3–6×.** The dominant lever: today
   llama.cpp layer-split runs **one GPU per token** (7 idle). `-sm row` / RPC / SGLang-TP puts
   all 8 GPUs' bandwidth on every token. → #3075.
2. **L2 — continuous batching, 10–40× aggregate.** Stands up Ceiling B (concurrency), currently
   unmeasured. → #3079, with KV-paging sizing #3080.
3. **L4 — flash-attention + CUDA graphs, 1.2–1.8×.** → #3076.
4. **L5 — quant sweep for the CUDA resident-Q4_K fast path, 1.1–1.5×.** UD-Q4_K_S / Q4_K-pure. → #3077.
5. **L3 — speculative decoding (draft/EAGLE/prompt-lookup), 1.5–2.5×.** → #3078.
6. **L9 — real prefill path (chunked + FA)** — stands up Ceiling C. → #3085/#3086.
7. **L7 — true DSA sparsity (not full MLA)** — long-context attention. → #3088.
8. **L8 — warm-start + persisted compile caches** — kills the ~500 s cold tax per iteration. → #3051/#3052/#3083/#3084.
9. **L10 — native fak resident-EP device-NCCL** — the pure-fak kernel witness (separate from the
   llama.cpp baseline). → #1482.

The #413 serving witness through fak against the live server-3 endpoint
(`tools/glm52_e2e_after_serve_dgx3.sh`) is the natural end-to-end checkpoint on top of any lever.

## GPU server 2 — 8×40 GiB/card (320 GiB) — GLM-5.2 does NOT fit resident

### Baseline (GLM-5.2 is cpu-offload-only here; smaller models fit)

| What | Value | Label | Source |
|---|--:|---|---|
| GLM-5.2 fak cpu-offload steady-state TPOT | **0.2324 tok/s** (4.30 s/tok) | WITNESSED (fak kernel) | 06-28 overnight |
| GLM-5.2 llama.cpp GPU + `--n-cpu-moe` | **2.62** single / 4.84@2 | OBSERVED | 06-25 cross-engine |
| GLM-5.2 concurrency (2 streams) | **0.27×** (worse — serializes) | WITNESSED | 06-28 (shared host bottleneck) |
| Stock SGLang, Qwen3.6-27B (fits 40 GiB), tp8 bf16 | **93.05** single / 607.7@8 (0 err) | OBSERVED | 06-25 cross-engine |

**The bottleneck is host-bound, not GPU-count-bound.** Under `--cpu-offload-experts` the ~13 GiB/token
routed-expert stream (K=8 × 1.619 GiB, DERIVED from the header) is read from host RAM and the expert GEMM runs on the CPU — so concurrency makes it
*worse* (0.27×), and the 7 idle GPUs can't be batched into use as configured. This is the closed-#971
finding; on 40 GiB cards the resident-experts fix that closed it for server 3 is **not available**.

### Next steps (server 2 has its own law — do NOT put the resident drive here)

1. **L11 / Lane G-offload — move the expert GEMM off the host.** On 40 GiB cards the realizable
   forms are a PCIe-resident *partial* expert set (hot experts on-GPU, cold on host) and/or an
   **int8 host expert GEMM** (the VNNI reducers already show 8–18× on the host path). Target:
   escape the 0.23 tok/s wall toward the llama.cpp 0.89 CPU baseline and past it.
2. **L12 / Lane G-fit — the largest GLM-5.2 quant that fits 320 GiB resident.** Below ~2.9 bits/weight
   avg (Q2_K / IQ2 / Q3_K_S territory) the model may fit 320 GiB with KV headroom — turning server 2
   into a *resident* GLM-5.2 node at lower quality. Sweep quant × {fit, decode tok/s, quality delta};
   ties to #2336 (native q3 serve). This is the one path to a real per-token GPU roofline on server 2.
3. **Lane H-stock — the stock-engine capability baseline** for models that DO fit 40 GiB
   (Qwen3.6-27B recorded; extend the SGLang/vLLM concurrency sweep) — the honest "what a stock
   engine does on this box" floor every fak number is measured against.
4. **Lane F — ceiling ground-truth (box-agnostic).** Pin GLM-5.2 active-params + active-bytes/token
   from the GGUF header (no resident serve needed), which re-derives every ceiling from measured
   truth instead of the ESTIMATE. → #3074 (this runs fine on server 2).

## Ticket re-homing (correcting the epic #3073 fleet)

The drive tickets were created from the earlier "two identical 80 GiB nodes" draft, so the
GLM-5.2-**resident** levers were mis-assigned to GPU server 2, which cannot host them. Corrected
assignment (also posted to the tickets):

| Ticket | Was | Now | Why |
|---|---|---|---|
| #3075 (L1 tensor split), #3076 (L4 FA/graphs), #3085 (prefill sweep) | server 3 | **server 3** (unchanged) | resident — correct already |
| #3080 (KV paging), #3079 (L2 batching) | server 2 · Lane B | **server 3 · Lane B** | resident concurrency needs the model in VRAM |
| #3078 (L3 spec-decode) | server 2 · Lane A-spec | **server 3 · Lane A** | resident decode |
| #3077 (L5 quant sweep) | server 2 · Lane A | **server 3 · Lane A** | resident Q4_K fast path |
| #3086 (L9 prefill/FA) | server 2 · Lane D | **server 3 · Lane D** | resident prefill |
| #3087 (L6 int8 expert GEMM), #3088 (L7 DSA) | server 2 · Lane E | **server 3 · Lane E** | resident expert/attn kernels |
| #3074 (Lane F active-set) | server 2 · Lane F | **server 2 · Lane F** (unchanged — box-agnostic, correct) | GGUF-header read needs no resident serve |
| #3090 (roofline dashboard) | server 2 · Lane F | **either** (reads artifacts) | box-agnostic |
| **new** L11 cpu-offload-off-host | — | **server 2 · Lane G-offload** | server 2's real GLM-5.2 lever |
| **new** L12 smallest-quant-that-fits | — | **server 2 · Lane G-fit** | the one path to resident GLM-5.2 on 40 GiB |
| **new** stock-engine baseline | — | **server 2 · Lane H-stock** | the capability floor of the box |

*Companions:* [theoretical ceiling + lever map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) ·
[8-GPU full-resident serve (WITNESSED, server 3)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) ·
[expert-parallel multi-GPU](GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md) ·
[decode path to 10 tok/s](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md) ·
[cold-start vs caching ablation](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md).
