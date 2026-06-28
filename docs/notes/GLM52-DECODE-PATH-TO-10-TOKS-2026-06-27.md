---
title: "GLM-5.2 on fak: the decode path to 10 tok/s (root cause + levers)"
description: "Why the real 753B GLM-5.2 cpu-offload serve decodes at <0.1 tok/s on DGX3, the host expert-kernel ceiling measured on a 32-core dev box, and the prioritized lever decomposition (pure-CPU vs hybrid, batched expert dispatch, a vectorized int8 reducer) that gets the pure fak kernel to 10 tok/s — with the DGX3 experiment matrix to confirm it."
---

# GLM-5.2 decode: the path to 10 tok/s on the pure fak kernel

_2026-06-27._ Companion to
[GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md)
and [GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md).
It targets the **real 753B `glm_moe_dsa` serve** (UD-Q4_K_M, ~436 GB), not the synthetic
reduced-layer `glmdsatput` micro-number. The goal: drive sustained **decode ≥ 10 tok/s**
on the **pure fak kernel** on DGX3 (8× datacenter GPU-80GB, sm_80, ~2 TB host RAM).

## TL;DR

The cpu-offload hybrid (`--backend cuda --cpu-offload-experts`) measured **< 0.1
tok/s** (prior session — 16/32-token smokes timed out at 0 tokens). The number is
**not** a kernel-arithmetic wall and **not** a load-speed problem. The leading
cause is **structural device glue**: the GLM-DSA forward issues ~12 *synchronous*
device↔host round-trips **per layer** (one `cudaMemcpy`+`be.Read` stream-sync per
dense projection), and at ~92 layers that is **~1100 synchronous round-trips per
decoded token**. The expert kernel quant lever (int8 vs f32 dequant) is real but
**secondary** (~1.3–1.8×).

So the climb from < 0.1 to 10 tok/s decomposes into three independent levers, in
priority order:

| # | Lever | Est. gain | Where | Code? |
|---|---|---|---|---|
| 1 | **Drop the device glue** — serve pure-CPU (no `--backend`), or make the dense forward device-resident | ~30× | serve config / future device-resident forward | none (config) |
| 2 | **Batch the MoE expert dispatch** — one parFor across all active experts per proj, not ~24 tiny parFors/layer | ~1.8× | `internal/model` host expert path | yes |
| 3 | **Vectorize the int8 reducer** — AVX2/AVX512-VNNI Q4_K / K-quant GEMV (scalar caps ~0.55 GiB/s/core) | ~2–4× | `internal/model` amd64 kernel | yes (asm) |

Levers 1×2 alone land in the **~5–8 tok/s** band on a 256-core host (estimate
below); clearing 10 reliably wants lever 3 (which is also why llama.cpp's CPU
GLM-class decode beats fak's today — its AVX512-VNNI kernels).

## What is MEASURED vs INFERRED (kept honest)

**Measured:**

- The hybrid serve decodes **< 0.1 tok/s** on DGX3 (prior session; smoke `max_tokens 8`
  timed out at 120 s ⇒ > 15 s to first token). Witnessed in the dgx3 collection note.
- The **host expert-GEMV kernel ceiling** (this session, 32-core amd64 dev box, the
  same `!arm64` scalar path DGX3's host CPUs run; reproducible via
  `BenchmarkGLMExpertDispatch{Looped,Batched}` in `internal/model`):
  - single-core Q4_K GEMV ≈ **0.55 GiB/s** (int8 ≈ f32 single-core — the int8 win is
    parallel-only, from quantizing the activation once and reusing it across rows);
  - parallel scaling is **poor**: a per-expert loop of GEMVs reaches only **~6–9×** of
    32 cores; **batching all experts into one parFor recovers ~1.8×** (4.3 → 7.4 GiB/s
    int8, still only ~14× of 32 cores; the looped path also does ~7× the allocations);
  - the int8-vs-f32 *aggregate* speedup at MoE-layer scale is **~1.3×** scalar.
- The forward issues **~12 device round-trips per layer** in the hybrid (code-counted:
  `glmDsaIndexStep` 4 projections + `indexSelect`, `glmDsaAppendAttentionKV` 4
  projections, `glmDsaAttendCached` `sparseAttend` + `o_proj`, `glmMoeFFN` router — the
  experts themselves are host-resident in the split). `q_a_proj` is even computed
  **twice** per layer (index step *and* kv step).

**Inferred (the leading hypothesis, not yet directly timed):** that the ~1100
synchronous round-trips/token dominate the hybrid's < 0.1 tok/s. Each `backendKernel.mul`
does `uploadHostF32Class` (H2D) → `MatMul` → `be.Read` (D2H **stream-synchronize**); a
contended GPU makes each sync ms-scale, and ~1100 × ~10 ms ≈ 11 s/token ≈ 0.09 tok/s —
which matches the measured number. The code comment names the design explicitly: *"The
device↔host copy per GEMM keeps the glue simple (correctness-first); a fully
device-resident GLM-DSA forward is the next slice."* **Next checkable step:** a per-op
timing pass on DGX3 (decode-only, after prefill) to confirm the round-trip share before
investing in lever 1's device-resident variant.

## Why not "resident across all 8 GPUs"?

The 436 GB model would fit in 640 GB of aggregate VRAM, but fak's CUDA backend is
**single-GPU** today: `fcuda_init` pins `cudaSetDevice(0)`, `DeviceMemory()` reports one
GPU's `totalGlobalMem`, and there is **no** device-side collective (NCCL/RCCL) — the
tensor-parallel seam is shipped but CPU-only (hardware-gated, #295). So multi-GPU
resident sharding is a separate, larger effort; it is **not** the near-term path to 10
tok/s. The near-term path keeps the experts on the host (where they already are under
`--cpu-offload-experts`) and removes the per-token tax around them.

## The pure-CPU ceiling estimate (DGX3 host)

Treating the experts as a host weight stream (they dominate the parameter count), and
using the measured batched-int8 throughput:

- active expert stream/token ≈ `(K + shared) × moe_layers × per-expert-bytes`. At
  estimated GLM-5.2 dims (H≈5120, expert-intermediate≈1536, K≈8, ~89 MoE layers) this is
  **~10 GiB/token** (dims are estimates — the real values come from the GGUF header; the
  tok/s scales **inversely** with this number).
- batched-int8 host throughput measured **~7.4 GiB/s on 32 cores** ⇒ ~0.7 tok/s here.
  The kernel is **compute-bound** (far below memory bandwidth), so it scales with cores
  until bandwidth saturates: a 256-core DGX host at ~8× ⇒ **~5–6 tok/s** (lever 1+2,
  scalar int8). Lever 3 (vectorized reducer, ~2–4×) carries it past 10.

This is an **estimate**, not a serve measurement — the real number needs a free host
(see matrix). But it already says the right config is **pure-CPU, not the hybrid**: the
hybrid pays ~1100 device syncs/token to run a *small* fraction of the FLOPs on the GPU,
while the experts (the bulk) sit on the host either way. Serving pure-CPU on DGX3's host
cores also **frees the GPUs** for other users — a good-neighbor bonus on a shared box.

## DGX3 experiment matrix (run when a host is free)

One GLM-class load at a time; gate on `free -g`; NVMe-first weights
(`/mnt/sglang_dv3/glm52-q4/`); the GPUs/ports 8000-8001 may belong to another user.

| # | Config | Command sketch | Measures |
|---|---|---|---|
| A | hybrid, default | `fak serve --gguf S1 --backend cuda --cpu-offload-experts` | reproduce < 0.1 tok/s baseline |
| B | hybrid + int8 experts | `FAK_KQ_INT8=1 … --backend cuda --cpu-offload-experts` | int8 lever on the hybrid (expect small — glue-bound) |
| C | **pure-CPU + int8** | `FAK_KQ_INT8=1 fak serve --gguf S1` (no `--backend`) | the lever-1 win: decode tok/s with NO device glue |
| D | per-op decode timing | C + a decode-only timing pass | confirm the round-trip share / kernel share |

Decode tok/s should be read **after** prefill completes (the goal is sustained decode,
not first-token latency). Capture `fak_*` /metrics + `FAK_WORKERS`/`FAK_BUDGET`.

## Status

**Not yet at 10 tok/s.** Root cause localized (device glue, with the host-kernel ceiling
quantified); levers prioritized; the confirming measurement is host-gated (DGX3 and the
CPU-only DA33 node were both in active peer use this session). Next: run matrix row C/D
on a free host to validate lever 1, then implement levers 2 and 3.
