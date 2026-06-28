---
title: "fak H100 kernel: the next steps to a 5–10× speedup, decomposed"
description: "An evidence-backed decomposition of the measured fak-on-H100 gap (decode 3.75× behind, prefill ~380× behind) into ranked, code-anchored levers — Q8 device decode, a reusable replay-many CUDA graph, the prefill-amortization defect, and tensor-core prefill — with expected multipliers, confidence, and the next checkable step for each."
---

# H100-KERNEL-5X-ROADMAP — how fak's own CUDA kernel gets 5–10× faster on Hopper

> **The honest frame up front.** Every speedup number below is a *projection* gated
> on a measured Hopper run — this host has no NVIDIA GPU / CUDA toolkit, so the
> figures here are derived from the measured H100 baseline + the code that is already
> in tree, not from a new run. The baseline itself is real and measured
> ([GCP-H100-RESULTS.md](GCP-H100-RESULTS.md)). This document PLANS the next steps and
> ships the first executable one (the apples-to-apples Q8 device-decode bench row); the
> kernel changes it scopes are tracked, GPU-gated follow-ons.

## The measured baseline (Qwen2.5-3B-Instruct Q8_0, single-stream, 1× H100 80GB, sm_90)

| Engine | Precision | Prefill tok/s | Decode tok/s |
|---|---|--:|--:|
| llama.cpp CUDA | Q8_0 | 19,310.5 | 361.6 |
| **fak-cuda** | **f32** | **51.0** | **96.3** |
| fak-cpu (pure-Go) | Q8_0 | 109.7 | 15.7 |

Two gaps, with very different shapes:

- **Decode: 3.75× behind** (96.3 vs 361.6). This is almost entirely a **precision /
  memory-bandwidth** gap, not a compute gap — see Lever 1.
- **Prefill: ~380× behind** (51 vs 19,310). And note the tell: **fak-cuda's prefill
  per-token (51 tok/s) is *slower* than its own decode (96 tok/s)**, which is
  structurally backwards — prefill amortizes one weight stream across P tokens and
  should be *much* faster per token than decode. That inversion is a **batching /
  per-op-overhead defect**, not a GEMM-quality problem (the GEMMs already call cuBLAS).
  See Lever 3.

## Why the decode gap is bandwidth, not arithmetic

Single-stream decode is a stack of **GEMV**s (one activation row × each weight matrix).
GEMV is **memory-bandwidth-bound**: the cost is dominated by streaming the weight bytes,
not the multiply-adds. fak-cuda decode runs **f32 weights** (4 bytes/weight); llama.cpp
runs **Q8_0** (1 int8 code + a thin per-block f32 scale ≈ **~1.03 bytes/weight**). So
fak streams **~3.9× more bytes per token** than llama.cpp — which lands almost exactly on
the measured **3.75×** decode gap. The repo's own CPU head-to-head reaches the same
verdict from the other side: "decode is fundamentally memory-bandwidth-bound (streaming
Q8 weights)" ([LLAMACPP-HEADTOHEAD-RESULTS.md](LLAMACPP-HEADTOHEAD-RESULTS.md)).

**The correction this roadmap makes to the record:** the H100 results note and the bench
harness both said fak-cuda runs f32 because "the cuda backend does not advertise
`UploadDtype`." **That is stale.** The CUDA backend advertises `UploadDtype: true`
(`internal/compute/cuda.go:450`) and already implements native Q8_0, Q4_K, F16, and AWQ
device GEMMs (`uploadQ8`/`uploadQ8Resident`/`uploadQ4K`/`uploadF16` in `cuda.go`;
`k_q8_gemm`/`k_q4k_gemm` in `internal/compute/cuda_kernels.cu`). The f32 H100 number was
not a missing capability — **the bench simply never *requested* Q8.** modelbench routes
the HAL through the Q8 device path whenever `-quant`/`-lean` is set against a backend that
advertises `UploadDtype` (`cmd/modelbench/main.go:852`, gate at `:447`).

## The levers, ranked

Ranked by (expected multiplier × confidence ÷ cost). File anchors are exact.

### Lever 1 — Q8 device weights for decode  ·  ~3.9× decode  ·  HIGH confidence  ·  SHIPPED (wiring); GPU-run pending

**What.** Run fak-cuda decode on resident Q8_0 weights (int8 codes + per-block f32 scales,
native `k_q8_gemm` GEMV) instead of f32. Streams ~1 byte/weight instead of 4.

**Where.** Already implemented end-to-end: `cuda.go` `uploadQ8Resident` → `k_q8_gemm`
(`cuda_kernels.cu:336`), HAL Q8 routing at `cmd/modelbench/main.go:852`. Off-GPU cosine
witnesses exist (#485, `cuda_quant_test.go`, `tools/run_485_acceptance_on_gpu.sh`,
floor `cudaQ8CosineMin = 0.999`).

**Shipped here.** `tools/gcp_bench.py` now has a **`fak-cuda-q8`** engine (`-lean -backend
cuda`) — the apples-to-apples row vs llama.cpp Q8_0. It is opt-in until a green Hopper run
witnesses the device Q8 GEMV, then it promotes into the default `all` set.

**Next checkable step.** On a 1× H100:
`python tools/gcp_bench.py --tier a3-high-h100-1g --spot --engine llama,fak-cuda,fak-cuda-q8`.
**Expectation:** fak-cuda-q8 decode ≈ **300–375 tok/s** (≈ llama.cpp Q8 parity), i.e.
~3.9× over the 96.3 f32 row — *and* a first on-hardware correctness pass of the device Q8
GEMV. This run also tells us whether decode is now launch-overhead-bound (Lever 2).

### Lever 2 — Reusable "replay-many" CUDA graph for decode  ·  ~1.5–2× on top of Q8  ·  MED-HIGH  ·  tracked (#35/#3), GPU-gated

**What.** A decode step issues **~500–700 kernel launches** (≈30 layers × {3–5 GEMVs,
2 RMSNorm, RoPE, flash-attn, SwiGLU, 3–4 adds, 2 KV writes} + argmax). Capturing that op
stream into a CUDA graph and **launching it once per token** collapses ~600 launches into
1. The catch already learned in-tree: **per-token re-capture is a no-win** — re-recording
a ~600-node graph every token costs about what the 600 launches cost
(`cuda.go:44`). The real win is **capture once, replay many**.

**What already exists.** The instantiate-once machinery is built: `fcuda_graph_end_launch`
keeps `g_exec` and uses `cudaGraphExecUpdate` rather than re-instantiating
(`cuda_kernels.cu:943-968`); the KV write is a scalar-offset kernel (`k_copyrow`) so the
exec is patchable as the cache grows (`cuda_kernels.cu:535`); `#969` pool pre-warm makes
capture allocation-free; `FAK_CUDA_GRAPH=1` gates it; `cudaKVMaxPos` fixes KV capacity so
no realloc happens during capture.

**The missing piece.** Make the decode graph **length-agnostic** so it is captured ONCE per
session and `cudaGraphLaunch`-ed every subsequent token with **zero per-token CPU/capture
work**. That needs `pos`/`nPos` to be **device-resident scalars** the kernels read, instead
of host launch-params that change every token (which is what forces re-capture today). The
kernels to convert: `k_rope` (reads `pos`), `k_flash_attention` (reads `nPos`), the KV-write
offset, `k_argmax`. Anchor for the position counter: `cuda_kernels.cu:535-544` (the
scalar-offset write is already the template).

**Why it compounds with Lever 1.** Once Q8 cuts weight traffic ~4×, the per-launch overhead
becomes the *dominant* residual: ~600 launches × ~3–5 µs ≈ **2–3 ms/token of pure launch
latency**, i.e. a ~330–500 tok/s ceiling sitting right at llama.cpp's number. Removing it is
what takes fak **past** parity, not just to it. The repo's own RTX-4070 note already shows
fak reaching **decode parity with llama.cpp Q8 using a reusable graph, at f32 precision**
([LLAMACPP-HEADTOHEAD-RESULTS.md](LLAMACPP-HEADTOHEAD-RESULTS.md) intro; `GPU.md` §3b) — so
"graph + Q8" is the combination that should clear it on Hopper.

**Next checkable step.** First, Lever 1's run with `FAK_CUDA_GRAPH=1` to re-confirm the
per-token-capture no-win on H100 (cheap, no code). Then implement device-resident `pos`/`nPos`
and re-measure; success = decode tok/s rising materially above the Q8-only number with the
forward still bit-faithful to cpuref.

### Lever 3 — Fix the prefill amortization defect  ·  large (10–100× on prefill)  ·  MED  ·  needs a phase profile first

**What.** fak-cuda prefill (51 tok/s) being *slower per token than decode* (96) is a
structural defect: prefill should stream each weight once for all P=512 tokens. The GEMMs
already use `cublasSgemm`/`cublasGemmEx` (`cuda_kernels.cu:210`, `:274`), so the defect is
**not** GEMM quality. The likely culprits, in order: per-op overhead with **no prefill
graph** (every op a separate launch + a `devTr` `cudaMalloc`), per-op stream serialization,
or a prefill attention path whose cost is not being amortized across the batch.

**Next checkable step (diagnose before fixing).** The flag already exists:
`modelbench-cuda -gguf <q8> -lean -backend cuda -phase-profile` emits per-phase ms
(`cmd/modelbench/main.go` `runPrefill` → `phaseTable`). Run it on H100, read which phase
dominates a 512-token prefill, and fix that phase specifically. This is the single largest
raw-number headroom and matters most for long-prompt agentic workloads (big system prompts,
tool outputs).

### Lever 4 — Tensor-core / TF32 prefill  ·  large on prefill  ·  HIGH (that it helps)  ·  cheap to wire

**What.** The f32 SGEMM runs on Hopper's **FP32 CUDA cores**, leaving the tensor cores idle.
Two cheap wins for the compute-bound prefill phase: (a) wire an **F16 prefill** row — the
`UploadDtype(F16)` → `cublasGemmEx` tensor-core path is already built (#484,
`uploadF16`/`fcuda_matmul_f16`, floor `cudaFP16CosineMin = 0.997`); and/or (b) enable **TF32**
math mode on the f32 SGEMM (`cublasSetMathMode(..., CUBLAS_TF32_TENSOR_OP_MATH)` at
`cuda_kernels.cu:210`) to route f32 GEMMs through tensor cores at a tiny, disclosed precision
cost. Both are bench-wiring-sized changes (like Lever 1), not kernel rewrites.

**Next checkable step.** Add a `fak-cuda-f16` engine (`-lean` with the F16 upload selected)
or a TF32 toggle, and measure the prefill row against llama.cpp on H100.

## The math to 5–10×

| Path | Levers | Compounded vs current f32 un-graphed fak-cuda |
|---|---|--:|
| **Decode** | Q8 (×3.9) → past parity with graph (×~1.5–2) | **~6–8×** → clears llama.cpp Q8 |
| **Prefill** | amortization fix + tensor-core/TF32 | **10–100×** (its own large, separate headroom) |

So the 5–10× is reachable and decomposed: **decode** via Q8 + the replay-many graph;
**prefill** via the amortization fix + tensor cores. None of it requires new silicon — it is
the same "tuning, not architecture ceiling" boundary the CPU and 4070 head-to-heads already
identified, now on Hopper.

## What shipped in this increment

- **`tools/gcp_bench.py`**: a new opt-in **`fak-cuda-q8`** engine (`-lean -backend cuda`) that
  measures fak's Q8 device decode apples-to-apples with llama.cpp Q8_0; reuses the
  `modelbench-cuda` binary the `fak-cuda` engine builds; corrected the stale "does not
  advertise `UploadDtype`" comments. Test coverage in `tools/gcp_bench_test.py`.
- **This roadmap** — the ranked, code-anchored next steps.

Everything past this increment (Levers 2–4) is a **GPU-gated `not yet`**: the code paths
exist or are scoped, but the measured Hopper number is the witness, and this host cannot
produce it. The first witness to collect is Lever 1's `--engine llama,fak-cuda,fak-cuda-q8`
run.

## Reproduce / drive the next run

```bash
# apples-to-apples Q8 decode head-to-head on a 1x H100 (spot), then teardown
python tools/gcp_bench.py --tier a3-high-h100-1g --spot \
    --engine llama,fak-cuda,fak-cuda-q8

# diagnose the prefill defect (Lever 3) on the same box, if --keep is used
modelbench-cuda -gguf <qwen2.5-3b-q8_0.gguf> -lean -backend cuda -phase-profile
```
