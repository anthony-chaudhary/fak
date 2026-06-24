---
title: "Native Q8_0 / Q4_K device GEMM, recorded on a datacenter sm_80 GPU node (2026-06-24): the native-753B Pillar-2 cosines, run to a number for the first time"
description: "First on-hardware capture of fak's native quantized device-GEMM acceptance (#485) on an idle 8x datacenter GPU-80GB (sm_80, compute 8.0) node, CUDA 12.8, Go 1.26.4. The Q8_0 and Q4_K dequant-fused GEMV/GEMM run on fak's own CUDA kernels (k_q8_gemm / k_q4k_gemm), graded against the cpu-ref f32 Reference: Q8_0 cosine 0.99999980 (gate 0.999), Q4_K cosine 1.00000000 (gate 0.995), both argmax-exact, with the resident weight 3.56x (Q8_0) and 7.11x (Q4_K) smaller than f32. This retires the named Pillar-2 honesty gap in the native-753B staged plan: the device quant cosines had never been run to a recorded number on a GPU node. The raw-FLOP throughput is honestly slower than f32 SGEMM (correctness kernels, not tensor-core tiled) - the win is the VRAM/bandwidth footprint that lets a 753B model fit."
---

# Native Q8_0 / Q4_K device GEMM, recorded on a datacenter sm_80 GPU node (2026-06-24)

> **What this is:** the first time fak's native quantized device-GEMM acceptance (the
> `#485` `-tags cuda` witness) has been **run to a recorded number on a real GPU node**.
> The native-753B staged plan
> ([`native-753b-track-staged-plan.md`](native-753b-track-staged-plan.md)) names this as
> a Pillar-2 gap in two places — "the CUDA Q4_K/Q8_0 cosines have **never been run to a
> recorded number** on a GPU node" and the honesty-ledger "Scale gap" — and this note
> closes it. The numbers are grounded in the `go test` exit code and the run log of
> `tools/run_485_acceptance_on_gpu.sh`, not a self-report.

All runs are on a fresh, idle **8× datacenter GPU-80GB (sm_80, compute 8.0), CUDA 12.8
(`V12.8.61`), ~2 TB host RAM** node, GPU 0, at `origin/main` HEAD `26fe933`, Go 1.26.4.
The kernels were built with `nvcc -arch=sm_80` into `libfakcuda.a`; the test backend
self-reported `device=cuda tier=sm_80 class=approx`.

## 1. The per-dtype cosine gates — all PASS, argmax-exact

Each device quantized GEMM is held to its **own recorded cosine floor** against the
cpu-ref f32 Reference (the `Approx` class — bit-identity is deliberately off the table;
the device quantizes). The weight stays NARROW in VRAM — int8 codes for Q8_0, raw Q4_K
super-block bytes for Q4_K — and the GEMM consumes it directly, no dequant-to-f32 round
trip (Q8_0 quantizes the activation on-device; Q4_K fuses the dequant into the GEMM tile).

| Witness | Shape | cosine | gate | max\|Δ\| | verdict |
|---|---|---|---|---|---|
| `TestCUDAQ8MatMulApproxMatchesRef` (decode GEMV, P=1) | [320,256] | **0.99999980** | 0.999 | 6.91e-03 | PASS, argmax-exact |
| `TestCUDAQ8BatchedMatMulApproxMatchesRef` (prefill GEMM, P=8) | [320,256] | **0.99999969** | 0.999 | 3.64e-02 | PASS |
| `TestCUDAQ4KMatMulApproxMatchesRef` (decode GEMV, P=1) | [320,256] | **1.00000000** | 0.995 | 5.47e-02 | PASS, argmax-exact |
| `TestCUDAQ4KBatchedMatMulApproxMatchesRef` (prefill GEMM, P=8) | [320,256] | **1.00000000** | 0.995 | 1.22e-03 | PASS |

Two facts worth reading carefully:

- **Q8_0 sits a hair below 1.0** because the gate measures *real* quantization error: the
  weight is narrowed to int8 codes + per-block f32 scales at H2D and the activation is
  quantized to int8 on-device, so the cosine is the genuine Q8 weight+activation quant
  error against the unquantized f32 reference. It clears its 0.999 floor with margin.
- **Q4_K reads as exactly 1.0** because that gate isolates the *device dequant-fused tile
  arithmetic*: its reference is an f32 dequant of the **same** super-block bytes, so a
  correct `getScaleMinK4` 6-bit unpack and `w = d·scale·code − dmin·min` reconstruction on
  the device reproduces the host f32 dequant to the cosine's printed precision. This is the
  exact trap the comment in `cuda_quant_test.go` calls out — a wrong GLSL/CUDA port of the
  k-quant geometry silently collapses the cosine — and on this node the port is faithful.
  (The looser 0.995 floor exists for the full-model true-f32 → Q4_K reconstruction residual,
  which this isolated gate does not exercise.)

## 2. VRAM witness — the weight stays int8/int4-sized

Read straight off the device buffers (`residentWeightBytes`), not a host estimate, for a
`[512,256]` weight:

- **f32:** 524 288 B
- **Q8_0:** 147 456 B — **3.56× smaller** (int8 codes + per-block(32) f32 scales)
- **Q4_K:** 73 728 B — **7.11× smaller** (raw super-block bytes, ≈0.56 byte/elem)

This is the whole point for 753B: the resident weight is a small fraction of the f32 bytes
a dequant-to-f32 upload would have paid.

## 3. Throughput — honestly slower in raw FLOPs; the win is footprint

The same harness ran the quantized GEMM benchmarks beside the F32 SGEMM baseline (a
4096×4096 weight, P=512 prefill tile):

| Kernel | ms / GEMM | GFLOP/s | vs F32 |
|---|---|---|---|
| F32 SGEMM (baseline) | 2.867 | 5992.3 | 1.00× |
| Q8_0 (resident int8, on-device act quant) | 34.655 | 495.7 | **0.08×** |
| Q4_K (resident super-blocks, dequant fused) | 81.380 | 211.1 | **0.04×** |

Read this straight: **the quantized kernels are slower than f32 SGEMM in raw FLOP/s.**
They are correctness-first kernels — a dequant-in-the-inner-loop GEMM, not a tiled,
tensor-core path — so they trade arithmetic throughput for a small resident weight. The
win they buy is the VRAM/bandwidth footprint in §2, which is what lets a model that does
not fit in f32 fit at all; it is **not** a speed win yet. Closing the speed gap (tiling +
tensor cores for the quantized path) is separate, later work and is **not** claimed here.

## 4. What this retires, and what is still open

**Retired (Pillar 2, "run the gates to a recorded number"):** the native quant device
GEMM now has a recorded, reproducible cosine on real sm_80 hardware. The staged plan's
P2 `Metal HAL + CUDA witness` milestone had two halves; the **CUDA-witness half is done**
and pinned to the numbers above. The honesty-ledger "Scale gap" line — "the CUDA quant
cosines have never been recorded on hardware" — no longer holds.

**Still open (unchanged, do not over-claim):**

- **Vulkan Q4_K** GEMM (AMD path is Q8_0-only) and the **Metal HAL Q4_K** exposure — the
  other half of the P2 milestone — still need a node each.
- **Full-model mixed-precision forward:** these gates isolate single GEMMs on synthetic
  super-blocks. The end-to-end Q4_K + Q8_0/f32 `glm_moe_dsa` forward witness (P2 "Full-model
  quant forward") still depends on the Pillar-1 GGUF load path.
- **753B serving** remains the labeled multi-month integration: real GGUF → mixed-precision
  device GEMM → multi-GPU TP/EP → tiered offload. This note moves one device-GEMM rung from
  "type-checks on the build host" to "recorded on the GPU"; it does not move the integration.

See [`native-753b-track-staged-plan.md`](native-753b-track-staged-plan.md) for the full
pillar map and the dependency-ordered milestones, and
[`GLM52-REAL-ORACLE-GENERATION-ON-PURE-FAK-2026-06-23.md`](GLM52-REAL-ORACLE-GENERATION-ON-PURE-FAK-2026-06-23.md)
for the matching real-checkpoint generation witness.
