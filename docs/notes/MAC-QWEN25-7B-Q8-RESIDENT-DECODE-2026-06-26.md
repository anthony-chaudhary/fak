---
title: "Mac Qwen2.5-7B Q8 GPU-resident decode forward — built, correct, near-parity with llama.cpp-Metal (0.95×) (2026-06-26)"
description: "The #67 one-command-buffer GPU-resident dense Q8 decode forward, built, optimized and measured on the M3 Pro. Numerically faithful to the CPU Q8 decode (logit cosine 0.999988, real-model token-parity). Five shipped optimizations took clean 7B Q8 decode from 10.0 -> 16.9 tok/s (0.56x -> 0.95x of llama.cpp-Metal 17.82): a parallel RMSNorm, the LM head on the GPU, f16 block scales, fused bias, and persistent on-GPU KV. The residual ~5% is the per-layer elementwise serialization (~4.8 ms) vs llama's fused forward; fak's projection GEMVs already run at ~154 GB/s, above llama's ~143."
date: 2026-06-26
---

# Mac Qwen2.5-7B Q8 GPU-resident decode forward (2026-06-26)

Run on `node-macos-a` (Apple M3 Pro / Mac15,7, 12 CPU core, 18 GPU core, 36 GB unified,
macOS 26.5, Metal 4), fresh clone built `-tags fakmetal`, over Tailscale SSH from a Windows
box. Every clean number stopped the launchd `com.fak.qwen36-model` llama-server (an EXIT trap
restored it) so fak and llama-bench each had the whole GPU.

## TL;DR

The **one-command-buffer GPU-resident dense Q8 decode forward** (issue #67, the decode twin of
forward.m's `mg_prefill`) is **built, correct, and at near-parity with llama.cpp-Metal**. It runs
the WHOLE decode token — the seven projections (Q8 dequant-GEMV), RMSNorm, RoPE, GQA attention
over the resident KV, SwiGLU, the residuals, the final norm AND the LM head — in ONE Metal command
buffer with the activation + KV resident on the GPU.

- **Correctness: proven.** Logit **cosine 0.999988** + greedy token-parity vs the CPU Q8 decode
  (`TestMetalDecodeResidentMatchesCPU`), and on the real Qwen2.5-1.5B Q8 it emits the same text as
  the CPU path (`"…capital of France? A:" → "Paris"`). Engages on Qwen2.5-7B Q8 (attention bias on).
- **Performance: near-parity, 0.95×.** Clean 7B Q8 decode (M3 Pro, 3 runs × 5 reps, stable) is
  **16.8–16.9 tok/s** vs **fak CPU 12.1** and **llama.cpp-Metal 17.82** — **0.948× of llama**.
- **The residual ~5% is the elementwise serialization, not the matmuls.** A matmul-only diagnostic
  shows the seven projection GEMVs run at **~154 GB/s — above llama's ~143** (fak's Q8 mat-vec is
  not the bottleneck). The full forward's ~59 ms vs the ~54 ms matmul+head floor is the per-layer
  norm/RoPE/attention/SwiGLU/residual chain (~4.8 ms) that llama's fused forward largely hides.

## The numbers (clean GPU, llama-server stopped)

| 7B Q8 decode (M3 Pro) | tok/s | vs llama |
|---|---:|---:|
| llama.cpp-Metal (`llama-bench -ngl 99`, build b9707) | **17.82** | 1.00× |
| **fak GPU-resident Q8 (this work, #67)** | **16.9** | **0.95×** |
| fak CPU Q8 (pure-Go NEON SDOT) | 12.1 | 0.68× |

## How it got there — five shipped optimizations (10.0 → 16.9 tok/s)

Each step was measured clean and committed; the diagnostic that found it is in parentheses.

| step | change | tok/s | vs llama | GPU ms |
|---|---|---:|---:|---:|
| 0 | resident forward, v0 (one command buffer) | 10.0 | 0.56× | ~90 |
| 1 | **parallel RMSNorm** (was 1 GPU thread over H, ×56/token) | 14.1 | 0.80× | ~58 |
| 2 | **LM head + final norm on the GPU** (was a CPU `headQ` tail) | 15.6 | 0.88× | ~62 |
| 3 | **f16 block scales** + fused projection bias (GGUF Q8_0 std, ~6% fewer bytes) | 16.3 | 0.92× | ~59 |
| 4 | **persistent on-GPU KV** (seed once, append in place; no per-step concat/upload) | 16.9 | 0.95× | ~59 |

The decisive find was the matmul-only timing split: the GEMVs were never the problem (154 GB/s),
the **1-thread RMSNorm** was — it cost ~29 ms/token and step 1 alone recovered most of the gap.

## What was built (all shipped on origin/main)

- `internal/metalgemm/decode.m` — the resident forward + MSL kernels: `q8dq_gemv` (f16-activation ×
  Q8-weight dequant-GEMV, f16 scales, 8 rows/threadgroup, fused bias), a threadgroup-reduction
  `d_rmsnorm`, `d_rope`, single-query `attn_decode`, `d_silumul`, `d_add`; `mg_decode_step` chains a
  whole token (incl. final norm + head) into one command buffer over the persistent KV (`gKVk`/`gKVv`).
- `internal/metalgemm/decode.go` — `DecodeConfig`/`DecodeLayer`/`DecodeHead`/`DecodeStep` bindings.
- `internal/metalgemm/q8.m` — accessors so the forward binds the q8.m weight table into its encoder.
- `internal/model/metal_decode.go` (+ `_off.go` stub, `_test.go` gate) — registers the dense model,
  tracks the resident (session, KV-length), runs the forward; hooked into `token()`'s fast Q8 path.
- Gate: `FAK_METAL_DECODE=1` (or `s.Metal`). Dense Qwen2.5 only (declines hybrid/MoE/QK-norm), so it
  composes with CPU prefill (no f16 upload) and leaves the default CPU decode untouched.

## The honest residual (the last ~5%)

fak's projection GEMVs are already faster than llama's (~154 vs ~143 GB/s), so the gap is **not** the
mat-vec kernel. It is the per-layer elementwise chain (2 norms + 2 RoPE + attention + SwiGLU + 2
residual adds), ~4.8 ms/token of dependent dispatches that serialize the forward; llama's hand-tuned
ggml-metal forward fuses/overlaps more of it. Fusing cheap ops did not help (bias fusion was
perf-neutral — dispatch count is not the cost), so closing the last 5% means deeper fusion (norm
into the projection, a key-parallel decode attention) — diminishing returns against a stable 0.95×.

## Honesty fences

- **0.95× is near-parity, not a clean 1.0×.** The fak number is stable (16.8–16.9, low variance) and
  llama is 17.82 ± 0.01, so the ~5% gap is real, not run-to-run noise.
- f16 block scales are the GGUF Q8_0 standard precision; correctness held (cosine 0.999988).
- One decode sequence is resident on the GPU at a time (the GPU lease serializes); a cross-session
  KV-length collision declines to the CPU path and re-seeds — correct, just not append-fast then.

## Reproduce

```bash
# on node-macos-a, clone built -tags fakmetal -> ~/fak-67/.bin/{modelbench,fakchat}
G=~/.cache/fak-models/gguf/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
launchctl bootout gui/$(id -u)/com.fak.qwen36-model   # clean GPU; restore after (EXIT trap)
FAK_METAL_DECODE=1 FAK_GPU_LEASE_NOWAIT=1 ./.bin/modelbench -gguf $G -lean -decode-steps 64 -decode-reps 5 -prefill-sizes 16  # fak resident
~/.local/llamacpp-b9707/llama-b9707/llama-bench -m $G -ngl 99 -p 0 -n 128                                                     # llama.cpp-Metal
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.qwen36-model.plist
# correctness: go test -tags fakmetal -run TestMetalDecodeResidentMatchesCPU ./internal/model
# matmul-only timing split: add FAK_DECODE_MATMUL_ONLY=1 FAK_DECODE_PROF=1
```
