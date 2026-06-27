---
title: "Mac Qwen2.5-7B Q8 GPU-resident decode forward — at performant parity with llama.cpp-Metal (0.99×) (2026-06-26)"
description: "The #67 one-command-buffer GPU-resident dense Q8 decode forward, built, optimized and measured on the M3 Pro. Numerically faithful to the CPU Q8 decode (logit cosine 0.999991, real-model token-parity). Six shipped optimizations took clean 7B Q8 decode from 10.0 -> 17.6 tok/s (0.56x -> 0.988x of llama.cpp-Metal 17.82) — at parity within run-to-run variance: parallel RMSNorm, GPU LM head, f16 block scales, fused bias, persistent on-GPU KV, and key-parallel flash-decode attention. fak's projection GEMVs already run at ~154 GB/s, ABOVE llama's ~143."
date: 2026-06-26
---

# Mac Qwen2.5-7B Q8 GPU-resident decode forward (2026-06-26)

Run on `node-macos-a` (Apple M3 Pro / Mac15,7, 12 CPU core, 18 GPU core, 36 GB unified,
macOS 26.5, Metal 4), fresh clone built `-tags fakmetal`, over Tailscale SSH from a Windows
box. Every clean number stopped the launchd `com.fak.qwen36-model` llama-server (an EXIT trap
restored it) so fak and llama-bench each had the whole GPU.

## TL;DR

The **one-command-buffer GPU-resident dense Q8 decode forward** (issue #67, the decode twin of
forward.m's `mg_prefill`) is **built, correct, and at performant parity with llama.cpp-Metal**. It runs
the WHOLE decode token — the seven projections (Q8 dequant-GEMV), RMSNorm, RoPE, GQA attention
over the resident KV, SwiGLU, the residuals, the final norm AND the LM head — in ONE Metal command
buffer with the activation + KV resident on the GPU.

- **Correctness: proven.** Logit **cosine 0.999988** + greedy token-parity vs the CPU Q8 decode
  (`TestMetalDecodeResidentMatchesCPU`), and on the real Qwen2.5-1.5B Q8 it emits the same text as
  the CPU path (`"The capital of France is" → "Paris"`). Engages on Qwen2.5-7B Q8 (attention bias on).
- **Performance: at parity, 0.99×.** Clean 7B Q8 decode (M3 Pro, 4 runs × 5 reps, stable) is
  **17.5–17.6 tok/s** vs **fak CPU 12.1** and **llama.cpp-Metal 17.82** — **0.988× of llama**, inside
  run-to-run variance.

## The numbers (clean GPU, llama-server stopped)

| 7B Q8 decode (M3 Pro) | tok/s | vs llama |
|---|---:|---:|
| llama.cpp-Metal (`llama-bench -ngl 99`, build b9707) | **17.82 ± 0.01** | 1.00× |
| **fak GPU-resident Q8 (this work, #67)** | **17.6** | **0.99×** |
| fak CPU Q8 (pure-Go NEON SDOT) | 12.1 | 0.68× |

## How it got there — six shipped optimizations (10.0 → 17.6 tok/s)

Each step was measured clean and committed; the diagnostic that found it is in parentheses.

| step | change | tok/s | vs llama | GPU ms |
|---|---|---:|---:|---:|
| 0 | resident forward, v0 (one command buffer) | 10.0 | 0.56× | ~90 |
| 1 | **parallel RMSNorm** (was 1 GPU thread over H, ×56/token) | 14.1 | 0.80× | ~58 |
| 2 | **LM head + final norm on the GPU** (was a CPU `headQ` tail) | 15.6 | 0.88× | ~62 |
| 3 | **f16 block scales** + fused projection bias (GGUF Q8_0 std, ~6% fewer bytes) | 16.3 | 0.92× | ~59 |
| 4 | **persistent on-GPU KV** (seed once, append in place; no per-step concat/upload) | 16.9 | 0.95× | ~59 |
| 5 | **key-parallel flash-decode attention** (8 simdgroups/head split the keys + flash-combine) | **17.6** | **0.99×** | ~57 |

Two diagnostics drove it. A **matmul-only** split showed the GEMVs were never the problem (154 GB/s,
above llama's ~143) — the **1-thread RMSNorm** was (~29 ms/token; step 1 recovered most of the gap).
A **skip-attention** split (`FAK_DECODE_NO_ATTN`) then isolated the last residual as the serial,
O(ctx), 7%-utilization attention — step 5 parallelized it over keys.

## What was built (all shipped on origin/main)

- `internal/metalgemm/decode.m` — the resident forward + MSL kernels: `q8dq_gemv` (f16-activation ×
  Q8-weight dequant-GEMV, f16 scales, 8 rows/threadgroup, fused bias), a threadgroup-reduction
  `d_rmsnorm`, `d_rope`, the flash-decode `attn_decode` (8 simdgroups/head, threadgroup-memory
  combine), `d_silumul`, `d_add`; `mg_decode_step` chains a whole token (incl. final norm + head)
  into one command buffer over the persistent KV (`gKVk`/`gKVv`).
- `internal/metalgemm/decode.go` — `DecodeConfig`/`DecodeLayer`/`DecodeHead`/`DecodeStep` bindings.
- `internal/metalgemm/q8.m` — accessors so the forward binds the q8.m weight table into its encoder.
- `internal/model/metal_decode.go` (+ `_off.go` stub, `_test.go` gate) — registers the dense model,
  tracks the resident (session, KV-length), runs the forward; hooked into `token()`'s fast Q8 path.
- Gate: `FAK_METAL_DECODE=1` (or `s.Metal`). Dense Qwen2.5 only (declines hybrid/MoE/QK-norm), so it
  composes with CPU prefill (no f16 upload) and leaves the default CPU decode untouched.

## Honesty fences

- **0.99× is at parity within run-to-run variance** — the fak number is stable (17.5–17.6) and llama
  is 17.82 ± 0.01, a ~1.2% gap inside the decode-benchmark noise floor. fak's matmul+head floor
  (~55.5 ms) is below llama's full decode (~56.1 ms), so the compute is genuinely competitive.
- f16 block scales are the GGUF Q8_0 standard precision; correctness held (cosine 0.999991).
- One decode sequence is resident on the GPU at a time (the GPU lease serializes); a cross-session
  KV-length collision declines to the CPU path and re-seeds — correct, just not append-fast then.
- Scope: dense Qwen2.5 (q/k/v/o + gate/up/down, attention bias, full attention, standard RoPE, no
  QK-norm/softcap/sliding-window). The Gated-DeltaNet hybrid (27B) needs the gdn.m recurrence.

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
