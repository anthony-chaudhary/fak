---
title: "Mac Qwen2.5-7B Q8 GPU-resident decode forward — built, correct, in the batched-bandwidth regime; the kernel-efficiency pass is what's left to parity (2026-06-26)"
description: "The #67 one-command-buffer GPU-resident dense Q8 decode forward, built and measured on the M3 Pro. It is numerically faithful to the CPU Q8 decode (logit cosine 0.999988, real-model token-parity) and engages on Qwen2.5-7B. Clean decode: fak resident 10.0 tok/s vs fak CPU 12.1 vs llama.cpp-Metal 17.73 (0.56x). The host overhead is ~1 ms/token; the whole cost is the GPU int8 weight stream at ~92 GB/s (61% of the 150 GB/s ceiling) vs llama's ~143 GB/s — so the remaining lever is the GEMV kernel-efficiency pass, not the orchestration."
date: 2026-06-26
---

# Mac Qwen2.5-7B Q8 GPU-resident decode forward (2026-06-26)

Run on `node-macos-a` (Apple M3 Pro / Mac15,7, 12 CPU core, 18 GPU core, 36 GB unified,
macOS 26.5, Metal 4), fresh clone built `-tags fakmetal`, over Tailscale SSH from a Windows
box. The clean arm stopped the launchd `com.fak.qwen36-model` llama-server (an EXIT trap
restored it) so fak and llama-bench each had the whole GPU.

## TL;DR

The **one-command-buffer GPU-resident dense Q8 decode forward** (issue #67, the decode twin
of forward.m's `mg_prefill`) is **built and correct**. It moves the WHOLE decode token —
the seven projections (Q8 dequant-GEMV), RMSNorm, RoPE, GQA attention over the resident KV,
SwiGLU, the residuals — into ONE Metal command buffer with the activation resident on the
GPU, so the per-token submit/sync the live decode pays ~7×nLayers times is paid **once**.

- **Correctness: proven.** Logit **cosine 0.999988** + greedy token-parity vs the CPU Q8
  decode on a synthetic dense model (`TestMetalDecodeResidentMatchesCPU`), and on the real
  Qwen2.5-1.5B Q8 the resident decode emits the same coherent text as the CPU path
  (`"The capital of France is" → "Paris"`). It engages on Qwen2.5-7B Q8 (attention bias on).
- **Performance: not yet at parity.** Clean 7B Q8 decode is **10.0 tok/s** vs **fak CPU 12.1**
  and **llama.cpp-Metal 17.73** — **0.56× of llama**.
- **The honest gap is the GEMV kernel, not the orchestration.** A per-token timing split shows
  host marshaling (buffer alloc + KV upload) **~1 ms** and command encode **~0.5 ms** — both
  negligible; the GPU command buffer is **~90 ms/token = the whole cost**. That is the int8
  weight stream (~8.3 GB/token) at **~92 GB/s — 61% of the 150 GB/s device ceiling**. llama.cpp
  reads its Q8 weights at **~143 GB/s (95% of ceiling)**. So the one-command-buffer lever
  landed the decode in the batched-bandwidth regime exactly as the prior diagnosis projected;
  closing the rest needs the **GEMV kernel-efficiency pass** (the hand-tuned-GGML-kernel boundary
  the MODEL-BASELINE doc names throughout).

## The numbers (clean GPU, llama-server stopped)

| 7B Q8 decode (M3 Pro) | tok/s | effective GB/s | vs llama |
|---|---:|---:|---:|
| llama.cpp-Metal (`llama-bench -ngl 99`, build b9707) | **17.73** | ~143 | 1.00× |
| fak CPU Q8 (pure-Go NEON SDOT) | 12.1 | ~100 | 0.68× |
| **fak GPU-resident Q8 (this work, #67)** | **10.0** | ~92 | **0.56×** |

Per-token timing split (`FAK_DECODE_PROF=1`, steady state): `host(alloc+kvup)=0.6 encode=0.5
gpu=89 total=90 ms`. The GPU command buffer is ~99% of the wall.

## What was built

- `internal/metalgemm/decode.m` — the resident decode forward + its MSL kernels: `q8dq_gemv`
  (f16-activation × Q8-weight dequant-GEMV, 8 output rows per threadgroup), `attn_decode`
  (single-query GQA online-softmax over the resident KV), and f16 `d_rmsnorm`/`d_rope`/
  `d_silumul`/`d_add` mirroring forward.m's validated kernels. `mg_decode_step` chains a whole
  token into one command buffer.
- `internal/metalgemm/decode.go` — `DecodeConfig`/`DecodeLayer`/`DecodeStep` bindings.
- `internal/metalgemm/q8.m` — accessors (`mg_q8_codes_buf`/`mg_q8_scales_buf`/`mg_q8_dims`) so
  the resident forward binds the q8.m weight table into its own encoder.
- `internal/model/metal_decode.go` (+ `_off.go` stub, `_test.go` gate) — registers the dense
  model once and runs the resident decode; hooked into `tokenHiddenQ`.
- Gate: `FAK_METAL_DECODE=1` (or `s.Metal`). Dense Qwen2.5 only (declines hybrid/MoE/QK-norm),
  so it composes with CPU prefill (no f16 upload) and leaves the default CPU decode untouched.

## The kernel-efficiency pass (what's left)

Four `q8dq_gemv` variants were measured clean — scalar lane-per-block (9.6), char4 lane-per-block
(10.0), scalar lane-per-column coalesced (9.8), char4 8-rows-per-threadgroup (10.0). They all
plateau at ~92 GB/s: the decode is DRAM-bandwidth-bound on the int8 weight stream and these
inner-loop / occupancy tweaks do not break past it. Reaching llama.cpp's ~143 GB/s is a dedicated
kernel port (its `mul_mv_q8_0` simdgroup choreography / f16 block scales / mat-vec intrinsics) —
a real, scoped follow-on, the same "hand-tuned assembly vs GGML" boundary fak's x86 decode crossed
with hand-written AVX-512. A secondary saver: fak's f32 block scales cost ~13% extra byte traffic
vs llama's f16 — switching the resident path to f16 scales recovers part of that.

## Honest fences

- **Performant parity with llama.cpp-Metal is NOT YET reached** (0.56×). What is proven is
  correctness parity + that the one-command-buffer orchestration lever works end-to-end; the
  residual is a single, named, GPU-only kernel pass.
- Single-run greedy numbers on one prompt; a multi-rep authority row follows once the kernel pass
  lands and the number is worth pinning.
- The resident decode v0 re-uploads the KV context f32→f16 each step (cheap at these lengths, ~0.6
  ms); persistent on-GPU KV is a follow-on but is NOT on the critical path (the GPU GEMV is).

## Reproduce

```bash
# on node-macos-a, clone built -tags fakmetal -> ~/fak-67/.bin/{modelbench,fakchat}
G=~/.cache/fak-models/gguf/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
# stop the launchd llama-server for a clean GPU, restore after (EXIT trap):
launchctl bootout gui/$(id -u)/com.fak.qwen36-model
./.bin/modelbench -gguf $G -lean -decode-steps 48 -decode-reps 2 -prefill-sizes 16            # fak CPU
FAK_METAL_DECODE=1 FAK_GPU_LEASE_NOWAIT=1 ./.bin/modelbench -gguf $G -lean -decode-steps 48 -decode-reps 2 -prefill-sizes 16  # fak resident
~/.local/llamacpp-b9707/llama-b9707/llama-bench -m $G -ngl 99 -p 0 -n 64                      # llama.cpp-Metal
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.qwen36-model.plist
# correctness: go test -tags fakmetal -run TestMetalDecodeResidentMatchesCPU ./internal/model
```
