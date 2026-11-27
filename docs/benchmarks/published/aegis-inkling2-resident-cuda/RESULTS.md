---
title: "Campaign record: Inkling-2-Mini resident CUDA decode on rtx5080-host RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-resident-cuda (2026-11-23): Inkling-2-Mini resident CUDA decode on rtx5080-host RTX 5080."
---

# Inkling-2-Mini resident CUDA decode on rtx5080-host RTX 5080 (2026-11-23)

Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs are under [`receipts/`](receipts/).

## Result

| | decode tok/s (127 steps, ABBA, profile off) | ms/token |
|---|---|---|
| fak d09931ed93 (host-driven per-matrix device MatVec) | 6.31 / 6.32 | 158.4 |
| fak 60b536f89f (resident forward + graph replay) | **82.55 / 82.54** | 12.11 |
| llama.cpp ad50351f5, llama-bench tg128 (same card, same GGUF) | 80.99 | 12.35 |

- The resident path is 13.1x faster than d09931ed93 and about 1.02x llama.cpp's tg128. The two use different harnesses, so read the llama.cpp ratio as parity rather than a lead.
- Greedy output: 64/64 tokens identical to the llama.cpp CPU oracle on the acceptance prompts (capitals, fibonacci), and 32/32 on chat-template.
- passphrase-recall-1946 (added in e4de7e2213) is red on **both** CUDA paths:
  - The resident path diverges at decode token 10; the older device path diverges at token 11.
  - llama.cpp's own top two candidates are 0.21 nats apart there, and they flip with `-ub 512`.
  - Tokens 0–9 match after 1946 context tokens, so the local ring KV works past the 1024-token window.

## What changed

The public lane `ticket-inkling2-cuda-decode-resident-20261122`, commit 60b536f89f:

- `internal/compute/cuda/inkling2_loop.cuh` holds the small-op kernels:
  - RMS and depth-mix
  - the fused loop-controller down GEMV and its apply step
  - short conv with dynamic taps
  - q/r/k/v post-processing with the KV ring append
  - relative-bias sink attention
  - warp top-8 routing
  - SITU, the MoE combine, and multi-block softcap+argmax
- The same file has expert-indexed GEMVs for NVFP4, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and Q3_K. The expert ids stay on the device, and the per-expert scale is applied when the output is written.
- `internal/ggufload/inkling2loop_resident.go` is the forward driver. A deterministic per-token arena makes device addresses repeat from token to token, so a single captured CUDA graph replays every token.
- Each token costs one H2D (embedding row plus position) and one 4-byte D2H (argmax).
- Opt-outs: `FAK_INKLING2_RESIDENT=0` switches back to the host-driven path, and `FAK_INKLING2_GRAPH=0` runs eagerly.

## Climb (token-time delta per step)

| step | decode tok/s | protocol |
|---|---|---|
| 1 resident forward, eager | 6.25/6.32 → 33.23/33.24 | ABBA 127 |
| 2 parallel route/qkv/argmax, NVFP4 shared-x | 46.2 | single run, 31 steps |
| 3 one-pass mix and RMS at 1024 threads; graph replay on by default | 60.32/60.26 | ABBA 127 |
| 4 expert scale folded into the GEMV; parallel softmax exp | 65.82 | single run |
| 5 attend weights in shared memory, 512 threads | 77.86/79.24 | ABBA 127 |
| 6 controller pair-RMS fused into the down GEMV | 82.55/82.54 | ABBA 127 |

## Next binding kernel

The profile-on run is [`receipts/profile-step6.txt`](receipts/profile-step6.txt): 15.8 GPU ms/token across 1689 ops.

1. `k_ink2_expert_nvfp4` is 16% of GPU time and reaches only about 540 GB/s of the card's 960. A wider-load or tensor-core NVFP4 expert GEMV is the next move.
2. The 197 Q8 activation quantizations per token are redundant: q, gate, rel and sink all quantize the same input.
3. The IQ3_XXS, IQ2_S and IQ3_S expert GEMVs take about 100 µs per call.
