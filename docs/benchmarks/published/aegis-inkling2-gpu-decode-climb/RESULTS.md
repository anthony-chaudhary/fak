---
title: "Campaign record: Inkling-2-Mini host-driven CUDA decode climb on rtx5080-host RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-gpu-decode-climb (2026-11-23): Inkling-2-Mini host-driven CUDA decode climb on rtx5080-host RTX 5080."
---

# Inkling-2-Mini host-driven CUDA decode climb on rtx5080-host RTX 5080 (2026-11-23)

Ticket `inkling2-gpu-decode-climb-20261122`. Public fak 03a73bb075.

## Verdict

The host-driven CUDA decode path (the per-matrix device path; behind the
resident forward it is the `FAK_INKLING2_RESIDENT=0` fallback) went from
6.30 to 17.48 tok/s, about 2.8x. That is a profile-OFF ABBA of 127 decode
steps on the capitals prompt, re-measured after rebasing onto the resident
forward a8030e394b (W4). The original capitals and fibonacci oracle gives
64/64 greedy tokens identical to the llama.cpp ad50351f5 CPU oracle.

The brief's stated gap was stale. The IQ2_S, IQ3_XXS and IQ3_S GEMVs had
already landed in d09931ed93. The real wall was found by profiling:

| Step | What changed | Decode tok/s | Where |
|---|---|---|---|
| base d09931ed93 | per-GEMV upload, MatMul, Read (2 blocking copies each) | 6.29 / 6.31 / 6.32 / 6.33 | W1/W3 A arms |
| p1 | host forward groups independent MatVecs (`Inkling2LoopBatcher`) | 6.55 | W1 single |
| p2 | `MatVecPacked`: one lock, one H2D, N GEMV + async D2D, one D2H per batch | 6.74 | W1 single |
| p3 | `SituMoE`: fused gate-up, situ, down, weighted sum, one round trip per MoE | 7.09 / 7.10 (ABBA) | W1 |
| p4 | warp-per-row NVFP4 W4A16 decode GEMV for the batched paths | 17.57 / 17.52 (ABBA) | W3 |

## Rebase onto the resident forward (W4)

The resident forward landed first as a8030e394b and is now the default device
path. This lane was rebased onto it, and W4 re-measured the
`FAK_INKLING2_RESIDENT=0` fallback with an ABBA against a8030e394b.

- **Fallback ABBA.** a8030e394b ran 6.28 and 6.31 tok/s; this lane ran 17.50
  and 17.46 tok/s.
- **Resident default, no regression.** 81.52 on base and 82.94 on this lane,
  one run each. This lane does not touch the resident kernels.
- **Parity on both paths.** The oracle test now has four cases. capitals,
  fibonacci and chat-template pass on both paths.
- **passphrase-recall-1946 fails on both paths.** It is a llama.cpp
  0.21-nat near-tie.
  - Fallback: diverges at decode token 9 here; the resident lane reported
    the base fallback diverging at token 11.
  - Resident: diverges at token 10.
  - This lane does not claim that case.

## What the profile showed

- **Syncs were not the wall.** p1 to p3 cut about 1150 blocking round trips
  per token to about 3 per layer visit, and gained only 12%.
- **pprof of p3.** `SituMoE.readLocked` spun for 15.0 s over 146 steps,
  about 103 ms/token waiting on the GPU MoE.
- **`FAK_CUDA_TIMESTAMP_PROFILE=1`, W2.** `k_nvfp4_w4a16_f32` took
  87.7 of 97.6 GPU ms/token (90%): 624 calls at 140.5 us each, grid 9x1.
  - That kernel is the reference-order baseline: one thread per output, a
    serial K loop and uncoalesced row reads, so about out/256 blocks on
    84 SMs.
- **p4 kernel.** `k_nvfp4_w4a16_warp_f32` uses the same per-element `_rn`
  math. Each lane owns 16-value scale blocks (one 8-byte packed load and
  four float4 activation loads), and the warp tree-sums the lanes.
  - Result: 9.27 us/call (15x), grid 288x1.
  - GPU time fell to 15.6 of 57.3 wall ms/token, so this path is now
    host-bound.
- **Scope of the warp kernel.** It runs only from the batched decode entry
  points (`MatVecPacked`, `SituMoE`). The generic `MatMul` keeps the serial
  kernel that `TestCUDANVFP4W4A16Parity` holds bit-exact to the CPU W4A16
  oracle, and that test still passes in W3.

## Parity

- `TestInkling2CUDAOracleGreedyMatchesLlamaCpp`: capitals and fibonacci,
  2 x 32 greedy tokens, 64/64 identical to the llama.cpp CPU oracle.
  - Passed for p2 and p3 (W1) and p4 (W3).
- **Reduction order and transcendentals.** The fused path uses CUDA double
  exp/tanh, and the warp kernel splits the K order. Neither is bit-equal to
  the Go host forward. The gate is the llama.cpp oracle, not host bit
  identity.
  - `TestCUDASituMoEMatchesHostComposition` measured 0 max abs diff on its
    fixture.
- **passphrase-recall-1946 is not claimed here.** That oracle case is a
  llama.cpp 0.21-nat near-tie, and the resident lane reports both CUDA
  paths diverge on it.

## Method

- **Node.** rtx5080-host RTX 5080 16 GB sm_120, driver 615.78.08 (`receipts/window-w*.log`).
  - Measure leases: hwl-db29d1047e61108b (W1), hwl-eb1bb6daf71c82e5 (W2),
    hwl-ad92c0782de91cfd (W3) and hwl-aa0381a78a5dbbcc (W4), each released.
  - The W1 release initially failed (`--holder` missing) and was redone by hand.
  - `fak-llama-qwen27b.service` was stopped and restored around each window;
    `/health` was ok afterwards.
- **Arms.** Each arm is a separate `ggufload.test` binary built from one
  lane snapshot plus one patch (`receipts/iwin.sh`, `receipts/window-driver.sh`).
  - Bench: `FAK_INKLING2_BENCH=128 ... -test.run TestInkling2CUDADecodeBench`.
    Decode is 127 steps after prefill, with profile env unset.
  - Profile-ON rows come from separate runs (W2, W3 row 10) and never from
    the token-time arms.
- **Model.** Inkling-2-Mini-UD-Q3_K_XL.gguf mainline, sha256 e2d03e8c92ac...
  (see rtx5080-host-inkling2-native-cuda-20261122).

## Next

- The resident forward (`ticket-inkling2-cuda-decode-resident-20261122`,
  82.5 tok/s) is the default device path.
- If that path still routes NVFP4 experts through the serial kernel, the
  warp kernel is its next climb.
- This host-driven path is now bound by the host forward: 57 ms wall
  against 15.6 GPU ms.
