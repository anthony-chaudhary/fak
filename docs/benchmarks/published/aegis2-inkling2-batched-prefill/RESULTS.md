---
title: "Campaign record: Inkling-2-Mini batched CUDA prefill on rtx5080-host-2 RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-2-inkling2-batched-prefill (2026-11-24): Inkling-2-Mini batched CUDA prefill on rtx5080-host-2 RTX 5080."
---

# Inkling-2-Mini batched CUDA prefill on rtx5080-host-2 RTX 5080 (2026-11-24)

Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs are under [`receipts/`](receipts/).

## Result

Prefill tok/s, ABBA (A B B A), profile off, same binary, arm chosen by `FAK_INKLING2_BATCHED_PREFILL`:

| prompt | A: per-token decode step | B: batched prefill | speedup |
|---|---|---|---|
| pp19 | 91.50 / 91.37 | 162.16 / 161.54 | 1.77x |
| pp512 | 103.88 / 103.87 | **862.74 / 862.05** | 8.30x |
| pp2048 | 79.22 / 79.22 | **704.51 / 704.66** | 8.89x |

- Landed: public fak 66bcd27c6e. It was measured as lane commit 4fd9c8bab0 rebased on 6a0e1f1fcc; the land replayed it onto ad095f8aa3. The per-token arm already carries the sibling decode climbs, about 118 tok/s decode.
- A and B pick the same next token at every length.
- For reference, llama.cpp upstream pp512 is 2360 tok/s and the realugbun fork 5106 tok/s on the same card (docs/benchmarks/rtx5080-inkling2-realugbun-fork-ab-20261122). The batched path is at 37% of upstream pp512.
- The flag is default **off** (`FAK_INKLING2_BATCHED_PREFILL=1` enables it). The per-token step stays the default and is the fallback.

## Parity

- Batched vs per-token: `TestInkling2CUDABatchedPrefillMatchesPerToken` passes. A 1300-token prompt crosses 3 chunks and wraps the 1024 local window, and the 8 greedy tokens after it are identical on both paths.
- Greedy oracle against llama.cpp CPU, batched arm: capitals, fibonacci and chat-template are green.
- passphrase-recall-1946 stays red on both arms, at the known 0.21-nat near-tie.
  - Per-token path: diverges at token 10 (153536 vs 150409), unchanged from the resident receipt.
  - Batched path: token 10 now matches llama.cpp (150409). It diverges at token 11 (150008 vs 149884).
  - The first batched build (B1, scalar FP32 expert GEMMs) passed all 32 tokens. The flip is reduction-order noise at the tie, not a cache-handoff defect. Not chased, per the ticket.

## What the batched path does

Token-axis batches the resident forward in 512-token chunks (`internal/ggufload/inkling2loop_prefill.go`, kernels in `internal/compute/cuda/inkling2_prefill.cuh`):

- Dense Q/K/V/rel/gate/o/router projections go through the existing MMQ/GEMM matmuls with P=T.
- Experts are bucketed by expert on the device (once per layer), then run as grouped GEMMs:
  - NVFP4: a BF16 tensor-core `mma.m16n8k16`. E2M1 times E4M3 is exact in BF16, so only summation order changes.
  - IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/Q3_K: an FP32 tile kernel that decodes each super-block once per 32-pair tile.
- Attention is masked rel-bias with sinks per (head, token): the rel-bias row is zero past extent 1024, and local layers have the SWA-1024 mask.
  - Local KV rings grow to window+chunk so a chunk never evicts a key an earlier query in it still needs.
- The short conv walks the token axis per channel and writes the final history, which the per-token decode then continues from.
- Mix and the loop controllers run per token. Only the last token runs the head.

## Climb (pp512 batched, profile ON single run)

| build | change | pp512 tok/s |
|---|---|---|
| B1 | correct baseline: scalar grouped NVFP4, per-pair IQ GEMV | 132 |
| B2 | NVFP4 grouped GEMM on BF16 tensor cores; float attention softmax; tiled ctrl_apply | 298 |
| B3 | grouped IQ/Q3_K tile kernel; attention weights normalized once (no per-key f64 divide) | 635 |
| B4 | 32-pair tiles; IQ activations reused across rows | 711 |
| B5 | IQ blocks of 64 rows (16 warps) in 16-pair halves | 789 |
| B6 | bucket once per MoE layer with a parallel stable fill; conv over 72 blocks | 811 |

The profile-off number for the same tree is 862 (ABBA above).

## Next binding kernel

From [`receipts/prof-R1-2048.txt`](receipts/prof-R1-2048.txt) (pp2048, profile on):

1. `k_ink2p_attend` is 28%. It recomputes q·k per (head, query) with one block each and has no KV-tile reuse across queries. A flash-style tiled kernel with the rel-bias and sink folded in is the next move.
2. `k_ink2p_expert_iq` is 16% (6.5 ms/call, about 2 GB of L2 activation staging).
3. `k_ink2p_expert_nvfp4` is 16% (1.5 ms/call, weight-bandwidth bound on per-tile re-reads).

Then come SITU (FP64 transcendentals), ctrl_down, and the Q8 MMQ projections.
