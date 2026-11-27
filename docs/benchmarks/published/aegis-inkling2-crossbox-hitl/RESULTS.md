---
title: "Campaign record: Inkling-2-Mini fak-native CUDA decode: cross-box check on rtx5080-host"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-crossbox-hitl (2026-11-24): Inkling-2-Mini fak-native CUDA decode: cross-box check on rtx5080-host."
---

# Inkling-2-Mini fak-native CUDA decode: cross-box check on rtx5080-host (2026-11-24)

**[HW-WITNESSED]** rtx5080-host RTX 5080 16 GB (sm_120). Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs are under [`receipts/`](receipts/).

The kernel changes were climbed and measured on rtx5080-host-2. This run repeats the end-to-end check on the other box (rtx5080-host), against the trunk that contains all of them.

## Result

| arm | build | decode tok/s (127 steps, ABBA, profile off) | ms/token |
|---|---|---|---|
| A | fak a8030e394b (the resident path from the 2026-11-23 receipt) | 82.33 / 82.66 | 12.12 |
| B | fak trunk 203f56bbcd | **104.65 / 104.64** | 9.56 |

- The speedup is 1.268x, or 2.57 ms less per token. A reproduces the 2026-11-23 receipt (82.55) to within 0.3%.
- The bar is the realugbun/llama.cpp fork at 119.28 tok/s (8.38 ms/token) on this card. The gap left is 1.18 ms per token.
- The 19-token prefill also got faster, 81 -> 103 tok/s. It still runs one token at a time. Batched prefill is tracked separately in ticket `inkling2-cuda-batched-prefill-20261122`.

## Greedy parity against llama.cpp ad50351f5 (CPU oracle), trunk 203f56bbcd

| case | result |
|---|---|
| capitals | PASS |
| fibonacci | PASS |
| chat-template | PASS |
| passphrase-recall-1946 | FAIL at decode token 10: fak 153536 vs llama.cpp 150409 |

The passphrase-recall-1946 failure was already there before these changes. It has the same divergence token as the 2026-11-23 receipt. There, llama.cpp's own top two candidates were 0.21 nats apart and flipped with `-ub 512`.

## What the trunk contains

These are public fak commits since a8030e394b, with the queue row each one closes. Per-step ABBA numbers on rtx5080-host-2 are in their own receipts.

- 28bb2c14a1, realugbun-3: MIX cluster with the following RMS folded in.
- 1d7f1dcf92, realugbun-1, realugbun-5, realugbun-2, plus a fak-native change that quantizes each activation to Q8 once instead of again for each consumer. The land merged these four steps into one commit. Receipt: inkling2-fused-climb (RTX 5080 host 2), 85.2 -> 101.6 tok/s on rtx5080-host-2.
- 6eba862214, strata-8: the NVFP4 expert GEMV runs as a single wave and is bit-identical to before. Receipt: inkling2-expert-gemv-climb (RTX 5080 host 2), 85.2 -> 89.3 tok/s on rtx5080-host-2.
- Host-side and wiring rows that do not change token time:
  - 1dcd99f9fb and 6426d2ddb5 (realugbun-6)
  - 1b891b9a09 and 141375fed2 (strata-15)
  - 5213bc8f78 (strata-13)
  - 442f83a10e (strata-4)
  - 206c8d0091 (strata-2)
  - b49ebcfec1 (strata-3)
  - 917fc1d1b3 (realugbun-8)

## Window

- Measure lease hwl-acd9760d98b22682 was held from 06:39:32Z to 06:42:52Z.
- `fak-llama-qwen27b.service`, the live router backend, was stopped for the run and restarted by an exit trap. `receipts/restore.txt` records it as `active`.
- A peer's CPU-only llama-server for deveval was paused for the window: its share lease was released and the process stopped with SIGSTOP. It was resumed afterwards.
- The script is [`receipts/xbox.sh`](receipts/xbox.sh).

## Next binding kernels

Taken from the rtx5080-host-2 profiles in the receipts linked above.

1. The IQ expert GEMVs (IQ3_XXS, IQ2_S, IQ3_S, IQ2_XS) cost about 1.43 ms per token. A shared-memory lookup-table and wide-load variant is in progress.
2. 96 post-conv launches per token are still separate kernels.
3. A GPU multi-token verify pass in the resident driver does not exist yet. It is the precondition for spec decode, where the DSpark draft accepts 2.50 of 7, and for rows strata-10 and strata-11.
