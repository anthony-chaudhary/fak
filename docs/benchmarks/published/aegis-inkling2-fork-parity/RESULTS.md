---
title: "Campaign record: Inkling-2-Mini fak-native CUDA decode vs the realugbun/llama.cpp fork on the same card, same window (rtx5080-host, 2026-11-24)"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-fork-parity (2026-11-24): Inkling-2-Mini fak-native CUDA decode vs the realugbun/llama.cpp fork on the same card, same window (rtx5080-host, 2026-11-24)."
---

# Inkling-2-Mini fak-native CUDA decode vs the realugbun/llama.cpp fork on the same card, same window (rtx5080-host, 2026-11-24)

**[HW-WITNESSED]** rtx5080-host RTX 5080 16 GB (sm_120). Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs are under [`receipts/window2`](receipts/window2/) and [`receipts/window3`](receipts/window3/).

Both windows ran the fak arms and the realugbun fork on the same box, interleaved within one window.

- Fak arms ran in ABBA order: `TestInkling2CUDADecodeBench`, capitals prompt, 127 decode steps, profile off.
- The fork arm is `llama-bench -ngl 99 -fa 1 -p 0 -n 128 -r 3` at 2eb2530e, the same harness as the 119.28 bar receipt.
- The harnesses are different programs, so read small differences as parity, not a lead.

## Result

| window | arm | build | decode tok/s |
|---|---|---|---|
| 2 (07:35Z) | A | fak 203f56bbcd | 104.04 / 104.65 |
| 2 | B | fak ad095f8aa3 | 118.40 / 118.40 |
| 2 | fork | realugbun 2eb2530e tg128 | 119.54 / 119.50 |
| 3 (07:44Z) | A | fak ad095f8aa3 | 114.36* / 118.44 |
| 3 | B | **fak b64405de76** | **119.90 / 119.88** |
| 3 | fork | realugbun 2eb2530e tg128 | **119.50 / 119.44** |
| 3 | fork, depth 1946 | tg32 @ d1946 | 113.81 |
| 3 | fak b64405de76, depth 1946 | oracle passphrase case, 31 decode steps | 88.97 |

\* The first A run of window 3 is a cold-start outlier. The closing A run reads 118.44, which matches window 2's B (118.40) for the same build.

- At depth 0, fak b64405de76 is at parity with the fork: 119.89 vs 119.47 tok/s (+0.35%).
- Since the 2026-11-23 receipt, fak's resident decode on this card has gone from 82.5 to 119.9 tok/s, a factor of 1.45.
- At 1,946 tokens of context, fak is still behind: 88.97 vs 113.81 tok/s (-22%).
  - The fak figure comes from a 31-step oracle run, not an ABBA, and the fork figure is tg32. Treat the gap as indicative.
  - Most of it is the decode attention's serial double-precision softmax denominator. See the attend step in inkling2-expert-gemv-climb (RTX 5080 host 2).

## Greedy parity against llama.cpp ad50351f5 (CPU oracle)

| build | capitals | fibonacci | chat-template | passphrase-recall-1946 |
|---|---|---|---|---|
| ad095f8aa3 | PASS | PASS | PASS | FAIL at token 10 |
| b64405de76 | PASS | PASS | PASS | FAIL at token 10 |

The passphrase failure was already there before this work, at the same token. llama.cpp's own top two candidates there are 0.21 nats apart.

## What b64405de76 adds over the earlier cross-box receipt (203f56bbcd)

| commit | change | rtx5080-host-2 ABBA |
|---|---|---|
| 6a0e1f1fcc | IQ expert GEMV: grid bytes go through 32 bits instead of I2F.U64 | 107.35 -> 117.56 |
| 2ba9ae6fce | IQ4_NL expert GEMV | +0.2% |
| ad095f8aa3 | Q8_0 decode GEMV with 16-byte loads issued ahead | 117.57 -> 120.85 |
| b64405de76 | attend split over key chunks and dim slices, bit-identical | 117.80 -> 119.22 at depth 0; 69.96 -> 89.31 at depth 1946 |

Batched prefill (66bcd27c6e) is behind `FAK_INKLING2_BATCHED_PREFILL=1` and off by default, so it does not affect these numbers. Its receipt is [rtx5080-host-2-inkling2-batched-prefill-20261124](../rtx5080-host-2-inkling2-batched-prefill-20261124/).

## Windows

- Window 2: measure lease from 07:35:26Z to 07:38:52Z.
- Window 3: measure lease hwl-e37c3c19a8f4ad3e from 07:44:00Z to 07:46:06Z.
- In both, `fak-llama-qwen27b.service` was stopped and restored by an exit trap (`restore.txt`: `active`).
- A peer's CPU-only deveval llama-server was paused with SIGSTOP and its share lease released for each window. Both were resumed afterwards.

## Next

1. Decode attention at depth. A bit-exact tree-sum softmax denominator is not possible, so the change would need to be gated by a KL check (`--kl-baseline`, landed 6426d2ddb5) rather than by greedy identity.
2. Batched prefill: tiled attention, which is 28% at pp2048, before making the flag default-on.
3. Spec decode: a GPU multi-token verify pass in the resident driver (rows strata-10 and strata-11).
