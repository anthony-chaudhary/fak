---
title: "Campaign record: Inkling-2-Mini DSpark speculative decode on the fak-native CUDA resident path (rtx5080-host, 2026-11-24)"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-spec-decode (2026-11-24): Inkling-2-Mini DSpark speculative decode on the fak-native CUDA resident path (rtx5080-host, 2026-11-24)."
---

# Inkling-2-Mini DSpark speculative decode on the fak-native CUDA resident path (rtx5080-host, 2026-11-24)

**[HW-WITNESSED]** rtx5080-host RTX 5080 16 GB (sm_120), measure lease hwl-c12d55d420e0ea06. Every window stopped `fak-llama-qwen27b.service` and restored it by exit trap (`restore.txt`: `active`). Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs: [`receipts/`](receipts/).

Landed public fak `2730bebdff` (ticket inkling2-spec-decode-20261124). Opt-in: `FAK_INKLING2_SPEC=<depth>` plus `FAK_INKLING2_MTP_DIR=<HF snapshot with mtp.safetensors>`; unset keeps plain decode unchanged.

## Result

| arm | build | decode tok/s, ABBA, 127 steps, profile off |
|---|---|---|
| plain | current tree (`wI`, public 93b941b07f) | 119.84 / 119.78 |
| **spec depth 2** | current tree (`wI`, public 93b941b07f) | **151.60 / 151.70** |
| plain / spec depth 2 | first landed tree (`wD`, public 2730bebdff) | 118.14 / 119.89 and 136.70 / 136.68 |

- At depth 2, spec decode is +26.6% over plain on the current tree, 151.7 against 119.8 tok/s: 2.40 tokens per verify, draft 1.67 ms and verify 14.13 ms per round. The first landed tree was +14.0% (verify 15.81 ms); the hill-climb table below has every step.
- Spec decode is opt-in and the batched verify is not bit-exact to the per-token step (see Correctness), so the default-path headline stays plain decode, 119.9 tok/s.
- Batched prompt prefill (on whenever spec is on) is 190 tok/s against 109 tok/s for the per-token prompt path.
- Correctness:
  - `TestInkling2CUDASpecDiag` passes in eager and graph modes. It covers warm rewind (keep 0) and partial commits, and every verify row equals plain greedy.
  - The llama.cpp oracle matches on capitals, fibonacci and chat-template. passphrase-recall-1946 is the pre-existing near-tie red.
  - The batched verify is not bit-exact to the per-token step: with a perfect draft, the first divergence came at token 65 (220 vs 187623), a GEMM-order near tie.

## What was built

- **Verify:** one batched forward over the anchor and the drafts, captured as a CUDA graph.
  - The position is read from device memory.
  - The batched conv logs its entry history and inputs, and a commit kernel keeps the accepted prefix. Rejected KV rows sit past the position and are overwritten.
  - Rows run decode-quality kernels:
    - few-row Q8_0/Q5_K GEMVs, with each weight read once and each row bit-identical to the decode GEMV;
    - the decode expert GEMV run pairwise over the T×8 routed pairs.
- **Draft:** the checkpoint's DSpark head runs on the device.
  - 3 dense layers, requantized to Q8_0, over an 8-row block.
  - Its context K/V are written inside every target forward from taps at visits 5/23/41 and the final norm.
  - The Markov (rnn_hidden) head scores the top-32 base logits of each row against Q5_K LM-head rows.

## Hill-climb steps (each row is one window)

| change | verify ms (T=3) | draft ms | spec d2 tok/s |
|---|---|---|---|
| prefill kernels, grouped MMA experts (`w1`) | T=1 17.5, T=3 25.1 | – | – |
| ungrouped experts (`w5`) | T=1 11.4, T=3 17.3 | – | – |
| first end-to-end, device draft (`w6`) | 17.9 | 3.7 | 113.0 (plain 115.6) |
| one-pass few-row Q5_K head + pairwise experts | 16.0 | 2.4 | 129.5 / 129.8 |
| few-row Q8_0 dense, draft and verify (`wD`) | 15.8 | 1.72 | 136.70 / 136.68 |
| verify mix: norm folded into a per-row cluster mix, Q8 activation codes reused (`wF`, public c54bdd2a2e) | 15.35 | 1.72 | 139.92 / 140.40 (plain 119.05 / 119.88) |
| one launch for a visit's three controllers (`wG`, public e0e2120f4e) | 14.99 | 1.72 | 143.36 / 143.44 (plain 119.91 / 119.91) |
| few-row Q6_K, adaptive Q5_K blocks, one post-controller rms (`wH`) | 14.3-14.6 | 1.72-1.89 | 145.53 / 149.60 (plain 116.50 / 119.88) |
| paired k/v conv launch, two rows per warp in the Q5_K GEMV (`wI`, public 5915ef42be) | 14.13 | 1.67 | **151.60 / 151.70** (plain 119.84 / 119.78) |

Knob probe on the `wG` tree (`wK`):

| setting | tok/s | tokens per verify |
|---|---|---|
| d2, shortlist 32 | 141.05 | 2.40 |
| d2, shortlist 64 | 140.31 | 2.40 |
| d3, shortlist 32 | 116.71 | 2.44 |
| d3, shortlist 64 | 114.89 | 2.44 |

- On this prompt the third draft adds 1 accepted token in 52 verifies.
- The fourth verify row costs about 4 ms.
- The defaults stay at depth 2, shortlist 32.

## Per prompt (`wO`, 32 decode tokens each, plain / spec d2 / plain)

| prompt | plain tok/s | spec d2 tok/s | gain |
|---|---|---|---|
| capitals | 121.73 / 121.79 | 151.86 | +25% |
| fibonacci | 121.67 / 121.81 | 164.69 | +35% |
| chat-template | 120.90 / 120.97 | 132.29 | +9% |
| passphrase-recall, depth 1946 | 88.97 / 88.97 | 101.97 | +15% |

## Acceptance

- Host, 4 oracle prompts × 25 blocks (`receipts/acc.txt`):
  - full Markov head (33623cc5cc): mean 2.50/7;
  - top-16 shortlist: 2.35/7;
  - base argmax: 1.29/7.
- Device spec d2: 74/106 drafts accepted (70%).
- Device depth sweep (`w10`): d2 136.6 > d5 115.9 > d4 114.5 > d3 106.2 tok/s.

## Where the remaining time goes

- The profile is `wP`, graphs off. Per round:
  - expert GEMVs 6.1 ms, at bandwidth;
  - unfused batched small ops (mix, controller down/apply, rms, conv, route) about 8 ms profiled;
  - dense few-row matmuls about 3 ms;
  - LM heads 1.2 ms.
- 128 routed experts with 8 per token means verify rows hardly share experts. Each verified row adds about 2.4 ms of expert reads, so depth > 2 loses.
- A T=1 batched verify costs about 10 ms against the 8.2 ms decode step, which uses fused mix/controller/norm kernels.

## Next

1. Port decode's fused mix+norm and multi-controller kernels to the row-batched verify path. Estimated verify 15.8 → about 13.5 ms, about 155 tok/s.
2. Cut the draft from 1.7 ms: fuse the 7 Markov steps, and make the Q5_K row kernel's staging cheaper.
3. 200 tok/s also needs the non-expert decode overhead cut. It is about 5.8 ms per forward against a 1.75 ms bandwidth floor, and the cut helps both plain and verify.
