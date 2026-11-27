---
title: "Campaign record: Inkling-2-Mini resident CUDA decode: fused small-op climb on rtx5080-host-2 RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-2-inkling2-fused-climb (2026-11-24): Inkling-2-Mini resident CUDA decode: fused small-op climb on rtx5080-host-2 RTX 5080."
---

# Inkling-2-Mini resident CUDA decode: fused small-op climb on rtx5080-host-2 RTX 5080 (2026-11-24)

Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs are under [`receipts/`](receipts/).

## Result

| step | queue row | change | ABBA decode tok/s (A → B, 127 steps, profile off) | ms/token |
|---|---|---|---|---|
| base | | public fak a4e0af4066 (resident path a8030e394b) | 85.23 / 85.26 | 11.73 |
| 1 MIX | realugbun-3 | depth mix as an 8-block cluster (DSMEM), per-lane float64 softmax, following RMS folded in | 85.23/85.26 → **87.70/87.73** (+2.9%) | 11.40 |
| 2 CTRL | realugbun-1 | the 3 attention controllers of a visit in one down + one apply launch; apply does the tanh on all 32 lanes; output RMS batched | 87.72/87.72 → **97.66/97.65** (+11.3%) | 10.24 |
| 3 Q8 dedupe | fak-native | one Q8_0 quantization per distinct activation per token (197 → 99 `k_q8_quant_act`) | 97.65/97.66 → **98.57/98.57** (+0.9%) | 10.15 |
| 4 SITU/gate | realugbun-5 | SITU at 64-thread blocks (all SMs), MoE combine folded into the FFN post conv, attention gate folded into the `wo` quantization | 98.57/98.57 → **100.37/100.37** (+1.8%) | 9.96 |
| 5 CONV | realugbun-2 | k and v short convs run inside `k_ink2_qkv_post` (v staged in its cache row, then RMS in place) | 100.37/100.38 → **101.56/101.56** (+1.2%) | 9.85 |

Net on rtx5080-host-2: 85.2 → 101.6 tok/s (+19.1%, 11.73 → 9.85 ms/token). Bar: realugbun/llama.cpp fork 119.28 tok/s (8.38 ms/token) on rtx5080-host.

The rtx5080-host-2 base reads 85.2 tok/s where rtx5080-host read 82.55 for a8030e394b; same card model and GGUF, different box and driver/runtime (CUDA 13.5 runtime here). Compare steps within this receipt, not across boxes.

## Landed

- Step 1 MIX: public fak 28bb2c14a1.
- Steps 2-5 (CTRL, Q8 dedupe, SITU/gate, CONV): public fak 1d7f1dcf92. The land squashed the four lane commits (5f0f9d74e2, 7bbbf651f3, 742b098f02, 34af2a1dd8) into one commit under the CONV subject; each step's ABBA row above is still its own measurement.

## Correctness (greedy oracle, llama.cpp ad50351f5 CPU)

| build | capitals 32 | fibonacci 32 | chat-template 32 | passphrase-recall-1946 |
|---|---|---|---|---|
| base | PASS | PASS | PASS | diverges at token 10 (known, llama.cpp top-2 0.21 nats apart) |
| 1 MIX | PASS | PASS | PASS | PASS (32/32) |
| 2 CTRL | PASS | PASS | PASS | diverges at token 10, same token as base |
| 3 Q8 dedupe | PASS | PASS | PASS | same as step 2 (bitwise-equal arithmetic) |
| 4 SITU/gate | PASS | PASS | PASS | same as step 2 |
| 5 CONV | PASS | PASS | PASS | same as step 2 |

The acceptance gate is capitals + fibonacci 64/64 and chat-template 32/32; all steps hold it. The passphrase case is the documented borderline flip and moves with any reduction-order change.

## Selected kernels (profile ON, graphs off, event pairs add about 3 us per launch)

| kernel | base us/call x calls | after |
|---|---|---|
| mix | `k_ink2_mix` 8.34 x 120 | `k_ink2_mixc` 7.18 x 120 |
| RMS rows | `k_ink2_rms_rows` 6.55 x 154 | 2 calls; 48 batched `k_ink2_rms_multi` |
| controller | `k_ink2_ctrl_down` 7.78 x 96 + `k_ink2_ctrl_apply` 13.45 x 96 (2.04 ms) | `ctrl_down4` + `ctrl_apply64` x 48 each (0.85 ms) |
| Q8 act quant | 3.41 x 197 + 3.51 x 44 | 3.47 x 99 + gated 3.64 x 44 |
| SITU | 7.38 x 48 | 5.28 x 48 |
| MoE combine | 3.42 x 48 | 0 (in the post conv) |
| sigmoid mul | 3.68 x 48 | 0 on Q8_0 `wo` |
| k/v conv | `k_ink2_conv` 3.5 x 96 | 0 (in `k_ink2_qkv_post`, 5.85 → 6.48 us) |
| profile ops/token | 1712 | 1210 |

Finding: the slow small kernels were float64 transcendentals issued by one lane per warp (controller tanh, mix softmax, SITU on 40 blocks). Consumer Blackwell runs FP64 at 1/64 rate, so the fix that paid most was spreading that FP64 work over every lane and SM, not removing launches.

## What is left

- `k_ink2_expert_nvfp4` (2.47 ms) and the IQ expert GEMVs (1.43 ms) are the largest items; chain B owns NVFP4.
- CONV: the 96 attention/FFN post convs (`k_ink2_conv` 9 blocks) still launch alone; folding them into the next mix is the next CONV move.
- RELBIAS (realugbun-4): fak never builds a bias mask; `k_ink2_attend` indexes the bias table per key (0.67 ms/token). No mask chain to fuse; the row is realized differently.

## Host state

- rtx5080-host-2 `ds41-serve` (transient, a peer's DS4.1 fak serve using 4 GB VRAM and 25 GB RAM) was stopped at 04:31Z because the Inkling resident load OOMed beside it. Recreate with `receipts/ds41-serve.recreate.sh` (captured from `systemctl cat`).
- One accidental unleased start of the base oracle binary at ~05:16Z was killed within seconds (load phase only, no measurement).

## Recipe

Build on rtx5080-host (CPU only, go + /opt/cuda 13.4 + gcc), run on rtx5080-host-2: `receipts/build.sh <tree>`, `receipts/ship.sh <tree> <name>`, then on rtx5080-host-2 `receipts/oracle.sh <name>`, `receipts/abba.sh <out> <A> <B>`, `receipts/profile.sh <name>`. The rtx5080-host-2 binary links `/opt/cuda/targets/x86_64-linux/lib/libcudart.so.13` and `libcublas*.so.13`.

## Step 6 (not landed): post convs deferred into the next mix, 2026-11-24

Ticket inkling2-fused-postconv-20261124. A = public trunk 6a0e1f1fcc (with chain B's IQ expert v3), B = A plus [`receipts/postconv-into-mix.patch`](receipts/postconv-into-mix.patch): the attention post conv and the FFN combine+post conv are deferred and run inside `k_ink2_mixc`, the next mix that reads them.

- Oracle: capitals, fibonacci and chat-template PASS; passphrase-1946 diverges at token 10, the same token as A.
- ABBA: 117.58/117.60 → 117.91/117.93 (+0.28%). That is below the 1% leaf threshold, so the change is recorded and not landed.
- Profile: `k_ink2_conv` 96 → 22 calls (0.33 → 0.08 ms) but `k_ink2_mixc` 6.78 → 8.02 us/call (+0.15 ms). The mix is latency-bound on its cluster syncs, and the extra conv load sits on its critical path.
- Profile top-5 with the patch: `k_ink2_expert_nvfp4_v2` 1.93 ms, `k_ink2_mixc` 0.96, `k_q8_gemm` (3072 rows) 0.96, `k_ink2_attend` 0.67, `k_q8_gemm` (1024 rows) 0.66.

## Step 7: Inkling Q8_0 decode GEMV, 2026-11-24

Ticket inkling2-q8-gemv-20261124. A = public trunk 6a0e1f1fcc, B = A + `k_ink2_q8_gemv` behind `Ink2MatVecQ8`. The kernel is Inkling-only; the shared `k_q8_gemm` is unchanged.

- Change: each lane issues all of its blocks' weight and activation loads as 16-byte loads (weights streamed with `__ldcs`) before the first DP4A. It keeps `k_q8_gemm`'s lane block order, term and shuffle tree.
- SASS: `k_q8_gemm` has 94 scalar `LDG.E`; the new kernel has 12 `LDG.*.128` plus 6 scalar scale loads. Neither has F2F, I2F.U64 or float64; the only conversion is `I2FP.F32.S32`, as before.
- Bit-exact: `TestCUDAInk2Q8GEMVFastMatchesReference` compares both kernels bitwise on 7 shapes and passes.
- Microbench ([`receipts/micro-G1.txt`](receipts/micro-G1.txt), 400 back-to-back launches): 3072x2304 6.09 → 3.33 us, 1024x2304 3.50 → 2.03, 2304x3072 6.14 → 3.01, 48x2304 2.55 → 1.41. The weights stay hot in the 64 MB L2 there, so its GB/s reads above the 960 GB/s DRAM roofline. In decode the weights come from DRAM: the profile puts 3072x2304 at about 9 us (11.9 us minus about 3 us of event overhead), roughly 80% of roofline.
- Oracle: capitals, fibonacci and chat-template PASS; passphrase-1946 diverges at token 10, the same as A.
- ABBA: 117.57/117.62 → **120.85/120.88** (+2.8%). That is past the realugbun bar of 119.28 tok/s.
- Profile top-5 after the step: `k_ink2_expert_nvfp4_v2` 1.93 ms, `k_ink2_q8_gemv` (3072 rows) 0.90, `k_ink2_mixc` 0.81, `k_ink2_attend` 0.67, `k_ink2_q8_gemv` (2304 rows) 0.52.
