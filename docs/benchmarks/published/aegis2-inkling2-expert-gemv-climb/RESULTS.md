---
title: "Campaign record: Inkling-2-Mini NVFP4 expert GEMV climb on rtx5080-host-2 RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-2-inkling2-expert-gemv-climb (2026-11-24): Inkling-2-Mini NVFP4 expert GEMV climb on rtx5080-host-2 RTX 5080."
---

# Inkling-2-Mini NVFP4 expert GEMV climb on rtx5080-host-2 RTX 5080 (2026-11-24)

Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Raw logs are under [`receipts/`](receipts/).

## Result

| | A: fak a4e0af4066 | B: lane commit d27b9cc5de, landed public 6eba862214 |
|---|---|---|
| decode tok/s, ABBA 127 steps, profile off | 85.22 / 85.24 | **89.28 / 89.28** |
| ms/token | 11.73 | 11.20 |
| NVFP4 expert GEMV, profile on (graphs off) | 31.77 us/call, 2.478 ms/token | 24.73 us/call, 1.929 ms/token |
| GPU ms/token, profile on | 15.34 | 14.78 |

- Decode is +4.75% (-0.53 ms/token). That matches the -0.55 ms/token the profile attributes to the NVFP4 expert GEMV.
- Outputs are bit-identical to the reference kernel:
  - The CUDA unit test `TestCUDAInk2ExpertNVFP4FastMatchesReference` passes. It covers gate_up, down, ragged rows, kb=8 and a SITU-staged input, and also checks against a float64 host dot (tolerance 1e-5·Σ|w·x|).
  - The microbench found 0 of 1.18M outputs differing for each of 10 variants and 64 expert-id sets.
- Greedy oracle (`TestInkling2CUDAOracleGreedyMatchesLlamaCpp`):
  - capitals, fibonacci and chat-template: 32/32 tokens each, identical to llama.cpp.
  - passphrase-recall-1946 diverges at decode token 10 on **both** A and B, with the same token stream (153536 vs llama.cpp 150409). This is the pre-existing red documented in docs/benchmarks/rtx5080-inkling2-resident-cuda-20261123. It is not a regression: the kernel is bit-exact.

## What changed (public fak)

- New file `internal/compute/cuda/inkling2_expert_gemv.cuh` holds `k_ink2_expert_nvfp4_v2`, included from `inkling2_loop.cuh`. The call site changes by one line.
- Arithmetic is the reference's: lane l owns 16-value blocks l, l+32, …; each block uses the same fmaf chain; the shuffle tree is unchanged.
- Only the schedule moves:
  - The grid is about 4 blocks of 8 warps per SM. Each warp walks rows with a grid stride, which removes the reference grid's nearly empty second wave (576 blocks on 84 SMs × 6).
  - A row's packed words and scale bytes are issued before its math. The next row's loads are issued before the current row's reduction.
  - The staged BF16 activation is swizzled per 16-value block, so the float4 reads hit distinct bank groups.
- Opt-out: `FAK_INKLING2_NVFP4_GEMV=0` keeps `k_ink2_expert_nvfp4`.

## Microbench (isolated, live rtx5080-host-2, 64 experts ≈ 190 MB > 64 MB L2)

`receipts/mb.cu`, the variants in `receipts/mb-variants.txt`, and ABBA inside each binary at 2000 calls per arm.

| variant | gate_up 2304×2304 k=8 | down 2304×1152 k=8 |
|---|---|---|
| reference | 38.2 us, 626 GB/s | 21.4 us, 558 GB/s |
| R=2, 4 blk/SM | 29.4 us, 811 GB/s | 15.7 us, 761 GB/s |
| R=4, 4 blk/SM | 30.2 us, 791 GB/s | 17.4 us, 686 GB/s |
| **R=1, 4 blk/SM (shipped)** | **26.1 us, 914 GB/s** | **14.9 us, 803 GB/s** |
| R=1, 3 blk/SM | 30.8 us, 775 GB/s | 18.0 us, 663 GB/s |
| R=1, 4 warps × 8 blk/SM | 27.0 us, 884 GB/s | 15.7 us, 761 GB/s |

## Flat/worse data (recorded, not shipped)

- An IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S expert GEMV with the same treatment was tested: R rows advanced together per super-block, plus one-wave grid sizing. It was bit-exact but **slower** in every variant. For example, IQ3_XXS gate_up went from 99.1 us (reference) to 107–120 us. See `receipts/mb-iq-sweep.txt`.
- These kernels run at 100–180 GB/s. They are bound by dependent byte loads and random grid-table lookups, not by bytes in flight, so the schedule change does not help. Next lever: grid tables in shared memory and wide block loads decoded in registers.
- The IQ expert kernels are now 1.44 ms/token (9.7%) on B.
- The tensor-core (FP16 MMA per 16-weight scale block) variant from Strata strata-8 targets batched windows. At batch 1 the v2 GEMV already reaches 914 GB/s of about 960, so MMA has no headroom for decode. That row is still prefill/verify work.

## Protocol

- Hardware: rtx5080-host-2, RTX 5080 16 GB, driver 620.73.04, CUDA 13.5 runtime. Measure lease hwl-67270ec41ba02d50, released.
- Builds: on rtx5080-host (CPU only, share lease hwl-6e7d2003d0a9d855, released) from a `git archive` of a4e0af4066 (A) and A plus the lane files (B), with `fak-dev build-cuda build --arch sm_120` and `go test -c -tags cuda`. The test binaries were copied to rtx5080-host-2.
- `receipts/run_e2e.sh`: unit test, oracle, ABBA (A B B A), then a separate profile-on run per arm (`FAK_INKLING2_GRAPH=0 FAK_CUDA_TIMESTAMP_PROFILE=1`).
- The sibling fused-mix lane (28bb2c14a1) landed on origin during this climb. A is the pre-fusion base, so B's +4.75% is this change alone.

## Step 2 (2026-11-24): IQ expert GEMVs (IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S)

Receipts: [`receipts-iq/`](receipts-iq/) and [`perf-receipt-v2-iq.json`](perf-receipt-v2-iq.json).

| | A: public origin/main 1d7f1dcf92 (includes 6eba862214 and the chain-A fusions) | B: A + `k_ink2_expert_iq_v3`, landed public 6a0e1f1fcc |
|---|---|---|
| decode tok/s, ABBA 127 steps, profile off | 107.35 / 107.36 | **117.55 / 117.57** |
| ms/token | 9.315 | 8.506 |
| IQ expert GEMVs, profile on | 18 calls, 79.7 us/call, 1.435 ms/token | 13 calls on v3 at 36.0 us/call (0.467 ms), plus 5 IQ4_NL/Q3_K calls left on the reference (0.163 ms) |
| GPU ms/token, profile on | 11.78 | 10.96 |

- Decode is +9.5% (-0.81 ms/token). The profile attributes -0.80 ms/token to the IQ GEMVs.
- Outputs are bit-identical:
  - `TestCUDAInk2ExpertIQFastMatchesReference` passes. It covers 4 formats × (2304×2304 shared x, 2304×1280 per-slot x, 37×512), and also checks against a float64 host dot over `DequantRawIQ`.
  - `TestCUDAIQ2SIQ3XXSIQ3SGemvMatchesRef` and the NVFP4 test pass in the same binary.
  - The microbench found 0 differing outputs in every variant.
- Oracle: capitals, fibonacci and chat-template are 32/32 identical on both A and B. passphrase-recall-1946 diverges at token 10 on both, with the same stream (pre-existing red).

### What bound the reference

Neither memory nor table lookups were binding. The SASS of `k_ink2_expert_iq` shows `fcuda_iq_dot8` converting each grid byte with `(float)` of an `unsigned long long`, which compiles to `I2F.U64`: 8 per lane per super-block. Time tracked super-block count, not bytes (IQ2_XS 74 B and IQ3_S 110 B both took about 96–101 us on gate_up).

`ink2_iq_dot8` converts each byte exactly through 32 bits (`__uint_as_float(0x4b000000|b) - 2^23`). Every fmaf operand is unchanged, so outputs are unchanged.

### Hill climb (microbench, sum over 4 formats × gate_up+down; reference 621 us)

| variant | result | verdict |
|---|---|---|
| grid/sign tables in shared memory only (64-bit conversion kept) | 96–106% of reference per call | flat, stopped (`mb-iq3-sweep.txt` T1/T2) |
| + warp stages its rows into shared memory with coalesced 16-bit loads | 98–123% | flat/worse, stopped (S1–S4) |
| exact 32-bit byte conversion, tables in smem, 2 rows/warp, 4 blk/SM | 179 us | win (`mb-iq4-sweep.txt` U2) |
| same + staging | 190–230 us | worse than no staging (V1–V3) |
| exact conversion, tables in global/L1 | 202–204 us | tables in smem are worth ~12% (`mb-iq5-sweep.txt` W3/W4) |
| **exact conversion, smem tables, 2 rows/warp, 6 blk/SM (shipped)** | **177 us (3.5x)** | win (W1) |

Per call on the shipped variant: IQ3_XXS gate_up 99.1 → ~28 us (164 → ~575 GB/s); IQ2_S 100 → ~28 us; IQ3_S 101 → ~28–32 us; IQ2_XS 96 → ~26 us.

### Left
- IQ4_NL and Q3_K experts (5 calls, 0.16 ms/token) still use `k_ink2_expert_iq`; the IQ4_NL path already converts through `signed char`.
- `fcuda_iq_dot8` in `iq_gemv.cuh` has the same `I2F.U64`. Every dense IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S GEMV in the tree pays it. The same exact fix applies there, owned by that file's lane.

Protocol: A and B are `git archive` exports built on rtx5080-host CPU (`fak-dev build-cuda build --arch sm_120`, `go test -c -tags cuda`), then run on rtx5080-host-2 under measure lease hwl-7512996769725f81 (released) with `receipts-iq/run_e2e_iq.sh`. The microbench used lease hwl-504ed6157a7948ec (released).

## Step 3 (2026-11-24): IQ4_NL / Q3_K experts (ticket inkling2-expert-iq4nl-q3k-20261124, landed public 2ba9ae6fce)

Receipts: [`receipts-iq4nl-dense/`](receipts-iq4nl-dense/).

| | A: origin/main 6a0e1f1fcc | B: A + `k_ink2_expert_iq4nl_v3` |
|---|---|---|
| decode tok/s, ABBA 127 steps, profile off | 117.60 / 117.59 | 117.83 / 117.86 (+0.2%, below the 1% climb bar) |
| IQ4_NL + Q3_K experts, profile on | 5 calls, 0.161 ms/token | IQ4_NL 4 × 23.2 us = 0.093 ms, Q3_K 1 × 51.4 us = 0.051 ms; 0.144 ms/token |

- IQ4_NL is bit-identical. The new kernel keeps the reference per-lane term but:
  - stages the codebook once as floats, instead of converting a signed char per weight;
  - reads the 4 code bytes as two 16-bit words;
  - advances 2 rows per warp, with the step-2 grid sizing.
- IQ4_NL microbench, `mb-iq4nl-q3k-and-dense.txt`:
  - gate_up 2304×2304, k=8: 37.9 → 29.4 µs.
  - down 2304×1280, k=8: 21.4 → 16.3 µs.
  - 0 differing outputs.
- `TestCUDAInk2ExpertIQFastMatchesReference` now covers IQ4_NL and passes.
- Q3_K tried the same way, with an exact magic-number conversion of the -4..3 code and 1, 2 or 4 rows per warp, was bit-exact but **slower**: gate_up 46.1 → 46.4–57.3 µs, down 23.4 → 25.3–25.7 µs. It stays on the reference kernel. This is recorded, not shipped.
- Oracle: capitals, fibonacci and chat-template are identical on A and B. passphrase-recall-1946 still diverges at token 10 on both (pre-existing).

## Step 4 (2026-11-24): dense IQ GEMVs, `fcuda_iq_dot8` (ticket iq-gemv-i2f-u32-20261124, landed public 636f8b3dc3)

- The step-2 fix went into the shared helper `fcuda_iq_dot8` in `iq_gemv.cuh`: each grid byte converts exactly through 32 bits instead of `I2F.U64`.
- It speeds every dense k_iq2xxs / k_iq2xs / k_iq2s / k_iq3xxs / k_iq3s GEMV. It also speeds the reference expert kernel `k_ink2_expert_iq`, which is now only reached with `FAK_INKLING2_IQ_GEMV=0`.
- Microbench `mbd.cu`, old origin kernels vs new in one binary, 4096×4096, 16 rotating matrices (larger than L2), ABBA, 0 differing outputs in every row:

| format | P=1 old → new | P=3 old → new |
|---|---|---|
| IQ2_XXS | 42.0 → 12.9 µs (3.26x) | 110.3 → 26.7 µs (4.13x) |
| IQ2_XS | 42.2 → 14.9 µs (2.83x) | 110.5 → 26.1 µs (4.23x) |
| IQ2_S | 42.6 → 15.3 µs (2.78x) | 110.6 → 27.6 µs (4.00x) |
| IQ3_XXS | 42.4 → 16.4 µs (2.59x) | 110.4 → 31.1 µs (3.55x) |
| IQ3_S | 43.4 → 17.0 µs (2.55x) | 110.6 → 31.6 µs (3.51x) |

- Device tests pass on rtx5080-host-2 for the patched tree: `TestCUDAIQ2SIQ3XXSIQ3SGemvMatchesRef` plus the IQ2_XXS, IQ2_XS, IQ1_M and IQ4_NL DeviceParity tests (`unit-C-dense.txt`). The IQ source-contract, lane-simulation and RawIQ tests pass on the host.
- These parity tests use a tolerance. The bit-exact witness is the microbench.
- Which models it affects:
  - Any GGUF whose dense (non-expert) tensors are IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S and that runs through the CUDA MatMul/BatchedMatMul IQ path. This is the low-bit unsloth UD-IQ / UD-Q2_K_XL style mixes.
  - The Inkling-2 host-routed fallback experts (`FAK_INKLING2_RESIDENT=0`).
  - The Inkling-2 resident path is unaffected: its dense weights are Q8_0/Q4_K/Q5_K/Q6_K per the profiles. So no Inkling ABBA was run for this step.

Protocol:
- A, B (IQ4_NL) and C (dense) are `git archive` exports built on rtx5080-host CPU under a share lease.
- They ran on rtx5080-host-2 under measure leases hwl-f5947f24131d1b55 (microbench) and hwl-697df65c947fd64d (`run_e2e_kq.sh`). Both are released.

## Step 5 (2026-11-24): `k_ink2_attend` split over key chunks (ticket inkling2-attend-splitkv-20261124, landed public b64405de76)

Receipts: [`receipts-attend/`](receipts-attend/).

| | A: origin/main 2ba9ae6fce | B: A + split attend |
|---|---|---|
| decode tok/s, ABBA 127 steps after a 19-token prompt, profile off | 117.80 / 117.83 | **119.22 / 119.24** (+1.2%) |
| decode tok/s at depth 1946, 31 steps (oracle passphrase case, ABBA) | 69.96 / 69.96 | **89.31 / 89.31** (+27.7%) |
| prefill of the 1946-token prompt | 23.97 s | 19.86 s |

- Outputs are bit-identical:
  - `TestCUDAInk2AttendSplitMatchesReference` passes on rtx5080-host-2. It covers local and global slots at pos 0, 82, 2000 and 5000 (the wrapped ring).
  - The microbench found 0 differing outputs in every case.
  - Oracle token streams are identical on all four A/B runs. capitals, fibonacci and chat-template pass; passphrase still diverges at token 10 (pre-existing).
- What the reference was bound by:
  - 24 blocks (one per head) on 84 SMs.
  - Uncached global value loads in a serial fmaf chain.
  - Above all, the serial key-order double denominator. It is about 40 ns per key even from shared memory: dependent DADD latency on this consumer part, and it must stay serial to be bit-exact.
  - The SASS has no `I2F.U64`. The float64 work (exp, denominator) is the reference's own arithmetic and is kept.
- What changed, in three launches:
  - Scores run on (head, 8 key chunks) blocks, one warp per key in the reference's order.
  - Exps run on (head, chunk) blocks, using a head max taken from the chunk maxes; fmax is exact in any order.
  - The output runs on (head, 32-dim slice) blocks. It stages the exps in shared memory for the serial denominator and prefetches values 8 keys ahead of each group's fmaf chain.
  - The chunk maxes park in the head's out row until the output launch writes it; the head max parks in the rel-query row, which only the scores launch reads.
- Microbench (`mba-sweeps.txt`, reference → shipped variant S8), local and global pos-82 rows:

| keys | reference | split |
|---|---|---|
| n=83 | 9.8 µs | 9.6 µs |
| n=1024 (local window) | 88 µs | 49 µs |
| n=2001 | 157 µs | 93 µs |
| n=4001 | 310 µs | 179 µs |

  - 16 chunks was 5–10% better at depth but 25% worse at n=83. 32 chunks was 2.8x worse at n=83.
- The profile-on run (graphs off, per-launch event timing) charges three launches: attend went from 0.704 to 0.857 ms/token. With graphs on (the ABBA) the split is faster.
- The serial denominator is now the floor, at about 40 ns per key. Going below it would need a non-bit-exact (tree) sum.
