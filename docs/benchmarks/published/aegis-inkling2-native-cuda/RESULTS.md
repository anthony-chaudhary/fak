---
title: "Campaign record: fak-native Inkling-2-Mini: token-identical to llama.cpp, first CUDA decode on RTX 5080 host RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-native-cuda (2026-11-22): fak-native Inkling-2-Mini: token-identical to llama.cpp, first CUDA decode on RTX 5080 host RTX 5080."
---

# RESULTS — fak-native Inkling-2-Mini: token-identical to llama.cpp, first CUDA decode on RTX 5080 host RTX 5080 (2026-11-22)

**[HW-WITNESSED]** rtx5080-host (RTX 5080 16 GB, sm_120, driver 615.78.08, CUDA 13.4).
Measure leases `hwl-21a815ec860c796e` (window 1) and `hwl-6eb995ebd7704248` (window 2). Both
leases were released. `fak-llama-qwen27b.service` was stopped for each window, restarted after it,
and confirmed `active` with `/health` ok.
Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json). Logs are in [`receipts/`](receipts/).

## Verdict

fak now runs Thinking Machines Inkling-2-Mini (`inkling2_loop`) natively, both as a host reference
and fully resident on CUDA. Each path reproduces llama.cpp `ad50351f5` greedy output **token for
token**: 2 prompts × 32 tokens, 64 of 64. The first CUDA decode measures **6.32 tok/s**, against
**80.4 tok/s** for llama.cpp on the same card and GGUF ([OSS baseline](../rtx5080-host-inkling2-mini-oss-baseline-20261122/RESULTS.md)), which is 0.079x.
That gap is expected for a first correct cut: every GEMV still runs as its own launch, Read and sync.

| arm | engine @ rev | weights | placement | parity vs oracle | decode tok/s | state |
| --- | --- | --- | --- | --- | --- | --- |
| H | fak host reference, fak `88dde7a2e8` | UD-Q3_K_XL mainline GGUF | rtx5080-host CPU, f32 | 64/64 tokens | ~1.4 incl. prefill | MEASURED |
| C | fak CUDA, fak `d09931ed93` | same | all on GPU, 12.35 GiB in 298 allocations | 64/64 tokens | **6.32** (6.32, 6.32; 127 steps) | MEASURED |
| L1 | llama.cpp `ad50351f5` (baseline) | same | all on GPU | n/a (it is the oracle's engine) | 80.4 | from the baseline receipt |

The oracle is llama.cpp `ad50351f5` llama-server on the CPU backend (`CUDA_VISIBLE_DEVICES=` empty,
`-ngl 0 -fa off`), greedy with `top_k 1`. Prompts are passed as token IDs with no BOS.
Script: [`receipts/oracle.sh`](receipts/oracle.sh); outputs: `receipts/llamacpp-cpu-oracle-*.json`.
The CUDA load takes 15.0 s. Prefill runs one token at a time at 6.56 tok/s.

## What was learned

1. **The llama.cpp graph is a complete spec.** The Go port of `src/models/inkling2.cpp` and
   `inkling2-loop.cpp` matched on the first run. Detail that had to be exact:
   - ggml NVFP4 (type 40) experts carry a per-expert F32 `.scale`, applied after each expert matmul.
   - The rel-bias table gets a zero pad row, and the sink is subtracted on every key.
   - Dynamic conv taps run over lags 0..3.
   - The loop bank reads only the core blocks of the previous pass.
   - An independent 4-lens review found 0 mismatches.
2. **Activation rounding did not flip any greedy token.** CUDA rounds NVFP4 W4A16 activations to
   BF16 and quantizes Q8_0 activations to int8, while the oracle runs f32 on the CPU. All 64
   tokens still matched.
3. **One device allocation per expert slice does not fit.** That layout needs about 6,100
   allocations, and the run hit 127 `cudaMalloc` OOMs on 16 GB (window 1). Each 128-expert
   stack is now uploaded once, and each expert reads through a non-owning `compute.RowView`
   (the same idea as llama.cpp `mul_mat_id`). The result is 298 allocations and 12.35 GiB.
4. **fak's CUDA backend had no kernels for IQ2_S, IQ3_XXS and IQ3_S.** Those types hold 2.4 GiB
   of this GGUF's gate_up experts. Ported warp-per-row GEMVs pass device parity
   (`TestCUDAIQ2SIQ3XXSIQ3SGemvMatchesRef`).

## Next

The highest-value next step is to remove the per-GEMV host sync. That means keeping the small ops on
the device (depth mix, controllers, short conv, clip, rel-bias attention with sink, SITU MoE) and
then capturing a CUDA graph per token. The current loop pays a launch plus Read for each of roughly
1000 GEMVs per token. Its 6.32 tok/s works out to 158 ms/token, against 12.4 ms/token for llama.cpp.
Not done: the DSpark MTP sidecar, a >1024-token SWA-window oracle, and the varloop checkpoint.
