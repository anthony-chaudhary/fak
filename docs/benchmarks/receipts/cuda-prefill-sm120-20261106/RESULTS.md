# CUDA Q4_K prompt GEMM defaults to MMQ on sm_80+ — same-card ABBA on RTX 5090 Laptop (sm_120)

**[HW-WITNESSED]** 2026-11-07T00:00Z-01:30Z. Host `raider-fak`, RTX 5090 Laptop
`GPU-2a2a837f-c82d-0f22-4538-e3a063903380` (driver 615.78.08, Arch 7.2.11), pinned via
`CUDA_VISIBLE_DEVICES`. Measure lease `hwl-6745c080f5c36df2` (holder
`claude-cuda-prefill-win-20261106`, epoch 159), released after the run. `fak.service` was inactive
and the GPU idle (2 MiB) at start and end.

Model `/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf` (the file from
[nvidia-cuda-perf-gap](../../published/nvidia-cuda-perf-gap/RESULTS.md)), ctx 4096, kv fp32, `--speculative ''`.
Client: pinned `fak-h2h-c9870f151d` (sha256 `fbcab944…`), cold 128/512/2048, 3 reps, 8 answer tokens.
Denominator: llama.cpp b11870 `llama-server` (sha256 `8453c97a…`), same GPU, model, and client.
Binaries were built in WSL2 (go1.26.10, nvcc 13.0, `build_cuda.sh binary ./cmd/fak`, SASS sm_120 only)
and copied to the node. See `raw/identity.txt` and `raw/runs/argv.txt`.

## Change

When `FAK_CUDA_Q4K_PREFILL` is unset, the arm now resolves through
`compute.DefaultCUDAQ4KPrefillArm(sm)`. That returns `mmq`, the llama.cpp int8 tensor-core Q4_K
MMQ already in `cuda/q4k_mmq.cuh`, on sm_80 and newer. It returns `compensated` below sm_80 or when
the arch is unknown. An explicit env value still wins, and `FAK_CUDA_Q4K_PREFILL=compensated`
restores the old default. No kernel changed. The prior raider receipts already used mmq, but only
because the service drop-in set it by hand. The out-of-box default was compensated: dequantize to
f32, then three FP16 cuBLAS GEMMs.

## Headline (ABBA, p50 of 3, client prefill tok/s = prompt/TTFT)

| arm (order) | binary | env | P=128 | P=512 | P=2048 |
| --- | --- | --- | ---: | ---: | ---: |
| A1 | base `1310932e` | unset (=compensated) | 108.4 | 83.5 | 118.7 |
| B1 | base `1310932e` | `FAK_CUDA_Q4K_PREFILL=mmq` | 158.4 | 118.8 | 154.7 |
| B2 | new `b9978b9a` | unset (=mmq, logged `q4k prefill arm=mmq`) | 164.1 | 119.2 | 153.8 |
| A2 | base `1310932e` | unset (=compensated) | 107.0 | 81.5 | 118.2 |
| **B/A** | | | **1.49x** | **1.44x** | **1.30x** |
| llama b11870 | | | 617.5 | 904.1 | 1053.2 |
| fak/llama before → after | | | 0.17x → **0.26x** | 0.09x → **0.13x** | 0.11x → **0.15x** |

TTFT p50 (ms), A → B: P=128 1752/1786 → 1206/1164; P=512 7455/7641 → 5246/5237; P=2048
19806/19894 → 15206/15296. Server-side `prompt_ms` tracks TTFT within 30-140 ms in every arm. Actual
prompt lengths were 191, 623, and 2353 tokens; llama's were 199, 631, and 2361 under the same client.
Decode tok/s is unchanged within noise.

**Greedy parity:** the fixed 283-token prompt (`raw/parity-prompt.txt`, temperature 0, 64 tokens,
thinking off) produces a byte-identical completion in all four ABBA arms (sha256 prefix
`386622dbcfdf258e`).

## Prefill profile at the new default (where the remaining gap is)

`FAK_CUDA_TIMESTAMP_PROFILE=1`, cold 512 + 2048, 1 answer token
(`raw/profile-P1-mmq-totals.tsv`, event-pair totals). Total GPU time was 27.7 s:

| kernel | ms | share |
| --- | ---: | ---: |
| `k_q6k_gemm_panel` (FP32 SIMT Q6_K prompt panel) | 10057 | **36%** |
| `cublas_gemm_ex` | 2197 | 8% |
| `k_qwen38_causal_attention_panel_hd256_splitkv_partial` | 2119 | 8% |
| `k_qwen35_gdn_fused_in_proj` (P-tiled) | 1912 | 7% |
| `mul_mat_q` (Q4_K MMQ) | 1629 | 6% |

The largest remaining item is the Q6_K prompt GEMM (`ffn_down`/`attn_v` in Q4_K_M). The retained
Q5/Q6 MMQ from eb6cc84fe (#14181) is compiled only with `--q56k-mmq-qualification`. It was measured
on the same card as arm C1: binary `ca18c17a`, built through `fak-dev build-cuda binary
--q56k-mmq-qualification`, with mmq set. **354 / 184 / 362 tok/s** (TTFT 548 / 3396 / 6515 ms), which is
**2.2-2.4x over B and 3.0-3.3x over A**, or 0.57x / 0.20x / 0.34x llama. **It fails the
byte-parity gate.** The completion diverges at about token 50 on a near-tie ("…computational
**intensity**…" vs "…**bottlenecks**…") and stays coherent. llama.cpp's own output on the same prompt
diverges from every fak arm at token 3, so it cannot arbitrate (`raw/runs/Lp.parity.json`). It was not
landed. Making it the default needs a numerical-parity or tolerance decision first. P=512 underperforms
P=128 and P=2048 in every arm (623 = 512 + 111, which suggests a sub-128 tail chunk falls off the fast
path). That is not investigated here.

Methodology correction to the 2026-10-31 receipt: its "prefill profile" folded in the 16 answer
tokens per request. `k_q4k_mmvq_gemv` and `k_q6k_gemm` (grid y=1) are decode kernels, not prefill.

## Secondary: MTP on vs off (same card, mmq, decode-128 ×3)

`--speculative ''` decodes at 17.90-17.93 tok/s and `--speculative mtp` at 17.81-17.83 tok/s
(`raw/runs/M-spec-*`). The serve log has no MTP/draft engagement line, so acceptance rate is
**not witnessed**. On this CUDA path, with this GGUF, MTP produced no measurable effect.

## eGPU lane (RTX 5070 Ti 16 GB, owned-rtx5070-egpu-a): attempted, not usable for the 27B

- llama.cpp b9724 on Windows, Qwen3.8-27B UD-Q2_K_XL: llama-bench pp128/512/2048 1268/1696/1670 tok/s
  (`raw/egpu-a-llama-bench-q2kxl.md`).
- fak (WSL2 build, same tree) plans 26.45 GiB of weights for that 9.8 GB file, against a 13.5 GiB fit
  budget, and refuses with a typed `FitTooBig` (`raw/egpu-a-fak-fit-refusal.log`). The 27B only runs
  there with layer offload, which another receipt measured at 0.2 tok/s decode. No fak eGPU prefill
  number exists.
- A native Windows `-tags cuda` link fails: MSVC-compiled `libfakcuda.a` against mingw ld gives
  unresolved `__security_cookie` and MSVC RTTI. WSL2 is the working local build lane.

## Status language

- mmq default flip: **[HW-WITNESSED]** ABBA on sm_120 with byte-identical greedy output. Other sm_80+
  arches (A100 sm_80, H100 sm_90, 4070 sm_89) now take mmq by default **unmeasured** on those cards.
  `CheckCUDAQ4KPrefillArm` already admits mmq from sm_80.
- The L3 prefill bar (≥0.5x llama) is **not met**: 0.13-0.26x after this change.
- Q5/Q6 MMQ: **[HW-WITNESSED] speed**, **parity gate FAILED** (strict sha). Not landed.

## Receipts

`raw/runs/`: per-arm serve logs, client stdout/stderr, h2h ledgers, nvidia-smi 1 Hz samples, parity
JSON, `argv.txt`. `raw/`: run scripts, `config.toml`, parity prompt, binary/client/llama sha256
(`identity.txt`), folded profile totals, eGPU artifacts.
