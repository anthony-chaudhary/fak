---
title: "Mac Qwen3.6-27B Metal Q4_K perf diagnosis — decode 1.2 / prefill 0.6 tok/s, the per-call command-buffer wall (2026-06-26)"
description: "A clean on-device measurement of fak's resident-Q4_K Metal path for Qwen3.6-27B on the M3 Pro, with FAK_QPROFILE phase profiles and the q4k_gemv/q4k_gemm microbenchmarks. The kernels are correct (cosine 1.0, decode token-parity) but the live per-token decode runs ~336 separate command-buffer GEMVs, each ~360 us launch-bound, so decode tops out at 1.2 tok/s — 0.16x of the 3x goal and 0.16x of the llama.cpp-Metal bar. The lever is a one-command-buffer GPU-resident decode forward plus a kernel-efficiency pass."
date: 2026-06-26
---

# Mac Qwen3.6-27B Metal Q4_K perf diagnosis (2026-06-26)

Run on `node-macos-a` (Apple M3 Pro / Mac15,7, 12 CPU core = 6P+6E, 18 GPU core, 36 GB
unified, macOS 26.5, Metal 4), fresh shallow public clone at HEAD `20fdf16`, built
`-tags fakmetal` (CGO Metal). Driven over Tailscale SSH from a Windows box. **The
llama-server (`com.fak.qwen36-model`, the 7.29 tok/s reference) was stopped for the clean
arm** (`launchctl bootout` → run → `launchctl bootstrap` restore) so fak had the whole 36 GB
GPU/unified pool; an EXIT trap guaranteed the service came back.

## TL;DR

fak's **resident-Q4_K Metal decode path is CORRECT but does not reach the speed bar.** The
27B's clean decode is **1.2 tok/s** (vs the CPU resident-q4k baseline 0.9, the 3× goal 2.7,
and the llama.cpp-Metal reference 7.29). The cause is **not** a wrong kernel — the Q4_K GEMV
matches the CPU f32 reference to cosine 1.000000 and the greedy decode token sequence is
bit-identical. The cause is **orchestration**: each decode token runs ~336 *separate* Metal
command-buffer GEMVs (≈7 projection/MLP matmuls × ~48–64 layers), and an isolated GEMV is
**~360 µs launch/sync-bound** on top of ~98 µs of actual bandwidth-limited work. The lever is
the **one-command-buffer-per-token GPU-resident decode forward** (the decode twin of the
already-shipped resident *prefill* forward), plus a **kernel-efficiency pass** on `q4k.m`.

## 1. Clean measurement (the headline)

`FAK_Q4K=1 FAK_METAL=1 FAK_GPU_LEASE_NOWAIT=1 FAK_QPROFILE=1 fakchat -gguf
~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf -tok ~/.cache/fak-models/tokenizers/qwen3.6
-p "Write three sentences about why the ocean is important." -n 64 -quiet`

backend banner: `fak in-kernel Gated-DeltaNet (resident Q4_K decode + Q8 fallback, cached)
[Metal q4_k prefill]`. Qwen3.6-27B is a Gated-DeltaNet **hybrid** (`cfg.IsQwen35Hybrid`, 48
GDN layers + periodic full-attention), so fakchat routes it through `runHybrid`, which sets
`s.MetalQ4K = q4k && FAK_METAL`.

| metric | clean (llama stopped) | contended (llama up, swap) | CPU resident-q4k | llama.cpp-Metal | 3× goal |
|---|---:|---:|---:|---:|---:|
| **decode tok/s** | **1.2** (64 tok / 54.12 s) | 0.8 | 0.9 | **7.29** | 2.7 |
| **prefill tok/s** | **0.6** (29 tok / 48.27 s) | 0.6 | — | **51.55** | — |
| load | ~40 s | 190 s | — | — | — |
| max RSS / peak footprint | 26.0 GB / 53 GB | 13.1 GB / 53 GB | — | — | — |
| swaps | 0 | 0 (but 15.4M page reclaims) | — | — | — |

The clean arm fits (26 GB RSS, 0 swaps). The contended arm has BOTH 27B models resident
(peak 53 GB footprint, 15.4M page reclaims) — its 0.8 tok/s is swap-noise, not a fak number.
Either way decode lands ~1 tok/s: **fak is at 0.16× of the 3× decode goal and 0.16× of the
llama.cpp-Metal bar.**

## 2. Where the time goes (FAK_QPROFILE phase profiles)

**Decode** (total 54122 ms for 64 tokens = 845 ms/token):

| phase | ms | % | calls | ms/call |
|---|---:|---:|---:|---:|
| `mlp_decode` | 29200 | **54.0%** | 4096 | 7.13 |
| `qwen35_linear_step_in_proj` | 8697 | 16.1% | 3072 | 2.83 |
| `qwen35_linear_step_out_proj` | 3453 | 6.4% | 3072 | 1.12 |
| `qwen35_linear_step_recurrent` | 3199 | 5.9% | 3072 | 1.04 |
| `full_attn_qkv_proj` | 2900 | 5.4% | 1024 | 2.83 |
| `lm_head_q8` | 1934 | 3.6% | 64 | 30.2 |
| `full_attn_o_proj` | 1066 | 2.0% | 1024 | 1.04 |
| rest (attn, conv, norms, gate) | ~3700 | 6.8% | — | — |

**Prefill** (total 48273 ms for 29 tokens): `mlp_gate_up_proj` 54.7% (412 ms/call),
`mlp_down_proj` 18.2%, `qwen35_linear_in_proj` 11.2%.

The matmuls (MLP + projections) are ~85% of both phases. `mlp_decode` alone is 54% at **7.1
ms per single-token matmul** — a Q4_K MLP weight read is ~40 MB, which at the device's ~150
GB/s should be ~0.27 ms. The 26× gap is per-call command-buffer overhead and CPU↔GPU
round-trips, not arithmetic.

## 3. The kernels are correct but the per-call overhead dominates (microbench)

`go test -tags fakmetal` on the same box:

- `TestMetalQ4KGemvMatchesCPU` **PASS** — q4k GEMV vs CPU f32 reference: cosine **1.000000**,
  maxRel ≤ 1.2e-6 at [256,256], [512,1024], [5120,5120].
- `TestMetalQ4KDecodeMatchesCPU` **PASS** — GPU greedy decode token sequence == CPU
  (`[433 92 166 106]`). The Metal decode path engages and is numerically faithful.
- `BenchmarkMetalQ4KGemv` ([5120,5120]): **457 µs/op = 32.2 GB/s** (≈21% of the ~150 GB/s
  device bandwidth). The ~14.7 MB row-read is ~98 µs of the 457 µs; **the other ~360 µs is
  fixed launch/sync per command buffer.**
- `BenchmarkMetalQ4KGemmSteady`: **10.77 ms/op = 4.66 GB/s / 364 GFLOP/s** (≈5% of the GPU's
  f32 FLOP ceiling) — the batched prefill GEMM is also far underutilized.

So the q4k.m kernels are *correct* but *slow in two compounding ways*: (a) per-call
command-buffer overhead (~360 µs), multiplied by ~336 matmuls/token in the live decode loop;
(b) low in-kernel utilization (GEMV ~21% bandwidth, GEMM ~5% FLOP) even ignoring launch cost.

## 4. The lever (and the bandwidth ceiling)

A 27B Q4_K decode reads ~15 GB of weights per token. At ~150 GB/s that is a **~100 ms/token
floor ≈ 10 tok/s ceiling**; llama.cpp-Metal achieves 7.29 there. fak's path leaves ~85% of
the wall-clock in launch overhead and round-trips, so it sits at 1.2.

**Primary lever — a one-command-buffer-per-token GPU-resident decode forward.** The repo
already ships the *prefill* analog: `internal/metalgemm/forward.m` `mg_prefill` runs the whole
fresh prefill in ONE command buffer with the activation resident on-GPU (`prefillMetalResident`
in `internal/model/metal_prefill.go`). The decode needs the twin: a `mg_decode_step` that, per
token, keeps the f32 activation on-GPU across all projection/MLP matmuls, reads the
GPU-resident KV cache, and submits ONE command buffer — paying the ~360 µs once per token
instead of ~336 times. The kv.go `MetalQ4K` doc already names this: *"a lone GEMV is
occupancy-bound; the decode bar needs the one-command-buffer forward, a tracked follow-up."*
Complication: the GDN recurrent scan + periodic full-attention must run on-GPU (or in a tight
hybrid that does not round-trip per matmul).

**Secondary lever — a kernel-efficiency pass on `q4k.m`.** Even amortized, the GEMV at 21%
bandwidth and the GEMM at 5% FLOP leave throughput on the table: revisit threadgroup/grid
sizing, vectorized/coalesced dequant, `simdgroup_matrix` for the GEMM tile, and larger
per-launch work so a single launch saturates more of the 18 GPU cores.

## 5. Honest fences

- **The clean 1.2 / 0.6 tok/s are single-run, greedy, one prompt** (29-token prefill, 64
  decode). They are a diagnosis baseline, not a multi-rep authority row; the phase split is
  the durable signal.
- **`[Metal q4_k prefill]` in the backend banner is set purely from `FAK_METAL`**, not from a
  live availability probe — but `TestMetalQ4KDecodeMatchesCPU` passing on this box proves the
  Metal path actually engages here.
- **The 53 GB "peak memory footprint"** is `/usr/bin/time -l`'s virtual figure (includes GPU
  mappings); the real resident set is 26 GB and there were 0 swaps in the clean arm.
- **llama.cpp 7.29 / 51.55** is the committed reference (`QWEN36-PARITY-RESULTS.md`,
  `model-ladder/qwen36-perf-gate-m3-20260619`), measured with fak not running.

## Reproduce

```bash
# on node-macos-a, fresh clone built -tags fakmetal -> ~/fak-3xbench/{fakchat,modelbench}
# stop the launchd-managed llama-server for a clean GPU, restore after (EXIT trap):
launchctl bootout gui/$(id -u)/com.fak.qwen36-model
FAK_Q4K=1 FAK_METAL=1 FAK_GPU_LEASE_NOWAIT=1 FAK_QPROFILE=1 ~/fak-3xbench/fakchat \
  -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  -tok ~/.cache/fak-models/tokenizers/qwen3.6 -p "..." -n 64 -quiet
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.qwen36-model.plist
# kernel microbench:
go test -tags fakmetal -run NONE -bench 'BenchmarkMetalQ4KGemv|BenchmarkMetalQ4KGemm' -benchtime 30x ./internal/model
```

Raw artifact: `experiments/benchmark/runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json`.
