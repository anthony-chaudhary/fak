---
title: "GLM-5.2 full-GPU-resident across all 8 GPUs: the resident-fit benchmark answered (2026-07-01)"
description: "GLM-5.2 UD-Q4_K_M now serves full-GPU-resident across all 8 A100-80GB GPUs on the GPU server via llama.cpp: ~434 GiB VRAM-resident, ≤6-minute load from local NVMe, measured 23.2 tok/s single-stream decode — ~8.9x the 2.62 tok/s cpu-offload baseline. Engine-honest: this is the llama.cpp baseline engine, not the pure-fak kernel; resident-EP on fak's kernel stays gated on the device collective."
---

# GLM-5.2 full-GPU-resident on all 8 GPUs (2026-07-01)

Companion to
[GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md](GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md),
which computed that the 8×80 GB GPU server's 640 GiB VRAM fits GLM-5.2 UD-Q4_K_M
(433.82 GiB) resident and left the live number as a `not yet`. This note records the
**measured** answer for the llama.cpp engine: the fit is real, the serve is up, and
single-stream decode lands at **23.2 tok/s** — about **8.9×** the measured 2.62 tok/s
`--n-cpu-moe` cpu-offload wall on the same box
([GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md)).

## What runs (WITNESSED on the box, 2026-07-01 ~21:30Z)

- `llama-server` (the pre-built sm_80 CUDA build at `/projects/llama.cpp/build/bin/`),
  shard 1 of the 11-shard unsloth UD-Q4_K_M checkpoint read from **local NVMe**
  (`/mnt/sglang_dv3/glm52-q4/`), `--n-gpu-layers 999`, **no** `--n-cpu-moe`,
  `--ctx-size 8192`, OpenAI `/v1` on `:8000`, alias `glm-5.2`.
- **All 8 GPUs carry weights** (`nvidia-smi` memory.used, MiB):
  `19875, 64599, 58799, 64599, 64599, 58799, 64599, 49181` — **≈434.6 GiB resident**,
  matching the computed 433.82 GiB weight bulk + KV/buffers. Mid-generation
  utilization is nonzero on every GPU (7–15% per-GPU snapshot during a 400-token decode;
  layer-split pipelines one GPU at a time, so per-GPU % is structurally low
  single-stream).
- **Load time ≤6 minutes** from launch to a passed smoke chat completion
  (launch 21:27:48Z → `READY8` observed on the 21:33Z poll) — the NVMe-first staging
  path doing what it was built for (the ~53× NFS→NVMe read-speed delta).

## Measured throughput (single-stream, temperature 0, 2026-07-01)

| metric | value | source |
|---|---|---|
| decode (256 new tokens) | **23.2 tok/s** (43.05 ms/token) | `llama-server` `print_timing`, two runs (23.23 / 23.22) |
| end-to-end wall clock | 22.9 tok/s (11.16 s for 256 tokens) | timed `curl`, warm run |
| prompt eval (11-token prompt) | 46.2 tok/s | `print_timing` (tiny prompt — NOT a prefill benchmark) |
| cpu-offload baseline (same box, same checkpoint) | 2.62 tok/s | 2026-06-25 plan note |

Evidence artifacts live on the box under `/mnt/sglang_dv3/glm52-8gpu/`
(`PHASE`, `launch.log`, `server.log`, `PERF.txt`); launcher at `/tmp/glm8.sh`.
The two long-idle single-GPU serves this replaced (:8000 fak-native GPU0-only and the
five-day-old :8001 sibling, both with **zero** established connections) were stopped
after verification; the small fak gateway process was left running.

## Honest scope

- **Engine-honest:** this is the **llama.cpp** engine — the repo's declared
  apples-to-apples throughput baseline — not the pure-fak kernel. Never put this
  number next to a fak per-token device-kernel cost without holding
  {weights, hardware, precision, ctx} equal.
- The pure-fak **resident-EP** serve remains gated exactly where the 06-29 note left
  it: the cross-device NCCL collective, EP live-wiring into `glmMoeFFN`, and a
  multi-GPU binary on the box. This note *raises the bar that path must beat*:
  23.2 tok/s resident-baseline, not 2.62 tok/s cpu-offload.
- Single-stream only; no concurrency sweep, no quality claim (UD-Q4_K_M dynamic 4-bit
  quant), ctx 8192. The 46 tok/s prompt-eval row is an 11-token prompt, not prefill
  at depth. The #413 serving witness against this endpoint is the natural next
  checkable step (`tools/glm52_e2e_after_serve_dgx3.sh`).
