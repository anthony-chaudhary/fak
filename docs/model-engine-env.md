---
title: "fak model/compute engine env knobs (FAK_*) reference"
description: "Every FAK_* environment variable the in-kernel model and compute engine read: GPU residency budget, quant/load format, matmul parallelism, SIMD kernel tiers, and the GPU build vars — each with type, default, and when to reach for it."
---

# model-engine-env.md — `FAK_*` knobs for the model & compute engine

A single reference for the environment variables read by the **in-kernel model
engine** (`internal/model`), the in-kernel lifecycle engine (`internal/modelengine`),
and the **compute backends** (`internal/compute`), plus the handful the front-end
binaries (`cmd/fak`, `cmd/fakchat`, `cmd/gpucheck`)
read to pick a load/device path. This is the compute-engine companion to
[`serve-config.md`](serve-config.md), which covers the `fak serve` gateway and
does **not** cover these.

Every variable, type, and default below is read directly from the code — the
**Source** column points at the exact `file:line` so the table can't silently
drift from the engine. Regenerate the candidate list any time with:

```sh
grep -rhoE "FAK_[A-Z_0-9]+" internal/model internal/modelengine internal/compute cmd | sort -u
```

Conventions used in the tables:

- **flag (set/unset)** — any non-empty value turns it on; unset is off.
- **`1`-flag** — only the literal string `1` turns it on; everything else is off.
- **off-words** — a boolean that is **on by default** and turned off by one of
  `0`, `false`, `False`, `FALSE`, `off`, `OFF`; any other value (or unset) is on.
- A malformed numeric value falls back to the listed default (it is ignored, not
  an error) — except `FAK_BUDGET`, which prints a one-line `[fak] …` notice to
  stderr and then runs at full width.

---

## 1. GPU / device residency & selection

The operator levers for *where weights live* and *which backend runs*. These are
the ones to reach for when a model is bigger than device VRAM, or when you are
choosing a GPU path.

| Variable | Type / units | Default | When to use | Source |
|---|---|---|---|---|
| `FAK_GPU_BUDGET_MB` | int, MiB | unset / `0` / invalid = **unbounded** for budgeted weight-upload paths | The **Spills** lever. Cap device-local weight residency on a card whose VRAM is smaller than the weight set; weights past the cap spill *in upload order* (early/hot layers stay device-local, the cold tail spills by choice instead of losing the allocation race): Vulkan uses host-visible memory, CUDA uses managed memory. | `internal/compute/vulkan.go:57`, `internal/compute/cuda.go:185` |
| `FAK_VULKAN_SPIRV` | filesystem path (compiled SPIR-V dir) | unset = **Vulkan backend not registered** | Point at the compiled shader dir to register and use the Windows Vulkan backend at runtime. `build_vulkan.ps1` sets it after a build. Without it, `init()` returns early and only the Reference floor is available. (Requires the `vulkan` build tag.) | `internal/compute/vulkan.go:28` |
| `FAK_METAL` | flag (set/unset) | unset (CPU forward) | Apple GPU. In `fak serve` it is the env-equivalent of `--metal`: with `--gguf` and no `--base-url` it runs the in-kernel chat through the metalgemm GPU — **prefill + GPU-resident Q8 decode** on a dense Qwen-class Q8 model (#67). In `cmd/fakchat` it routes the resident-Q4_K hybrid **prefill** q4_k GEMMs through the Metal dequant-GEMM (Qwen3.6-27B path; decode stays CPU there). The Metal backend is linked automatically on darwin/arm64 with cgo; it is a no-op on the pure-Go build (and `fak serve --metal` **fails loud** if requested without the build/device, rather than serving on CPU). | `cmd/fak/serve.go` (`--metal`/`resolveServeMetal`), `cmd/fakchat/main.go:205` |
| `FAK_CUDA_GRAPH` | `1`-flag | off | Enable the reusable-CUDA-graph decode path — the load-bearing GPU lever, not a marginal toggle. On a model that fits the GPU it captures the decode op stream once and replays it via `cudaGraphExecUpdate`, collapsing ~600 host launches/token into one: a **16× decode speedup (7.5 → ~120 tok/s), at parity with `llama.cpp` Q8_0** (`GPU.md` §3b). Off by default because the path pins a **fixed-capacity (1024-position) device KV** so capture never hits a `cudaMalloc`; dynamic/ring KV is the follow-up. (The earlier per-token *re-instantiate* approach was the measured no-win — not this reusable path.) | `internal/compute/cuda.go:191` |
| `FAK_CUDA_Q8` | `1`-flag | off | `gpucheck`: exercise the CUDA Q8 device path in the Approx-gate witness. | `cmd/gpucheck/main.go:101` |
| `FAK_CUDA_F16` | `1`-flag | off | `gpucheck`: exercise the CUDA f16 device path in the Approx-gate witness. | `cmd/gpucheck/main.go:101` |

Vulkan also has a hardware **single storage-buffer** ceiling, independent of the
aggregate residency budget. At backend init fak records the effective cap
(`min(maxStorageBufferRange, maxMemoryAllocationSize)` when both are known) and
refuses any one tensor/KV buffer that exceeds it with the offending buffer name.
`FAK_GPU_BUDGET_MB` can spill cold weights, but it cannot make one over-cap
resource legal; those tensors still need split/chunked upload.

> Note: `FAK_BACKEND` appears in `internal/compute` *comments* as the intended
> native backend selector (`FAK_BACKEND=cuda|metal|vulkan` → `Pick(name)`), but
> no shipped binary currently reads it — backends are selected in code today.
> It is listed here only so the comment reference doesn't read as a live knob.

## 2. Model load & quantization format

What format the weights are loaded/quantized into at process start.

| Variable | Type / units | Default | When to use | Source |
|---|---|---|---|---|
| `FAK_Q4K` | flag (set/unset) | unset (lean-Q8 path) | Load the resident-Q4_K decode path (raw q4_k blocks stay resident; decode streams ~1.8× fewer bytes). The Qwen3.6-27B route. | `cmd/fak/main.go:1316`, `cmd/fakchat/main.go:119` |
| `FAK_Q4_FORCE` | `1`-flag | unset | Acknowledge and run the int4 path whose q8-intermediate build peaks ~28 GB; gated so it can't silently pressure a shared fleet box. Without it the run refuses with an explanatory message. | `cmd/fakchat/main.go:189` |

## 3. Native lifecycle scheduler

The registered `inkernel` engine uses these knobs when it builds its process-local
continuous-batching scheduler.

| Variable | Type / units | Default | When to use | Source |
|---|---|---|---|---|
| `FAK_NATIVE_MAX_RUNNING` | int >= 1 (requests) | unset / invalid / <=0 = unbounded | Cap how many admitted in-kernel lifecycle requests run concurrently in `modelengine.NativeScheduler`; excess requests wait FIFO and are promoted as lanes finish. Use to bound the native batch width for local experiments or memory pressure. | `internal/modelengine/modelengine.go:226` |
| `FAK_NATIVE_KV_MAX_BLOCKS` | int >= 1 (paged-KV blocks) | unset / invalid / <=0 = preemption disabled | Enable the native scheduler's KV pressure path and set the live block budget. When the running set exceeds the budget, the scheduler preempts a victim at a decode-step boundary and readmits it later. | `internal/modelengine/modelengine.go:239` |
| `FAK_NATIVE_KV_BLOCK_TOKENS` | int >= 1 (tokens per block) | unset / invalid / <=0 = `16` | Override the block size used by the scheduler's paged-KV budget estimator and swap pool. | `internal/modelengine/modelengine.go:244` |
| `FAK_NATIVE_KV_PREEMPT_MODE` | enum: `swap`, `swap-to-host`, `recompute` | unset = `swap` | Select how a preempted lane releases KV: serialize paged KV to host bytes (`swap`) or drop KV and replay prompt+generated tokens on readmit (`recompute`). | `internal/modelengine/modelengine.go:249` |

## 4. Matmul parallelism (worker budget)

How many cores the matmuls spread across. Full precedence (first match wins):
`FAK_WORKERS` → `FAK_BUDGET` → `GOMAXPROCS` (all cores). Resolved once at package
init and recorded in the bench JSON so a run states the parallelism it was taken
at.

| Variable | Type / units | Default | When to use | Source |
|---|---|---|---|---|
| `FAK_WORKERS` | int ≥ 1 (absolute core count) | `GOMAXPROCS` | Pin an exact worker count — set `1` to reproduce the serial reference, or a fixed N to A/B serial-vs-parallel in one environment. | `internal/model/parallel.go:32`, `internal/model/budget.go:109` |
| `FAK_BUDGET` | fraction `(0,1]`, or percent (`75`, `75%`) | all cores | Machine-**portable** share — `0.75` = 75% of whatever box this is (24/32, 6/8…). Use to "leave headroom" across a fleet of differently-sized machines without per-box arithmetic. Resolved against `GOMAXPROCS`. | `internal/model/parallel.go:32`, `internal/model/budget.go:116` |
| `FAK_PAR_SPIN` | int64 ≥ 0 (idle spins) | `1048576` (`1<<20`, ~1 ms) | Tune the spin-before-park budget of the matmul worker pool for A/B. Must exceed the serial gap between decode matmuls or workers park mid-token; `1<<22` over-spins and regresses (M3 Pro sweep). | `internal/model/parallel.go:73` |

## 5. Quant kernel tiers & SIMD (A/B and benchmarking)

Developer/benchmark levers for *which kernel* runs. The defaults auto-detect the
best tier the hardware supports — you only set these to pin a tier for an A/B
measurement or to work around a misdetection. Pinning a tier above what the CPU
has is capped down to what's available.

| Variable | Type / units | Default | When to use | Source |
|---|---|---|---|---|
| `FAK_QKERNEL` | enum — amd64: `scalar`\|`avx2`\|`avx512`; arm64: `scalar`\|`neon`/`sdot`/`asimddp`\|`amort`/`i8mm`/`smmla` | hardware-detected | Pin the qdot8 / quant-GEMM SIMD tier for A/B. (`internal/compute` reads it as the ISA-neutral tier analogue too.) | `internal/model/quant_amd64.go:52`, `internal/model/quant_arm64.go:50` |
| `FAK_QGEMM` | string — `legacy` else tile | tile | Force the old per-element qdot8 prefill sweep (`legacy`) to A/B against the register-blocked tile kernel. | `internal/model/quant_gemm.go:22` |
| `FAK_QGEMM_GROUP` | off-words | on | Group the q/k/v and gate/up GEMM launches (avoids repeated launch barriers). Turn off to A/B. | `internal/model/quant_gemm.go:29` |
| `FAK_QGEMM_GROUP_MAXP` | int ≥ 1 (prompt panel) | `1024` | Max prompt-panel batch width that still groups. | `internal/model/quant_gemm.go:40` |
| `FAK_AWQ_KERNEL` | enum `scalar`\|`avx2`\|`avx512` | hardware-detected | Pin the AWQ matmul SIMD tier for A/B. | `internal/model/awq_amd64.go:19` |
| `FAK_ARM_TILE` | `1`-flag | off | Enable the arm64 register-blocked tile GEMM (opt-in for non-Apple arm64 parts / A/B). | `internal/model/quant_arm64_gemm.go:17` |
| `FAK_QATTN_GQA` | off-words | on | Fused GQA attention path. Turn off to A/B. | `internal/model/batch.go:52` |
| `FAK_FDOT3_SIMD` | off-words | on | SIMD `fdot3` (3-row dot) in attention. Turn off to fall back to scalar. | `internal/model/batch.go:88` |
| `FAK_FDOT3_SIMD_MINB` | int ≥ 1 (batch) | `64` | Minimum batch size at which SIMD `fdot3` kicks in. | `internal/model/batch.go:99` |
| `FAK_FDOT3_AVX512` | off-words | on | Use the AVX-512 `fdot3` asm (when the CPU has it). Turn off to use the AVX2 path. | `internal/model/fdot_amd64.go:16` |
| `FAK_SAXPY3_SIMD_MINPOS` | int ≥ 0 (positions) | `1` | Minimum positions at which SIMD `saxpy3` (V accumulate) kicks in. | `internal/model/batch.go:63` |
| `FAK_SAXPY3_SIMD_MINB` | int ≥ 1 (batch) | `1` | Minimum batch size for SIMD `saxpy3`. | `internal/model/batch.go:74` |
| `FAK_Q_FAST_SWIGLU` | off-words | on | Fast quantized SwiGLU. Turn off to A/B against the reference. | `internal/model/batch.go:110` |
| `FAK_HAL_Q8_BATCH_LAYERS` | int ≥ 0 (layers) | `2` | How many device-Q8 batched layers the HAL path uses. | `internal/model/hal.go:105` |
| `FAK_QPROFILE` | flag (set/unset) | off | Print coarse phase timing (quantize / GEMM / attention …) for batched-Q and Metal prefill. | `internal/model/quant_forward.go:21`, `internal/model/metal_prefill.go:30` |
| `FAK_GEMMA4_NO_ROPEFREQS` | `1`-flag | off | Gemma-4 numerics A/B: skip the RoPE inv-freq precompute. | `internal/model/gemma4.go:12` |
| `FAK_GEMMA4_SCALE_SQRT` | `1`-flag | off | Gemma-4 numerics A/B: apply the `sqrt(dim)` embedding scale. | `internal/model/gemma4.go:13` |

## 6. GPU build-time vars

Read by the offline shim build scripts, not the running engine.

| Variable | Type / units | Default | When to use | Source |
|---|---|---|---|---|
| `FAK_CUDA_ARCH` | `sm_XX` (e.g. `sm_80`, `sm_89`, `sm_90`; bare `89` also accepted) | `sm_89` (Ada / L4) | Target a different NVIDIA arch when building `libfakcuda` (e.g. `sm_80` for datacenter GPU). | `internal/compute/build_cuda.sh:51` |
| `FAK_NVCC_CCBIN` | filesystem path | `/usr/bin/g++` | Point `nvcc` at a specific host compiler. | `internal/compute/build_cuda.sh:55` |

> `FAK_VULKAN_SPIRV` is set by `internal/compute/build_vulkan.ps1` after compiling
> the shaders, but it is read by the engine **at runtime** to register the backend
> — see §1.

---

## Not listed here

Out of scope for the model/compute engine reference, and so deliberately omitted:

- **Test-only** vars (read only from `*_test.go`): `FAK_PERF*`, `FAK_BENCH_*`,
  `FAK_ORACLE_*`, `FAK_RESOLVER_CHECKPOINT_DIR`, `FAK_SINGLEFILE_CHECKPOINT`.
- **C header include guards** (`#ifndef FAK_*_BACKEND_H`) — not env vars:
  `FAK_CUDA_BACKEND_H`, `FAK_METAL_BACKEND_H`, `FAK_VULKAN_BACKEND_H`.
- **Gateway / serve** vars (`FAK_HTTP_*`, `FAK_PLANNER_*`, `FAK_RATELIMIT_*`,
  `FAK_MODEL_DIR`, …) — see [`serve-config.md`](serve-config.md).

See also: [`experiments/gpu/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/gpu/README.md) and
[`docs/notes/gpu-parity-tracking-480.md`](notes/gpu-parity-tracking-480.md) for the GPU
residency / parity context behind `FAK_GPU_BUDGET_MB` and the device paths.
