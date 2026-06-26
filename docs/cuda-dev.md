---
title: "fak CUDA backend development loop"
description: "How to edit, build, validate, and ship fak CUDA kernels across local no-GPU checks and remote GPU acceptance gates."
---

# Developing the CUDA backend

This is the one place that describes the whole loop for working on fak's CUDA backend:
edit a kernel, build it, prove it correct, and ship it without it silently rotting. If you
are adding a *new* device backend from scratch, read [`EXTENDING.md`](../EXTENDING.md) first
for the registration seam; this guide is the CUDA-specific loop that rides on top of it.

## Why the loop is shaped this way

The CUDA backend is three files that have to agree on one flat C ABI:

| File | Role |
|---|---|
| [`internal/compute/cuda_kernels.cu`](../internal/compute/cuda_kernels.cu) | the kernels (`__global__` + `extern "C" fcuda_*` entry points) |
| [`internal/compute/cuda_backend.h`](../internal/compute/cuda_backend.h) | the flat C ABI — the typed seam (data, never trust) |
| [`internal/compute/cuda.go`](../internal/compute/cuda.go) | the cgo binding that registers an *Approx* `cuda` backend |

The default `go build ./cmd/fak` excludes all of this and stays one pure-Go binary; the CUDA
path compiles only under `-tags cuda`. That keeps the shipped artifact clean, but it means
the canonical dev host (win32, **no CUDA toolkit, a walled GPU quota**) cannot build or run
the kernels at all. The build-and-prove half of the loop runs on a remote GPU node (WSL with
a consumer card, the GPU server, or a GCP DLVM). So the loop is deliberately split into a
**GPU-free half you run locally on every edit** and a **GPU half you run on a device**.

| Stage | Command | Where it runs |
|---|---|---|
| author (local gate) | `make cuda-check` | anywhere — no nvcc, no GPU |
| build | `make cuda-build` | a CUDA host (WSL / GPU server / cloud VM) |
| test (Approx witness) | `make cuda-test` | a CUDA host with a GPU |
| validate (every witness) | `make cuda-accept` | a CUDA host with a GPU |

## 1. Setup — stand up a toolchain

You only do this on a node that will actually build. On the win32 dev host you do *not* need
any of it: `make cuda-check` runs without a toolchain.

- **WSL, no sudo** (a laptop with an NVIDIA card and WSL2 GPU passthrough): run
  [`internal/compute/setup_cuda_wsl.sh`](../internal/compute/setup_cuda_wsl.sh). It installs a
  complete user-space CUDA 12.6 toolkit (real `nvcc`, cuBLAS-dev, headers) via micromamba into
  `~/cudaenv`, then `source ~/cudaenv.env`. The version is pinned so two devs build against the
  same `nvcc`.
- **Datacenter / GPU server / GCP Deep-Learning-VM**: `nvcc` is already on `PATH` and CUDA
  lives at `/usr/local/cuda`. Nothing to install — [`build_cuda.sh`](../internal/compute/build_cuda.sh)
  detects the system toolchain automatically.
- **Native Windows** (a Windows box with the CUDA Toolkit + a signing cert): use
  [`tools/build_cuda_windows.ps1`](../tools/build_cuda_windows.ps1), which ports the build off
  the WSL workaround and code-signs the binary (WDAC blocks unsigned fork/exec).

The GPU arch defaults to `sm_89`; override with `FAK_CUDA_ARCH=sm_80` for an older datacenter
card, or `sm_90` / `sm_100` for a newer one (the four advertised arches are `sm_80`, `sm_89`,
`sm_90`, `sm_100`).

## 2. Author — edit, then gate locally (no GPU)

Edit the kernel in `cuda_kernels.cu`, its prototype in `cuda_backend.h`, and the cgo binding
in `cuda.go`. Then run the local gate **before** you push to a GPU node:

```
make cuda-check                 # wraps: python tools/cuda_abi_parity.py --check
```

This runs [`tools/cuda_abi_parity.py`](../tools/cuda_abi_parity.py) — a pure-text cross-check
that every `fcuda_*` prototype in the header is defined in the `.cu` and declared for every
`C.fcuda_*` call in `cuda.go`. It needs no nvcc, no GPU, and no cgo, so it catches the most
common real break — a signature changed in one file but not the others — in milliseconds,
locally, instead of on a multi-host round trip. It runs on the no-toolchain Windows host too
(it is mirrored into [`scripts/ci.ps1`](../scripts/ci.ps1) and is part of `make ci`).

On a node that *has* a C compiler, add the cgo type-check (no CUDA toolkit needed — the header
is deliberately toolkit-free): `go vet -tags cuda ./internal/compute/`.

## 3. Build — compile the kernels on a CUDA host

```
make cuda-build                 # nvcc -> libfakcuda.a, then go build -tags cuda
```

This delegates to [`build_cuda.sh`](../internal/compute/build_cuda.sh), which compiles the
kernels into an in-tree `libfakcuda.a` and builds the `-tags cuda` variant against it. The
same script works unchanged on WSL, a datacenter image, and a GCP DLVM.

## 4. Validate — prove correctness against the CPU reference

The CUDA backend is an **Approx** peer of the `cpuref` f32 Reference: it is held to an
argmax-exact + logit-cosine gate, not bit-identity, because the device GEMM reorders the f32
contraction. Each numeric path records its cosine floor as a constant in `cuda.go`
(`cudaFP16CosineMin`, `cudaQ8CosineMin`, `cudaQ4KCosineMin`, `cudaAWQCosineMin`,
`cudaFlashAttnCosineMin`, …), and **every floor carries the same honest caveat**: the constant
*records* the threshold, it does not assert the path passes it — that is measured on a GPU node,
not read from the source.

Run the Approx witness on a GPU host:

```
make cuda-test                  # the -tags cuda CUDA/HALDevice witnesses (graphs off + on)
```

Run **every** on-GPU witness with one verdict:

```
make cuda-accept                # wraps: bash tools/cuda_acceptance.sh — one manifest, SKIP-is-not-PASS
```

[`tools/cuda_acceptance.sh`](../tools/cuda_acceptance.sh) runs each acceptance witness
(`run_479`/`run_482`/`run_483`/`run_484`/`run_485`/`run_486` and the GLM-DSA
[`tools/dgx_glm_gpu_witness.sh`](../tools/dgx_glm_gpu_witness.sh)) and prints a per-family
`PASS | SKIP | FAIL` manifest. A **SKIP is not a PASS**: on a host with no reachable GPU every
family skips and the aggregator exits non-zero, so a CPU-only box can never read as green.
Each individual witness — e.g. [`tools/run_484_acceptance_on_gpu.sh`](../tools/run_484_acceptance_on_gpu.sh)
for the fp16 HGEMM path — can still be run on its own.

## 5. Add a kernel

Adding a device op is a five-point change; `make cuda-check` catches the two you are most
likely to forget (a missing prototype or a missing binding):

1. **Kernel** — the `__global__` kernel plus an `extern "C" fcuda_yourop(…)` entry in
   `cuda_kernels.cu`.
2. **ABI** — the matching prototype in `cuda_backend.h`. Keep the header toolkit-free (pass
   `__half`/device handles as `void*`) so `cuda-check` and `go vet -tags cuda` stay GPU-free.
3. **Binding** — the method on `*cudaBackend` in `cuda.go` that calls `C.fcuda_yourop(…)`.
4. **Floor** — if the op is a new numeric path, record its cosine (or equality) floor as a
   constant in `cuda.go`, with the "do not read a pass from this value alone" caveat *and* a
   pointer to the acceptance witness that measures it.
5. **Witness** — a `-tags cuda` test (`TestCUDA…MatchesRef`) comparing the device op to the
   `cpuref` Reference under the floor, and an entry in `cuda_acceptance.sh` so the new path
   joins the one-command verdict.

## CI — what's automatic and what isn't

- [`.github/workflows/cuda-build.yml`](../.github/workflows/cuda-build.yml) runs
  **automatically** on every CUDA-touching push/PR, on plain GitHub-hosted runners with **no
  GPU**: the ABI parity check, the pure-Go cgo-leak guard (the default build must stay
  pure-Go), `go vet -tags cuda` (no toolkit), and — in an `nvidia/cuda:12.6.2-devel` container
  — an `nvcc` compile of the kernels plus `go build -tags cuda` (which *links* against the
  image's cudart/cublas stub libs but never *runs*, so no device is needed). This buys "a peer
  can't break the kernels or the cgo seam unnoticed" for ~2 minutes on the rare CUDA PR.
- [`.github/workflows/windows-cuda.yml`](../.github/workflows/windows-cuda.yml) is the
  **manual** GPU lane: a self-hosted Windows+CUDA runner that runs the signed-binary Approx
  gate. It needs a real GPU and a signing cert, so it is `workflow_dispatch` only.

## Honest residuals

The AWQ 4-bit path (`AWQMatMul` / `AWQBatchedMatMul`, kernels `fcuda_awq_gemv` / `fcuda_awq_gemm`)
was the one device op family without a cpuref-parity gate — it shipped a binding but had no
recorded cosine floor and no acceptance witness. **That gap is now closed (#926, the #905
selection):** it carries a `cudaAWQCosineMin` constant + a `-tags cuda` witness
(`internal/compute/cuda_awq_test.go`, `TestCUDAAWQMatMul…` / `…BatchedMatMul…`) + a `run_*`-style
acceptance script (`tools/run_926_acceptance_on_gpu.sh`, in the `cuda_acceptance.sh` manifest),
the same recorded-then-measured pattern as every other floor. The realized cosine is still a GPU
residual (the build host records the threshold; the acceptance run measures it).

The health of this whole loop is itself scored: [`tools/cuda_dev_scorecard.py`](../tools/cuda_dev_scorecard.py)
re-derives a **process-debt** number from the tree (is there a local gate? an automatic CI
compile gate? a one-command witness? a documented loop?) so the dev process can't quietly
regress. Snapshot: [`CUDA-DEV-SCORECARD.md`](CUDA-DEV-SCORECARD.md).
