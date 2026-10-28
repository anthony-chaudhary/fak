---
title: "fak CUDA backend development loop"
description: "How to edit, build, validate, and ship fak CUDA kernels across local no-GPU checks and remote GPU acceptance gates."
---

# Developing the CUDA backend

This is the one place that describes the whole loop for working on fak's CUDA backend:
edit a kernel, build it, prove it correct, and ship it without it silently rotting. If you
are adding a *new* device backend from scratch, read [`EXTENDING.md`](https://github.com/anthony-chaudhary/fak/blob/main/EXTENDING.md) first
for the registration seam; this guide is the CUDA-specific loop that rides on top of it.

## Why the loop is shaped this way

The CUDA backend is three source layers that have to agree on one flat C ABI:

| File | Role |
|---|---|
| [`internal/compute/cuda_kernels.cu`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/cuda_kernels.cu) + [`internal/compute/cuda/*.cuh`](../internal/compute/cuda/README.md) | thin single-translation-unit aggregator + focused kernel modules (`__global__` + `extern "C" fcuda_*` entry points) |
| [`internal/compute/cuda_backend.h`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/cuda_backend.h) | the flat C ABI — the typed seam (data, never trust) |
| [`internal/compute/cuda.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/cuda.go) | the cgo binding that registers an *Approx* `cuda` backend |

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
| validate (real model + every device-family witness) | `make cuda-accept` | a CUDA host with a GPU and compatible HF weights |

## 1. Setup — stand up a toolchain

You only do this on a node that will actually build. On the win32 dev host you do *not* need
any of it: `make cuda-check` runs without a toolchain.

- **WSL, no sudo** (a laptop with an NVIDIA card and WSL2 GPU passthrough): run
  [`internal/compute/setup_cuda_wsl.sh`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/setup_cuda_wsl.sh). It installs a
  complete user-space CUDA 12.6 toolkit (real `nvcc`, cuBLAS-dev, headers) via micromamba into
  `~/cudaenv`, then `source ~/cudaenv.env`. The version is pinned so two devs build against the
  same `nvcc`.
- **Datacenter / GPU server / GCP Deep-Learning-VM**: `nvcc` is already on `PATH` and CUDA
  lives at `/usr/local/cuda`. Nothing to install — [`build_cuda.sh`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/build_cuda.sh)
  detects the system toolchain automatically.
- **Native Windows** (a Windows box with the CUDA Toolkit + a signing cert): use
  `fak-dev build-cuda build|binary`, which ports the build off the WSL workaround and
  code-signs the binary (on Windows, WDAC blocks unsigned fork/exec).
  [`tools/build_cuda_windows.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/build_cuda_windows.ps1) is the legacy path; it
  writes no archive stamp (section 3), so a kernel-only edit built through it can relink stale.

The GPU arch defaults to `sm_89`; override with `FAK_CUDA_ARCH=<arch>`. The authoritative
supported set is [`internal/compute/cuda_arch.txt`](../internal/compute/cuda_arch.txt), which every
build entry point validates before invoking `nvcc`. Single-arch builds remain the fast developer
path, including first-class Blackwell targets:

```bash
make cuda-build-sm100  # B200 / GB200
make cuda-build-sm103  # B300 / GB300 (CUDA 12.9+)
make cuda-build-sm120  # RTX 50-series / consumer Blackwell
```

## 2. Author — edit, then gate locally (no GPU)

Edit the kernel in the focused `cuda/*.cuh` module, its prototype in `cuda_backend.h`, and
the cgo binding in `cuda.go`. Keep `cuda_kernels.cu` as the thin compile entry and preserve
its module include order; see the [`cuda/` module guide](../internal/compute/cuda/README.md).
Then run the local gate **before** you push to a GPU node:

```
make cuda-check                 # wraps: bash internal/compute/build_cuda.sh check
```

This runs [`build_cuda.sh check`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/build_cuda.sh): first the pure-text
[`tools/cuda_abi_parity.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/cuda_abi_parity.py) cross-check that every `fcuda_*`
prototype in the header is defined in the `.cu` and declared for every `C.fcuda_*` call in
`cuda.go`, then a standalone `cuda_backend.h` parse with a strict host C compiler when one is
available. It needs no nvcc, no GPU, and no cgo, so it catches both signature drift and
`uint8_t`-class missing-header portability bugs locally instead of on a paid GPU VM. The
no-toolchain Windows host keeps the Python mirror in [`scripts/ci.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/scripts/ci.ps1),
and the same check is part of `make ci`.

On a node that *has* a C compiler, add the cgo type-check (no CUDA toolkit needed — the header
is deliberately toolkit-free): `go vet -tags cuda ./internal/compute/`.

## 3. Build — compile the kernels on a CUDA host

```
make cuda-build                 # nvcc -> libfakcuda.a, then go build -tags cuda
```

This delegates to [`build_cuda.sh`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/build_cuda.sh), which compiles the
kernels into an in-tree `libfakcuda.a` and builds the `-tags cuda` variant against it. The
same script works unchanged on WSL, a datacenter image, and a GCP DLVM.

Go keys builds on source content, not on cgo link inputs, so the archive alone never
invalidates a linked binary. Both `build_cuda.sh` and `fak-dev build-cuda` therefore write
the git-ignored `internal/compute/cuda_archive_stamp_gen.go`, which carries the archive digest
(nvcc's per-process `tmpxft_<pid>` names masked). A kernel-only edit then relinks every
`-tags cuda` binary and test binary, including a hand-run `go test -tags cuda -c`, while an
unchanged rebuild still hits the cache. An archive rebuilt by hand, or by the legacy
`tools/build_cuda_windows.ps1`, bypasses the stamp; rerun the entrypoint. The masked digest is
witnessed byte-stable only on Linux (GNU ar's deterministic default); native Windows objects are
not normalized, so there an unchanged rebuild may re-stamp and recompile the compute package:
slower, never stale.

### Build provenance (`fak.cuda-build-receipt.v1`)

A successful `build-cuda build` or `build-cuda binary` observes the built artifact back with
`cuobjdump` before the receipt is written, so the receipt proves which architectures the
artifact actually contains instead of echoing what was requested. On success the receipt schema
switches from `fak.compute-build-receipt.v1` to the successor `fak.cuda-build-receipt.v1`
exactly when the `cuda` block is attached; `--json` output reports both via `receipt_schema`
and `provenance`.

The block carries:

| Field | Meaning |
|---|---|
| `artifact_kind` | `binary` (the linked executable) or `libfakcuda` (the archive) — whichever artifact was observed |
| `artifact_sha256` | content digest of the observed artifact |
| `requested_archs` | image targets derived from the gencode arguments that `ParseCUDAArchMatrix` builds from `internal/compute/cuda_arch.txt` (`sm_*` SASS targets plus the last arch's `compute_*` PTX target), sorted by numeric compute capability then suffix |
| `sass_archs` | SASS images observed in the artifact via `cuobjdump --list-elf`, parsed as `sm_<cc>` with an optional `a` or `f` suffix |
| `ptx_archs` | PTX images observed via `cuobjdump --list-ptx`; a PTX image named `sm_<cc>` is recorded as its requested target form `compute_<cc>` |
| `nvcc` | `{version, role, executable, sha256}` — version parsed from the `nvcc --version` banner, executable SHA-256 without any host path |
| `cublas_version` | build-time cuBLAS version read from `cublas_api.h` under the toolchain's CUDA include directories (falling back to the toolkit include dir); the header the kernels were compiled against, not the runtime-loaded library |

Presence and failure rules:

- The block is attached **only** on a successful `build` or `binary`. Failed receipts keep
  `fak.compute-build-receipt.v1` and never carry a `cuda` block; the tracker clears the block
  and resets the schema on any later failure (including a post-attach smoke failure).
- If a requested SASS arch or requested PTX target has no matching image in the artifact, the
  build fails with the exported sentinel `computebuild.ErrCUDAArchMissing` (use `errors.Is`).
  The receipt is written failed with no CUDA block.
- If `cuobjdump` is missing beside the selected nvcc, a listing or the nvcc version banner is
  unparseable, or `cublas_api.h` cannot be read, the build fails with
  `computebuild.ErrCUDAProvenanceInvalid`. An unobserved CUDA block is never written.
- `ArchiveCUDALib` records `libfakcuda.a`'s SHA-256 in the archive artifact identity,
  independent of the provenance block.
- The `cuda` field lives only on the successor wire envelope; the shared receipt JSON type keeps
  it out of V2 marshaling, so Vulkan v2 unknown-field rejection is unchanged.

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

```bash
FAK_CUDA_ACCEPT_HF=/srv/models/Qwen2.5-1.5B-Instruct make cuda-accept
```

`cuda-accept` first runs `cuda-accept-model`. It refuses before checking the CUDA
toolchain or building anything unless `FAK_CUDA_ACCEPT_HF` is explicit. The directory
must contain a real HuggingFace `config.json` and compatible, unsharded
`model.safetensors`; a GGUF file is not accepted. `FAK_CUDA_ACCEPT_TOKENS` selects the
positive greedy-token count and defaults to 12. The model gate builds `cmd/gpucheck`
through `build_cuda.sh binary`, then emits one JSON receipt on stdout while build progress
goes to stderr.

For RTX PRO 6000 and RTX 50-series first light (`sm_120`), use the current Blackwell
toolchain (CUDA 12.9.1 in CI) and select the architecture explicitly:

```bash
FAK_CUDA_ARCH=sm_120 \
FAK_CUDA_ACCEPT_HF=/srv/models/Qwen2.5-1.5B-Instruct \
FAK_CUDA_ACCEPT_TOKENS=12 \
make cuda-accept
```

The model receipt compares the same greedy continuation from the legacy pure-Go f32 CPU
reference and the requested native CUDA backend. `-require-native` refuses when the
session's observed backend metadata does not select CUDA with device memory. That is a
session-selection and output-parity gate; it does not prove that every individual operation
avoided a host fallback. The per-family acceptance witnesses retain that narrower diagnostic
role. Run their manifest directly when diagnosing an operation family without rebuilding or
loading the model:

```bash
bash tools/cuda_acceptance.sh
```

[`tools/cuda_acceptance.sh`](https://github.com/anthony-chaudhary/fak/blob/main/tools/cuda_acceptance.sh) runs each acceptance witness
(`run_479`/`run_482`/`run_483`/`run_484`/`run_485`/`run_486` and the GLM-DSA
`private GPU witness runner`) and prints a per-family
`PASS | SKIP | FAIL` manifest. A **SKIP is not a PASS**: on a host with no reachable GPU every
family skips and the aggregator exits non-zero, so a CPU-only box can never read as green.
Each individual witness — e.g. [`tools/run_484_acceptance_on_gpu.sh`](https://github.com/anthony-chaudhary/fak/blob/main/tools/run_484_acceptance_on_gpu.sh)
for the fp16 HGEMM path — can still be run on its own. The current `gpucheck` receipt records
the model-directory basename, backend/tier selection, and compared tokens; it is parity and
selection evidence for this invocation, not a source-SHA, model-digest, full-configuration, or
hardware-identity provenance receipt. Provenance qualification, performance measurement, and
maturity promotion remain separate gates.

For native preparation without shell scripts, use the reached developer command:

```bash
fak-dev build-cuda first-light --repo-root /src/fak --arch sm_120 \
  --hf /srv/models/Qwen2.5-1.5B-Instruct --plan --json \
  --receipt .fak/cuda-first-light-plan.json
fak-dev build-cuda first-light --repo-root /src/fak --arch sm_120 \
  --hf /srv/models/Qwen2.5-1.5B-Instruct --timeout 20m --json
```

The plan writes pending stages without discovering a toolchain, compiling or running
a device. Execution checks model files and architecture, and checks that `cuobjdump`
sits beside the selected `nvcc`, because the archive image audit needs it. A toolkit
without it refuses before anything compiles, with `refusal` set to
`CUDA_PREFLIGHT_CUOBJDUMP_MISSING`; on an apt-installed toolkit, install
`cuda-cuobjdump-<major>-<minor>`. Execution then parses the C header strictly,
builds `gpucheck` through the native CUDA builder, and invokes its strict native HF
parity gate. It stops on the first failure and writes `.fak/cuda-first-light.json`,
including failed stages, the binary digest and the observed acceptance receipt.
`--arch` must match the observed device tier exactly. `--timeout` bounds the header,
build and acceptance commands together. GGUF serving coherence, per-family CUDA
acceptance, same-card reference ABBA and performance receipts remain pending;
`hardware_qualified` stays false even when this preparation passes.

## 5. Add a kernel

Adding a device op is a five-point change; `make cuda-check` catches the two you are most
likely to forget (a missing prototype or a missing binding):

1. **Kernel** — the `__global__` kernel plus an `extern "C" fcuda_yourop(…)` entry in the
   focused `cuda/*.cuh` module. Add a new module through the thin `cuda_kernels.cu` aggregator
   only when the [`cuda/` module guide](../internal/compute/cuda/README.md) calls for one.
2. **ABI** — the matching prototype in `cuda_backend.h`. Keep the header toolkit-free (pass
   `__half`/device handles as `void*`) so `cuda-check` and `go vet -tags cuda` stay GPU-free.
3. **Binding** — the method on `*cudaBackend` in `cuda.go` that calls `C.fcuda_yourop(…)`.
4. **Floor** — if the op is a new numeric path, record its cosine (or equality) floor as a
   constant in `cuda.go`, with the "do not read a pass from this value alone" caveat *and* a
   pointer to the acceptance witness that measures it.
5. **Witness** — a `-tags cuda` test (`TestCUDA…MatchesRef`) comparing the device op to the
   `cpuref` Reference under the floor, and an entry in `cuda_acceptance.sh` so the new path
   joins the one-command verdict.

Wrap every launch, cuBLAS call and async copy/fill on `g_stream` in
`FCUDA_TS("<kernel symbol>", grid, …)` and put `fcuda_ts_untimed_transfer();` on the line
before every synchronous `cudaMemcpy`. The GPU-free `TestCUDATimestampSourceCoversEveryLaunch`
fails on an unwrapped site, and `TestKernelOpClassCUDANames` fails on a new row name that
classifies as `other` without being listed (section 6).

## 6. Profile — per-kernel GPU time

`FAK_CUDA_TIMESTAMP_PROFILE=1` (exactly `1`, like `FAK_VULKAN_TIMESTAMP_PROFILE`) turns on the
CUDA kernel timestamp profile. The shim reads it once at load (`cuda/kernel_profile.cuh`), so
it needs no Go env read. With it unset, each launch site costs one predictable branch and the
command stream gets no event calls, and no kernel-profile reader is registered.

- **Timing = `event_pairs`.** Each `FCUDA_TS` site records a pre-created start and stop
  `cudaEvent_t` on `g_stream` around its launch. The pairs resolve in one host wait, never per
  launch: at the token boundary (`Recycle`), at a reader snapshot, or when the 4096-pair ring
  fills.
- **Rows** use the Vulkan `KernelRow` shape and land in the same reader (`cuda`). A row is a
  kernel symbol without namespace or template arguments plus its grid x × y as `groups`, for
  example `k_q4k_gemm`, `mul_mat_q` (the P-04 MMQ arm) or `k_qwen38_fattn_mma_hd256` (the P-05
  mma arm). cuBLAS rows are `cublas_sgemm`, `cublas_gemm_ex` and `cublaslt_matmul`, with the
  GEMM m × n as `groups`. Async device copies and fills are the pseudo-kernels `d2d_copy` and
  `fill` (`0x0`). `KernelOpClass` maps `gemv` to `matvec`; `gemm` and `mul_mat` to `matmul`;
  and `fattn`, `attend`, `spec_verify` and `dsa_index` to `attention`. The rows that stay
  `other` are listed in `cudaKernelOtherAllowlist`.
- **Counters.** `GPUNS` is the sum of the pairs, and the rows sum to it. `untimed_transfer`
  counts synchronous host copies (uploads and readbacks on the legacy stream) and the
  peer-device copies (`fcuda_d2d_on` and the `fcuda_handoff_*` peer handoff). The ledger
  reports it without clearing `complete`.
  `truncated` counts pairs whose events failed. `untimed_compute` counts launches inside a
  capture. `barriers_elided` stays 0 because there is one in-order stream.
- **Graphs off.** With the profile on, `GraphBegin` returns false, so a profiled row times the
  ungraphed launch stream even when `FAK_CUDA_GRAPH=1`. The stderr header says
  `graphs=disabled`. One device and one `g_stream` only: per-session streams (P-07) and
  multi-GPU ranks are not profiled.
- **Stderr**, every `FAK_CUDA_TIMESTAMP_PROFILE_EVERY` token boundaries (default 32), in the
  Vulkan format with a `fak-cuda` prefix:
  `fak-cuda ts-profile batches=… ops/batch=… gpu_ms/batch=… wall_ms/batch=… truncated=… timing=event_pairs graphs=disabled`
  and then up to 40
  `fak-cuda ts-stage kernel=… groups=XxY calls/batch=… ms/batch=… us/call=… share=…%` lines.
  `fak-dev kernel-oracle rank --fak-profile` reads them. `--fak-ledger` keeps the lineage of a
  `cuda` structured `disabled-by-profile` ledger row, with method
  `structured_device_event_pair_intervals`.
- **Token-time caveat.** A profiled decode is not token time: each launch adds two event
  records, graphs are off, and a pair recorded on an idle stream also books the host's launch
  gap between the start event and the kernel. On a host-bound decode this overstates short
  kernels. Read shares and per-kernel A/B deltas from profile-on rows, and read tok/s from
  profile-off rows. The overhead has not been measured yet. The pending CUDA-host rung
  records profile-on against profile-off decode tok/s on the same card.

## CI — what's automatic and what isn't

- [`.github/workflows/cuda-build.yml`](https://github.com/anthony-chaudhary/fak/blob/main/.github/workflows/cuda-build.yml) runs
  **automatically** on every CUDA-touching push/PR, on plain GitHub-hosted runners with **no
  GPU**: the ABI/header preflight, the pure-Go cgo-leak guard (the default build must stay
  pure-Go), `go vet -tags cuda` (no toolkit), and — in official NVIDIA NGC CUDA `-devel`
  containers — an `nvcc` compile of the kernels plus `go build -tags cuda` (which *links*
  against the image's cudart/cublas stub libs but never *runs*, so no device is needed).
  The matrix keeps CUDA 12.6.2 for `sm_89` and CUDA 12.9.1 for `sm_100`/`sm_103`/`sm_120`, all on
  Ubuntu 22.04. Each image in the workflow is pinned by its multi-platform index digest.
  On 2026-10-09, anonymous manifest reads from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda)
  and NVIDIA's Docker Hub repository returned byte-identical indexes and `linux/amd64`
  manifests for both tags. The NGC origin avoids the observed Docker Hub unauthenticated
  pull-limit failure before checkout; it preserves the toolkits and every compile gate.
  [NVIDIA lists both tags as unsupported](https://gitlab.com/nvidia/container-images/cuda/-/raw/master/doc/unsupported-tags.md).
  Digest pinning fixes content selection, but does not guarantee registry availability or
  remove existing image vulnerabilities. Full image pulls and compiler results still need
  an exact-head CI witness; manifest equality alone does not prove those steps succeeded.
- [`.github/workflows/windows-cuda.yml`](https://github.com/anthony-chaudhary/fak/blob/main/.github/workflows/windows-cuda.yml) is the
  **manual** GPU lane: a self-hosted Windows+CUDA runner that runs the signed-binary Approx
  gate. It needs a real GPU and a signing cert, so it is `workflow_dispatch` only.

## Honest residuals

The AWQ 4-bit path (`AWQMatMul` / `AWQBatchedMatMul`, kernels `fcuda_awq_gemv` / `fcuda_awq_gemm`)
was the one device op family without a cpuref-parity gate — it shipped a binding but had no
recorded cosine floor and no acceptance witness. **That gap is now closed (#926, the #905
selection):** it carries a `cudaAWQCosineMin` constant + a `-tags cuda` witness
(`internal/compute/cuda_awq_test.go`, `TestCUDAAWQMatMul…` / `…BatchedMatMul…`) + a `run_*`-style
acceptance script (`tools/run_926_acceptance_on_gpu.sh`, in the `cuda_acceptance.sh` manifest),
the same recorded-then-measured pattern as every other floor. **The realized cosine is now
measured:** the acceptance witness ran on a datacenter GPU (sm_80, CUDA 12.8, at fak HEAD
`265ce03`) and recorded **cosine 1.00000000** for both `AWQMatMul` and `AWQBatchedMatMul` vs the
cpuref f32 Reference — above the recorded `cudaAWQCosineMin = 0.995` floor, which stays conservative
as on every sibling lane (e.g. Q8 measured 0.99999969 over its 0.999 floor, Q4_K 1.0 over 0.995).
That GPU datapoint completes the recorded-then-measured loop and closes #926's last residual by
data; the floor is left at the conservative 0.995 (a measured-cosine-of-1.0 gate would be fragile to
fp reduction-order drift, so — as on every sibling — the recorded floor sits below the measurement).

The health of this whole loop is itself scored: [`tools/cuda_dev_scorecard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/cuda_dev_scorecard.py)
re-derives a **process-debt** number from the tree (is there a local gate? an automatic CI
compile gate? a one-command witness? a documented loop?) so the dev process can't quietly
regress. Snapshot: [`CUDA-DEV-SCORECARD.md`](CUDA-DEV-SCORECARD.md).
