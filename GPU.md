# GPU — the in-kernel forward pass, on the GPU (on-box witness + honest gap to llama.cpp)

> This is the doc the repo's "no GPU claim without an on-box witness" rule
> (`IN-KERNEL-MODEL-DESIGN.md`) was waiting for. GPU was **SIMULATED / forbidden to
> claim**. It now isn't: a modular CUDA backend behind the frozen `compute.Backend` seam
> runs a real Llama decode forward pass on this box's RTX 4070, **witnessed against the CPU
> reference**, and — with a reusable CUDA graph — **decodes at parity with `llama.cpp` Q8_0
> (≈120 tok/s) on a model that fits the GPU** (§3b). Every number below is from a real run
> captured in the test/bench output, not asserted; the one honest caveat (the parity path is
> gated + uses a fixed-capacity KV) is stated just as plainly.

## 0. What shipped (this lane)

| Piece | Where | Status |
|---|---|---|
| CUDA backend (cgo + cuBLAS + custom kernels) implementing `compute.Backend` | `internal/compute/cuda.go`, `cuda_kernels.cu`, `cuda_backend.h` | **shipped** |
| Registered as **Approx** peer `"cuda"`; `cpu-ref` stays the Reference Default | `cuda.go:init` | **shipped** |
| Device-resident KV cache (VRAM K/Kraw/V; on-device AppendKV/Clone) | `cuda.go:cudaKV` | **shipped** (Evict via host round-trip, deferred → #39) |
| Model leaf runs decode through the HAL — same model on CPU (bit-exact) or GPU (Approx) | `internal/model/hal.go` | **shipped** |
| **Reusable CUDA graph (`cudaGraphExecUpdate`) + kernel KV-append + cross-session weight share → decode parity with llama.cpp Q8_0 (≈120 tok/s)** | `cuda_kernels.cu`, `cuda.go` | **shipped, gated `FAK_CUDA_GRAPH=1`** (§3b) |
| No-sudo WSL CUDA toolchain + two-step nvcc→static-lib→cgo build | `setup_cuda_wsl.sh`, `build_cuda.sh` | **shipped** |
| Default `go build` stays one **pure-Go** artifact; CUDA is opt-in `-tags cuda` | — | **verified** |

## 1. The on-box witness (real runs, RTX 4070 Laptop, sm_89, CUDA 12.6, WSL2)

The toolchain itself is witnessed first — `nvcc` compiles an sm_89 kernel and it executes
on the GPU (`setup_cuda_wsl.sh` → `DEVICE NVIDIA GeForce RTX 4070 Laptop GPU sm_89`,
`RESULT y[0]=5.0`). Then the backend:

| Witness | Result | Test |
|---|---|---|
| Device SGEMM vs cpuref `fdot` matmul | cosine **1.00000000**, maxAbs **1.91e-06** | `TestCUDAMatMulApproxMatchesRef` |
| Full 3-layer Llama decode, 6 prompt + 8 greedy tokens, GPU vs cpuref | **argmax-exact every step**, final logit cosine **1.0** | `TestCUDAForwardMatchesRef` |
| **Real in-kernel model** decode on the GPU vs native path | **argmax-exact over 10 tokens**, prefill cosine **1.0** | `TestHALDeviceForwardMatchesNative` |
| HAL adoption preserves the proven path on cpuref | **Float32-bit-identical** logits + token-for-token greedy | `TestHALPrefillLogitsMatchNativeBitExact`, `TestHALGenerateMatchesNativeBitExact` |

The last row is the load-bearing safety property: routing the model through the HAL is
**byte-for-byte** identical on the Reference backend (so R2/R14 and the HF oracle cannot be
perturbed), while the **same code** runs Approx on the device. `compute.RequireReference`
keeps the device off the bit-identity rungs by construction.

Reproduce from `fak/`: `bash internal/compute/setup_cuda_wsl.sh` once, then
`bash internal/compute/build_cuda.sh test`. From the repo root, the laptop lane runner wraps
the same path and also exposes the CPU/Intel lane. On Anthony's Windows laptop, use the
PowerShell wrapper:

```powershell
.\tools\fak_laptop_test.ps1 accept
.\tools\fak_laptop_test.ps1 accept --full-cpu   # same acceptance path, but CPU runs ./...
.\tools\fak_laptop_test.ps1 accept --cpu-only   # CPU/Intel proof, no NVIDIA required
```

If the usual checkout has unrelated dirty work and `git pull` would be risky, run the proof
from a detached clean worktree instead:

```powershell
git fetch origin
git worktree add --detach ..\fleet-laptop-proof origin/master
cd ..\fleet-laptop-proof
.\tools\fak_laptop_test.ps1 accept
```

That single command runs the required laptop acceptance sequence: CPU/NVIDIA passthrough
preflight, CPU smoke, CUDA setup, CUDA tests, post-setup toolchain checks, and report
verification. Use `--full-cpu` when the CPU/Intel lane needs the full `./...` suite instead
of the focused smoke tests. Use `--cpu-only` when validating the CPU/Intel lane on its own;
it writes `fak\experiments\gpu\laptop-cpu-check.json` and
`fak\experiments\gpu\laptop-cpu.json`, then verifies those artifacts without requiring
NVIDIA passthrough. To isolate a failure, run the lower-level steps directly:

```powershell
.\tools\fak_laptop_test.ps1 check --require-nvidia --out fak\experiments\gpu\laptop-check.json
.\tools\fak_laptop_test.ps1 cpu --smoke            # focused CPU/HAL smoke
.\tools\fak_laptop_test.ps1 cpu -- ./...           # full CPU suite
.\tools\fak_laptop_test.ps1 check --out fak\experiments\gpu\laptop-cpu-check.json
.\tools\fak_laptop_test.ps1 cpu --smoke --out fak\experiments\gpu\laptop-cpu.json
.\tools\fak_laptop_test.ps1 cpu --out fak\experiments\gpu\laptop-cpu.json   # full CPU report
.\tools\fak_laptop_test.ps1 nvidia --setup --out fak\experiments\gpu\laptop-nvidia.json
.\tools\fak_laptop_test.ps1 check --require-nvidia --require-cuda-toolchain --out fak\experiments\gpu\laptop-post-setup.json
.\tools\fak_laptop_test.ps1 all --smoke --setup --out fak\experiments\gpu\laptop-all.json
.\tools\fak_laptop_test.ps1 verify
.\tools\fak_laptop_test.ps1 verify --cpu-only
.\tools\fak_laptop_test.ps1 verify --cpu-only --full-cpu
.\tools\fak_laptop_test.ps1 status
.\tools\fak_laptop_test.ps1 status --cpu-only
```

On Linux/macOS, call the same runner as `python tools/fak_laptop_test.py ...`; the NVIDIA
lane intentionally refuses to run on macOS except in `--dry-run` mode.
Use `--wsl-distro NAME` or `FAK_WSL_DISTRO=NAME` to pin a distro; otherwise the Windows
runner matches `fak/test.ps1` by preferring `Ubuntu-24.04` when it is installed and
falling back to WSL's default distro.
The `accept` lane writes `fak\experiments\gpu\laptop-check.json`,
`fak\experiments\gpu\laptop-all.json`, and
`fak\experiments\gpu\laptop-post-setup.json`; relative report paths are resolved from the
repo root even when the wrapper is launched from another directory. The `verify` lane
reads the post-setup check and all-lane reports by default and fails unless CPU, NVIDIA
passthrough, the CUDA toolchain, CUDA setup, and CUDA tests all passed. CPU check reports
also record the Go target (`GOOS`/`GOARCH`) so the laptop artifacts show which CPU lane was
actually tested. Reports include proof metadata (`mode` and, for run reports,
`cpu_scope`) so `verify --cpu-only` rejects NVIDIA/CPU artifacts and `verify --full-cpu`
rejects smoke-only CPU runs. Reports also include git revision and dirty-worktree summary
metadata, including a bounded dirty-path sample, so laptop artifacts can be tied back to
the exact repo state that produced them.
Each report also has a compact `summary` block for quick pass/fail review.
Use `status` to print those summaries, proof metadata, repo metadata, and per-check /
per-command results from the existing report files without rerunning any tests; it exits
non-zero if the selected reports are missing, from the wrong proof mode, failed, or only
dry-run artifacts. `verify` and `status` also reject reports from a different git `HEAD`
or dirty-worktree fingerprint by default; add `--allow-stale-repo` only when deliberately
inspecting older artifacts.

## 2. Why this is "native as reasonable" on THIS box

- **WSL2 + CUDA is real CUDA on the real GPU.** WSL2 GPU passthrough is live
  (`nvidia-smi` sees the 4070 from Ubuntu, `/usr/lib/wsl/lib/libcuda.so` present); the
  kernels are compiled by a real `nvcc` for `sm_89` and run on the Ada silicon. The only
  thing "virtual" is the driver shim, which is how essentially all Linux-on-Windows GPU
  compute works.
- **It sidesteps the WDAC policy** that blocks unsigned native-Windows binaries from
  `%TEMP%` — the same reason the Go test suite already runs in WSL (`CLAUDE.md`). A native
  Windows `-tags cuda` build (signed) is the portability follow-up (#37).
- **No sudo, no system mutation.** The toolkit is a user-space micromamba CUDA 12.6 env;
  the default `go build` needs none of it and stays pure-Go.
- **It honors DIRECTION.md.** CUDA C++ is a sanctioned *hardware seam* in a statically
  typed compiled language, off the request path, behind the re-validated typed boundary
  (the flat C ABI carries device pointers + shapes, never trust). The default artifact is
  still one pure-Go binary (`GoFiles=[compute.go cpuref.go]`; `cuda.go` is a CgoFile only
  under `-tags cuda`).

## 3. Parity baseline — `llama.cpp` CUDA on this 4070 (measured)

> The model is **Qwen2.5-7B-Instruct Q4_K_M** — the GPU parity target chosen in
> `GPU-MODEL-PICK.md` (strongest agentic model that fully fits 8 GB; weights ~4.68 GB,
> ~1.6–1.8 GiB headroom at 4–8K ctx). `llama.cpp` is built with CUDA from source in the
> same WSL+CUDA environment the fak backend uses, so the GPU + passthrough are identical.

**Measured** (RTX 4070 Laptop, 8188 MiB / 7090 free, sm_89, CUDA 12.6, WSL2;
`llama-bench -ngl 99` full GPU offload, median of 5 — a cold first invocation read low
and was discarded):

| test | t/s |
|---|---:|
| prefill `pp512` | **2256 ± 45** |
| prefill `pp256` | 2245 ± 73 |
| prefill `pp64` | 1432 ± 298 |
| decode `tg128` | **48.0 ± 0.4** |
| decode `tg64` | 46.3 ± 1.0 |

Weights resident: **4.36 GiB** on the ~7.0 GiB free budget — the `GPU-MODEL-PICK.md` fit
confirmed on real hardware. Reproduce (after `setup_cuda_wsl.sh` and a CUDA llama.cpp
build): `llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 5`.

**This is the bar — and the honest asymmetry must be stated.** fak's Go-CUDA backend
cannot yet run Qwen2.5-7B-Q4_K_M end-to-end, so there is **no fak-vs-llama tok/s at 7B in
this session, by construction**: the in-kernel loader holds the whole **f32** blob (a 7B
f32 set is ~28 GB and will not fit WSL's ~15 GB RAM), and there is no GGUF / Q4_K device
path yet. A 7B head-to-head is gated on the loaders ([#40](https://github.com/anthony-chaudhary/fleet/issues/40),
[#41](https://github.com/anthony-chaudhary/fleet/issues/41)) + quantized device GEMM
([#33](https://github.com/anthony-chaudhary/fleet/issues/33)), tracked under
[#38](https://github.com/anthony-chaudhary/fleet/issues/38). What this session delivers is
the **baseline to beat** (above) and a **witnessed modular GPU path** (§1) to grow toward
it — not a parity claim fak has not yet earned. (For the CPU axis, the existing
`LLAMACPP-HEADTOHEAD-RESULTS.md` already measures fak at-best-parity / otherwise-behind
llama.cpp on SmolLM2-135M; this GPU doc is the orthogonal GPU axis.)

## 3b. The head-to-head fak CAN run today — SmolLM2-135M, GPU, same box

The 7B is loader-gated, but fak **does** run SmolLM2-135M on the GPU today (the proven
in-kernel checkpoint), so this is a real fak-vs-`llama.cpp` head-to-head on a model that
fully fits the GPU — measured on the same RTX 4070 / WSL+CUDA, decode median over 128
steps.

| engine | precision | decode t/s | prefill t/s (P=64) |
|---|---|---:|---:|
| fak-CUDA, op-per-call (lean path) | f32 | 7.5 | 24 |
| **fak-CUDA, reusable CUDA graph** (`FAK_CUDA_GRAPH=1`) | f32 | **119–120** | **~125** |
| `llama.cpp` | Q8_0 | 120 ± 15 | — |
| `llama.cpp` | F16 | 261 ± 10 | — |

**Decode throughput parity REACHED.** With the reusable-graph path, fak-CUDA decodes at
**~120 tok/s — dead even with `llama.cpp` Q8_0 (120 tok/s)** on a model that fits the GPU,
and at higher *precision* (fak runs **f32**; llama.cpp Q8_0 is 8-bit, so fak moves 4× the
weight bytes per token and still matches it). Against llama.cpp **F16** (261, 2-byte weights)
fak f32 is ~46% — the residual is the f32-vs-f16 memory-bandwidth difference, the lever
issue #34 (fp16) addresses, not an architecture gap. Output parity also holds (greedy
argmax-exact vs the reference). **Getting here was a 16× decode speedup (7.5 → 120 tok/s).**

**Step 1 — the diagnosis (a microbench of the bare WSL CUDA floor).** fak's compute is
trivial (a 135M token is ~270 MFLOP, microseconds on a 4070); the cost was **CUDA-API
launch/submission overhead on WSL**. Timing the bare floor (N kernel launches + N cuBLAS
GEMVs, one device sync at the end):

| op (WSL, async submit) | cost |
|---|---:|
| bare kernel launch | **0.071 ms** |
| kernel launch + sync | 0.283 ms |
| cuBLAS sgemm GEMV (1×2048×2048) | **0.134 ms** |

⇒ the ~210 cuBLAS GEMVs/token alone floor an op-per-call decode at **~80 ms ≈ 12 tok/s**,
*independent of compute*. So the op-per-call backend cannot reach llama.cpp on WSL — the
fix had to collapse the ~600 host CUDA calls/token into ~one. (Cleaning the non-compute
overhead — resident weight cache, pooled allocator + recycling, async copies, single stream
— first took it 2.7 → 7.5 t/s and confirmed the residual *was* the launch count.)

**Step 2 — the dead end.** Per-token `cudaStreamBeginCapture`/`Instantiate`/`Launch` did
NOT help (7.0 t/s): re-**instantiating** a ~600-node graph every token costs ~what the 600
launches cost.

**Step 3 — the win: a REUSABLE graph (instantiate-once).** Three changes made one captured
graph replayable across the whole growing decode, so the per-token cost is one launch:
1. **`cudaGraphExecUpdate`** — keep the instantiated exec; each token capture a fresh graph
   (identical topology) and *patch* the changed params (position, nPos, KV offset) into the
   kept exec instead of recompiling.
2. **kernel-form KV-append** (`k_copyrow`, scalar offset) replacing the `cudaMemcpy` whose
   destination pointer grew every token — a moving pointer ExecUpdate could not patch, which
   was silently forcing a re-instantiate on every decode token.
3. **cross-session weight sharing** (`Upload` caches device buffers by host pointer) — without
   it every session re-uploaded the whole model and VRAM exhausted mid-bench, which had been
   masking the decode result entirely.

Result: **7.5 → 119–120 tok/s decode (16×), at parity with `llama.cpp` Q8_0.** It ships
gated `FAK_CUDA_GRAPH=1` with one honest caveat: the device KV is **fixed-capacity (1024
positions)** so capture never hits a `cudaMalloc`; lifting that to dynamic/ring is the
follow-up (device KV [#39](https://github.com/anthony-chaudhary/fleet/issues/39)), and fp16
([#34](https://github.com/anthony-chaudhary/fleet/issues/34)) is the lever to also reach the
F16 number. Native-Linux (no WSL per-call tax) would start from a far lower floor.

Reproduce: fak `FAK_CUDA_GRAPH=1 go run -tags cuda ./cmd/modelbench -dir
internal/model/.cache/smollm2-135m -backend cuda` (or `internal/compute/build_cuda.sh
bench`); llama `llama-bench -m SmolLM2-135M-Instruct-{f16,Q8_0}.gguf -ngl 99 -n 128`.

## 4. What's still open (decode parity reached; these extend it)

Decode parity with `llama.cpp` Q8_0 is **reached** on a fitting small model (§3b). These
follow-ups extend it — to the F16 number, to a 7B, and to a non-gated default — mapped to
tracked issues on `anthony-chaudhary/fleet`:

| Gap vs llama.cpp | Consequence | Issue |
|---|---|---|
| F32 compute only (no fp16 / tensor cores) | leaves the 4070's tensor cores idle; ~2–4× on the table | [#34](https://github.com/anthony-chaudhary/fleet/issues/34) |
| No quantized device GEMM (Q8_0 / Q4_K) | a 7B Q4 model can't run weights-in-int4 on device; would dequant-to-f32 (won't fit) | [#33](https://github.com/anthony-chaudhary/fleet/issues/33) |
| Naive decode attention (per-call scratch, one block/head) | no flash/paged attention; launch + memory overhead | [#32](https://github.com/anthony-chaudhary/fleet/issues/32) |
| Synchronous, one kernel launch per op | batch-1 decode is launch-bound | async [#36](https://github.com/anthony-chaudhary/fleet/issues/36), CUDA Graphs [#35](https://github.com/anthony-chaudhary/fleet/issues/35) |
| No GGUF / sharded / quant-on-load | the 7B Q4 target can't be **loaded** by the in-kernel path yet (28 GB f32 won't fit 15 GB WSL RAM) | GGUF [#41](https://github.com/anthony-chaudhary/fleet/issues/41), quant-on-load [#40](https://github.com/anthony-chaudhary/fleet/issues/40) |
| `Evict` round-trips host-ward | quarantine correct but slow on device | [#39](https://github.com/anthony-chaudhary/fleet/issues/39) |
| 7 hand-copied blocks not yet consolidated | arch-dispatch + clean HAL adoption blocked | SEAM-0 [#42](https://github.com/anthony-chaudhary/fleet/issues/42) |

The umbrella that closes the loop — a measured Go-CUDA tok/s next to the `llama.cpp`
baseline on Qwen2.5-7B-Q4_K_M, same batch-1 protocol — is
[#38](https://github.com/anthony-chaudhary/fleet/issues/38).

## 5. Bottom line

- **GPU is no longer SIMULATED.** A modular CUDA backend behind `compute.Backend` runs the
  in-kernel model's decode on the RTX 4070, witnessed argmax-exact vs the CPU reference,
  with the HAL adoption proven byte-identical on the Reference path.
- **Modularity held the line.** Adding the GPU was a new `Backend` registration + a build
  tag; the forward loop and the default pure-Go artifact were untouched, exactly as the HAL
  was designed for.
- **Decode throughput parity: REACHED.** On SmolLM2-135M (a model that fully fits the GPU),
  fak-CUDA decodes at **119–120 tok/s — even with `llama.cpp` Q8_0 (120)** — and at higher
  precision (fak f32 vs llama Q8_0), with output (greedy) parity too. That is a **16× jump
  (7.5 → 120 tok/s)** this session, via a reusable CUDA graph: `cudaGraphExecUpdate`
  (instantiate-once), a kernel-form positioned KV-append (so the graph stays patchable as
  the cache grows), and cross-session weight sharing (so VRAM stops exhausting). The
  diagnosis that drove it was a microbench proving the op-per-call path is launch-bound on
  WSL (~80 ms/token floor) — the kernel-program boundary, now crossed. vs llama.cpp **F16**
  (261) fak f32 is ~46%, an f32-vs-f16 bandwidth gap (#34), not architecture.
- **Honest caveat:** the parity path is gated `FAK_CUDA_GRAPH=1` and uses a fixed-capacity
  device KV (1024 positions, so capture never hits a `cudaMalloc`); making it dynamic and
  default-on is the follow-up (#39). 7B-on-GPU is still loader-gated (#40/#41). Nothing here
  is claimed beyond what the committed `modelbench` + `llama-bench` numbers show.
