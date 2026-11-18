---
title: "Level 4 — ROCm: how it actually works"
description: "AMD's native compute stack explained: HIP's CUDA-compatible reuse path, the shipped GCN/CDNA/RDNA device taxonomy, and why 64-lane vs 32-lane wavefronts decide your tuning."
slug: accelerators-level-4
keywords:
  - rocm explained
  - hip programming
  - amd gpu compute
  - cdna vs rdna
  - wavefront 64 vs 32
  - hipify cuda to hip
  - fak accelerator ladder
date: 2026-11-18
---

# Level 4 — ROCm: how it actually works

*You are on **Level 4 of 5** of the [fak accelerator ladder](README.md).*

> **Audience.** A systems engineer who knows GPUs and has read [Level 3 — Vulkan](level-3-vulkan-how-it-works.md).
> You know what a workgroup, a dispatch and a shared-memory allocation are. This rung adds
> AMD's *native* stack on top of that: what ROCm is, how HIP relates to CUDA, and what fak's
> ROCm backend actually consists of today.

> **Short answer.** ROCm is AMD's own compute platform — a runtime, a compiler toolchain and
> libraries for running compute on AMD GPUs. Its programming language, **HIP**, is deliberately
> source-compatible with CUDA, so fak's plan is not to hand-write a second kernel set but to
> compile the *existing* `cuda_kernels.cu` through `hipcc`/`hipify`. What is shipped and tested
> today is the hardware-independent half: a device-architecture taxonomy that classifies every
> supported AMD target and, crucially, records whether it runs a **64-lane** or **32-lane**
> wavefront. Everything that needs silicon is deferred and named.

> **Status: scaffold, not silicon.** fak's ROCm backend is a **scaffold with a named blocker**.
> The device-architecture taxonomy is shipped and unit-witnessed on any host; the HIP kernel
> compilation and **every acceptance bullet that runs on silicon are deferred to an
> AMD-GPU-on-Linux node** — the dev host has no AMD GPU and no `hipcc`. **No ROCm throughput or
> correctness number was run, estimated, or fabricated.** The source of truth is
> `internal/compute/ROCM-C002-NOTES.md`; read it in full.

## What is ROCm, and how is it different from Vulkan?

ROCm is AMD's open compute platform: a runtime, a compiler toolchain and a set of libraries for
running compute on AMD GPUs. Its GPU programming language is **HIP**, and HIP is deliberately
**source-compatible with CUDA by design** — `hipMalloc` maps to `cudaMalloc`, `hipMemcpy` to
`cudaMemcpy`, and so on. The point of that compatibility is mechanical: a CUDA kernel can be
*translated* to HIP rather than rewritten, and AMD ships a tool, `hipify`, that does exactly
that translation. `internal/compute/ROCM-C002-NOTES.md:26` states the design intent directly:
HIP is source-compatible with CUDA by design, and AMD ships `hipify` to translate CUDA to HIP.

Contrast that with [Vulkan](level-3-vulkan-how-it-works.md). Vulkan is a **cross-vendor
standard** you reach through a driver: the same API works on AMD, NVIDIA and Intel, and the
driver is the thing that knows the hardware. ROCm is AMD's **own native stack**. Because it is
not obliged to be vendor-neutral, it can expose AMD-specific hardware features — matrix cores,
the exact wavefront width, LDS bank geometry — that a cross-vendor API cannot surface. That is
the whole trade: Vulkan buys portability, ROCm buys reach into the silicon.

## The HIP-reuse strategy — the key architectural decision

fak's plan, recorded in `internal/compute/ROCM-C002-NOTES.md`, is **not** to hand-write a second
kernel set. It is to compile the **existing** `cuda_kernels.cu` through `hipcc`/`hipify` — the
same path llama.cpp's ROCm backend takes. One kernel source, two vendors.

```
   internal/compute/cuda_kernels.cu        (the ONE kernel source)
              |
              |  hipify  (CUDA -> HIP source translation)
              v
   internal/compute/rocm_kernels.hip       (mechanically translated)
              |
              |  hipcc --offload-arch=<gfx>   (gfx from ROCmOffloadArch)
              v
   internal/compute/libfakrocm.a           (static device library)
              |
              |  cgo link  (-lfakrocm -lamdhip64)
              v
   internal/compute/rocm_backend.go        //go:build rocm
              |
              |  compute.Register(...)   (the HAL seam, unchanged)
              v
   Approx backend named "rocm"   <-- selected only via --backend rocm
```

Three pieces make up the deferred backend (`internal/compute/ROCM-C002-NOTES.md:31`):

1. **`internal/compute/rocm_backend.go`** — `//go:build rocm`, a near-clone of `cuda.go`'s cgo
   wrapper that registers an `Approx` backend named `"rocm"` over a flat C ABI
   (`rocm_backend.h`), keeping `cpu-ref` the `Reference` default so nothing runs on the GPU
   unless explicitly selected (`FAK_BACKEND=rocm` / `--backend rocm`). (The source note calls
   this file `rocm.go` at `internal/compute/ROCM-C002-NOTES.md:31`; the shipped file is
   `rocm_backend.go`, which is what the diagram above and this line name.) The host already
   reports an unbuilt backend honestly: `cmd/fak/serve_model_load.go:346` says a device backend
   "needs both a matching build tag (e.g. `-tags rocm`) and a reachable device at runtime".
2. **`build_rocm.sh`** — the `build_cuda.sh` twin: `hipify`/`hipcc` over `cuda_kernels.cu` →
   `libfakrocm.a`, with `--offload-arch` taken from `ROCmOffloadArch(<device gfx>)`. (The
   source note spells this `librocfak.a` at `internal/compute/ROCM-C002-NOTES.md:37`; the
   tree's convention is `libfak<backend>.a` — `libfakcuda.a`, `libfakvulkan.a` — so the
   diagram above and this line use `libfakrocm.a`.)
3. **The witnesses** — op-level and full-forward tests (`-tags rocm`) mirroring `cuda_test.go`
   and `vulkan_test.go`.

**The payoff:** one kernel source, two vendors. **The cost:** you inherit CUDA's assumptions
wholesale, and you depend on `hipify`'s fidelity — a translation that is *almost* right is a
silent numerical drift, not a compile error.

## The device-architecture taxonomy — the part that IS shipped

This is the page's technical centrepiece, and it is real, tested code.
`internal/compute/rocm_arch.go` is **pure Go with no build tag**, so it compiles and is
unit-tested on any host. That is how a hardware-dependent backend still ships a host-witnessed
half: the file's own doc comment calls it "the always-compiled, hardware-independent half of the
ROCm (AMD Linux) backend". The same split the prefill work used — ship the exact, host-tractable
part; defer the device run.

`ROCmFamily` (`internal/compute/rocm_arch.go:30`) classifies every supported AMD target:

- `ROCmGCN5` — gfx906, Instinct MI50/MI60 (Vega 20)
- `ROCmCDNA1` — gfx908, Instinct MI100
- `ROCmCDNA2` — gfx90a, Instinct MI200
- `ROCmCDNA3` — gfx942, Instinct MI300A/MI300X
- `ROCmRDNA1` — gfx101x, Radeon RX 5000
- `ROCmRDNA2` — gfx103x, Radeon RX 6000
- `ROCmRDNA3` — gfx110x, Radeon RX 7000
- `ROCmRDNA3_5` — gfx1151, Strix Halo / Ryzen AI Max+ 395 APU

The `rocmArches` table at `internal/compute/rocm_arch.go:105` is the supported-target list, each
row carrying `GFX`, `Family`, `Wavefront` and `Examples`. It is the single most useful artifact
here, so it is reproduced verbatim:

| GFX | Family | Wavefront | Examples |
|---|---|---|---|
| `gfx906` | GCN5 | 64 | Instinct MI50/MI60 (Vega 20) |
| `gfx908` | CDNA1 | 64 | Instinct MI100 |
| `gfx90a` | CDNA2 | 64 | Instinct MI210/MI250/MI250X |
| `gfx942` | CDNA3 | 64 | Instinct MI300A/MI300X |
| `gfx1010` | RDNA1 | 32 | Radeon RX 5700 (XT) |
| `gfx1030` | RDNA2 | 32 | Radeon RX 6800/6900 |
| `gfx1032` | RDNA2 | 32 | Radeon RX 6600 (XT) |
| `gfx1100` | RDNA3 | 32 | Radeon RX 7900 XTX/XT |
| `gfx1102` | RDNA3 | 32 | Radeon RX 7600 (Vulkan-witnessed) |
| `gfx1151` | RDNA3.5 | 32 | Ryzen AI Max+ 395 (Strix Halo APU, 40 CUs) |

The `ROCmArch` struct itself (`internal/compute/rocm_arch.go:96`) is small; the `Wavefront` field
at `internal/compute/rocm_arch.go:99` is the load-bearing one:

```go
// ROCmArch is one supported AMD compile target: its canonical LLVM AMDGPU id, its family,
// the native wavefront width hipcc tunes for, and a representative product.
type ROCmArch struct {
	GFX       string     // canonical `--offload-arch` token, e.g. "gfx90a"
	Family    ROCmFamily // generation
	Wavefront int        // native wavefront lanes: 64 on GCN/CDNA, 32 on RDNA
	Examples  string     // representative product(s)
}
```

`Datacenter()` (`internal/compute/rocm_arch.go:92`) is the predicate separating the Instinct
line from the Radeon line — `GCN5` plus every `CDNA` generation. The multi-GPU acceptance bullet
is a datacenter concern, so this is the predicate a fleet planner keys on. The rest of the
surface is `LookupROCmArch` (`:154`), `ROCmOffloadArch` (`:162`), `KnownROCmArches` (`:173`) and
`CompilerFlags` (`:182`).

### The load-bearing distinction: wavefront width

A **wavefront** is the group of lanes that execute one instruction together — AMD's analogue of
NVIDIA's warp. Sixty-four lanes all execute the same instruction at the same time; that is the
machine's unit of work, and everything you size is sized in those units. GCN and CDNA parts
execute a **64-lane wavefront**; RDNA parts execute a native **32-lane wavefront**. The file's
own doc comment is blunt about the consequence: picking the wrong family "silently mistunes
occupancy and LDS".

Here is the same 128-thread workgroup mapped onto each machine:

```
   64-lane wavefront (GCN / CDNA)          32-lane wavefront (RDNA)
   ------------------------------          ------------------------------
   workgroup = 128 threads                 workgroup = 128 threads
   = 2 wavefronts of 64 lanes              = 4 wavefronts of 32 lanes

   wave 0: lanes  0..63  ##########        wave 0: lanes  0..31  #####
   wave 1: lanes 64..127 ##########        wave 1: lanes 32..63  #####
                                           wave 2: lanes 64..95  #####
                                           wave 3: lanes 96..127 #####

   occupancy: 2 waves resident             occupancy: 4 waves resident
   LDS: sized for 64-wide rows             LDS: sized for 32-wide rows
```

Why it changes your sizing: a workgroup of 128 threads is *two* waves on CDNA but *four* on
RDNA. If you sized a shared-memory tile assuming 64 lanes per row, on a 32-lane part you either
waste half the LDS or, worse, index it with the wrong stride and get bank conflicts that look
like a mysterious slowdown. And a **silent mistune is worse than a crash**: a crash tells you
you are wrong; a mistune just runs at half the occupancy you designed for, and every number you
measure afterwards is quietly wrong.

```
   AMD target family tree
   ======================

   GCN5  gfx906  MI50/MI60 ......... 64-lane  \  Datacenter() == true
   CDNA1 gfx908  MI100 ............. 64-lane   |  (Instinct line,
   CDNA2 gfx90a  MI200 ............. 64-lane   |   matrix cores)
   CDNA3 gfx942  MI300A/MI300X ..... 64-lane  /

   RDNA1 gfx101x RX 5000 ........... 32-lane  \  Datacenter() == false
   RDNA2 gfx103x RX 6000 ........... 32-lane   |  (Radeon consumer line)
   RDNA3 gfx110x RX 7000 ........... 32-lane   |
   RDNA3.5 gfx1151 Strix Halo ...... 32-lane  /
```

## `ROCmOffloadArch` — normalizing noisy device strings

`internal/compute/rocm_arch.go:162`. ROCm reports targets with case noise and optional feature
suffixes — `"GFX90A"`, `"gfx90a:sramecc+:xnack-"`, `"gfx1100  "` — that all denote the same
compile target. `normalizeGFX` (`internal/compute/rocm_arch.go:132`) lowercases, trims
whitespace anywhere, and drops the feature suffix at the first `:`. Then `ROCmOffloadArch`
resolves the normalized id against the table and returns the canonical token, or `("", false)`.

The important half is the failure mode: it **fails closed on an unsupported part** rather than
inventing a target. `LookupROCmArch` (`internal/compute/rocm_arch.go:154`) is described in its
own comment as "the fail-closed admission a build/runtime path uses so an unknown AMD part is
never silently compiled for the wrong wavefront". A wrong `--offload-arch` produces a binary
that either will not load at all, or — worse — loads and runs **mistuned**, because the compiler
tuned for a wavefront width the silicon does not have. Failing closed turns a silent mistune
into a loud "unsupported part".

## The build recipe

From `internal/compute/ROCM.md:7`: install a matching HIP compiler/runtime for the target GPU,
and note the file's own warning that **`hipcc` and the runtime libraries must come from the same
ROCm installation**. The real commands, from the repository root:

```sh
export ROCM_PATH=/opt/rocm
"$ROCM_PATH/bin/hipcc" --version
"$ROCM_PATH/bin/rocm_agent_enumerator"
"$ROCM_PATH/bin/hipcc" -O3 -std=c++17 -fPIC --offload-arch=gfx1151 \
  -c internal/compute/rocm_kernels.hip -o internal/compute/rocm_kernels.o
ar rcs internal/compute/libfakrocm.a internal/compute/rocm_kernels.o
export CGO_ENABLED=1
export CGO_LDFLAGS="-L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib"
export LIBRARY_PATH="$ROCM_PATH/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
go build -tags rocm -o /tmp/fak-rocm ./cmd/fak
```

The `--offload-arch=gfx1151` token is exactly what `ROCmOffloadArch` produces from a device
string. Rebuild the static library whenever its HIP source or header changes.

## The execution scope — what the backend is designed to cover

From `internal/compute/ROCM.md:38`, the core implementation provides: resident F32 tensors;
packed Q8_0/Q4_K/Q5_K/Q6_K projection weights; matmul and batched matmul; RMSNorm; RoPE; SwiGLU;
residual/bias addition; grouped attention; device argmax; and resident KV append/clone/eviction.
Unsupported storage formats or geometries **fail explicitly** — no silent fallback.

The honest boundary is stated in the same file: this core slice targets standard transformer
forwards. Hybrid Qwen GDN, automatic backend priority, multi-GPU collectives and comparative
performance qualification each require their own implementation and witnesses. The existing model
admission gate rejects a hybrid model whose selected backend lacks the whole-operation GDN
capability.

## The single-stream design

`internal/compute/rocm_backend.go:22` declares `var rocmMu sync.Mutex`, and the file's comment
says it "serializes the first ROCm backend's single HIP stream and device". The backend selects
visible device zero; `HIP_VISIBLE_DEVICES` can constrain which physical device that represents
(`internal/compute/ROCM.md:25`).

Compare this to Vulkan's `vulkanMu` (`internal/compute/vulkan.go:69`) — same shape, same
tradeoff. A serialized single stream costs you overlap: no compute/transfer overlap, no
multi-stream pipelining, one dispatch at a time. It is the right first cut because it makes the
correctness question answerable before the performance question: with one stream there is no
interleaving to reason about, so a numerical mismatch is a kernel bug, not a synchronization bug.

## The witness, and why it is not yet green

The real command, from `internal/compute/ROCM.md:30`:

```sh
FAK_ROCM_REQUIRE_DEVICE=1 go test -tags rocm ./internal/compute ./internal/model \
  -run 'Test(ROCmBackend|HALROCm)' -count=1 -v
```

The design point is in the environment variable. `FAK_ROCM_REQUIRE_DEVICE=1` makes missing
registration a test **failure**, so a green run cannot be a silently-skipped run. A default-build
host test, a successful compilation, or a skipped device test does not establish hardware
correctness.

**And this command has not been run on silicon.** The dev host has no AMD GPU and no `hipcc`.
The four acceptance bullets that remain unwitnessed are, verbatim from
`internal/compute/ROCM-C002-NOTES.md:44`:

- [ ] Run on AMD GPU on Linux
- [ ] Within 1.5× Vulkan throughput
- [ ] Bit-exact results
- [ ] Multi-GPU support

The 1.5× rule is worth explaining. The benchmark gate needs no new machinery: it reuses the
existing `WithinTarget(measured, baseline, factor)` helper (`internal/compute/prefill.go:281`),
so the device node would feed `WithinTarget(rocmSeconds, vulkanSeconds, 1.5)`
(`internal/compute/ROCM-C002-NOTES.md:19`). The ROCm backend is expected to beat Vulkan's
per-dispatch tax — Vulkan-native is roughly 2 ms/op against CUDA-on-WSL's roughly 0.07 ms/op —
and 1.5× is the bar it must clear to justify the native path.

## The cross-vendor AMD path already works

Here is the genuinely interesting fact. The **Vulkan** backend already executes the full forward
pass on a real AMD Radeon RX 7600 (gfx1102 / RDNA3) with numerical parity — argmax-exact greedy
decode, prefill-logit cosine 1.0. See `docs/benchmarks/VULKAN-AMD-RESULTS.md`. So on AMD hardware
today, **Vulkan is the working path and ROCm is the native path being built to beat it**. That
is the exact throughput baseline this ROCm backend benchmarks against
(`internal/compute/ROCM-C002-NOTES.md:54`). Level 5 takes that head-to-head apart.

## Checkpoint

**1. What is HIP, and why does its CUDA compatibility matter to fak?**
HIP is AMD's GPU programming language, deliberately source-compatible with CUDA (`hipMalloc` ↔
`cudaMalloc`). It matters because fak can compile its *existing* `cuda_kernels.cu` through
`hipify`/`hipcc` instead of hand-writing a second kernel set — one kernel source, two vendors.

**2. What is a wavefront, and why is 64 vs 32 load-bearing?**
A wavefront is the group of lanes that execute one instruction together — AMD's analogue of
NVIDIA's warp. GCN/CDNA parts run 64-lane wavefronts; RDNA parts run native 32-lane wavefronts.
It is load-bearing because it changes how you size a workgroup and lay out shared memory, and
picking the wrong family "silently mistunes occupancy and LDS".

**3. Why is a silent mistune worse than a crash?**
A crash tells you that you are wrong. A mistune runs at the wrong occupancy and produces numbers
that are quietly wrong — every measurement afterwards is suspect, with no error to point at it.

**4. What does `ROCmOffloadArch` do with `"gfx90a:sramecc+:xnack-"`?**
It lowercases, trims, and drops the feature suffix at the first `:`, yielding `gfx90a`, then
resolves it against the table. On an unsupported part it fails closed — `("", false)` — rather
than inventing a target.

**5. Why does the witness set `FAK_ROCM_REQUIRE_DEVICE=1`?**
So that a missing device registration is a test *failure* rather than a skip. A green run then
cannot be a silently-skipped run.

**6. Has any ROCm number been measured?**
No. No ROCm throughput or correctness number was run, estimated, or fabricated. The four
acceptance bullets that run on silicon are deferred to an AMD-GPU-on-Linux node.

## See also

- [The accelerator ladder (README)](README.md) — the index for all five rungs
- [Level 2 — the HAL seam](level-2-the-hal-seam.md) — how a backend registers at all
- [Level 3 — Vulkan: how it works](level-3-vulkan-how-it-works.md) — the cross-vendor path
- [Level 5 — Vulkan vs ROCm](level-5-vulkan-vs-rocm.md) — the head-to-head
- [Hardware portability](../hardware-portability.md) — the wider portability story
- [Vulkan AMD results](../../benchmarks/VULKAN-AMD-RESULTS.md) — the RX 7600 parity witness

*[Full ladder (README)](README.md) · prev ← [Level 3 — Vulkan](level-3-vulkan-how-it-works.md) · next → [Level 5 — Vulkan vs ROCm](level-5-vulkan-vs-rocm.md)*
