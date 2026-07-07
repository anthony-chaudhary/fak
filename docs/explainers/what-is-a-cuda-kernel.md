---
title: "What is a CUDA kernel? (And why \"kernel\" means three things in an agent-kernel repo)"
description: "A from-scratch disambiguation of the word every reader trips on: the OS-metaphor agent kernel (fak's reference monitor), the HPC compute kernel (which arithmetic a matmul runs through), and the literal CUDA kernel (__global__ code executing on the GPU) — what a .cu file actually looks like, the depth ladder from header file to silicon, and the four-part test for when going deeper is worth it."
slug: what-is-a-cuda-kernel
keywords:
  - what is a CUDA kernel
  - CUDA kernel vs OS kernel
  - compute kernel meaning
  - .cu file explained
  - __global__ function
  - PTX vs SASS
  - tensor cores vs CUDA cores
  - agent kernel
  - fak kernel disambiguation
date: 2026-07-08
---

# What is a CUDA kernel? (And why "kernel" means three things here)

fak calls itself an **agent kernel**, ships a Go file named `kernel.go`, *and*
contains real GPU kernels in a `.cu` file. Three unrelated machines share one
word, and every newcomer eventually asks some form of "wait — are our kernels
*real* kernels?" This page pins each sense to the file that embodies it, shows
what a CUDA kernel literally is, and maps the layers below it so you know when
"going closer to the hardware" is a real lever and when it is a category error.

*Who this is for:* anyone reading fak docs or code who has hit the word
"kernel" in two places that cannot possibly mean the same thing. No GPU
background assumed.

## One word, three machines

| Sense | What it is | Where it lives here | Runs on |
|---|---|---|---|
| **OS-metaphor kernel** — fak as reference monitor | The trust boundary that adjudicates every tool call: the model proposes, the kernel disposes. Borrowed from operating systems, where the kernel is the code user programs cannot bypass. | `internal/adjudicator`, `internal/kernel` (the verdict fold); the whole [tool-call-is-a-syscall](tool-call-is-a-syscall.md) story | Plain CPU, pure Go. **No GPU involved.** |
| **Compute kernel** — the HPC sense | The inner-loop math routine an operation dispatches to; "which arithmetic does this matmul run through?" | `matKernel` in `internal/model/kernel.go` — the f32 vs Q8 arithmetic paths share one block skeleton | Host CPU (Go), orchestrating the forward pass |
| **CUDA kernel** — the literal NVIDIA sense | A `__global__` function launched onto the GPU, executed by thousands of threads at once | `internal/compute/cuda_kernels.cu` — 21 of them (`k_q8_gemm`, `k_flash_attention`, `k_rmsnorm`, `k_rope`, `k_argmax`, …) | The GPU itself |

The three never overlap: no GPU code participates in a tool-call verdict, and
the reference monitor never touches the forward pass. When a fak doc says
"in-kernel", context decides which machine it means — the glossary's
[kernel-cluster section](../glossary.md) keeps the adjacent vocabulary
(vDSO, MMU, adjudicator, rung) from blurring the same way.

## What a CUDA kernel file actually looks like

There is no `.mcu` — the extensions are **`.cu`** (CUDA C++ source) and
**`.cuh`** (CUDA headers). A `.cu` file is ordinary C++ plus a few extensions,
holding two kinds of code.

**Device code** — the kernels themselves, marked `__global__`. Here is fak's
int8 matmul kernel, trimmed, from `internal/compute/cuda_kernels.cu`:

```cuda
__global__ void k_q8_gemm(const signed char *W, const float *Wscale,
                          const signed char *qX, const float *xScale,
                          float *Y, int out, int in, int P, int block) {
  int o = blockIdx.x, t = blockIdx.y;   // which output row / token THIS block owns
  __shared__ float red[256];            // on-chip scratch shared by the block's threads
  float local = 0.f;
  for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
    int acc = 0;
    for (int i = 0; i < block; i++)
      acc += (int)wb[i] * (int)xb[i];   // scalar int8 dot product
    local += (float)acc * wsc[b] * xsc[b];
  }
  red[threadIdx.x] = local;
  __syncthreads();                      // barrier, then tree-reduce across the block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { /* ... */ }
  if (threadIdx.x == 0) Y[t * out + o] = red[0];
}
```

What makes it a kernel rather than a function: it is written once but executed
by a **grid** of thread blocks simultaneously, and each of the thousands of
concurrent instances uses `blockIdx`/`threadIdx` coordinates to figure out
which slice of the problem it owns. `__shared__` memory and `__syncthreads()`
let the threads in one block cooperate.

**Host code** — plain C++ in the same file that launches kernels with the
triple-chevron syntax:

```cuda
k_q8_gemm<<<dim3(out, P), 256, 0, g_stream>>>(...);  // out×P blocks, 256 threads each
```

Most of the `.cu` file is this host-side plumbing: memory pools, streams,
cuBLAS handles, CUDA-graph capture. The build never touches the default Go
toolchain — `nvcc` compiles the `.cu` offline into a static library that the
cgo wrapper links only under `-tags cuda`.

## The header is a contract, not a shallower kernel

A common misreading: that the `.h` file is a "level" above the `.cu`, so going
below the header is exotic. It isn't. The header
(`internal/compute/cuda_backend.h`) is deliberately a flat C ABI with **zero
compute in it** — function signatures only, kept parseable without a CUDA
toolchain. Going "below the .h" is just writing the implementation. The real
depth ladder:

| Layer | In this repo | Do you write it? |
|---|---|---|
| Go model code | `internal/model` (`matKernel` picks the arithmetic path) | yes |
| cgo + flat C ABI | `cuda.go` + `cuda_backend.h` | yes — the trust/build boundary |
| Host CUDA C++ (launches, streams, graphs) | the `fcuda_*` functions in the `.cu` | yes |
| Device CUDA C++ (`__global__`) | the 21 kernels | yes — **this is where performance lives** |
| Intrinsics / inline PTX (`dp4a`, `mma.sync`, `cp.async`) | not yet — a costed lever | only when chasing tensor cores by hand |
| PTX (virtual GPU assembly) | emitted by `nvcc` | almost never by hand |
| SASS (real machine code per `sm_XX`) | inspected with `cuobjdump` / Nsight | never written; only read |
| Silicon: CUDA cores, tensor cores, per-generation units | `sm_80`-class vs `sm_90`-class GPUs | — |

Three facts from the bottom of that ladder shape real decisions:

1. **A real kernel is not automatically a fast kernel.** `k_q8_gemm`'s inner
   loop (`acc += wb[i] * xb[i]`) is scalar int32 work on the plain CUDA cores —
   it never engages the tensor cores (the dedicated matrix-multiply units).
   That is deliberate: fak's CUDA lane is a first-generation **Approx** peer of
   the CPU reference, gated on correctness (argmax-exact decode, logit cosine
   against `cpu-ref`), not on throughput. Rewriting that loop as int8
   tensor-core MMA is a costed lever — days of kernel work, estimated (not
   measured) up to ~2× — and it only pays once batching creates a
   compute-bound region.
2. **Instruction floors are hardware facts, not software ones.** Fused kernels
   published for `sm_90`-class GPUs use instructions that simply do not exist
   on an `sm_80`-class datacenter GPU. No amount of `.cu` authoring adds a
   missing instruction; the honest move on older silicon is to measure the
   prize before anyone writes a kernel.
3. **You can reach tensor cores without writing any of this.** One flag
   (`fcuda_set_tf32`) reroutes fak's f32 GEMMs through tensor cores, because
   those GEMMs delegate to cuBLAS — NVIDIA's library of pre-written kernels.
   Libraries (cuBLAS, CUTLASS, Triton) are the normal road to the deep layers;
   hand-written MMA is the last resort.

## When is going deeper worth it?

fak's optimization costing already encodes the answer: **S** = a config flag,
**M** = a script or sweep with *no kernel code*, **L** = new kernel code
(days). Descend the ladder only when all four hold:

1. **You are on the GPU serving lane at all.** For the gateway, dispatch,
   guard, or DOS work, "kernel" is the reference-monitor metaphor — hardware
   depth is a category error there.
2. **The op is compute-bound.** Single-stream decode is memory-bandwidth-bound;
   tensor cores idle waiting on VRAM. Batching or prefill is where the fancy
   math units earn their keep.
3. **No flag or library gets you there first.** Exhaust the S and M levers
   (TF32 flag, quantization sweep, batching) before writing L-cost kernel code.
4. **The hardware has the instruction, and a measurement says the prize
   exists.** Size the win before building the kernel.

## Common questions

- **Are fak's kernels real CUDA kernels?** Yes — literal `__global__`
  functions launched on the GPU. They are also deliberately simple,
  correctness-gated first-generation kernels, not tuned-engine competitors;
  raw tokens/sec is the job of vLLM/SGLang/llama.cpp, which fak fronts
  (see [what fak is not](what-fak-is-not.md)).
- **Is fak-the-agent-kernel a GPU kernel?** No. That sense is the OS metaphor:
  the reference monitor that adjudicates tool calls, in pure Go, on the CPU.
- **What file extension do CUDA kernels use?** `.cu` for source, `.cuh` for
  CUDA headers. (There is no `.mcu`.)
- **Do I ever write PTX or SASS?** Almost never. You write CUDA C++ and
  occasionally intrinsics; you *read* PTX/SASS through profiling tools to
  understand what the compiler and hardware did.
- **How do I use tensor cores without writing kernel code?** Route the op
  through a library that already does (cuBLAS via the TF32 flag here), or use
  a quantized path whose library kernels carry MMA. Writing `mma.sync` by hand
  is the last rung, not the first.

## Related pages

- [Hardware portability via the compute HAL](hardware-portability.md) — how the
  CUDA backend registers behind `internal/compute` instead of forking the
  forward pass.
- [The cross-platform spine](cross-platform-spine.md) — the deployment axis the
  hardware-depth axis sits beside.
- [The canonical glossary](../glossary.md) — the wider disambiguation table
  (agent, model, steering, and the kernel-adjacent cluster: vDSO, MMU, rung).
- [Model/compute engine env knobs](../model-engine-env.md) — the `FAK_*`
  variables that steer the in-kernel engine, including the GPU build vars.
- [What fak is not](what-fak-is-not.md) — the honest serving boundary.
