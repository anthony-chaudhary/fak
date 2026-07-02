---
title: "Backend SOTA matrix: where fak should borrow, bind, or stay minimal"
description: "The inward engineering map issue #905 asked for. For each compute operation fak's kernel actually performs, this records the current fak path (verified against the tree), the production kernel stack to learn from, the candidate integration route (borrow / bind / stay minimal), the verification oracle, and the expected bottleneck. It then selects one native-engine optimization with a falsifiable acceptance target and an oracle witness, grounds benchmark lineage in the existing authority docs, and records the issue-update text an authed operator posts. Honest about the one gate this environment cannot reach."
slug: backend-sota-matrix
keywords:
  - backend matrix
  - CUTLASS
  - FlashInfer
  - TensorRT-LLM
  - Marlin
  - AWQ
  - GPTQ
  - GGUF
  - verification oracle
---

# Backend SOTA matrix — learn the production stacks before the next native push

This is the deliverable issue [#905](https://github.com/anthony-chaudhary/fak/issues/905) asked
for. It is the **inward** engineering counterpart of
[`docs/industry-scorecard/`](../industry-scorecard/README.md): the scorecard positions fak
against the field (an *outward* competitive map); this page maps each operation fak's kernel
**actually performs** to the production stack worth learning from, the route to integrate it,
and the oracle that would prove it. Every fak path below was read out of the tree on
2026-06-26, not assumed.

> **Read this right.** This is a research artifact. It carries analysis and a *selection*,
> not a shipped capability. The one optimization it selects (closing the AWQ CUDA witness gap)
> is recorded here as a falsifiable target with an oracle; it is **not** implemented by this
> note, and no `CLAIMS.md` row is added for it. "Selected, not shipped" is the honest status.

## The matrix

Columns are the issue's implementation-ladder #1: operation, current fak path, SOTA reference
(official docs or primary repo), candidate integration route, verification oracle, expected
bottleneck.

| Operation | Current fak path (verified) | SOTA stack to learn | Route | Oracle | Expected bottleneck |
|---|---|---|---|---|---|
| Dense f32 GEMM | `internal/compute/cpuref.go` (Reference, max\|Δ\|=0) + parallel/`fdot` in `internal/model/parallel.go`, `fdot_amd64.s` | [cuBLAS](https://docs.nvidia.com/cuda/cublas/) / [CUTLASS](https://docs.nvidia.com/cutlass/) GEMM | **Stay minimal.** fak's value is the bit-exact contract, not beating cuBLAS at GEMM. | `cpuref` f32 (bit-identity) | reduction order (only matters once device-reordered) |
| Dense fp16 / Q8_0 / Q4_K device GEMM | `internal/compute/cuda.go` + `cuda_kernels.cu` (`k_q8_gemm`, `k_q4k_gemm`, `cublasGemmEx` fp16). Floors: `cudaFP16CosineMin=0.997`, `cudaQ8CosineMin=0.999`, `cudaQ4KCosineMin=0.995` | cuBLAS (fp16/tensor-core); [llama.cpp](https://github.com/ggml-org/llama.cpp) Q8_0/Q4_K fused dequant-GEMM | **Bind** (cuBLAS fp16 already bound); **borrow** the fused tile-dequant pattern for the int lanes | `cpuref` f32, argmax-exact + cosine ≥ floor | decode = weight memory BW; prefill = tensor-core occupancy |
| **AWQ 4-bit GEMV/GEMM** | `internal/compute/cuda.go:1101` (`AWQMatMul`/`AWQBatchedMatMul`) + `cuda_kernels.cu:884` (`k_awq_gemv`/`k_awq_gemm`/`k_awq_dequant_row`) + `internal/model/awq.go`, `awq_cuda.go` (`-tags "cuda awqcuda"`) | [Marlin](https://github.com/IST-DASLab/marlin) mixed-precision INT4 GPTQ kernel; AutoAWQ | **Borrow** (Marlin's fused dequant-MMA), **after** the witness exists | `cpuref` f32 + HF AWQ reference (greedy-continuation + cosine) | dequant fusion / weight BW — **and the witness gap (see Selection)** |
| GPTQ resident (CPU) | `internal/model/gptq.go` (`LoadGPTQ`, resident 4/8-bit, `g_idx`) — `[SHIPPED]` CPU-resident, honest-fenced (no native GPU kernel) | AutoGPTQ / [Marlin](https://github.com/IST-DASLab/marlin) | **Bind later**: a Marlin-style GPU kernel is the GPU rung; CPU-resident parity is the floor | HF GPTQ dequant / `cpuref` | CPU-bound until a device kernel lands |
| EXL2 | `internal/model/exl2.go` (loader) | [ExLlamaV2](https://github.com/turboderp-org/exllamav2) | **Stay minimal** on loader parity; no kernel claim | ExLlamaV2 reference generation | loader completeness, not kernel speed |
| GGUF quant-at-load (Q4_K/Q5_K/Q6_K resident) | `internal/ggufload/quant_q4k_loader.go`, `gguf_glm_tensors.go`; `internal/model/quant_q4k.go`, `qwen35_prefill_q4k.go` | [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF quant + backend | **Bind** to the GGUF format; fak's resident-q4k is the parity target | llama.cpp oracle (2-token parity documented at `BENCHMARK-AUTHORITY` Qwen3.6-27B row) | prefill throughput (0.01× on 27B) — the compute, not the load |
| MoE expert dispatch | `internal/model/moe.go`, `glm_dsa.go`, `moe_offload.go` | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) / DeepEP expert-parallel | **Borrow** grouped-decode cleanup; fak's value is the contract around it | dense reference / HF | expert weight load BW |
| Fused attention (MHA/GQA/MQA) | `internal/compute/cuda_kernels.cu` `k_flash_attention` (#486, `cudaFlashAttnCosineMin=0.999`) | [FlashInfer](https://docs.flashinfer.ai/) / FlashAttention | **Stay minimal** (own fused online-softmax ships); consider FlashInfer only for paged/variable-length if pursued | `cpuref` (cosine ≥ 0.999) | KV BW / context length |
| GLM sparse attention (DSA) | `internal/compute/dsa.go`, `dsa_index.go` (`cudaDsaSparseAttnCosineMin=0.999`) | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) custom sparse | **Stay minimal** (host-index + device-sparse ships) | `cpuref` | host-side index-selection roundtrip |
| KV cache (paging / prefix reuse) | `internal/model/kv.go`, `paging.go`, RadixAttention (bit-exact eviction) | [vLLM PagedAttention](https://docs.vllm.ai); [FlashInfer](https://docs.flashinfer.ai/) | **Stay minimal.** fak's differentiator is exact eviction/reuse on an owned f32 cache, not paged throughput | bit-identity (`max\|Δ\|=0`) | fragmentation is deliberately not pursued |
| Metal Q4_K GEMM | `internal/metalgemm/`; `internal/model/metal_q4k*.go`, `metal_prefill.go` | [llama.cpp Metal](https://github.com/ggml-org/llama.cpp) / MLX | **Borrow** one-command-buffer resident-decode fusion | `cpuref` (GEMV cosine 1.000000) | per-token command-buffer launch (~336 GEMVs @ ~360 µs each) — epics #59/#67 |

## Selection — the one near-term win (issue Done-when #2)

**Close the AWQ 4-bit CUDA witness gap.** It is the one device operation family in the tree
that ships a binding and a kernel but has **no recorded cosine floor and no acceptance
witness**. Every sibling lane has all three pieces:

- fp16: floor `cudaFP16CosineMin=0.997` (`cuda.go:76`) + `cuda_fp16_test.go` + `tools/run_484_acceptance_on_gpu.sh`
- Q8_0/Q4_K: `cudaQ8CosineMin=0.999` / `cudaQ4KCosineMin=0.995` (`cuda.go:104-105`) + `cuda_quant_test.go` + `tools/run_485_acceptance_on_gpu.sh`
- flash: `cudaFlashAttnCosineMin=0.999` (`cuda.go:125`) + `cuda_flash_test.go` + `tools/run_486_acceptance_on_gpu.sh`
- GLM-DSA: `cudaDsaSparseAttnCosineMin=0.999` (`cuda.go:141`) + acceptance in `cuda_acceptance.sh`

AWQ has `AWQMatMul`/`AWQBatchedMatMul` (`cuda.go:1101-1136`) and kernels `k_awq_gemv` /
`k_awq_gemm` / `k_awq_dequant_row` (`cuda_kernels.cu:884-992`) but **no** `cudaAWQCosineMin`
constant, **no** `cuda_awq_test.go`, and **no** `tools/run_awq_acceptance_on_gpu.sh` row in
`tools/cuda_acceptance.sh`. `docs/cuda-dev.md` "Honest residuals" already names this exactly.

### Why this one

- **Smallest correct step.** It adds no new kernel and changes no critical path; it lifts an
  existing, unwitnessed binding to the same recorded-then-measured bar as the other floors.
- **Falsifiable by construction.** A recorded cosine floor is a predicate a GPU run can *fail*
  (a `SKIP is not a PASS` acceptance row, per `cuda_acceptance.sh`), so the claim cannot
  silently read green on a CPU-only box.
- **No premature kernel bet.** It does not require choosing Marlin vs a hand kernel yet; the
  witness is what makes that choice *evidence-based* later.

### Falsifiable acceptance target + oracle

- **Target (record).** Add `cudaAWQCosineMin` to `internal/compute/cuda.go`, initially pinned
  conservatively at the AWQ source's own stated bar — `internal/model/awq.go` already
  documents "cosine similarity ≥ 0.995 vs FP32 baseline" — then tightened from the measured
  GPU run (the recorded-then-measured pattern of every other floor).
- **Oracle (witness).** The `cpuref` f32 Reference backend (`internal/compute/cpuref.go`,
  held to `max|Δ|=0`) — the same oracle the fp16/Q8/Q4_K/flash/DSA lanes are held to.
- **Predicate.** `fcuda_awq_gemv` (decode) and `fcuda_awq_gemm` (batched/prefill) outputs
  match `cpuref` with **argmax-exact** dominant-channel selection AND **cosine ≥
  `cudaAWQCosineMin`**; a measured cosine below the floor fails the acceptance row.
- **Mechanization (the GPU job).**
  1. A `-tags "cuda awqcuda"` test (`internal/compute/cuda_awq_test.go`, matching the
     `awq_cuda.go` build tag since `internal/model` carries SIMD `.s` kernels that forbid a
     plain cgo build) comparing both AWQ entry points to `cpuref` under the floor — the
     analogue of `cuda_fp16_test.go`.
  2. A `tools/run_awq_acceptance_on_gpu.sh` row appended to `tools/cuda_acceptance.sh`, so
     the new path joins the one-command `make cuda-accept` verdict.
  3. The GPU run records the realized cosine; the floor constant is then tightened to the
     measured value. **GPU-gated:** this measurement needs a CUDA host + device; the decision
     and the scaffolding are what ship now, the number is the named next action.

This is a **selection**, recorded to make the next change evidence-based. It is not the
implementation, and it adds no `CLAIMS.md` row.

## Benchmark lineage (issue Done-when #3)

The lineage the issue asks for (commit, machine, model, quant, backend, and whether the result
is native fak / llama.cpp / vLLM / SGLang) is **already the discipline** of the two authority
docs; this matrix does not duplicate it, it points at it:

- **Numbers + commit + artifact:** [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
  Quick-Reference — every row carries Number, Model, Baseline, Commit, Artifact, and the
  provenance-vs-reproducibility fence (private-lineage SHA vs the public artifact + reproduce
  command). The AWQ row is the one lineage entry the **Selection** above would fill.
- **Machine + backend + native-vs-engine:** [`HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md) —
  four platforms, each row naming the GPU backend (Metal / Vulkan / CUDA Ada / CUDA Ampere) and
  whether a number is native-fak or llama.cpp/vLLM/SGLang (e.g. RTX 4070 CUDA decode "~120
  tok/s, parity with llama.cpp Q8_0"; 8-GPU server "1085.6 vs 1451.6 tok/s vs raw SGLang").
- **The two-regime honesty rule** both enforce: deterministic metrics (token speedup, hit
  rate, bit-identity) are hardware-independent and reproduce byte-for-byte; only wall-clocks
  are single-box and stay labeled per-machine.

One honest gap the Selection resolves: AWQ has kernels but no measured row in either doc —
exactly because the witness does not exist yet.

## Format-vs-kernel prerequisite split (issue "Questions to answer" #1)

The matrix makes the split the issue asked for explicit, because it changes which stack to
learn from:

- **Format / load prerequisites** (GGUF quant-at-load, sharded safetensors, GPTQ/EXL2/AWQ
  loaders, resident Q4_K): learn from [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF
  and the per-format primary repos. fak's value here is loader correctness and the resident
  contract, not GEMM tuning.
- **Kernel prerequisites** (the actual quantized contraction): learn from
  [CUTLASS](https://docs.nvidia.com/cutlass/) / [Marlin](https://github.com/IST-DASLab/marlin)
  / [FlashInfer](https://docs.flashinfer.ai/). Here is where borrow-vs-bind-vs-stay-minimal
  actually decides.

The rule the matrix encodes: **borrow a kernel only after the witness for the current path
exists.** That is why the AWQ witness — not a new Marlin bind — is the selected first step.

## Issue-update text (issue Done-when #4) — gate not reached here

Done-when #4 requires the selected *implementation* issue to be updated with the SOTA decision
and verification plan. The decision and plan are above, ready to post; **posting it requires an
authed `gh` token, which this environment does not have** (`gh auth status` → not logged in;
no `GH_TOKEN`/`GITHUB_TOKEN`; credential helper is the OS manager and not wired to the issue
tracker). This is the one gate this note does not reach, by design rather than by fabrication.

Ready-to-paste comment for the AWQ implementation issue (target: the device-quant tracking
doc [`docs/notes/gpu-parity-tracking-480.md`](gpu-parity-tracking-480.md), and loader issues
#277 / #279 / #289 / #483 / #870 per #905's ladder):

> Backend SOTA matrix landed in `docs/notes/RESEARCH-backend-sota-matrix-2026-06-26.md`.
> Selected near-term win: close the AWQ 4-bit CUDA witness gap — add `cudaAWQCosineMin`
> (record ≥0.995 per `internal/model/awq.go`, tighten from the measured run), a `-tags "cuda
> awqcuda"` `cuda_awq_test.go` against the `cpuref` f32 oracle (argmax-exact + cosine ≥ floor),
> and a `tools/run_awq_acceptance_on_gpu.sh` row in `cuda_acceptance.sh`. Falsifiable: a
> measured cosine below the floor fails the acceptance row (SKIP-is-not-PASS). GPU-gated
> measurement; the selection ships now, the number is the next action.

When an authed operator runs that `gh issue comment`, Done-when #4 closes and the issue is
fully resolved. Until then the resolution is 3-of-4 criteria, with #4 blocked only on
credentials — not on any missing work product.
