---
title: "The SOTA prior-art matrix — check known art before writing a kernel from scratch"
description: "The maintained, load-bearing map of every compute operation fak's kernel performs to the production / SOTA stack to learn from first, the route (borrow / bind / stay-minimal), and the verification oracle. Read this before optimizing a kernel; the PRIOR_ART gate, `fak sota`, and the sota-coverage scorecard all read its source of truth."
---

# The SOTA prior-art matrix

**Read this before you optimize a kernel.** Almost every contraction fak performs — a
quantized GEMM, a fused attention, a KV-cache reuse, a MoE dispatch — has a production
reference that already solved it: llama.cpp's GGML kernels, Marlin, CUTLASS, FlashInfer,
vLLM PagedAttention, SGLang RadixAttention, a named paper. The default answer to "should I
write this from scratch?" is **no** — study the reference, decide *borrow / bind /
stay-minimal*, and hold the result to an oracle. Re-deriving known art badly is the failure
mode this matrix exists to stop.

This is the **inward** engineering counterpart of
[`docs/industry-scorecard/`](../industry-scorecard/README.md): the scorecard positions fak
against the field (an *outward* competitive map); this matrix maps each operation fak's
kernel **actually performs** to the production stack worth learning from, the route to relate
to it, and the oracle that proves a fak version correct.

## Why this is load-bearing, not a note

Prior art was *always* being researched here — `tools/idea_scout.py` files adjacent work as
issues, the [`docs/notes/RESEARCH-*`](../notes/) corpus is deep, and this matrix began as a
single dated note
([`RESEARCH-backend-sota-matrix-2026-06-26.md`](../notes/RESEARCH-backend-sota-matrix-2026-06-26.md)).
The gap was that the research was **inert**: nothing on the kernel-commit path forced an agent
to consult it, so an agent could (and did) reach for "implement the Mac Q6_K fused MLP from
scratch" without first reading the Metal kernel llama.cpp already ships.

The fix is to make the prior-art map a maintained datum that three surfaces read:

| Surface | What it does | Source |
|---|---|---|
| **`fak sota <op\|file>`** | Agent-facing lookup. Run it *before* writing a kernel: prints the SOTA reference, route, oracle, and the link to read. | `internal/sotamatrix` |
| **`PRIOR_ART` pre-commit gate** | When a commit touches a kernel file, prints the matching op's SOTA reference and suggests a `Prior-art:` trailer. **Advisory — never blocks.** | `internal/hooks/gate_priorart.go` |
| **`tools/sota_coverage_scorecard.py`** | Cross-checks the matrix against the tree: every kernel file is covered by a row, every row's fak-path file exists, every row carries a primary link + oracle. Folds gaps into one `sota_debt` integer. | `tools/sota_coverage_scorecard.py` |

The single source of truth is the flat literal in
[`internal/sotamatrix/sotamatrix.go`](../../internal/sotamatrix/sotamatrix.go) — the same
in-binary-registry discipline as `internal/benchcatalog`. The gate, the command, and the
scorecard all read it; none keeps a rival copy. **Adding a kernel operation means adding one
row there.**

## The process (what to actually do)

1. **Before** you write or optimize a kernel, run `fak sota <operation>` (or
   `fak sota <the-file-you're-about-to-edit>`). Read the `PrimaryLink`. Decide your route.
2. **Route honestly.** `stay-minimal` (the bit-exact contract is fak's value, not raw speed),
   `bind` (use the production library/format directly), or `borrow` (adapt the reference
   technique) — and **borrow a kernel only after a witness for the current path exists**, so
   the choice is evidence-based rather than a premature bet.
3. **Prove it against the oracle** the row names (almost always `cpuref` f32 with a cosine
   floor, or bit-identity). A kernel with no oracle is not done.
4. **Stamp the commit** with a `Prior-art:` trailer naming what you consulted (e.g.
   `Prior-art: Marlin fused dequant-MMA (IST-DASLab/marlin); cosine ≥ 0.995 vs HF AWQ`). This
   silences the advisory gate and leaves a durable record of the reference for the next person.

## The matrix

The rows below are rendered from `internal/sotamatrix`; run `fak sota list` for the live
table and `fak sota <slug>` for one operation's full detail (route note, oracle, papers).

| Operation | fak path | SOTA to learn | Route | Oracle |
|---|---|---|---|---|
| Dense f32 GEMM | `internal/compute/cpuref.go` + `model/parallel.go` | cuBLAS / [CUTLASS](https://docs.nvidia.com/cutlass/) | stay-minimal | `cpuref` bit-identity |
| Dense fp16/Q8/Q4_K device GEMM | `internal/compute/cuda.go` + `cuda_kernels.cu` | cuBLAS; [llama.cpp](https://github.com/ggml-org/llama.cpp) | bind | `cpuref`, argmax + cosine ≥ floor |
| AWQ 4-bit GEMV/GEMM | `internal/compute/cuda.go:1101` + `cuda_kernels.cu` | [Marlin](https://github.com/IST-DASLab/marlin); AutoAWQ | borrow | `cpuref` + HF AWQ (cosine ≥ 0.995) |
| GPTQ resident (CPU) | `internal/model/gptq.go` | AutoGPTQ / [Marlin](https://github.com/IST-DASLab/marlin) | bind | HF GPTQ dequant / `cpuref` |
| EXL2 loader | `internal/model/exl2.go` | [ExLlamaV2](https://github.com/turboderp-org/exllamav2) | stay-minimal | ExLlamaV2 reference |
| GGUF quant-at-load (Q4/5/6_K) | `internal/ggufload/`, `internal/model/quant_q4k.go` | [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF | bind | llama.cpp 2-token parity |
| MoE expert dispatch | `internal/model/moe.go`, `glm_dsa.go` | [DeepEP](https://github.com/deepseek-ai/DeepEP) / TensorRT-LLM | borrow | dense reference / HF |
| Fused attention (MHA/GQA/MQA) | `internal/compute/cuda_kernels.cu` `k_flash_attention` | [FlashInfer](https://docs.flashinfer.ai/) / FlashAttention | stay-minimal | `cpuref` (cosine ≥ 0.999) |
| GLM sparse attention (DSA) | `internal/compute/dsa.go` | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) custom sparse | stay-minimal | `cpuref` (cosine ≥ 0.999) |
| KV cache (paging / prefix reuse) | `internal/model/kv.go`, `internal/radixkv` | [vLLM PagedAttention](https://docs.vllm.ai); SGLang RadixAttention | stay-minimal | bit-identity |
| Metal Q4_K / Q6_K GEMM | `internal/metalgemm/`, `internal/model/metal_q4k*.go` | [llama.cpp Metal](https://github.com/ggml-org/llama.cpp) / MLX | borrow | `cpuref` (GEMV cosine 1.000000) |

## The rule the matrix encodes

**Borrow a kernel only after the witness for the current path exists.** The reference is
inspiration and a correctness target — read it first, route deliberately, prove against the
oracle, and record what you read. Throughput is a real program (the
[kernel-optimization program](../../AGENTS.md) is never "done"), but it is pursued *on top of*
the known art, not in ignorance of it.
