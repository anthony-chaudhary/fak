---
title: "Dense Qwen3.8-27B Metal OSS hot-path study"
description: "Dated source study for closed issue #8823 and closed parent target #8697; not an implementation or performance result."
---
# Dense Qwen3.8-27B Metal OSS hot-path study

> Issue: [#8823](https://github.com/anthony-chaudhary/fak/issues/8823) (closed 2026-08-25) · Historical parent performance target: [#8697](https://github.com/anthony-chaudhary/fak/issues/8697) (closed 2026-08-26)

Date: 2026-08-24. This is a source study, not an implementation or performance result. The three upstream trees were already cloned locally; all paths and symbols below were inspected at the exact revisions shown.

## 2026-08-28 identity and tracker reconciliation

This study concerns the dense `Qwen/Qwen3.8-27B` artifact. The official model card calls it a dense native vision-language model, while its official configuration uses `model_type=qwen3_5`, `architectures=[Qwen3_5ForConditionalGeneration]`, and `text_config.model_type=qwen3_5_text`. That is why the pinned upstream and fak implementation symbols below retain `qwen35` / `qwen3_5`; those source names are compatible with the dense Qwen3.8-27B artifact and are not Qwen3.6 receipts.

The separate [`QwenLM/Qwen3.8-Flash-Next`](https://github.com/QwenLM/Qwen3.8-Flash-Next) source repository and `Qwen/Qwen3.8-Flash-Next` weights are a multimodal MoE preview whose README calls it an early preview of the architecture used in Qwen4 and whose design citation names the Qwen3.8-Next architecture. Use `QwenNext` only as explicitly Flash-Next/Qwen3.8-Next-scoped shorthand. The preview has 125B main-model parameters plus 51B N-gram embeddings, 6B activated per token, GDN + QSA hybrid attention, Gated Residual, and N-gram Embedding. None of this study's source mapping, receipts, or performance thresholds is evidence for that separate release. In particular, the pinned MLX-LM file named `qwen3_next.py` is a shared component imported by its Qwen3.5 implementation; the filename is not evidence that this 2026-08-24 study inspected the later Qwen3.8-Flash-Next/Qwen3.8-Next release.

Official cached inputs observed 2026-08-28:

| Input | Official source | Cached SHA-256 | Reconciliation use |
|---|---|---|---|
| `Qwen3.8-27B` README | [`Qwen/Qwen3.8-27B` model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md) | `57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641` | Dense identity and published architecture facts |
| `Qwen3.8-27B` config | [`config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` | Exact config identifiers and dimensions |
| `Qwen3.8-Flash-Next` README | [`QwenLM/Qwen3.8-Flash-Next` README](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/README.md) | `34d45d3486c29dcc23dade1472b5cbf1347ffe0a1adc3334aec83b3dc4e08c50` | Separate-preview boundary only |

These architecture facts classify artifacts; they do not establish a performance gain. Historical Qwen3.6 receipts elsewhere in the repository remain historical Qwen3.6 evidence and must not be rewritten as Qwen3.8 evidence.

Live issue state observed 2026-08-28: reconciliation #9739 is open; #8823 and #8697 are closed historical tickets; mechanism-only #9488 is closed with no M4 KEEP; #8324 remains the open broader M4 same-artifact campaign parent; and #9513 is the open M10 parity close-out leaf. This document remains the historical source map for #8823 rather than a live dispatch packet.

## Pinned sources and license posture

| Project | Origin URL | Exact HEAD | License found in clone | Decision consequence |
|---|---|---|---|---|
| llama.cpp | https://github.com/ggml-org/llama.cpp.git | `ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0` | MIT, `LICENSE`, “Copyright (c) 2023-2026 The ggml authors” | Code may be adapted with the MIT notice retained for copied substantial portions. Prefer behavioral reimplementation with source attribution. |
| MLX | https://github.com/ml-explore/mlx.git | `43d2f06cb87e76895bf9a152bade4fee83408643` | MIT, `LICENSE`, “Copyright © 2023 Apple Inc.” | Same: compatible for adaptation, but preserve the notice if code is copied. |
| MLX-LM | https://github.com/ml-explore/mlx-lm.git | `cc8521569694a3240b52c98acffd100d59b4c755` | MIT, `LICENSE`, “Copyright © 2023 Apple Inc.” | Same. Use primarily as an architecture/correctness oracle rather than importing Python. |

These license notes are repository-level observations, not legal advice. Before landing copied code, check file-local notices and record the source revision in the change.

## Exact upstream evidence

### llama.cpp

Architecture and graph construction:

- `src/models/qwen35.cpp`: `llm_build_qwen35`, including its constructor, `build_layer_attn`, `build_layer_gdn`, and `build_forward` path. This is the dense Qwen3.5-family hybrid decomposition relevant to fak's `qwen3_5_text` loader; `qwen3_5_text` is also the official dense Qwen3.8-27B config identifier.
- `src/models/qwen35moe.cpp`: `llm_build_qwen35moe` and its corresponding attention/Gated-DeltaNet/MoE graph construction. It is adjacent architecture evidence only, not evidence for the dense target.
- `src/llama-arch.cpp`, `src/llama-arch.h`, `src/llama-hparams.h`, and `src/llama-model.cpp`: Qwen3.5 architecture/tensor naming and hyperparameter plumbing.

Metal Q4_K dispatch and submission:

- `ggml/src/ggml-metal/ggml-metal-device.cpp`: `ggml_metal_library_get_pipeline_mul_mv_ext`, `ggml_metal_library_get_pipeline_mul_mm`, `ggml_metal_library_get_pipeline_mul_mv`, `ggml_metal_library_get_pipeline_mul_mm_id_map0`, `ggml_metal_library_get_pipeline_mul_mm_id`, and `ggml_metal_library_get_pipeline_mul_mv_id`. The `GGML_TYPE_Q4_K` cases and generated `kernel_mul_mv_*`/`kernel_mul_mm_*` names are the relevant format-to-pipeline selection evidence.
- `ggml/src/ggml-metal/ggml-metal-device.h`: declarations for those pipeline selectors.
- `ggml/src/ggml-metal/ggml-metal.cpp`: operation encoding and command-buffer lifecycle around the selected pipelines.
- `ggml/src/ggml-metal/ggml-metal-impl.h` and generated Metal kernel sources under `ggml/src/ggml-metal`: Q4_K type bindings and matrix/vector kernel instantiations.

The useful llama.cpp idea is the explicit separation of tensor-format dispatch, pipeline specialization, and graph execution. Its graph/runtime and generated kernel framework are not a drop-in fit for fak.

### MLX

Command-buffer and residency control:

- `mlx/backend/metal/device.cpp`: `CommandEncoder::needs_commit` and `CommandEncoder::commit`. The latter attaches newly created residency sets before commit because Metal fixes residency at commit time, then commits the retained command buffer. `Device::get_command_encoder`/encoder-pool code in the same file is the ownership seam.
- `mlx/backend/metal/device.h`: `CommandEncoder`, `Device`, and residency-set declarations.
- `mlx/backend/metal/resident.cpp` and `resident.h`: `ResidencySets` management used by `CommandEncoder::commit`.

GEMM/Q4 machinery:

- `mlx/backend/metal/matmul.cpp`: Metal matmul dispatch, including Steel selection and encoded GEMM paths.
- `mlx/backend/metal/jit_kernels.cpp`: `get_steel_gemm_fused_kernel`, `get_steel_gemm_splitk_kernel`, and related JIT pipeline construction.
- `mlx/backend/metal/jit/includes.h`: `steel_gemm_fused`, `steel_gemm_masked`, `steel_gemm_splitk`, `steel_gemm_gather`, and `steel_gemm_segmented` source accessors.
- `mlx/backend/metal/kernels/steel/gemm/` and `mlx/backend/metal/kernels/quantized.h`: tiled GEMM and quantized load/dequantization building blocks.
- `mlx/backend/metal/quantized.cpp`: quantized matmul dispatch and kernel specialization.

MLX's reusable lesson is encoder ownership plus explicit commit thresholds/residency, not wholesale adoption of Steel: Steel assumes MLX's array metadata, JIT source assembly, allocator, stream, and dispatch stack.

### MLX-LM

- `mlx_lm/models/qwen3_5.py`: `TextModelArgs`, `GatedDeltaNet`, `GatedDeltaNet.__call__`, and the model layer dispatch. Exact dimensions include `linear_num_value_heads`, `linear_num_key_heads`, `linear_key_head_dim`, `linear_value_head_dim`, and `linear_conv_kernel_dim`.
- `mlx_lm/models/gated_delta.py`: `gated_delta_update`, the recurrent Gated-DeltaNet update used by the model.
- `mlx_lm/models/qwen3_5_moe.py`: Qwen3.5 MoE composition and sparse expert wiring; adjacent evidence only, not part of the dense target path.
- `mlx_lm/models/qwen3_next.py`: shared `Qwen3NextAttention`, `Qwen3NextMLP`, `Qwen3NextRMSNormGated`, and `Qwen3NextSparseMoeBlock` imported by the pinned Qwen3.5 model. This filename predates and must not be conflated with the separate Qwen3.8-Flash-Next/Qwen3.8-Next preview.

The dense-model-relevant subset is the clearest compact semantic oracle for tensor shapes, convolution state, normalized q/k, recurrent state update, and gated output. The adjacent MoE files do not define this study's target, and none of these sources specifies fak's Q4_K Metal scheduling.

## Current fak seams inspected

The complete `internal/metalgemm/*` directory was searched for kernel definitions, pipeline creation, buffer ownership, dispatch, command-buffer commit/wait, and Q4_K/GEMM references. The directly relevant seams are:

- `internal/metalgemm/qwen35_decode.go`: `Qwen35DecodeState`, `NewQwen35DecodeState`, `(*Qwen35DecodeState).Step`, and `ResetQwen35Decode`. This is the Go-to-Metal stateful token-step boundary.
- `internal/metalgemm/qwen35_decode.m`: `mg_qwen35_decode_create`, `mg_qwen35_decode_step`, and `mg_qwen35_decode_reset`, plus the embedded Qwen35 conv, q/k normalization, recurrent update, and gated-norm kernels. `mg_qwen35_decode_step` currently copies host inputs, creates/encodes a Metal command buffer for one step, commits, waits, and copies the result back.
- `internal/metalgemm/qwen35_decode_stub.go`: non-Metal build boundary for the decode API.
- `internal/metalgemm/q4k.go`, `q4k.m`, and `q4k_stub.go`: Q4_K Go/C bridge, packed-weight ownership, Metal pipeline setup, GEMV/GEMM dispatch, and fallback boundary.
- `internal/metalgemm/qwen35_decode_test.go`, `qwen35_decode_metal_test.go`, `q4k_test.go`, and `q4k_metal_test.go`: correctness and build-tag witnesses available for a regression test.
- Other files under `internal/metalgemm/*` supply adjacent kernels and tests, but no inspected file provides a reusable whole-model command graph joining the Q4_K projections and the GDN step.
- `internal/model/qwen35.go`: `Config.IsQwen35Hybrid`, `Config.isLinearAttnLayer`, `materializeQwen35Tensors`, `newLinearAttnCache`, `Config.linearAttnDims`, `Model.linearAttnSeq`, and `Session.linearAttnStep`. `linearAttnStep` is the model seam that prepares projections/state and can call the Metal GDN step; it remains above the Q4_K projection calls.

## Source-to-fak seam map

| Upstream behavior | Upstream source/symbol | Current fak seam | Fit |
|---|---|---|---|
| Identify hybrid layer type and construct attention versus GDN graph | llama.cpp `src/models/qwen35.cpp`: `llm_build_qwen35::{build_layer_attn,build_layer_gdn,build_forward}`; MLX-LM `qwen3_5.py`: `GatedDeltaNet.__call__` and model layer dispatch | `internal/model/qwen35.go`: `IsQwen35Hybrid`, `isLinearAttnLayer`, `linearAttnSeq`, `linearAttnStep` | **Borrow semantics**, not graph/runtime code. Use as a parity oracle. |
| Stateful conv + normalized q/k + recurrent delta update + gated RMSNorm | MLX-LM `qwen3_5.py:GatedDeltaNet`; `gated_delta.py:gated_delta_update` | `qwen35_decode.m:mg_qwen35_decode_step`; `qwen35.go:linearAttnStep` | **Adapt tests/invariants**. The operation already exists in fak; importing Python adds no hot-path value. |
| Q4_K format selects specialized MV/MM pipeline | llama.cpp `ggml-metal-device.cpp`: `ggml_metal_library_get_pipeline_mul_mv[_ext]`, `...mul_mm`, `...mul_mv_id`, `...mul_mm_id` | `q4k.go`/`q4k.m` pipeline and dispatch code | **Borrow dispatch questions and tuning methodology**; adapt only a measured specialization. Do not transplant ggml tensor/runtime machinery. |
| Quantized/tiled GEMM implementation | MLX `quantized.cpp`, `matmul.cpp`, `kernels/quantized.h`, `kernels/steel/gemm/*`; `jit_kernels.cpp:get_steel_gemm_*_kernel` | `q4k.m` embedded Metal Q4_K kernels | **Reject wholesale; adapt isolated loader/tile concepts only after an ablation**. MLX Steel's JIT/allocator/array contracts make direct import high-risk and non-minimal. |
| Reuse encoder ownership and delay commit while preserving residency | MLX `device.cpp:CommandEncoder::{needs_commit,commit}` and `ResidencySets` | Per-call command-buffer creation/commit/wait in `qwen35_decode.m:mg_qwen35_decode_step` and analogous Q4_K bridge calls | **Borrow the lifecycle pattern; adapt narrowly** to a fak-owned decode-step command-buffer/encoder boundary. |
| Whole graph scheduling and memory planner | llama.cpp ggml graph/backend; MLX stream/allocator/JIT runtime | No equivalent whole-model graph in `internal/metalgemm`; orchestration is in `internal/model` | **Reject for #8697**. Building a runtime is much larger than the measured seam and would confound attribution. |
| GPU residency sets/heaps | MLX `resident.{h,cpp}:ResidencySets`; `CommandEncoder::commit` | Persistent `MTLBuffer` objects in Qwen35/Q4_K states, but no equivalent residency-set manager | **Defer**. First determine whether submission/wait overhead is material; residency work without that witness is speculative. |

## Borrow / adapt / reject decisions

**Borrow:** the dense-target observable contracts: Qwen hybrid operation ordering and shapes from MLX-LM/llama.cpp; explicit Q4_K pipeline specialization from llama.cpp; and MLX's long-lived encoder/command-buffer ownership with commit policy separated from individual kernel calls.

**Adapt:** only fak's existing `mg_qwen35_decode_step` boundary first. Preserve persistent state buffers and pipelines, but add an experimental path capable of encoding the existing conv → normalize → recurrent → gated-norm sequence without forcing an avoidable intermediate host synchronization. Keep the old path selectable for rollback. If profiling instead attributes most time to Q4_K projection kernels, move the same experiment boundary to `q4k.m`; do not claim that attribution before measuring it.

**Reject now:** importing ggml's graph executor, MLX's Steel/JIT/allocator stack, Python model code in production, changing Q4_K packing merely to match another runtime, or combining command reuse, residency, kernel tiling, and model changes in one patch. Each would obscure which change moved the matched decode result.

## Historical first ablation for #8697

#8697 is closed. The contract below is preserved as the 2026-08-24 proposed ablation and receipt threshold; it is not a current dispatch instruction. Mechanism-only #9488 is also closed; #8324/#9430 retain the missing exact-artifact performance receipt, and #9513 owns the final matched M10 parity close-out.

**Ablation A: remove one-token GDN submission overhead without changing math.** Add an experiment switch that compares the current `mg_qwen35_decode_step` command-buffer/encoder lifecycle with a fak-owned reusable/batched encoding lifecycle modeled on MLX `CommandEncoder`, while invoking the same four existing GDN kernels with identical buffers, thread geometry, and operation order. The experiment must record command-buffer count, encoder count, GPU elapsed time where available, wall decode time, and output parity. It must not include Q4_K tile changes or residency-set work.

Run the fak-native baseline and candidate on the #8697 pinned witness: the same receipt-pinned Q4_K_M artifact for dense `Qwen/Qwen3.8-27B`, Apple M3 Pro 18-GPU-core/36 GiB, temperature zero, exactly `P=32/T=64`, and three measured repetitions after the same warm-up policy. Preserve `engine=inkernel backend=metal forward_path=metal/qwen35-hybrid-session-v1 q4k=true`; any non-native execution or fallback invalidates the candidate. Use llama.cpp b9828 only as #8697's explicitly selected parity/reference benchmark, not as an execution path or fallback. Report every repetition, mean, standard deviation, artifact hash, fak commit, comparator revision, flags/environment, and command-buffer counts.

Acceptance requires all of the following:

1. All 64 generated-token steps complete on the intended Metal path with no fallback, error, leak, or state-reset failure.
2. Token IDs are identical to the current fak control for the deterministic `P=32/T=64` run; the existing per-step kernel witness also remains within its established numeric tolerance.
3. The ablation reduces the targeted submission/synchronization count as designed and improves the three-run fak decode mean by at least **5%**, with each ablation repetition faster than the control mean. This is an experiment-retention threshold, not a predicted gain.
4. No measured prompt/prefill regression greater than **2%** and no peak-memory increase greater than **5%** under the same campaign.
5. The result is separately reported against #8697's product gate: fak-native Metal must reach at least **95%** of the matched llama.cpp decode throughput. Passing the ablation threshold does not by itself close #8697.

Reject or revert the ablation if any correctness criterion fails, if it silently falls back, if the submission count does not fall, if the throughput threshold is missed, if gains appear only after excluding a repetition, or if the prefill/memory guardrail fails. A clean rejection still answers whether command submission is the first material seam; the next checkable step is then a profiler-attributed, single-variable Q4_K projection-kernel ablation in `q4k.m`.

## Historical recommendation and current disposition

The 2026-08-24 recommendation was to start with Ablation A because it was the smallest reversible adaptation supported by an exact upstream lifecycle analogue and an exact fak boundary. Subsequent work landed and closed the enabling #9488 block mechanism; #8324/#9430 still own its absent exact-artifact fak-native performance receipt. Treat all throughput movement as unproven until the matched `P=32/T=64` campaign exists; this report makes no performance claim for any borrowed design or for either Qwen3.8 architecture.
