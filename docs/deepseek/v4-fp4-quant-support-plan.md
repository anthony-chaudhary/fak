---
title: "DeepSeek V4 FP4 expert-weight + FP4 indexer quantization support plan"
description: "Support plan for DeepSeek V4's mixed FP4 (MoE experts + attention-indexer QK) / FP8 (rest) checkpoint: per-tensor precision admission table, checkpoint-metadata detector, dequant/GEMM backend options by hardware class, FP4 indexer QK path requirements, a numerical-parity oracle design, and a recommended disposition — all mapped onto real fak seams. Plan only; no weights, no benchmark numbers."
---

# DeepSeek V4 FP4 expert weights + FP4 indexer — quantization support plan

**2026-07-09.** Issue **#3019**, parent epic **#3006** (native DeepSeek-V4 kernel track).
This is the canonical home for the #3019 plan under `docs/deepseek/`; it supersedes and
absorbs the working note `docs/notes/DEEPSEEK-V4-FP4-QUANT-PLAN-2026-07-08.md`, and it is a
**sibling** of the other V4 design notes:
`docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md` (#3016 attention/indexer seam map),
`docs/notes/DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md` (#3018 MoE dispatch), and
`docs/benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`.

**Plan + fixture plan only.** No native FP4 GEMM lands here, no in-tree weights, and per the
issue any tooling is **Go**, not Python. Every current-state claim is witnessed against the
exact `path:line` cited (read 2026-07-09 on `main`). No throughput/latency/accuracy number is
stated as measured anywhere in this doc; modeled figures carry the **MODELED** label and
anything needing a GPU or a tuned engine carries **host-gated**.

## Thesis — V4 is a *mixed-precision checkpoint*, not another FP8 model

V4's checkpoint is **mixed precision**: **MoE expert parameters are FP4**, **most other
parameters are FP8**, and the **attention indexer QK path is also FP4** — post-training
includes FP4 quantization-aware training (QAT) for the expert weights and the indexer. fak's
loader already has an **FP8 dtype tag** (`internal/compute/compute.go:62`) and two distinct
**4-bit decode-to-f32** paths (MXFP4 for gpt-oss, AWQ/GPTQ integer 4-bit), plus a proven
**quantize-on-load per-tensor admission gate** (`isQuantWeight`,
`internal/model/safetensors_quant.go:30`). But it has **no native FP4 GEMM dtype, no
NVFP4/E2M1 format, and no per-tensor admission that fails closed on an unknown FP4 tensor
class**. So V4 quant must be treated as a **new quantization/backend problem** — begun with a
metadata detector and an admission table, **not** a dequant kernel.

## The V4 quant facts that drive the plan (from the issue grounding)

Sources: the HF model card <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro> and the V4
technical report <https://arxiv.org/html/2606.19348v1> (as cited in #3019's Grounding; the
same constants fak already encodes in `internal/gateway/deepseek_budget.go:70-97`).

| V4 quant fact | Value | Consequence for the plan |
|---|---|---|
| MoE expert parameters | **FP4** | the parameter *bulk* (1.6T-total / 49B-active for V4 Pro) needs an FP4 dequant/GEMM path |
| Most other parameters | **FP8** | fak carries an FP8 dtype tag but *widens at load* today, not a native FP8 GEMM |
| Attention indexer QK | **FP4** | the lightning indexer (#3016) needs an FP4 QK score path |
| Post-training | FP4 QAT for experts + indexer | numerics must match a **QAT reference**; naive round-to-FP4 will not match |
| Precision widths | FP4 = 0.5 B/param, FP8 = 1.0 B/param | resident weight size sits between an all-FP4 floor and an all-FP8 ceiling (MODELED bound, `deepseek_budget.go:90-91`) |

## Per-tensor precision admission table (acceptance item 1)

The load-bearing deliverable is a **per-tensor-class → expected-precision** table that the
loader admits against and **fails closed** on any class it cannot place. Column 3 names the
real fak seam each class already routes through today (GLM-5.2 `glm_moe_dsa` is DeepSeek MLA +
a DSA lightning indexer, so V4's tensor classes reuse GLM's canonical names and splitters).

| Tensor class | Expected V4 precision | fak seam that already routes this class (`path:line`) |
|---|---|---|
| **Routed experts** (`mlp.experts.<e>.{gate,up,down}_proj.weight`) | **FP4** | batched-expert split `internal/ggufload/gguf_glm_tensors.go:228` `glmMoeDsaBatchedExpert` / `:265` `splitGLMMoeDsaExperts`; raw-quant resident split `:326` `splitGLMMoeDsaExpertsRawQuant`; safetensors admit `internal/model/safetensors_quant.go:52-54` |
| **Shared experts** (`mlp.shared_experts.{gate,up,down}_proj.weight`) | **FP8** | GGUF map `internal/ggufload/gguf_glm_tensors.go:114-116`; safetensors admit `internal/model/safetensors_quant.go:55-57` |
| **Router / gate** (`mlp.gate.weight` + `e_score_correction_bias`) | **FP8** | GGUF map `internal/ggufload/gguf_glm_tensors.go:111-112`; safetensors admit `internal/model/safetensors_quant.go:48` |
| **LM head** (`lm_head.weight`, or tied embed) | **FP8** | admit `internal/model/safetensors_quant.go:60`; tied-head quantize `:369-371` |
| **Attention MLA projections** (`q_a/q_b/kv_a_proj_with_mqa/kv_b_proj`) | **FP8** | GGUF map `internal/ggufload/gguf_glm_tensors.go:101-108`; KV-b 2→1 merge `:448` `mergeGLMMoeDsaKVB`; safetensors admit `internal/model/safetensors_quant.go:36-39` |
| **Attention indexer QK** (`indexer.wq_b/wk/weights_proj`) | **FP4** | GGUF map `internal/ggufload/gguf_glm_tensors.go:118-122`; layer classifier `:84` `glmLayerHasIndexer`; safetensors admit `internal/model/safetensors_quant.go:40-42` |
| **Norms** (`attn_norm`, `ffn_norm`, `q_a_layernorm`, `kv_a_layernorm`, `indexer.k_norm`) | **FP8 / kept f32** | fall-through base map (not in `isQuantWeight`) → stays f32 in `raw` `internal/model/safetensors_quant.go:360-366` |
| **MTP / "nextn" head** | FP4/FP8 (spec-decode, **not wired**) | drop-by-default `internal/ggufload/gguf_glm_tensors.go:183` `glmMoeDsaSkipGGUFTensor`; retained under `model.RetainMTP` (#3078/#3197) |

**Admission invariant:** for every class above, the detector asserts the expected precision
and **refuses an unrecognized FP4 tensor class** with the existing typed refusal
`internal/model/arch_support.go:33` `UnsupportedArchError` (already used for unknown hybrid
arch at `:69` `refuseUnsupportedHybridArch`) rather than silently mis-loading. This is the
"fail closed on an unknown FP4 tensor class" property fak does not have today.

## Checkpoint-metadata detection (the first landable witness)

A **Go** fixture (not Python, per the issue) that parses **representative V4 tensor metadata**
— tensor names, shapes, dtypes, and the checkpoint's `quantization_config` / `quant_method`
block — **without downloading full weights**, and:

- Classifies every tensor into the eight classes of the table above (routed expert / shared
  expert / router / head / attention / norm / MTP / **indexer QK**), reusing the existing
  name classifiers (`parseGLMBlkLayerSuffix`, `glmMoeDsaBatchedExpert`, `glmLayerHasIndexer`,
  `isGLMMoeDsaMTPTensor`) so classification is one code path with the real GLM-5.2 loader.
- Asserts the expected precision per class (experts → FP4, indexer QK → FP4, rest → FP8).
- **Fails closed** on an unrecognized FP4 tensor class via `UnsupportedArchError`.

Detection sources, in order of preference: (a) the safetensors `_blocks`/`_scales` block/scale
pairing convention the MXFP4 path already keys on (`internal/model/safetensors_quant.go:286`
`quantizeMXFP4TensorInto`) — V4's NVFP4 experts ship an analogous packed-block + scale layout;
(b) the checkpoint's `quantization_config` metadata; (c) the HF card's tensor index. No
weights, no in-tree binaries — the fixture is metadata + shapes only.

## Dequant / GEMM backend options by hardware class (acceptance item 2)

fak already carries a **per-hardware-class 4-bit backend split** for AWQ (integer 4-bit); an
FP4 path would mirror that seam family, not invent a new one. Every runtime row below is
**host-gated** (needs the actual device + a tuned kernel); the table names *options*, and does
not claim any of them is wired for FP4.

| Hardware class | Existing 4-bit backend seam (`path:line`) | FP4 (NVFP4/E2M1) option | Status |
|---|---|---|---|
| **x86 scalar / AVX2 / AVX512** | `internal/model/awq_amd64.go`, `awq_scalar.go`, `awq_amd64_asm.go`; MXFP4 SIMD `internal/ggufload/gguf_dequant_simd_amd64.go` | dequant-to-f32 then existing f32 GEMM (correctness-first, no FP4 tensor cores) | **host-gated**, MODELED-only |
| **NVIDIA CUDA (Hopper/Blackwell)** | `internal/model/awq_cuda.go` (CUDA dequant/GEMM) | native NVFP4 GEMM via CUTLASS/FlashInfer FP4; W4A16 on Hopper | **host-gated**, native path fenced |
| **AMD ROCm** | `internal/amdgpu/` (device facts); `awq_nocuda.go` fallback | MXFP4/OCP-FP4 GEMM where ROCm exposes it, else decode-to-f32 | **host-gated** |
| **Apple Metal** | `internal/metalgemm/` (Metal GEMM) | decode-to-f32 (no FP4 tensor path on Metal) | **host-gated** |
| **Portable fallback (any box)** | MXFP4 decode-to-f32 `internal/model/safetensors.go` `decodeMXFP4Blocks`; AWQ scalar | **decode FP4 → f32 → existing quant/f32 GEMM** — the correctness reference every device path is diffed against | **landable, CPU-only** |

The portable decode-to-f32 row is the numerical **reference implementation**: it reuses the
E2M1 nibble decode already proven bit-exact for MXFP4
(`internal/ggufload/dequant_mxfp4_test.go`), extended to V4's NVFP4 block/scale geometry. It
is not a fast path; it is the oracle-side ground truth the device kernels must match.

## FP4 indexer QK path requirements (acceptance item 3)

The lightning indexer's QK dot is the hot inner loop (#3016 records the selector as the
fast, high-recall path; its **2×-speedup / 99.7%-recall** figures are `PAPER_CLAIMED` —
upstream DSA numbers carried by #3016, **not re-measured under fak**). V4 runs its **QK in
FP4**. Requirements to support it:

1. **FP4-typed index projections.** `indexer.wq_b` / `indexer.wk` admit as FP4 (table above);
   the index *scores* are computed on the dequantized/native-FP4 QK, feeding the existing
   score formula `internal/model/dsa_index.go:20` `dsaIndexScores`
   (`sum_h w[q,h]·relu(scale·dot(idx_q,idx_k))`) unchanged — only the QK dtype changes.
2. **Top-k selection is dtype-agnostic.** `internal/model/dsa_index.go:66` `dsaTopKIndices`
   selects key *positions*; `topK` (V4 = 1024) is a config value, so FP4 QK does not touch the
   selection code — it only changes the precision of the scores fed in.
3. **k-norm stays higher precision.** `indexer.k_norm` is a norm (kept f32/FP8), so the FP4
   boundary is the QK matmul only, not the RMSNorm on the index key.
4. **Parity is scored on the selection set, not the raw scores.** Because FP4 QK perturbs
   scores, the oracle compares the *selected top-k index set* (and its digest
   `internal/model/dsa_index.go:232` `dsaIndexDigest`) against the reference, tolerating
   score-value noise below the selection boundary. This is the same digest fak already uses to
   share a selection across layers (`:208` `dsaIndexShare`).

## Numerical-parity oracle design (acceptance item 4)

FP4 **QAT** numerics mean naive round-to-FP4 will *not* reproduce the reference — the oracle
must compare against a **real FP4 reference**, never a fak-internal re-derivation.

- **Reference A (weights/logits):** vLLM or SGLang FP4 output on a representative prompt, or an
  HF-metadata fixture with reference-dequantized expert tiles. **host-gated** (needs the tuned
  engine or the published tiles).
- **Reference B (indexer):** the SGLang FP4-indexer selection trace — compared as the top-k
  **index set + digest**, per the indexer requirements above.
- **fak side:** the portable decode-to-f32 reference row drives fak's existing forward; the
  oracle diffs (a) per-expert dequant tiles bit-exact where reference tiles are published, and
  (b) end-to-end logits within a QAT-tolerance band where only engine output is available.
- **Provenance discipline:** the oracle emits rows under the **same closed label vocabulary**
  fak's budget calculator already enforces — `SOURCE_DOCUMENTED` / `PAPER_CLAIMED` / `MODELED`
  / `WITNESSED` (`internal/gateway/deepseek_budget.go:41-68`) — and **fails closed** if a row
  cannot carry an honest label. No parity row is `WITNESSED` until it is measured under fak.

The oracle is recorded here as a **design**; the metadata fixture is the witness that must land
*before* any native FP4 ticket is filed (the issue's done-condition for a loader follow-on).

## Recommended disposition (acceptance item 5) — DECISION

Three dispositions were on the table (mirroring #3016's implement/wrap/observe decision):
**metadata-only inspection**, **route-to-tuned-engine-first**, and **native FP4 path**.

**Recommendation: metadata-only inspection FIRST, then route-to-tuned-engine for serving;
native FP4 is REFUSED until the metadata fixture + parity oracle land.**

Rationale:

1. **Metadata-only inspection is the only immediately landable, weight-free, CPU-only witness.**
   It reuses seams that already exist and are proven (the GLM name classifiers, `isQuantWeight`
   admission, `UnsupportedArchError` fail-closed, the MXFP4 block/scale detector). It carries
   zero device dependency and zero fabricated numbers — it is the acceptance gate.
2. **Route-to-tuned-engine-first is the right *serving* disposition** once the detector exists:
   fak owns admission + routing + the KV/long-context budget (`deepseek_budget.go`), and the
   tuned engine (SGLang FP4 indexer / w4a4 MegaMoE / FlashInfer NVFP4) owns the FP4 kernel. This
   avoids fak claiming a native FP4 GEMM it cannot yet witness, while still serving V4.
3. **Native FP4 is refused now** — a new `compute.Dtype` (NVFP4/E2M1) + per-hardware GEMM is a
   real kernel program that must not be started before a metadata fixture proves the format and
   a parity oracle proves the numerics. This plan fences it (see below).

Any model-loader follow-on **must declare its prior art** (matrix below) per the issue.

## Seam map — V4 quant requirement → fak seam (`path:line`) or proposed

| V4 requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| Dtype enum with FP8 | `internal/compute/compute.go:62` `FP8` (E4M3/E5M2; variant in QuantSpec) | **Partial** — FP8 is a first-class tag but *widened at load*, not a native FP8 GEMM |
| Native FP4 / NVFP4 / E2M1 dtype | *No enum member* — `Dtype` list ends at `Q4_K` (`compute.go:56-63`) | **Gap** — a native FP4 dtype is a new enum member + kernel; fenced |
| 4-bit weight decode (existing) | MXFP4 decode `internal/model/safetensors.go` `decodeMXFP4Blocks`; block/scale pairing `internal/model/safetensors_quant.go:286`; SIMD `internal/ggufload/gguf_dequant_simd_amd64.go` | **Partial fit** — decode-to-**f32**, not a native FP4 GEMM; MXFP4 ≠ V4 NVFP4/E2M1 |
| Per-tensor admission | `internal/model/safetensors_quant.go:30` `isQuantWeight`; resolver `internal/model/tensor_resolver.go:464` `deepSeekMLASpec` (SCAFFOLD, #25) | **Fit** — admission lives here; V4 needs the class→precision table above wired in |
| Batched-expert split (the FP4 bulk) | `internal/ggufload/gguf_glm_tensors.go:228/265/326` | **Fit** — expert splitters exist; the raw-quant resident split `:326` is where an FP4 block format plugs in |
| Fail-closed on unknown quant/arch | `internal/model/arch_support.go:33` `UnsupportedArchError`, `:69` | **Strong fit** — the typed-refusal pattern is exactly how an unknown FP4 class fails closed |
| 4-bit GEMM backends (template) | `internal/model/awq.go:59` `awqDequantRow` + `awq_amd64.go` / `awq_cuda.go` / `awq_scalar.go` | **Fit as template** — per-hardware dequant/GEMM split an FP4 path mirrors; not FP4 itself |
| GEMM backend by hardware class | `internal/metalgemm/`, `internal/amdgpu/`, `awq_cuda.go`, `awq_amd64.go` | **Fit** — hardware-class seams exist; FP4 dispatch extends them |
| Mixed-precision residency policy | `internal/model/dynamic_precision.go` `DynamicPrecisionPolicy` (test `dynamic_precision_test.go`) | **Fit** — a precision-tier policy exists; FP4-expert / FP8-rest residency is a new policy over it |
| Long-context / weight-storage budget | `internal/gateway/deepseek_budget.go` (FP4=0.5, FP8=1.0 B/param bounds) | **Fit** — the mixed FP4/FP8 storage bound is already MODELED and fail-closed |
| DeepSeek reasoning/usage conformance | `internal/agent/deepseek_conformance_test.go`, `deepseek_reasoning_test.go`, `deepseek_usage_test.go` | **Fit** — the API-side V4 conformance already exists; unaffected by quant, cross-referenced only |

## Prior-art matrix (a loader follow-on must declare this)

Per the issue grounding (SGLang V4 roadmap <https://github.com/sgl-project/sglang/issues/23602>):

| Prior art | What it provides | Relation to fak |
|---|---|---|
| **SGLang FP4 indexer** | FP4 QK path for the lightning indexer | The indexer-FP4 parity oracle (ties to #3016) |
| **W4A16 Hopper / NVFP4 checkpoint** | FP4 weight formats + Hopper support | The format fak's proposed new dtype would target |
| **MXFP4 / FlashInfer** | MXFP4 blocks + FP4 GEMM primitives | fak decodes MXFP4→f32 today; FlashInfer is the native-kernel candidate |
| **w4a4 MegaMoE** | 4-bit fused MoE | The FP4 expert-dispatch reference (ties to #3018) |
| **vLLM FP4 recipes** | Reference config + numerics | Secondary numeric oracle (Reference A) |

`fak sota` / `internal/sotamatrix` is the in-repo home for keeping this matrix live; this doc
records the snapshot, it does not re-run the tool.

## Honest fences (what is NOT decided or built)

- **No native FP4 dtype** — `compute.Dtype` has FP8 and integer 4-bit (I4/Q4_K), not
  FP4/NVFP4/E2M1 (`internal/compute/compute.go:56-63`).
- **MXFP4 is decode-to-f32**, not a native FP4 GEMM, and is a *different format* from V4's
  NVFP4 expert FP4.
- **FP8 is widened at load**, not a native FP8 GEMM — the "FP8 rest" is correctness-only today.
- **No V4 metadata detector / admission-table wiring yet** — that is this ticket's witness.
- **No FP4 indexer QK path** — shared gap with #3016.
- **No numerical-parity oracle run** — designed here, not executed; every device/serving row is
  **host-gated** and no number in this doc is WITNESSED.
- **`deepSeekMLASpec` is still a #25 scaffold** (`tensor_resolver.go:464`) — the resolver's
  DeepSeek MLA required-tensor table is deliberately minimal until a real manifest pins it.

## Acceptance mapping

| Issue acceptance / witness criterion | Where satisfied in this doc |
|---|---|
| Per-tensor precision table (experts/shared/router/head/attention/norm/MTP + indexer QK) | "Per-tensor precision admission table" — 8-class table with expected FP4/FP8 per class and the real fak seam each routes through |
| Checkpoint-metadata detection for FP4+FP8 mixed weights | "Checkpoint-metadata detection" — Go, weight-free, reuses MXFP4 block/scale + GLM name classifiers, fails closed via `UnsupportedArchError` |
| Backend-by-hardware options | "Dequant / GEMM backend options by hardware class" — x86 / CUDA / ROCm / Metal / portable rows, each mapped to an existing AWQ/MXFP4 seam, all host-gated |
| FP4 indexer QK path requirements | "FP4 indexer QK path requirements" — 4 requirements over `dsa_index.go` scores/top-k/digest |
| Numerical-parity oracle design | "Numerical-parity oracle design" — vLLM/SGLang + HF-metadata references, selection-set digest for the indexer, closed provenance vocabulary, fails closed |
| Recommended disposition (route vs metadata-only vs native) — decide one | "Recommended disposition — DECISION" — metadata-only inspection first → route-to-tuned-engine for serving; native FP4 refused, with rationale |
| Model-loader follow-on must declare prior art | "Prior-art matrix" — SGLang FP4 indexer / W4A16-Hopper / MXFP4-FlashInfer / w4a4-MegaMoE / vLLM |
| No fabricated performance numbers | House rules honored: only MODELED (storage bound) and host-gated (device/serving) figures; no WITNESSED number emitted |
| Parent epic + sibling docs cross-linked | Header — #3006 parent; siblings #3016 attention seam map, #3018 MoE dispatch, self-host baseline runbook |

## Next rungs

1. Land the **Go metadata fixture + per-tensor admission table + fail-closed test** (the
   witness) — no weights, reusing the GLM classifiers + `UnsupportedArchError`.
2. Wire the admission table into `tensor_resolver.go` (promote `deepSeekMLASpec` past its #25
   scaffold) and the safetensors/GGUF admit gates.
3. Land the **numerical-parity oracle** against an external FP4 reference (host-gated).
4. **Only then** file the native-FP4-GEMM / route-to-engine follow-on, declaring its prior art
   per the issue's done-condition.

## Sources (researched July 2026)

- DeepSeek V4 Pro HF model card — <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
  (mixed FP4 MoE-expert + FP8-rest precision; FP4 indexer QK).
- DeepSeek V4 technical report — <https://arxiv.org/html/2606.19348v1> (FP4 QAT for experts +
  indexer; architecture constants).
- SGLang DeepSeek V4 roadmap — <https://github.com/sgl-project/sglang/issues/23602> (FP4
  indexer, W4A16 Hopper, MXFP4/FlashInfer, NVFP4 checkpoint, w4a4 MegaMoE prior art).
- In-repo grounding (read 2026-07-09 on `main`): `internal/compute/compute.go`,
  `internal/model/safetensors_quant.go`, `internal/model/tensor_resolver.go`,
  `internal/model/arch_support.go`, `internal/model/dsa_index.go`,
  `internal/model/awq.go`, `internal/ggufload/gguf_glm_tensors.go`,
  `internal/ggufload/dequant_mxfp4_test.go`, `internal/gateway/deepseek_budget.go`,
  `internal/agent/deepseek_conformance_test.go`.
</content>
</invoke>
