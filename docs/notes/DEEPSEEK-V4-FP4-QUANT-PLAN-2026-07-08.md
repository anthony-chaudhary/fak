# DeepSeek V4 FP4 expert weights + FP4 indexer path — quantization support plan

**2026-07-08.** Issue **#3019**, parent epic **#3006** (native DeepSeek-V4 kernel track).
**Support plan + fixture plan only** — no native FP4 GEMM lands here, no in-tree weights, and
per the issue any tooling is **Go**, not Python. Current-state claims are witnessed against
the exact `path:line` cited (read 2026-07-08 on `main`).

## Thesis — V4 is a *mixed-precision checkpoint*, not another FP8 model

V4's checkpoint is mixed precision: **MoE expert parameters are FP4, most other parameters
are FP8**, and the **attention indexer QK path is also FP4** (post-training includes FP4
quantization-aware training for the expert weights and the indexer). fak's loader has an FP8
*dtype tag* and a 4-bit *decode-to-f32* path (MXFP4, AWQ), but it has **no native FP4 GEMM
dtype, no NVFP4/E2M1, and no per-tensor admission that fails closed on an unknown FP4 tensor
class**. So this must be treated as a **new quantization/backend problem** — starting with a
metadata detector and an admission plan, not a dequant kernel.

## The V4 quant facts that drive the plan (from the issue grounding)

Sources: HF model card https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro and the V4 report
https://arxiv.org/html/2606.19348v1 (per #3019 Grounding).

| V4 Pro quant fact | Value | Consequence |
|---|---|---|
| MoE expert parameters | **FP4** | the parameter bulk needs an FP4 GEMM / dequant path |
| Most other parameters | **FP8** | fak has an FP8 dtype tag but widens at load |
| Attention indexer QK | **FP4** | the lightning indexer (#3016) needs an FP4 QK path |
| Post-training | FP4 QAT for experts + indexer | numerics must match a QAT reference, not naive round |

## Seam map — V4 quant requirement → fak seam (`path:line`) or proposed

| V4 requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| **Dtype enum with FP8** | `internal/compute/compute.go:53` `Dtype`; `:62` `FP8` (E4M3/E5M2, variant in QuantSpec) | **Partial** — FP8 is a first-class dtype tag, but it is *widened at load* today, not a native FP8 GEMM. |
| **Native FP4 / NVFP4 / E2M1 dtype** | *No enum member* — the `Dtype` list ends at `Q4_K`; there is no FP4/NVFP4/E2M1 | **Gap.** A native FP4 dtype is a **new** enum member + kernel; fence it. |
| **4-bit weight decode (existing)** | `internal/model/safetensors.go:497` `decodeMXFP4Blocks`, `:492` `mxfp4Values`, `:461` MXFP4 block/scale pairing; quantize path `internal/model/safetensors_quant.go:286` `quantizeMXFP4TensorInto` | **Partial fit** — fak *decodes* MXFP4 blocks **to f32** at load (gpt-oss). It is a decode-to-f32, **not** a native FP4 GEMM, and MXFP4 ≠ V4's NVFP4/E2M1 expert format. |
| **4-bit GEMM backends (existing)** | `internal/model/awq.go:59` `awqDequantRow` (+ `awq_amd64.go`, `awq_cuda.go`, `awq_scalar.go`) | **Fit as a template** — AWQ shows the per-hardware dequant/GEMM backend split (scalar / AVX2 / AVX512 / CUDA) an FP4 path would mirror; it is not FP4 itself. |
| **Per-tensor admission** (experts / shared / router / head / attn / norm / MTP) | `internal/model/tensor_resolver.go` (per-family required-tensor tables; gpt-oss MXFP4 scaffold `:440`) | **Fit** — the resolver is where per-tensor class → dtype admission lives; V4 needs a table mapping each class to FP4 vs FP8. |
| **Fail-closed on unknown quant/arch** | `internal/model/arch_support.go:33` `UnsupportedArchError`, `:69` `refuseUnsupportedHybridArch` | **Strong fit** — the typed-refusal-before-mid-request-panic pattern is exactly how an unknown FP4 tensor class should fail closed. |
| **GEMM backend by hardware class** | `internal/metalgemm/` (Metal), `internal/amdgpu/` (ROCm facts), `internal/model/awq_cuda.go` / `awq_amd64.go` (CUDA / x86) | **Fit** — the hardware-class backend seams exist; FP4 dispatch would extend them. |
| **Mixed-precision residency policy** | `internal/model/dynamic_precision.go:8` `PrecisionTier`, `:36` `DynamicPrecisionPolicy` | **Fit** — a precision-tier policy object exists; FP4-expert / FP8-rest residency is a new policy over it. |

**Key honest fact:** the `compute.Dtype` enum (`compute.go:53`) is `F32, F16, BF16, Q8_0, I8,
I4, FP8, Q4_K` — **there is no FP4/NVFP4/E2M1**. fak's only 4-bit paths are `I4`/`Q4_K`/AWQ
(integer 4-bit) and MXFP4-decoded-to-f32. V4's FP4 expert format is genuinely new.

## Checkpoint-metadata detection (the first deliverable)

A **Go** fixture (not Python, per the issue) that parses **representative V4 tensor
metadata** — tensor names, shapes, dtypes — *without downloading full weights*, and:
- Classifies each tensor into: routed expert, shared expert, router/gate, head, attention,
  norm, MTP, **indexer QK**.
- Asserts the expected precision per class (experts → FP4, indexer QK → FP4, rest → FP8).
- **Fails closed** on an unrecognized FP4 tensor class (reusing the `UnsupportedArchError`
  pattern) rather than silently mis-loading it.

This can be sourced from the HF model card's tensor index; no weights, no in-tree binaries.

## The route / inspect / native decision (recommended)

Three options, increasing commitment (mirrors #3016's decision):

1. **Metadata-only inspection first (recommended start).** Land the detector + admission
   table + fail-closed test. Zero dequant, zero GEMM. This is the acceptance gate.
2. **Route to a tuned engine** (SGLang FP4 indexer / w4a4 MegaMoE) — fak owns admission +
   routing, the engine owns the FP4 kernel.
3. **Native FP4 path** — a new `compute.Dtype` (E2M1/NVFP4) + per-hardware GEMM, **only**
   after the metadata fixture and a numerical-parity oracle land. This plan **refuses** the
   native path until then.

## Numerical parity oracle plan

Parity against **vLLM/SGLang FP4** output or **HF metadata fixtures** — the FP4 QAT numerics
mean naive round-to-FP4 will *not* match; the oracle must compare against a real FP4 reference,
not a fak-internal re-derivation. Recorded here as a plan; the fixture is the witness that
must land before any native ticket is filed.

## Prior-art matrix

Per the issue grounding (SGLang V4 roadmap, https://github.com/sgl-project/sglang/issues/23602):

| Prior art | What it provides | Relation to fak |
|---|---|---|
| **SGLang FP4 indexer** | FP4 QK path for the lightning indexer | Ties to #3016; the indexer-FP4 oracle. |
| **W4A16 Hopper / NVFP4 checkpoint** | FP4 weight formats + Hopper support | The format fak's new dtype would target. |
| **MXFP4 / FlashInfer** | MXFP4 blocks + FP4 GEMM primitives | fak decodes MXFP4→f32 today; FlashInfer is the native-kernel candidate. |
| **w4a4 MegaMoE** | 4-bit fused MoE | Ties to #3018 (FP4 expert dispatch). |

## Honest fences (what is NOT decided or built)

- **No native FP4 dtype** — `compute.Dtype` has FP8 and integer 4-bit, not FP4/NVFP4/E2M1.
- **MXFP4 is decode-to-f32**, not a native FP4 GEMM, and is a *different format* from V4's
  expert FP4.
- **No V4 metadata detector / admission table yet** — that is this ticket's witness.
- **No FP4 indexer QK path** — shared gap with #3016.
- **No numerical parity oracle run** — planned, not executed.

## Next rungs

1. Land the **Go metadata fixture + per-tensor admission table + fail-closed test** (the
   witness) — no weights.
2. Wire the admission table into `tensor_resolver.go` and reuse `UnsupportedArchError`.
3. Land the **numerical-parity oracle** against an external FP4 reference.
4. **Only then** file the native-FP4-GEMM / route-to-engine follow-on (per the issue's
   done-condition: a follow-on exists *only after* the fixture and oracle land).
