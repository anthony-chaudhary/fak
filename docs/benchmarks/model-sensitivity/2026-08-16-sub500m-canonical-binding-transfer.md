# Sub-500M canonical-routing transfer

**Run date:** 2026-08-16 UTC  
**Hardware:** one NVIDIA L4-class accelerator  
**Models:** Qwen2.5-0.5B-Instruct FP16 and SmolLM2-360M-Instruct FP16  
**Raw artifact:** [`2026-08-16-sub500m-canonical-binding-transfer.json`](2026-08-16-sub500m-canonical-binding-transfer.json)  
**Raw SHA-256:** `1fb90b5844ad79af03e2578965838d900a387e28ca6e44f99978535b2d36a94f`

## Verdict

The successful canonical named-tool plus deterministic-binding route does **not** transfer to the tested sub-500M models. Qwen2.5-0.5B scored **32/96** exact, or 8/24 unique requests. SmolLM2-360M scored **0/96** and never selected the expected tool on any unique request. The binder improved neither model because neither produced a correct-tool email call that was eligible for safe binding.

Canonicalization still did its narrower job: every model output and post-bind call was identical across all four upstream source orders. Order invariance is therefore necessary but not sufficient; at this model scale the dominant failure is semantic tool selection.

## Protocol

This is a frozen transfer test, not a newly tuned prompt:

- same 24 held-out requests and 24-tool catalog;
- same original, reversed, and two deterministic shuffled source orders;
- same lexicographic named-tool canonicalization;
- same concise output contract and per-tool JSON Schema constraint;
- same fail-closed `send_email` literal binder and six controls from #6964;
- greedy decode with a 128-token cap;
- 24 requests × four source orders × two models = **192 calls**.

Runner SHA-256: `9e3d43863fc0f655f840971f738fa50a3ce25798a8c13bd9b3b04aed62f1ee1f`.

## Results

| Model | Valid | Exact before | Exact after | Unique exact | Unique correct-tool selections | Output tokens | Bound / passthrough / rejected | Order-invariant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 92/96 | 32/96 | **32/96** | 8/24 | 8/24 | 2,628 | 0 / 92 / 4 | 24/24 |
| SmolLM2-360M | 96/96 | 0/96 | **0/96** | 0/24 | 0/24 | 3,012 | 0 / 96 / 0 | 24/24 |

Each source-order arm repeated exactly:

- Qwen2.5-0.5B: 23/24 structurally valid and 8/24 exact;
- SmolLM2-360M: 24/24 structurally valid and 0/24 exact.

Qwen's four aggregate rejections are one repeated request: on `email-04` it generated an invalid partial call, so the fail-closed binder rejected it. Its other email outputs selected unrelated tools and passed through unchanged. SmolLM selected unrelated but schema-valid tools for every request, so all calls passed through and none were bindable.

## Comparison with larger small models

| Route | Qwen family | SmolLM family |
|---|---:|---:|
| Sub-500M canonical + binder | Qwen2.5-0.5B: **32/96** | SmolLM2-360M: **0/96** |
| 1–2B canonical names | Qwen2.5-1.5B: 72/96 | SmolLM2-1.7B: 68/96 |
| 1–2B canonical + binder | Qwen2.5-1.5B: **84/96** | SmolLM2-1.7B: **80/96** |

The representation and binder gains depend on the model first making a viable semantic selection. They cannot compensate for the sub-500M models' dominant wrong-tool behavior on a 24-tool confusable catalog.

## Interpretation

- **Keep canonicalization:** it deterministically removes upstream-order variation even for weak models.
- **Do not admit these models:** structural validity from grammar constraints masks semantic failure, especially SmolLM2-360M's 96/96 valid but 0/96 exact result.
- **Do not broaden the binder:** changing unrelated selected tools or inventing arguments would violate its safety contract and turn a measured model failure into silent execution risk.
- **Practical floor observed here:** the useful route begins around the tested 1.5B–1.7B class, not below 500M, for this 24-tool catalog and exact-match contract.

## Reproduction and provenance

The raw artifact records the binder controls, environment, model/tokenizer/weight hashes, source and canonical digests, every prompt digest, all raw outputs and confidence traces, every pre/post call and binder decision, tokens, synchronized timings, summaries, and cross-order invariants.

Independent validation recomputed validity, exactness, binding counts, unique correct-tool selections, and all output invariants. It also verified that the binder never changed a model-selected tool. The local raw SHA-256 matched the accelerator copy.

Larger-model baseline: [`2026-08-16-deterministic-argument-binding-ablation.json`](2026-08-16-deterministic-argument-binding-ablation.json), SHA-256 `37ba35278664aacf0df1a033b3707dfb3065583e9475af32bc94206ee7d14296`.

## Scope

This is one held-out synthetic routing set with a confusable 24-tool catalog. It establishes non-transfer for these exact model checkpoints and frozen route, not a universal parameter-count law. A different training recipe or a much smaller candidate set may shift the floor, but that requires a separately captured comparison.
