# Canonical named-tool catalog ablation

**Run date:** 2026-08-16 UTC
**Hardware:** one NVIDIA L4-class accelerator
**Models:** Qwen2.5-1.5B-Instruct FP16 and SmolLM2-1.7B-Instruct FP16
**Raw artifact:** [`2026-08-16-canonical-named-catalog-ablation.json`](../benchmarks/model-sensitivity/2026-08-16-canonical-named-catalog-ablation.json)
**Raw SHA-256:** `0ad557364a593c05aa122b497ec3ca60ace0d2087494164859e444e6e0ddaf38`

## Verdict

Canonicalizing a catalog while retaining meaningful tool names removed all measured source-order sensitivity **and improved both small models**. Across four independently reordered inputs, Qwen2.5-1.5B produced byte-identical outputs and scored **72/96**, up from 59/96 with raw catalogs. SmolLM2-1.7B likewise became fully invariant and scored **68/96**, up from 66/96. Both used fewer output tokens than their raw-catalog baselines.

This is the strongest tested representation for Qwen2.5-1.5B and the first one in this series to improve semantics, tokens, and order robustness together. It is still not execution-admissible: the unique-request scores were only 18/24 and 17/24, with systematic exact-value failures.

## Question

The stable-ID study proved that canonical preprocessing can eliminate catalog-order variation, but opaque IDs collapsed SmolLM2-1.7B quality. This follow-on isolates the ID indirection: sort and serialize the catalog canonically, but preserve ordinary tool names in the prompt and output.

## Protocol

- Requests and tools: the same 24 held-out paraphrases and 24-tool catalog as the prior held-out studies.
- Source orders: original input, reverse input, and deterministic shuffles with seeds 20260816 and 6692.
- Canonicalization: sort tools lexicographically by name and serialize with sorted JSON keys and compact separators.
- Output: `{"tool":"name","args":{...}}`, constrained by the same per-tool exact JSON Schema strategy as the raw-catalog baselines.
- Decode: greedy, 128-token cap, `lm-format-enforcer==0.11.3` under Transformers 5.14.1.
- Work: 24 requests × four source orders × two models = **192 calls**.
- Runner SHA-256: `53dfaf4a52cc00e23220d9150486ac948e0f26122cc262f3c92c5e566ba18178`.

The source catalogs have different digests. All four canonical catalogs have SHA-256 `d3eb3d474f154b7ee6243f6a2975f44f4c191130869ab586e36ab169877e31c3`, and every request's prompt digest is identical across source orders.

## Results

| Model | Canonical valid | Canonical exact | Raw-catalog exact | Delta exact | Canonical tokens | Raw-catalog tokens | Delta tokens | Byte-identical across orders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 96/96 | **72/96** | 59/96 | **+13** | 2,776 | 2,767 | +9 | 24/24 |
| SmolLM2-1.7B | 96/96 | **68/96** | 66/96 | **+2** | 3,052 | 3,253 | **-201** | 24/24 |

Each source-order arm repeated the same result:

| Model | Exact per source order | Valid per source order | Output tokens per source order |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 18/24 | 24/24 | 694 |
| SmolLM2-1.7B | 17/24 | 24/24 | 763 |

The first arm includes loading and warmup. Across the three warm arms, median end-to-end latency was 1.161–1.172 seconds for Qwen and 0.834–0.840 seconds for SmolLM. These synchronized in-process timings are runner-specific, not production-serving claims.

## Interpretation

### Canonicalization, not opaque IDs, is the useful intervention

The named and stable-ID studies used the same source-order matrix and canonical preprocessing. Both attained exact output invariance, so opaque IDs added no order-robustness benefit. Their semantic outcomes diverged sharply:

| Model | Raw catalogs | Canonical names | Canonical stable IDs |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 59/96 | **72/96** | 60/96 |
| SmolLM2-1.7B | 66/96 | **68/96** | 16/96 |

Meaningful names preserve the model's learned association between operation semantics and output tokens. Canonical preprocessing removes arbitrary upstream order without forcing a second opaque-ID association.

### Remaining errors are prompt/value failures

With order variation eliminated, the misses are easy to localize. Qwen's six unique misses were all search or email tasks. SmolLM's seven were three searches, one order lookup, and three emails. Typical outputs chose the expected operation but paraphrased an argument instead of reproducing the exact expected value. The representation is robust to source ordering, but neither model meets an exact execution threshold.

### Recommended design direction

Canonicalize named catalogs before model inference. Do not expose arbitrary upstream ordering, and do not replace meaningful names with opaque IDs for these models. The next optimization target is exact argument extraction or a deterministic argument-binding layer after semantic tool selection, not another confidence threshold over order-sensitive raw prompts.

## Reproduction and provenance

The raw artifact captures environment and decode settings, full model and weight hashes, all source/canonical/prompt digests, the exact schema and canonical catalog, every raw output and parsed call, mapped tool, exact validation result, token-confidence trace, synchronized timings, aggregate summaries, and per-task cross-order invariance.

Independent validation reparsed all 192 outputs, recomputed schema validity and exact correctness, recomputed summaries, and confirmed every digest and byte-level output invariant. The local raw SHA-256 matched the accelerator copy.

Baseline provenance:

- Qwen2.5-1.5B raw-catalog direct: [`2026-08-16-heldout-selective-adjudication.json`](../benchmarks/model-sensitivity/2026-08-16-heldout-selective-adjudication.json), SHA-256 `b215141aae5fa4bc17bf2ebdbf420f7bff57d24c3459a9e6ca57ace498f7e400`.
- SmolLM2-1.7B raw-catalog constrained: [`2026-08-16-smollm17-heldout-order-robustness.json`](../benchmarks/model-sensitivity/2026-08-16-smollm17-heldout-order-robustness.json), SHA-256 `626e493a9d5369a8187f668c7eea5102e8972837f45f661d6481cb08be5d0fd7`.
- Stable-ID comparison: [`2026-08-16-stable-id-catalog-ablation.json`](../benchmarks/model-sensitivity/2026-08-16-stable-id-catalog-ablation.json), SHA-256 `b5f7524e47e2596a001aa7232c2ce2be4c213442dc2af7e096d072da421e59c8`.

## Scope

This is a synthetic exact-match routing set, not a production distribution. The same canonical prompt was intentionally repeated four times to prove source-order erasure, so aggregate gains are not four independent samples. The evidence supports deterministic named-catalog canonicalization for this tested seam, not universal model quality or execution admission.
