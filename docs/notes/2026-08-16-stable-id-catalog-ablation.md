# Stable-ID canonical catalog ablation

**Run date:** 2026-08-16 UTC
**Hardware:** one NVIDIA L4-class accelerator
**Models:** Qwen2.5-1.5B-Instruct FP16 and SmolLM2-1.7B-Instruct FP16
**Raw artifact:** [`2026-08-16-stable-id-catalog-ablation.json`](../benchmarks/model-sensitivity/2026-08-16-stable-id-catalog-ablation.json)
**Raw SHA-256:** `b5f7524e47e2596a001aa7232c2ce2be4c213442dc2af7e096d072da421e59c8`

## Verdict

Canonical serialization with stable tool IDs completely removed the measured catalog-order seam: both models produced byte-identical outputs for all **24/24** requests across four independently reordered source catalogs. It did **not** provide a useful semantic-routing improvement.

Qwen2.5-1.5B moved only from 59/96 to **60/96** exact while spending 765 more output tokens. SmolLM2-1.7B collapsed from 66/96 to **16/96**, despite identical prompts and outputs across source orders. Stable IDs are therefore an effective representation-level order-invariance mechanism, but they are not an execution-admissible routing strategy for these small models and this prompt.

## Question

Earlier held-out runs found that equivalent tool-catalog ordering changed small-model routing. This ablation asks whether separating a stable ID index from argument schemas and canonicalizing both before prompting can eliminate upstream-order sensitivity without sacrificing semantic selection.

## Protocol

- Requests: the same 24 held-out paraphrases used by the selective-adjudication and SmolLM2-1.7B order-robustness studies.
- Tools: the same 24-tool catalog.
- Source orders: original input, reverse input, and deterministic shuffles with seeds 20260816 and 6692.
- Canonicalization: sort tools lexicographically by name, assign `T01` through `T24`, then serialize a `tool_index` and per-ID `schemas` map with sorted JSON keys and compact separators.
- Output: `{"tool_id":"Txx","args":{...}}`, constrained by a 24-branch JSON Schema with an exact ID and exact argument schema per branch.
- Validation: map the selected ID back to its tool name, require exact keys and argument types, then compare exact tool and arguments with the expected call.
- Decode: greedy, 128-token cap inherited from the held-out runner, with `lm-format-enforcer==0.11.3` under Transformers 5.14.1.
- Work: 24 requests × four source orders × two models = **192 calls**.
- Original accelerator runner SHA-256: `9711c7265ad71f0d78967323be2823b4945165076083adcffb58e37eb2020e28`.

The four source catalogs have different SHA-256 digests in the raw artifact. After canonicalization all four have digest `6a04ededf496470e45c6aabbea003ccaf90ed58411897ae54979e58bde1a5077`. Per-request prompt digests are also identical across all four source orders.

## Results

| Model | Stable-ID valid | Stable-ID exact | Raw-catalog exact | Delta exact | Stable-ID tokens | Raw-catalog tokens | Delta tokens | Byte-identical across orders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 96/96 | **60/96** | 59/96 | +1 | 3,532 | 2,767 | +765 | 24/24 |
| SmolLM2-1.7B | 92/96 | **16/96** | 66/96 | -50 | 3,876 | 3,253 | +623 | 24/24 |

Because every canonical prompt was identical and greedy decoding was deterministic, each model repeated the same 24 outputs in every source-order arm:

| Model | Exact per source order | Valid per source order | Output tokens per source order |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 15/24 | 24/24 | 883 |
| SmolLM2-1.7B | 4/24 | 23/24 | 969 |

The first source-order arm includes model loading and warmup, so its latency is not directly comparable with the later warm arms. Across the three warm arms, Qwen median end-to-end latency was 1.372–1.374 seconds; SmolLM median was 1.032–1.038 seconds. These synchronized in-process timings describe this runner only, not production serving.

## Interpretation

### The order seam is removable before inference

The invariance result is exact rather than statistical: canonical catalog digests match, every task's prompt digest matches, and all 24 raw outputs from each model match byte-for-byte across the four source orders. Systems that need order neutrality should canonicalize before inference rather than ask a model to ignore arbitrary order.

### Opaque IDs change the routing problem

The ID layer did not preserve model quality. Qwen selected the correct exact call on 15/24 unique requests, the same canonical-order score as its prior raw-name baseline, yielding only one additional replicated correct call across the four-order aggregate. SmolLM selected only 4/24 unique requests correctly. Its four structural failures were the same max-token output repeated once per order.

The likely mechanism is representational: the model must first associate an opaque token such as `T16` with a named operation and then generate it, instead of directly emitting a semantically meaningful tool name. This run establishes the outcome, not a causal attribution; a named canonical catalog without opaque IDs is the clean next ablation.

### Reject stable IDs as the small-model route

This representation solves one robustness property while worsening the overall route. It does not make either model execution-admissible, and the SmolLM regression is decisive. Do not replace named-tool routing with opaque stable IDs based on the order-invariance result alone.

## Reproduction and provenance

The raw artifact includes:

- full environment and decode settings;
- canonical and source-catalog digests;
- every per-task prompt digest;
- exact output schema and canonical catalog;
- model configuration/tokenizer/weight file hashes;
- all 192 raw outputs, parsed IDs, mapped tool names, validity, exact correctness, token counts, confidence traces, and synchronized timings;
- per-order and aggregate summaries plus cross-order invariance details.

Independent local validation reparsed every output, remapped every ID, recomputed validity and exact correctness, recomputed summaries, and confirmed all catalog, prompt, raw-output, and parsed-output invariants. The original pulled raw SHA-256 matched the accelerator copy. A metadata-only correction changed the decode label from 96 to the actual inherited 128-token cap; `artifact_corrections` records the original hash and correction, and no result row or summary changed.

Baseline provenance:

- Qwen2.5-1.5B raw-catalog direct: [`2026-08-16-heldout-selective-adjudication.json`](../benchmarks/model-sensitivity/2026-08-16-heldout-selective-adjudication.json), SHA-256 `b215141aae5fa4bc17bf2ebdbf420f7bff57d24c3459a9e6ca57ace498f7e400`.
- SmolLM2-1.7B raw-catalog constrained: [`2026-08-16-smollm17-heldout-order-robustness.json`](../benchmarks/model-sensitivity/2026-08-16-smollm17-heldout-order-robustness.json), SHA-256 `626e493a9d5369a8187f668c7eea5102e8972837f45f661d6481cb08be5d0fd7`.

## Scope

This is a routing-sensitive synthetic held-out set, not a production traffic distribution. Exact argument matching intentionally penalizes semantically nearby normalization. The test isolates one catalog representation and two small instruction-tuned models on one accelerator class; it does not claim stable IDs universally harm tool routing.


