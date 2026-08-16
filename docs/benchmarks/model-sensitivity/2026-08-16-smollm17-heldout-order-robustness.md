# SmolLM2-1.7B held-out catalog-order robustness

**Run date:** 2026-08-16 UTC  
**Hardware:** one NVIDIA L4-class accelerator  
**Model:** SmolLM2-1.7B-Instruct, FP16  
**Raw artifact:** [`2026-08-16-smollm17-heldout-order-robustness.json`](2026-08-16-smollm17-heldout-order-robustness.json)  
**Raw SHA-256:** `626e493a9d5369a8187f668c7eea5102e8972837f45f661d6481cb08be5d0fd7`

## Verdict

The prior SmolLM2-1.7B 24/24 result did **not** survive held-out paraphrases. Freeform routing scored 15/24 to 17/24 across four equivalent catalog orders; grammar-constrained routing scored 15/24 to 18/24. Aggregate constrained correctness was **66/96**, better than the direct Qwen2.5-1.5B baseline's 59/96 on the same held-out matrix, but still far below execution admission.

SmolLM2-1.7B was substantially more order-stable than the earlier 0.5B cascade routes: 19/24 constrained calls were invariant across all four orders. However, five calls changed with order, and correctness changed on four of those. Neither grammar constraints nor a strong single-order score establishes semantic robustness.

## Why this test

SmolLM2-1.7B with the concise contract was the smallest prior arm to achieve 24/24 on the original requests and canonical catalog. The held-out selective-routing study then showed that equivalent catalog order can drastically change small-model predictions. This test applies the same 24 unseen paraphrases and four catalog orders to that former winner.

## Protocol

- Held-out requests: exactly the 24 paraphrases captured in `2026-08-16-heldout-selective-adjudication.json`.
- Catalog: the same 24 tools and argument schemas in canonical, reversed, and deterministic seed-1729/seed-947 shuffled orders.
- Prompt: concise one-JSON-object contract.
- Arms:
  - **freeform:** ordinary greedy generation;
  - **constrained:** greedy generation filtered by the catalog-derived JSON Schema.
- Decode: temperature 0, 128 generated-token cap.
- Call order: freeform/constrained order alternated by task and catalog index.
- Baseline: direct constrained Qwen2.5-1.5B results from the held-out selective-routing artifact, SHA-256 `b215141aae5fa4bc17bf2ebdbf420f7bff57d24c3459a9e6ca57ace498f7e400`.

## Results

| Catalog order | Freeform valid | Freeform correct | Constrained valid | Constrained correct | Qwen2.5-1.5B direct |
|---|---:|---:|---:|---:|---:|
| Canonical | 24/24 | 15/24 | 24/24 | 15/24 | 15/24 |
| Reversed | 22/24 | **16/24** | 24/24 | **16/24** | 14/24 |
| Shuffle 1729 | 24/24 | **17/24** | 24/24 | **17/24** | 13/24 |
| Shuffle 947 | 23/24 | 17/24 | 24/24 | **18/24** | 17/24 |
| **Aggregate** | **93/96** | **65/96** | **96/96** | **66/96** | **59/96** |

Output tokens across all 96 calls were identical at 3,253 for both SmolLM arms. Qwen2.5-1.5B used 2,767, so constrained SmolLM2-1.7B spent 486 more output tokens for seven additional exact calls.

Median per-call latency:

| Catalog order | Freeform | Constrained |
|---|---:|---:|
| Canonical | 944 ms | 1,181 ms |
| Reversed | 963 ms | 1,195 ms |
| Shuffle 1729 | 973 ms | 1,193 ms |
| Shuffle 947 | 977 ms | 1,197 ms |

Grammar filtering added 218–237 ms median latency in this in-process implementation.

## Prompt-generalization failure

The original concise-contract witness was 24/24. With canonical catalog order but unseen paraphrases, both arms fell to 15/24. The nine canonical misses included:

- value normalization: `two factor auth` instead of exact `two-factor authentication`;
- wrong confusable tool: `search_orders` instead of `get_order`;
- title truncation or substitution in ticket creation;
- missing or changed email subject text;
- string-typed refund amounts in freeform;
- cancellation routed to a lookup or search tool.

Several are exact-value failures rather than broad intent failures, but the tool boundary requires the exact declared arguments.

## Catalog-order sensitivity

Parsed calls invariant across all four orders:

- freeform: **18/24**;
- constrained: **19/24**.

The constrained arm's five order-sensitive tasks were:

- `search-02`—different but always incorrect query text;
- `order-01`—incorrect in canonical order, correct in the other three;
- `ticket-03`—incorrect in canonical order, correct in the other three;
- `email-02`—correct only under seed-947 order;
- `refund-02`—incorrect only in reversed order.

Thus, equivalent ordering moved constrained correctness from 15/24 to 18/24. This is milder than the earlier 0.5B draft's 2/24 invariant calls, but still operationally material.

## Effect of grammar constraints

- Canonical and seed-1729 arms produced identical parsed calls between freeform and constrained decoding.
- Across the whole matrix, constraints changed only four parsed calls.
- Constraints raised structural validity from 93/96 to 96/96 and exact correctness from 65/96 to 66/96.
- Under reversed order, constraints fixed one malformed/wrong call but also changed one correct freeform call into an incorrect legal call.
- Under seed-947, constraints repaired one additional call.

Grammar constraints therefore provided a small net quality gain and complete structural validity, but did not address the dominant semantic failures.

## Comparison with smaller routes

On the identical 96-case held-out matrix:

| Route | Exact correct | Output tokens |
|---|---:|---:|
| Qwen2.5-0.5B draft | 21/96 | 3,353 |
| Frozen 0.5B→1.5B selective cascade | 41/96 | 5,202 |
| Always-on candidate adjudication | 47/96 | 6,500 total path |
| Direct constrained Qwen2.5-1.5B | 59/96 | **2,767** |
| Constrained SmolLM2-1.7B | **66/96** | 3,253 |

SmolLM2-1.7B is the strongest tested small-model route on correctness, but direct Qwen2.5-1.5B remains lower-token. Neither is execution-admissible.

## Interpretation

1. **The 24/24 result was prompt-specific.** Held-out paraphrases reduced canonical correctness to 15/24 without changing tools or expected calls.
2. **SmolLM2-1.7B is more order-robust, not order-invariant.** Nineteen constrained calls were stable, leaving five order-sensitive cases.
3. **Grammar constraints solve syntax more than semantics.** They supplied three missing valid outputs and only one net exact call across 96 cases.
4. **A canonical order is part of the route contract.** Reordering tools cannot be assumed behavior-preserving for this model.
5. **The strongest small model still needs semantic rejection.** A 66/96 aggregate score cannot cross a fail-closed tool boundary without another reliable signal.
6. **Exact-value evaluation is intentionally strict.** Normalization that changes a requested query, title, subject, identifier, or amount is not silently accepted.

## Reproduction and validation

The raw artifact contains exact held-out tasks, all four ordered catalogs, model configuration and weight hashes, schema, call order, every freeform/constrained output, confidence trace, parse, correctness result, token count, synchronized timing, and order-invariance IDs.

Post-capture checks established:

- four catalog runs, two arms, and 24 rows per arm/run;
- every validity, correctness, token, and latency summary recomputes from row data;
- order-invariance IDs recompute by comparing parsed calls for each task across all four runs;
- freeform/constrained correction and regression counts recompute independently;
- the comparison baseline digest matches the committed held-out selective-routing artifact;
- local and accelerator SHA-256 digests match.

## Bounded deployment conclusion

Withdraw the earlier implication that SmolLM2-1.7B's 24/24 canonical result establishes a generally reliable route. It is the best tested small-model route on this held-out matrix, but **66/96 is not admissible**. Future routing claims must include held-out paraphrases and catalog permutations by default; optimizing the original canonical prompt further would not address the observed robustness failure.
