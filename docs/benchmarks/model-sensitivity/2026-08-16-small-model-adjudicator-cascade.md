# A 0.5B draft with a 1.5B semantic adjudicator

**Run date:** 2026-08-16 UTC  
**Hardware:** one NVIDIA L4-class accelerator  
**Models:** Qwen2.5-0.5B-Instruct draft and Qwen2.5-1.5B-Instruct adjudicator, FP16 and simultaneously resident  
**Raw artifact:** [`2026-08-16-small-model-adjudicator-cascade.json`](2026-08-16-small-model-adjudicator-cascade.json)  
**Raw SHA-256:** `06813a163812d3413b93340664cd9b9381627366e2aee8cc4dd564423144da7a`

## Verdict

A grammar-constrained 1.5B model reviewing the 0.5B candidate reached **22/24** exact calls, versus **20/24** for direct 1.5B constrained routing and **13/24** for the 0.5B draft. The adjudicator corrected nine draft misses without regressing any draft hit, but cost 1,547 output tokens and about 2.21 seconds median warm path latency per request—more than twice the direct route's 708 tokens and 1.12 seconds.

This is evidence that candidate context can help a small adjudicator resolve some semantic ambiguities. It is not evidence for a net production gain: the cascade still missed 2/24, and no selective trigger was available because every constrained draft was structurally valid.

## Question and real alternatives

Grammar constraints exposed latent routing capability in Qwen2.5-0.5B but left 11 semantic misses. This study compares three real routes on the same 24-tool confusable catalog:

1. **0.5B draft:** constrained concise-contract generation.
2. **Direct 1.5B:** constrained generation from the original request—the next-best non-cascade baseline.
3. **Adjudicated cascade:** the 1.5B model sees the original request, catalog, and 0.5B candidate, and is told to verify the tool and every exact argument before returning a corrected call.

The adjudicator receives no expected answer, correctness flag, or oracle-generated error.

## Protocol

- Catalog/tasks: the established 24-tool confusable catalog and 24 exact expected calls.
- Grammar: the same catalog-derived JSON Schema used in the preceding constrained-decoding study.
- Decode: greedy, temperature 0, 128 generated-token cap.
- Repetitions: three complete rounds—72 drafts, 72 direct calls, and 72 adjudicated calls.
- Execution: both models resident on one accelerator; synchronous in-process generation; direct/adjudicated call order alternated by task and repetition to limit order bias; no network or external queue.
- Residency: 4,326,424,576 accelerator bytes reserved after both models loaded.
- Timing: synchronized accelerator generation. The first round includes lazy grammar/cache construction; rounds two and three represent warm in-process execution.

## Results

Every output, parse, correctness result, and token count reproduced exactly across all three rounds.

| Route | Structurally valid | Exact correct | Draft misses corrected | Draft hits regressed | Output tokens / 24 |
|---|---:|---:|---:|---:|---:|
| 0.5B draft | 24/24 | 13/24 | — | — | 787 |
| Direct 1.5B | 24/24 | 20/24 | 7 | 0 | **708** |
| 0.5B → 1.5B adjudication | 24/24 | **22/24** | **9** | 0 | 1,547 total path |

Warm latency from rounds two and three:

| Route | Median route latency | Median total path latency |
|---|---:|---:|
| 0.5B draft | 990-991 ms | 990-991 ms |
| Direct 1.5B | 1,124 ms | 1,124 ms |
| 1.5B adjudication alone | 1,176-1,178 ms | — |
| 0.5B → 1.5B cascade | — | **2,214-2,216 ms** |

The cascade used 839 more output tokens per 24 requests than direct 1.5B routing, a 118% increase. Its warm median path latency was about 1.09 seconds higher, a 97% increase. These are observed synchronous generation costs, not provider prices or concurrent throughput measurements.

## Correction anatomy

The 0.5B draft's 11 misses included wrong confusable tools and wrong exact values. The adjudicator corrected nine:

- all three address updates;
- all three emails, including restoring exact subject capitalization that direct 1.5B missed on one case;
- two ticket creations and the ticket-listing request;
- other draft hits were preserved byte-semantically.

The adjudicated route's two stable misses were:

1. `order-03`: returned `search_orders {"query":"A-17"}` rather than `get_order {"order_id":"A-17"}`—the same miss as direct 1.5B.
2. `ticket-02`: returned the correct `create_ticket` tool and priority but shortened `"Typo on homepage"` to `"Typo"`—also the same direct-route miss.

Candidate context fixed two direct-route mistakes (`list-02` and `email-03`) but did not fix the two ambiguities shared by both routes.

## Interpretation

1. **A small candidate can improve a larger small model.** The cascade gained two exact calls over direct 1.5B routing without oracle feedback.
2. **Always-on cascading is not net-cheaper here.** The quality gain cost roughly double latency and more than double output tokens relative to direct 1.5B.
3. **Structural gating cannot trigger this cascade selectively.** The 0.5B grammar made all drafts structurally valid, including all 11 semantic misses. A useful selective route needs calibrated uncertainty or an independent semantic signal.
4. **Agreement is not proof.** The 0.5B and 1.5B paths can share a wrong confusable interpretation, as shown by the two residual failures.
5. **The adjudicator is not an execution safety boundary.** Its 22/24 result remains below exact admission; fail-closed policy and semantic rejection must remain separate.

## Reproduction and validation

The raw artifact contains platform and accelerator identity, library/CUDA versions, model configurations, per-weight SHA-256 manifests, exact schema/catalog/tasks/prompts, route order, every draft/direct/adjudicated output, input/output token counts, synchronized timings, parsed objects, and exact correctness.

Post-capture checks established:

- three rounds and 24 rows per round;
- every summary recomputes from row data;
- corrected-miss and regressed-hit counts recompute independently;
- route order alternates as declared;
- every output, parse, validity result, correctness result, and token count is identical across all rounds;
- local and accelerator SHA-256 digests match.

## Bounded deployment conclusion

Do not replace direct 1.5B routing with this always-on cascade solely on the observed 22/24 score: it is better but still non-admissible and approximately doubles measured cost. The useful next seam is **selective** adjudication driven by a non-oracle uncertainty signal; absent that signal, the previously witnessed SmolLM2-1.7B concise route remains both more accurate at 24/24 and cheaper than this two-model path on the tested catalog.
