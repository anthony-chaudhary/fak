# Qwen2.5-0.5B non-oracle shortlist routing — 2026-08-16

**Verdict: reject the route at the frozen admission bar.** A deterministic lexical
shortlist raised Qwen2.5-0.5B from the prior 24-tool baseline of 8/24 to **17/24**
exact calls with two candidates, one call below the predeclared 18/24 threshold.
Four candidates increased retriever recall but reduced exact calls to **11/24**.
The result converts the oracle curve into an end-to-end witness without claiming a
retriever win or admitting a route that missed its bar.

## Question

Can a CPU-only shortlist, computed from request text and declared tool metadata with
no expected-label access, preserve enough of the two-candidate oracle gain to admit a
sub-500M local routing tier?

This is the minimal spine for [#6973](https://github.com/anthony-chaudhary/fak/issues/6973),
under cross-model sensitivity umbrella
[#6692](https://github.com/anthony-chaudhary/fak/issues/6692).

## Frozen protocol

- **Model:** Qwen2.5-0.5B-Instruct; weights provenance digest
  `95cac2f4a62df51bce4b5039213e7379bc1c55371826de0d33f0ddb3b358762c`.
- **Hardware:** one sanctioned NVIDIA L4-class accelerator.
- **Workload:** the same frozen 24 held-out requests and 24 declared tools as the
  canonical named-tool, binding, sub-500M transfer, and oracle-size witnesses.
- **Retriever:** lowercase alphanumeric token sets over the request versus tool name,
  argument names, and schema type/enum values; sort by overlap descending, Jaccard
  descending, then tool name ascending; take the first 2 or 4; lexicographically sort
  the final catalog. The expected tool is **never inserted**.
- **Decode:** unchanged canonical named-tool prompt, candidate-derived JSON Schema
  constraint, greedy decoding, and a 128-token cap.
- **Postprocessor:** unchanged fail-closed `send_email` literal binder; it cannot change
  the selected tool.
- **Admission rule, declared before the run:** at least 18/24 exact at one size without
  a token or accelerator-time regression that erases the value.
- **Calls:** 48 total (24 requests × 2 candidate sizes), one primary model. The
  predeclared SmolLM2-360M extension was not run because Qwen missed admission, leaving
  no decision value worth another 48 calls.

## Results

| Candidate size | Retriever recall | Correct selected tool | Exact before binding | Exact after binding | Input tokens | Output tokens | Median generation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | **18/24** | **18/24** | 13/24 | **17/24** | 2,255 | 607 | 1,186.958 ms |
| 4 | **22/24** | **12/24** | 10/24 | **11/24** | 2,903 | 603 | 1,073.518 ms |

Reference points from the immediately preceding witnesses:

| Route | Exact calls |
|---|---:|
| Qwen2.5-0.5B, full 24-tool catalog | 8/24 |
| Qwen2.5-0.5B, oracle-included pair | 18/24 |
| Qwen2.5-0.5B, **non-oracle pair** | **17/24** |
| Qwen2.5-0.5B, oracle-included four | 11/24 |
| Qwen2.5-0.5B, **non-oracle four** | **11/24** |

The pair retrieved the expected tool for 18 requests; the model selected it for all
18, and the binder repaired four email arguments. The remaining included-tool miss was
`search-03`: the selected tool was correct but its argument was not exact. The six
retrieval misses were two paraphrased knowledge-base searches, three `get_order`
requests, and one closed-ticket list request.

At size four, recall rose by four requests but selected-tool correctness fell by six.
The added semantically adjacent distractors caused confusions among order actions,
message channels, and payment actions. This separates the two bottlenecks: pair failure
is mostly shortlist recall, while four-way failure is model discrimination despite
high recall.

## Interpretation and boundary

The non-oracle pair captures 9 of the 10 exact-call gains in the oracle pair and more
than doubles the 24-tool baseline, but it still misses the frozen 18/24 admission bar.
Therefore this evidence **rejects production admission** for this exact retriever/model
route. It does not show that every retriever must fail, and it does not weaken the
already-shipped larger-model results. A broader retriever search would be new work with
new overfitting controls, not a post-hoc reinterpretation of this frozen run.

This remains a model-routing benchmark, not a deployment throughput benchmark. Latency
is observed generation time on one accelerator-backed run; no concurrency, cost, or
production serving claim is made.

## Reproduction and validation

- Runner SHA-256: `157b178d1427a21a9baebfa76d072bc024caa21672c64bd7910a0729f2d7aec9`
- Raw artifact SHA-256: `97b69880df354867b37ed6019f1a61a2050858ed31885cd33c62517c77c0ae22`
- Captured log SHA-256: `e44dfc0e9addf00206a8475a8c0dbeea1951e07b555b00bbf8c8edc2ed3f919a`
- Runtime: 412.984 seconds
- Environment: Python 3.10.12, PyTorch 2.13.0+cu130,
  Transformers 5.14.1, `lm-format-enforcer`
  0.11.3, CUDA device `NVIDIA L4`.

An independent local validator recomputed all 48 top-k catalogs from the captured
ranking fixture, checked exact catalog sizes and hashes, confirmed label inclusion was
only an observed recall field, recomputed each summary, and verified binding never
changed a selected tool. Remote and local raw hashes matched. The accelerator returned
to 1,022 MiB and 0% utilization after the run.

Captured artifact:
[`2026-08-16-sub500m-nonoracle-shortlist.json`](2026-08-16-sub500m-nonoracle-shortlist.json).
