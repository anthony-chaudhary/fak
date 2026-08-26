# Sub-500M oracle candidate-size sensitivity

**Run date:** 2026-08-16 UTC
**Hardware:** one NVIDIA L4-class accelerator
**Models:** Qwen2.5-0.5B-Instruct FP16 and SmolLM2-360M-Instruct FP16
**Raw artifact:** [`2026-08-16-sub500m-oracle-candidate-size.json`](../benchmarks/model-sensitivity/2026-08-16-sub500m-oracle-candidate-size.json)
**Raw SHA-256:** `51e887c63493d14dba40bed7c95cea32063d0430d6c9b598e6a3f373ea1a9769`

## Verdict

Candidate load is a major part of the sub-500M failure, but even an oracle-inclusion two-tool shortlist does not make either checkpoint execution-admissible. Qwen2.5-0.5B rose from 8/24 with 24 tools to **18/24** with two candidates after binding. SmolLM2-360M rose from 0/24 to only **7/24**. Accuracy declined monotonically as the shortlist grew.

This is an upper bound, not a retriever result: every candidate set was given the expected tool by oracle. A real shortlist must first achieve recall, so its end-to-end accuracy cannot exceed these numbers under the same downstream route.

## Question

The frozen 24-tool canonical route produced perfectly order-invariant but semantically weak sub-500M results. This study asks whether that failure is mostly intrinsic or caused by selecting among many confusable tools. It measures candidate sizes 2, 4, and 8 while holding the prompt, decoding, grammar constraints, and deterministic email binder fixed.

## Candidate construction

For each request and size:

1. forcibly include the expected tool;
2. tokenize the request and every tool's name, argument names, and schema type/enum strings into lowercase alphanumeric sets;
3. rank non-expected tools by token overlap descending, Jaccard similarity descending, then tool name ascending;
4. fill remaining slots from that deterministic ranking;
5. sort the final candidate catalog lexicographically before prompting.

The forced inclusion is explicitly label-dependent. The raw artifact records the full 24-tool ranking and exact candidate catalog for every call.

## Protocol

- 24 held-out requests;
- candidate sizes 2, 4, and 8;
- two model checkpoints;
- one canonical named-tool prompt per request and size;
- greedy decode, 128-token cap, candidate-derived JSON Schema filtering;
- same fail-closed `send_email` literal binder as #6964;
- **144 total model calls**.

The source-order matrix was not repeated because prior evidence proved that canonical serialization makes those prompts and deterministic outputs identical. The committed 24-tool baseline uses the same canonical route and binder.

Runner SHA-256: `48375af5b2a1ed01b43ea88e28f0dc2f3f52971f988e2a466889f3725cb26c35`.

## Results

### Exact correctness after binding

| Model | 2 candidates | 4 candidates | 8 candidates | 24-tool baseline |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | **18/24** | 11/24 | 10/24 | 8/24 |
| SmolLM2-360M | **7/24** | 2/24 | 0/24 | 0/24 |

### Correct selected tool before binding

| Model | 2 candidates | 4 candidates | 8 candidates | 24 tools |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 19/24 | 12/24 | 10/24 | 8/24 |
| SmolLM2-360M | 8/24 | 2/24 | 0/24 | 0/24 |

All 144 outputs were structurally valid. The binder improved Qwen by four calls at size 2 and one call each at sizes 4 and 8. It improved SmolLM by three calls at size 2 and none at larger sizes. No binder operation changed a selected tool.

### Output tokens and latency

| Model | Size | Output tokens | Median synchronized call latency |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 2 | 611 | 1.191 s |
| Qwen2.5-0.5B | 4 | 603 | 1.085 s |
| Qwen2.5-0.5B | 8 | 629 | 2.970 s |
| SmolLM2-360M | 2 | 779 | 1.203 s |
| SmolLM2-360M | 4 | 738 | 1.168 s |
| SmolLM2-360M | 8 | 730 | 1.191 s |

These in-process timings include prompt preparation/tokenization and synchronized generation. Qwen's size-8 arm experienced substantially slower calls in this run; no serving-performance claim is inferred from one sequential matrix.

## Interpretation

### Qwen2.5-0.5B may be useful only behind an extremely strong shortlist

Reducing 24 tools to an oracle pair recovers ten unique exact calls, showing that distractor load—not only model incapacity—drives the 24-tool failure. But 18/24 is still below the larger models' 20/24–21/24 bound route, and a real retriever would introduce misses before generation. The result supports testing retrieval economics, not admitting the model.

### SmolLM2-360M remains unsuitable

Even when choosing between the expected tool and one deterministic distractor, SmolLM selects the correct tool only 8/24 times and is exact after binding only 7/24. Its failure is not rescued by a small candidate set.

### Structural validity remains a weak signal

Grammar filtering produced 144/144 valid objects while exact correctness ranged from 0/24 to 18/24. Valid JSON and valid argument types do not establish correct routing.

## Reproduction and provenance

The raw artifact includes environment and model hashes, the lexical ranking rule, full rankings and candidate fixtures, every candidate catalog and schema, prompt/catalog digests, raw outputs and confidence traces, pre/post binding calls, exact correctness, tokens, and synchronized timings.

Independent validation recomputed candidate membership and canonical digests, verified oracle inclusion and exact catalog size for all 144 calls, recomputed every summary, and confirmed the binder never changed a selected tool. The local SHA-256 matched the accelerator copy.

24-tool baseline: [`2026-08-16-sub500m-canonical-binding-transfer.json`](../benchmarks/model-sensitivity/2026-08-16-sub500m-canonical-binding-transfer.json), SHA-256 `1fb90b5844ad79af03e2578965838d900a387e28ca6e44f99978535b2d36a94f`.

## Scope

This study deliberately uses expected-label inclusion to measure downstream sensitivity. It neither implements nor validates retrieval, and its candidate ranking is used only to choose distractors. Production value requires a separate end-to-end retriever recall, latency, and net-cost witness.
