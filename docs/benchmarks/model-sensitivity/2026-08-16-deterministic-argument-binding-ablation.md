# Deterministic argument-binding ablation

**Run date:** 2026-08-16 UTC  
**Hardware:** one NVIDIA L4-class accelerator  
**Models:** Qwen2.5-1.5B-Instruct FP16 and SmolLM2-1.7B-Instruct FP16  
**Raw artifact:** [`2026-08-16-deterministic-argument-binding-ablation.json`](2026-08-16-deterministic-argument-binding-ablation.json)  
**Raw SHA-256:** `37ba35278664aacf0df1a033b3707dfb3065583e9475af32bc94206ee7d14296`

## Verdict

A narrow, fail-closed deterministic binder fixed the explicit email-literal failures without changing tool selection, model tokens, or order invariance. Qwen2.5-1.5B improved from 72/96 to **84/96** exact; SmolLM2-1.7B improved from 68/96 to **80/96**. On the 24 unique requests, those are 21/24 and 20/24.

The layer passes this scoped prove-out, but the combined routes remain below execution admission. All residual Qwen misses are semantic search-query normalization; SmolLM also retains one wrong-tool order lookup. The binder must remain narrow: it is evidence for extracting uniquely explicit literals, not permission to infer ambiguous or absent values.

## Why this test

Canonical named-tool serialization eliminated catalog-order sensitivity and exposed a concentrated failure pattern. Both models usually selected `send_email` correctly but corrupted an explicitly delimited recipient, changed subject capitalization, or appended a domain. Issue #6964 asked whether a deterministic binder could preserve semantic tool selection while recovering exact arguments without creating executable calls from missing or ambiguous inputs.

## Binder contract

The binder is intentionally smaller than a general entity extractor:

- it supports only `send_email`;
- it never changes the model-selected tool;
- it binds a recipient only when the request contains exactly one bracketed `[redacted:pii:<digits>B]` literal;
- it binds a subject only when exactly one value follows one of four explicit exact-subject phrasings used by the requests;
- absent or multiple recipient/subject candidates reject the entire call with a null post-bind output;
- unsupported tools pass through byte-for-byte unchanged.

Six pre-run controls exercised the contract:

| Control | Expected | Observed |
|---|---|---|
| One explicit recipient and subject | Bind | Bound |
| Recipient absent | Reject | Rejected, null output |
| Two recipients | Reject | Rejected, null output |
| Exact subject absent | Reject | Rejected, null output |
| Two exact subjects | Reject | Rejected, null output |
| Unsupported tool | Pass through | Passed through unchanged |

## Protocol

- Base route: the committed canonical named-tool representation from `2026-08-16-canonical-named-catalog-ablation.json`.
- Requests: the same 24 held-out paraphrases.
- Source orders: original, reversed, and deterministic shuffles with seeds 20260816 and 6692; all canonicalize to identical prompts.
- Decode: greedy, 128-token cap, grammar constrained with `lm-format-enforcer==0.11.3` and Transformers 5.14.1.
- Work: 24 requests × four source orders × two models = **192 model calls**, each evaluated before and after binding.
- Runner SHA-256: `5f93f51ef5b966965c6c67f0b7d7623975c01557f91f567aa1cfc62e2d2ab2e1`.

## Results

| Model | Before binder | After binder | Delta | Unique after | Output tokens | Median binder time | Order-invariant post calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 72/96 | **84/96** | **+12** | 21/24 | 2,776 | 4.123 µs | 24/24 |
| SmolLM2-1.7B | 68/96 | **80/96** | **+12** | 20/24 | 3,052 | 4.005 µs | 24/24 |

Each model had 16 bound calls and 80 unchanged passthrough calls across the four repeated source-order arms. There were no rejections in the evaluated set because all four email requests contain one explicit recipient and one explicit subject. The negative controls establish the rejection path separately.

Per unique request:

- Qwen: three previously wrong email calls became correct; the already-correct fourth email remained correct.
- SmolLM: three previously wrong email calls became correct; the already-correct second email remained correct.
- No correct call regressed.
- No selected tool changed.
- Every model output and every post-bind call remained identical across the four upstream source orders.

Generation output tokens are unchanged because binding occurs after generation. The measured binder medians are synchronized process-local measurements of a few microseconds and should be read as “negligible in this harness,” not as a production latency guarantee.

## Residual failures

Qwen's remaining unique misses are `search-01`, `search-02`, and `search-03`: it selects `search_kb` but produces semantically nearby rather than exact benchmark query strings. SmolLM retains those three plus `order-01`, where it selects `archive_order` instead of `get_order`.

These are outside the binder's safe contract. Correcting search queries would require a declared normalization dictionary, retrieval over canonical query concepts, or label-aware logic; correcting `order-01` would require changing tool selection. Neither should be smuggled into a literal extractor.

## Reproduction and provenance

The raw artifact captures:

- the full binder contract and all six control inputs/results;
- environment, model configuration, tokenizer, and weight hashes;
- catalog and per-request prompt digests;
- all 192 raw outputs and confidence traces;
- every pre-bind call, post-bind call, status, reason, selected-tool-preservation result, exact correctness, generation tokens/timing, and binder timing;
- per-order and aggregate summaries and cross-order invariants.

Independent validation reimplemented the binder separately, recomputed all 192 pre/post outcomes and summaries, verified that no selected tool changed, reran all digest/invariance checks, and matched the accelerator artifact SHA-256.

Baseline: [`2026-08-16-canonical-named-catalog-ablation.json`](2026-08-16-canonical-named-catalog-ablation.json), SHA-256 `0ad557364a593c05aa122b497ec3ca60ace0d2087494164859e444e6e0ddaf38`.

## Scope

This is a synthetic exact-match set whose email literals are explicitly marked. The positive result applies to deterministic extraction of unique explicit values under this contract. It does not establish safe extraction from ordinary natural-language addresses, implicit subjects, multiple candidates, or production traffic. Those cases stay rejected until independently proved.
