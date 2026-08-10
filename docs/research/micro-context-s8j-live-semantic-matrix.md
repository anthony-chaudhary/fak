# Micro-context S8j: live semantic matrix

**Status:** observed live OpenAI-compatible endpoint evidence<br>
**Issues:** #6110, parent #6033; label-stability follow-up #6140<br>

## Question

On the same non-zero semantic held-out workload, what do tuned retrieval-shaped, long-context, chunk map-reduce, and micro-context executions actually cost in requests, tokens, cache observations, wall time, TTFT, retries, and strict blind quality?

## Frozen comparison

All four alternatives use the same live `gpt-5.6-sol` route, temperature 0, streaming usage, the same 16 held-out S8i records, and two repeated trials. The answer bundle remains outside prompts. Each first-trial submission is independently graded against S8i consensus.

The operational shapes are intentionally distinct:

- **Retrieval + rerank:** one record per request, with the 16 tune answers supplied as retrieved/reranking examples.
- **Long context:** all 16 held-out issues in one request.
- **Chunk map-reduce:** four four-record map requests; the common typed answer schema is the reduce boundary.
- **Micro-context:** one issue per bounded request, up to eight concurrent windows, no tune-answer payload.

This is a live shape comparison, not a claim that these compact implementations exhaust the best possible version of each research family.

## Endpoint capability and provenance

- Endpoint class: separate OpenAI-compatible live route.
- Model: `gpt-5.6-sol`.
- Hardware: provider-managed and undisclosed.
- Native batch API: unsupported by the chat route used here.
- Prefix cache: usage-field observed only; no server-internal inference.
- Pricing: no authoritative pricing snapshot for this exact route/model, so dollar cost is explicitly `null` rather than modeled.
- Tool calls: zero. The matrix predicts tool eligibility but has no common external evidence source, so it does not fabricate one.

## Observed aggregate results

Counts below sum two trials; wall and latency values are trial means.

| Pipeline | Requests | Prompt tokens | Output tokens | Cached tokens | Mean wall | TTFT p50 | TTFT p95 | Request latency p95 | Retries | Exact held-out records |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Retrieval + rerank | 32 | 63,784 | 5,326 | 5,376 | 14.82 s | 5.07 s | 9.22 s | 9.22 s | 1 | 1/16 |
| Long context | 2 | 17,246 | 2,380 | 7,936 | 26.04 s | 26.04 s | 26.04 s | 26.04 s | 0 | 0/16 |
| Chunk map-reduce | 8 | 17,808 | 2,771 | 3,584 | 40.33 s | 10.62 s | 13.53 s | 13.53 s | 0 | 0/16 |
| Micro-context | 32 | 19,976 | 4,588 | 0 | 17.29 s | 4.84 s | 12.64 s | 12.64 s | 0 | 0/16 |

## What changed relative to S8f

S8f’s exact frontier correctly admitted zero model calls. S8i now admits real calls and exposes a different failure: **none of the live alternatives meets the current strict consensus quality floor**. The best exact score is only 1/16.

This is not evidence that every model is bad or that semantic decomposition is useless. The gold contract preserves adjudicator disagreement as `abstain`, while a single inference model tends to choose a concrete label. Most errors are therefore semantic/tool abstention mismatches. This is precisely why the matrix must report quality beside cost.

## Net-true comparisons

No pipeline is a benchmark winner because all fail the quality floor. Conditional cost observations are still useful:

- Retrieval-shaped prompting is fastest in mean wall time but pays the largest token bill because tune examples are repeated per issue.
- Long context minimizes requests and output tokens, and reports the most cached tokens, but has the largest TTFT.
- Chunking uses nearly the same prompt tokens as long context but is slowest in this serial map implementation; a parallel map scheduler is an obvious stronger baseline.
- Micro-context cuts prompt tokens by about 69% versus retrieval-shaped prompting and has the lowest TTFT p50, but reports no cached tokens on this route and still fails strict quality.

These are observed scoped facts, not an efficiency claim. The real alternative is quality-constrained; until at least one configuration passes, cost ranking cannot establish net value.

## Steelmanned perspectives

- **Long-context advocate:** one request preserves global relationships and maximizes provider-visible reuse. The current 26-second TTFT may be acceptable for an offline task, and the strict abstention grader may understate useful answers.
- **Retrieval advocate:** repeating all 16 tune examples is intentionally simple, not a tuned semantic index. A real retriever would select fewer, more relevant demonstrations and could eliminate much of the 63.8k prompt-token load.
- **Chunk advocate:** serial maps are an unfair latency baseline for inherently parallel work. Parallel maps plus a true learned reduce deserve measurement before ruling out the design.
- **Micro-context advocate:** bounded per-record windows have the lowest observed TTFT and good wall time while retaining local cancellation/cache keys. Their missing ingredient is calibrated abstention, not more input context.
- **Benchmark skeptic:** two trials and 16 records are mechanism evidence, not population-level performance. Tool-need agreement is only 21.875%, and #6140 may materially change gold.
- **Systems/security advocate:** a model’s willingness to emit a concrete tool label is not authority. Tool admission, budgets, and read-back receipts remain deterministic even if prediction quality improves.

## Reproduce

```powershell
go run ./cmd/microcontextdemo `
  -live-matrix-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -live-matrix-gold experiments/microcontext/s8i-semantic-gold-2026-08-10.json `
  -live-matrix-output experiments/microcontext/s8j-live-matrix-openai-2026-08-10.json `
  -live-matrix-endpoint $env:OPENAI_BASE_URL `
  -live-matrix-api-key $env:OPENAI_API_KEY `
  -live-matrix-model gpt-5.6-sol `
  -live-matrix-endpoint-class separate-openai-compatible-live `
  -live-matrix-hardware provider-managed-undisclosed `
  -live-matrix-native-batch unsupported-by-chat-route `
  -live-matrix-prefix-cache usage-field-observed `
  -live-matrix-pricing unavailable-for-gpt-5.6-sol-route `
  -live-matrix-trials 2 `
  -live-matrix-workers 8

go run ./cmd/microcontextdemo `
  -verify-live-matrix experiments/microcontext/s8j-live-matrix-openai-2026-08-10.json
```

## Decision boundary

S8j satisfies the live execution/accounting mechanism and falsifies any current winner claim. It does not complete the parent decision:

1. #6140 should stabilize the abstention/tool rubric.
2. Retrieval needs a real index/select step rather than all-example repetition.
3. Chunk maps need parallel execution.
4. Candidate prompts need tune-only abstention calibration.
5. A priced route is required before dollars can be reported.
6. At least one candidate must pass the held-out quality floor before #6111 can draw a Pareto frontier.
