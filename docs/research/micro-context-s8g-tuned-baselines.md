# Micro-context S8g: tuned baselines and the exact frontier

**Status:** observed local dry run; no provider/model measurement<br>`n**Issues:** #6033, #6109; semantic-residual follow-up #6124<br>`n**Artifacts:** `experiments/microcontext/s8g-tuned-baselines-2026-08-10.json`

## Question

Can exact SQL/search, retrieval plus reranking, one long-context call, coarse chunk map-reduce, and the micro-context operator be tuned fairly on the S8f 1,000-issue corpus before live endpoint measurement?

## Verdict

The exercise falsified its intended model comparison. S8f has **zero semantic residual under its current answer contract**. Candidate-visible state, labels, timestamps, and text plus deterministic parsing recover every scored fact exactly. The tuned policy therefore stops at the exact SQL/search/parser frontier for all five named pipelines. Each reaches the held-out quality floor with zero model calls; starting a retrieval, long-context, chunk, or micro-window model stage would add work without admitting a better answer.

This is not evidence that exact search generally dominates model reasoning, nor that the four model-bearing designs are equal in cost or quality. It is evidence that a benchmark whose labels are structural projections cannot measure their semantic differences.

## Leakage boundary and selection protocol

1. S8f fixes grouped train/tune/held-out splits (212/108/680); duplicate and reference-connected components cannot cross splits.
2. Each baseline declares candidate configurations before held-out evaluation.
3. The selector chooses the least-work candidate meeting the strict tune floor: zero false positives, false negatives, aggregate errors, or citation errors.
4. Exact derivation reads only candidate-visible public records. The hidden answer bundle is read by tune and held-out graders, not by the derivation path.
5. Held-out answers are opened only after the selected configurations are frozen in the report construction.

The common artifact records candidate configurations, the selected configuration, selection evidence, operator/prompt version, retry policy, and provider-native optimization posture for every pipeline.

## Selected configurations

| Pipeline | Tune candidates | Selected posture | Why |
|---|---|---|---|
| Exact SQL/search | indexed fields + regex variants | `exact-frontier-v1` | Strict tune quality passes with deterministic derivation. |
| Retrieval + rerank | BM25 k=4/8/16 with rerank | exact frontier; semantic stage not run | No residual remains for retrieval to resolve. |
| Long context | 32k/128k/272k single-call variants | exact frontier; long-context call not run | Added tokens cannot improve an already exact contract. |
| Chunk map-reduce | 25/50/100-record chunks | exact frontier; map/reduce model calls not run | Structural partitioning does not create a semantic task. |
| Micro-context | 1/4/8-record windows with selector | exact frontier; windows not run | The selector correctly cancels all zero-value model windows. |

“Supported but zero eligible calls” is the honest result for provider batching and prefix caching. Their live behavior remains outside this leaf.

## Held-out observation

All five postures produce the same exact 680-record held-out submission and pass the strict grader. Local work is measured as deterministic record parsing and fact derivation. For each logical posture the artifact reports:

- 680 records read;
- 680 deterministic filter evaluations;
- 2,823 emitted facts;
- zero model calls;
- zero provider batches;
- zero input/output model tokens;
- zero tool calls.

These repeated operation counts describe the shared exact-frontier execution, not five physically repeated paid endpoint runs. No token, dollar, TTFT, tail-latency, retry, prefix-cache, or native-batch claim follows from them. Those require #6110.

## Steelmanned readings

- **Exact-system view:** stopping here is the correct optimization. A router should not invoke an LLM merely because one is available; deterministic facts are cheaper, auditable, and exact.
- **Micro-context view:** zero windows is a successful adaptive-routing outcome. The general pattern includes deciding which filters or tools to run, and cancelling stages whose expected value of information is zero.
- **Long-context/retrieval view:** this corpus does not test their strongest case. Both become relevant when facts require semantic synthesis, latent linkage, ambiguity resolution, or evidence beyond literal fields.
- **Benchmark-skeptic view:** all five labels collapse to one implementation, so this artifact validates tuning discipline and falsification behavior—not comparative model performance.
- **Tool-routing view:** no read-only enrichment is justified when source fields suffice. A stronger corpus should include cases where bounded tool calls can resolve uncertainty and where abstention is preferable to unnecessary calls.

## Reproduce

```powershell
go run ./cmd/microcontextdemo `
  -tuned-baselines-public experiments/microcontext/s8f-github-issues-public-2026-08-09.json `
  -tuned-baselines-answers experiments/microcontext/s8f-github-issues-answers-2026-08-09.json `
  -tuned-baselines-output experiments/microcontext/s8g-tuned-baselines-2026-08-10.json

go run ./cmd/microcontextdemo `
  -verify-tuned-baselines experiments/microcontext/s8g-tuned-baselines-2026-08-10.json

go test ./cmd/microcontextdemo -run 'TestTunedBaselines|TestVerifyTuned' -count=1
```

The generated report binds the public corpus digest and corrected hidden-answer digest. `-verify-tuned-baselines` rejects a malformed schema, wrong split/result cardinality, failed quality result, non-zero semantic/model work, or a decision that does not preserve the falsification.

## What must happen next

#6124 owns the missing benchmark: independently adjudicated semantic labels, ambiguity/abstention cases, and positive-value filter/tool routing under the same leakage controls. #6110 remains the live endpoint witness for tokens, dollars, latency, retries, batching, and cache provenance. Until those exist, #6033 has no net-true winner claim.
