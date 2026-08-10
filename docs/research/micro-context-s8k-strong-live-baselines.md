# Micro-context S8k: strengthened live baselines

**Status:** observed live endpoint evidence with tune-only calibration<br>
**Issues:** #6151, parent #6033; predecessor #6110<br>

## Changes from S8j

S8j exposed two weak alternatives and no abstention calibration. S8k strengthens them without touching held-out answers:

1. **Retrieval:** tokenize title/body, score Dice overlap against tune records, select top-k=3 examples, and retain every selected ID/score. Tune evaluation is leave-one-out, so a tune record can never retrieve its own hidden answer.
2. **Chunk map-reduce:** run four independent four-record maps concurrently behind an eight-request admission limit. The reduce is typed concatenation by record ID, not another unmeasured model call.
3. **Abstention calibration:** every prediction includes confidence per field. Candidate thresholds `{0,.5,.7,.85,.95,1.01}` are scored on the 16 tune records only; maximum exact tune count wins, ties choose the lower threshold. The selected threshold is frozen before held-out execution.

Long-context and micro-context remain in the same matrix so baseline strengthening cannot hide a regression elsewhere.

## Results

All alternatives use the same live `gpt-5.6-sol` route and two held-out trials.

| Pipeline | Selected threshold | Best tune exact | Prompt tokens | Output tokens | Cached | Mean wall | TTFT p50/p95 | Held-out exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-k retrieval | 0.85 | 8/16 | 29,994 | 6,803 | 0 | 15.96 s | 5.77 / 10.33 s | 2/16 |
| Long context | 0.00 | 0/16 | 17,318 | 3,107 | 7,936 | 32.79 s | 32.79 / 32.79 s | 0/16 |
| Parallel chunks | 0.95 | 3/16 | 18,112 | 3,593 | 1,792 | **11.71 s** | 9.06 / 11.71 s | 2/16 |
| Micro-context | 0.95 | 2/16 | 21,128 | 5,224 | 0 | 316.02 s | 5.76 / 310.60 s | 2/16 |

The micro-context tail is a real endpoint outlier: one retry and approximately five-minute p95. It is retained rather than trimmed, illustrating why independent windows need cancellation/deadline policy and why mean-only reporting is unsafe.

## Comparison with S8j

- Retrieval prompt tokens fall from 63,784 to 29,994 (53% lower) after replacing all-example repetition with top-k=3. Held-out exact rises from 1 to 2.
- Parallel chunks reduce mean wall time from 40.33 seconds to 11.71 seconds (71% lower) while held-out exact rises from 0 to 2.
- Tune-only abstention calibration improves the best held-out exact count to 2, but **no alternative passes the 16/16 strict floor**.
- Micro-context retains low median TTFT but exhibits the worst tail in this run. Local decomposition bounds blast radius; it does not automatically guarantee a good tail without deadlines, hedging, or cancellation.

## Retrieval trace

The artifact records 16 held-out query IDs, each with exactly three tune IDs and deterministic similarity scores. The verifier requires the complete trace. The unit witness also proves leave-one-out tune retrieval excludes the query itself.

## Steelmanned interpretations

- **Retrieval:** top-k makes the baseline materially stronger and cheaper, but lexical overlap is not a semantic index. Embedding or learned reranking may improve quality.
- **Chunks:** parallel maps demonstrate the expected systems advantage. A learned reduce could recover cross-chunk relations but adds tokens and latency.
- **Long context:** cache reuse remains strongest, but the model does not learn the abstention contract from examples here. Few-shot long-context calibration is a fair next variant.
- **Micro-context:** p50 remains competitive and per-record scheduling permits cancellation, but one slow/retried window can dominate all-of-set completion. Early stopping requires an explicit task-level sufficiency rule rather than waiting for every record.
- **Benchmark skeptic:** 16 records and two trials reveal mechanisms and regressions, not stable population rankings. The consensus gold itself remains abstention-heavy.

## Reproduce

```powershell
go run ./cmd/microcontextdemo `
  -strong-matrix-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -strong-matrix-gold experiments/microcontext/s8i-semantic-gold-2026-08-10.json `
  -strong-matrix-output experiments/microcontext/s8k-strong-live-matrix-2026-08-10.json `
  -strong-matrix-endpoint $env:OPENAI_BASE_URL `
  -strong-matrix-api-key $env:OPENAI_API_KEY `
  -strong-matrix-model gpt-5.6-sol `
  -strong-matrix-endpoint-class separate-openai-compatible-live `
  -strong-matrix-hardware provider-managed-undisclosed `
  -strong-matrix-native-batch unsupported-by-chat-route `
  -strong-matrix-prefix-cache usage-field-observed `
  -strong-matrix-pricing unavailable-for-gpt-5.6-sol-route `
  -strong-matrix-trials 2 `
  -strong-matrix-workers 8 `
  -strong-matrix-retrieval-k 3 `
  -strong-matrix-chunk-size 4

go run ./cmd/microcontextdemo `
  -verify-strong-matrix experiments/microcontext/s8k-strong-live-matrix-2026-08-10.json
```

## Decision

#6151 removes the two obvious structural objections—repeated-all-example retrieval and serial chunks—and adds leakage-controlled calibration. It still falsifies a quality-qualified winner. #6111 must report `not-yet`, not draw a cost-only frontier across quality-inferior candidates. The next micro-context-specific experiment should measure deadline/cancellation/partial-fold policies against the observed five-minute tail rather than assuming decomposition alone solves tail latency.
