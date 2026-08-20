---
title: "Micro-context S8o: live filter and tool scheduler"
description: "A live quality-qualified matrix that tests adaptive filter and tool scheduling against a tuned fixed cascade and records cancellation limits."
---

# S8o: live quality-qualified filter/tool scheduler matrix

Date: 2026-08-10  
Issue: #6185  
Controlled prerequisite: S8n (`bfbcff3e86`)

## Verdict

`not-yet`: the tuned fixed cascade is the only best-quality live arm (12/16 = 75% in both cold and
warm passes). No adaptive arm matches that quality, so none is eligible for a latency/token winner
claim. The result directly falsifies carrying S8n's controlled adaptive winner into the live headline.
It also exposes two useful operational facts: this endpoint returned zero cached tokens on identical
second passes, and client cancellation acknowledgement cannot reveal cancelled-but-billed usage.

## Frozen contract and real tools

Every arm uses the same S8i packet, S8m majority fold, test split, model (`gpt-5.6-sol`), rubric, and
bounded output schema. Adaptive/run-all arms execute a real read-only GitHub issue-state call, reduced
to `{state, updated_at, locked}` before the model sees it. Unauthenticated REST exhaustion falls back
to authenticated `gh api`; no write/effect stage exists.

The matrix includes planner, tuned fixed cascade, adaptive, selective hedge, run-all, and universal
hedge in cold then identical warm passes. Four requests run concurrently. Streaming receipts record
TTFT, wall, returned prompt/output/cache tokens, retries, tool URL, hedge/cancel request, cancellation
acknowledgement, prediction, majority gold, and unanimity.

## Results

| Phase | Policy | Exact | Mean wall ms | p95 ms | Prompt | Output | Cached | Hedges |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cold | planner | 11/16 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| cold | **fixed cascade** | **12/16** | 3,993.5 | 6,257.4 | 9,078 | 1,768 | 0 | 0 |
| cold | adaptive | 11/16 | 5,655.3 | 8,865.3 | 9,494 | 2,013 | 0 | 0 |
| cold | selective hedge | 11/16 | 5,437.3 | 11,338.6 | 9,494 | 2,012 | 0 | 0 |
| cold | run-all | 9/16 | 5,532.2 | 9,489.7 | 9,494 | 2,363 | 0 | 0 |
| cold | universal hedge | 9/16 | 5,565.5 | 13,625.1 | 9,494 | 1,939 | 0 | 16 |
| warm | planner | 11/16 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| warm | **fixed cascade** | **12/16** | 4,326.1 | 8,979.6 | 9,078 | 1,702 | 0 | 0 |
| warm | adaptive | 10/16 | 5,344.4 | 7,539.3 | 9,494 | 2,235 | 0 | 0 |
| warm | selective hedge | 10/16 | 5,075.0 | 9,359.7 | 9,494 | 1,934 | 0 | 1 |
| warm | run-all | 11/16 | 5,353.1 | 8,713.9 | 9,494 | 1,896 | 0 | 0 |
| warm | universal hedge | 10/16 | 4,396.0 | 7,271.1 | 9,494 | 1,918 | 0 | 16 |

The planner has zero model/tool cost because its declared baseline always predicts `read_only`; its
11/16 quality is below the fixed cascade. Fixed cascade therefore wins the only quality-qualified
comparison. Adaptive tool enrichment does not repay its added prompt and latency on this slice.

The unanimous test slice contains one record and every cold arm gets it right; warm adaptive variants
miss it. That slice is reported but far too small for a headline.

## Cancellation and cache accounting

Universal hedging records 16 client cancel requests and 16 local cancellation acknowledgements in each
pass. Selective hedging opened no cold duplicate and one warm duplicate because only one primary crossed
the fixed delay. Returned usage belongs to completed winner streams only. The endpoint exposes no
provider read-back for loser billing, so cancelled-but-billed tokens remain **unknown**, not zero.

Every arm reports zero cached tokens in both passes. “Warm” means an identical second pass; it does not
imply the provider accepted or exposed a prefix-cache hit.

## Artifact and rerun

Artifact: `experiments/microcontext/s8o-live-filter-tool-2026-08-10.json`.

```powershell
go run ./cmd/microcontextdemo `
  -live-filter-tool-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -live-filter-tool-fold experiments/microcontext/s8m-semantic-tool-fold-2026-08-10.json `
  -live-filter-tool-output experiments/microcontext/s8o-live-filter-tool-2026-08-10.json `
  -semantic-endpoint $env:OPENAI_BASE_URL -semantic-api-key $env:OPENAI_API_KEY `
  -live-matrix-model gpt-5.6-sol

go run ./cmd/microcontextdemo `
  -verify-live-filter-tool experiments/microcontext/s8o-live-filter-tool-2026-08-10.json
```

## Steelman and boundary

- The fixed cascade's win may reflect the tiny abstention-heavy majority-gold slice, not a universal
  routing law. A larger human panel could change the ordering.
- Planner quality is surprisingly strong at zero calls; this is exactly why every adaptive comparison
  must include a structural baseline.
- Real GitHub state receipts are bounded and useful, but the majority labels mostly ask whether current
  state is *necessary*, not whether current state improves generic classification.
- Universal hedging can reduce some wall samples but doubles openings and has unknowable loser billing;
  it is not net-true here.
- Provider-native batch and a real cache-visible route may change economics. This route reported no
  native batch witness and no cached usage.

S8o closes #6185 with an honest `not-yet`. It advances #6033/#6111 by ruling out an adaptive live winner
on the observed envelope; it does not establish a general large-input winner.
