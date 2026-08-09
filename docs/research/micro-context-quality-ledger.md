---
title: "Micro-context quality and observability ledger"
description: "The `fak-microcontext-quality-ledger/1` surface: submitted equals retired plus failed plus cancelled, with three claim families kept structurally separate."
---

# Micro-context quality and observability ledger

**Status:** witnessed on 2026-08-06 for a 10,000-context synthetic run and a keyed model-backed API run.

`fak-microcontext-quality-ledger/1` ingests the existing run witness rather than inventing a second execution runtime. It reconciles submitted = retired + failed + cancelled, makes verifier failures part of the full submitted denominator, and keeps three claim families structurally separate:

1. **orchestration:** logical contexts and bounded physical workers;
2. **inference:** observed prompt/completion tokens and usage responses;
3. **useful work:** independently verified completions per wall-second.

Per-context IDs are sampled under an explicit cardinality cap; aggregate totals remain exact. Error classes are bounded labels rather than context IDs. The ingest contract requires a caller-supplied result verifier; the captured fixtures use a deterministic nonempty/accounting verifier and therefore do not claim semantic task quality.

## Captured ledgers

```powershell
go run ./cmd/microcontextdemo -verify-quality experiments/microcontext/s0-local-10k-quality-ledger-2026-08-06.json
go run ./cmd/microcontextdemo -verify-quality experiments/microcontext/s6-groq-api-only-4-quality-ledger-2026-08-06.json
```

| Ledger | Submitted | Verification | Samples | Inference usage |
|---|---:|---:|---:|---:|
| Synthetic S0 | 10,000 | 10,000 / 10,000 | 16 | explicitly zero / not inference |
| API-only S6 | 4 | 4 / 4 | 4 | 4 usage-bearing responses |

The test corpus also injects one verifier rejection and proves useful completions fall to 3/4 rather than silently retaining a successful-run denominator. A malformed ledger that omits failures is refused.

## Claim boundary

The synthetic useful-work rate means only that the fixture verifier accepted each synthetic completion; it is not model throughput. The model-backed ledger reports inference usage separately and does not equate token rate with verified task quality. Queue, cache, scheduler, retry, and cancellation fields are schema-ready aggregate classes; absent evidence remains zero rather than inferred.
