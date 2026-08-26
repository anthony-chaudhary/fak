# Qwen tool-error family re-audit (#8700)

`fak trajectory audit` now publishes deterministic family rows ranked by failed calls, with accounted tokens, repeated identical failures, mutation churn, and first/last event positions. The parser derives every field from the same content-free call signature and mutation target used by the aggregate trajectory counters; no model inference is involved.

## Reproducible cohort command

```powershell
fak trajectory audit --since 4d --user-contains Qwen --jsonl qwen-tool-errors.jsonl --md qwen-tool-errors.md
```

Run this against the retained 2026-08-23 four-day cohort to publish the current top-family table. The historical cohort receipt reports 1,276 tool errors from 16,072 calls (7.94%), six repeated-failure events, and two mutation-churn events. Raw transcripts are not committed, so this repository does not fabricate a per-family split for that retained aggregate.

## Lifecycle evidence

- **Promotion:** representative Claude/Codex fixtures prove stable family attribution and that family repeat/churn totals reconcile to existing aggregate counters; focused package tests are the promotion gate.
- **Demotion/retirement:** retire this attribution join if a source supplies typed provider error families with an independently witnessed, versioned schema; until then the deterministic local classifier remains reversible and content-free.
- **Invalidating assumption:** attribution is invalid if an adapter no longer preserves the tool-call ID linking an error result to its call, or if accounted token ownership cannot be assigned to the error event. Such evidence must remain zero/unknown rather than be inferred.

The conservative runtime loop breaker remains independently owned by #8716.
