---
title: "Cache frontier review ledger"
description: "Dated review entries for fak's cache-frontier operating plan: what cache value we used ourselves, what a new person can demo, and what witness or product surface is missing next."
---

# Cache frontier review ledger

This folder is the recurring review artifact for the
[cache frontier operating plan](../CACHE-FRONTIER-OPERATING-PLAN.md). Each entry answers
the three weekly questions from evidence:

1. What cache-frontier value did we use ourselves this week?
2. What can a new person demo this week?
3. What is the next missing witness or product surface?

Entries should cite the command output they were derived from and keep Track 1
WITNESSED kernel reuse separate from Track 2 OBSERVED provider-dollar savings.
The ranked default-enablement backlog lives in
[`DEFAULT-ENABLEMENT-NEXT-50.md`](DEFAULT-ENABLEMENT-NEXT-50.md); it keeps pure
fak/API work, O(1) context/query, provider-cache preservation, and external-engine
adapters on separate evidence planes.
Each review has two artifacts:

- a human-readable markdown note under `reviews/YYYY-MM-DD.md`;
- one machine-readable row appended to [`review-ledger.jsonl`](review-ledger.jsonl)
  with the same evidence fields.

Generate new dated artifacts from the ledgers rather than hand-writing them:

```bash
go run ./cmd/fak cachevalue review \
  --since 2026-06-22 \
  --date 2026-06-29 \
  --source-markdown reviews/2026-06-29.md \
  --append-ledger docs/cache-frontier/review-ledger.jsonl \
  --markdown-out docs/cache-frontier/reviews/2026-06-29.md
```

Use `--json` without `--append-ledger` to inspect the row before mutating the
append-only ledger.

## Entries

- [2026-06-29 cache-frontier review](reviews/2026-06-29.md)
