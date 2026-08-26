---
title: "Issue #8544 open-witnessed reconciliation"
description: "Reference documentation for Issue #8544 open-witnessed reconciliation, preserving the page's implementation details, evidence, and operating context."
---

# Issue #8544 open-witnessed reconciliation

This directory preserves the failing-before receipt and final row-complete ledger for
[#8544](https://github.com/anthony-chaudhary/fak/issues/8544).

The stale #8238 complete-window receipt classified 54 open issues as witnessed-closeable.
A fresh run of the repository's acceptance-aware close arm found that none can be closed
honestly today:

| Final disposition | Rows |
|---|---:|
| `OPEN_TYPED_REOPENED` | 15 |
| `OPEN_TYPED_UNWITNESSED` | 1 |
| `OPEN_TYPED_INCOMPLETE_EVIDENCE` | 32 |
| `OPEN_TYPED_PARTIAL` | 2 |
| `OPEN_TYPED_NONRESOLVING` | 4 |

`failing-before.json` captures the dry-run that exposed the stale aggregate: all 54 rows
were checked and zero passed every close gate. `reconciliation.json` captures the live
close arm plus independent read-back: 54 DOS audits, 54 ancestry checks against the recorded
`origin/main`, and 54 current GitHub states. Fifty-three recorded commits remain ancestors;
#6349 is both non-ancestral and `CLAIM_UNWITNESSED` / `subject-only`. All 54 issues remain
open with an explicit typed reason, so the live arm performed no close or reopen mutation.

Reproduce the deterministic artifact check from WSL:

```bash
go test ./docs/_witnesses/issue-8544-open-witnessed-closure -count=1
```
