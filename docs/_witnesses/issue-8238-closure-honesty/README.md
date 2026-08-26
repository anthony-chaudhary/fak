---
title: "Issue #8238 closure-honesty witness"
description: "This directory preserves the live receipts and the row-complete reconciliation for 8238."
---
# Issue #8238 closure-honesty witness

This directory preserves the live receipts and the row-complete reconciliation for
[#8238](https://github.com/anthony-chaudhary/fak/issues/8238).

- `default-window-before.json` is the failing-before run: both the issue and commit windows
  truncated, so its `ACTION` result could not safely drive issue state.
- `complete-window.json` is the complete run at
  `7825ba897a4dff736fdbdf748908732e4e37e8e5`: 8,505 issues and all 12,328 reachable
  commits, with `coverage.complete=true`.
- `reconciliation.json` maps every complete-audit issue number to a disposition. Every one
  of its 56 action rows links to an open repair issue (#8544, #8545, or #8546).
- `artifact_test.go` proves the receipts parse, the complete receipt hash matches the ledger,
  every complete-audit row appears exactly once, and no action row lacks an open repair link.

The parent issue preserved only the initial truncated run's aggregate 73
`CLAIMED_CLOSED` and 11 `OPEN_WITNESSED` counts, not its row identities. The reconciliation
therefore uses the complete 8,505-row repository population as an explicit superset. It does
not invent an unrecoverable 84-row subset.

The live complete receipt's raw SHA-256 is preserved in the reconciliation ledger. Publication
redacts only the title strings for issues #265, #6128, #4781, #4777, and #4769 because those
public issue titles contain strings blocked by this repository's `PUBLIC_LEAK` gate. Issue
numbers, states, buckets, commit bindings, counts, and ordering are unchanged.

Reproduce the artifact check from WSL:

```bash
go test ./docs/_witnesses/issue-8238-closure-honesty -count=1
```
