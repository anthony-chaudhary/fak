# Dispatch closure honesty: complete-window reconciliation (#8238)

Verdict: the truncated closure audit was not safe to act on. A complete-window run found a
large historical exception cohort and only 56 rows that still require action; every action row
now has an open repair issue.

## Receipts

The durable machine artifacts are in
[`docs/_witnesses/issue-8238-closure-honesty/`](../_witnesses/issue-8238-closure-honesty/README.md).
The live default run captured the failure first: 1,000 of 1,000 fetched issues and 2,000 of
12,321 commits produced `COVERAGE_INCOMPLETE`. The complete run then scanned all 8,505 issues
and all 12,328 commits reachable from snapshot
`7825ba897a4dff736fdbdf748908732e4e37e8e5`; its receipt reports
`coverage.complete=true` and `COVERAGE_COMPLETE`.

The complete audit took 641.699 seconds. The prior implementation started one `dos
commit-audit` process at a time, making the full-history command impractical. The #8238 change
uses a bounded eight-worker audit pool and captures the commit-count denominator before the
slow witness phase. The live run showed exactly eight concurrent DOS children.

## Reconciliation

The 8,505-row ledger records one of these dispositions for every complete-audit issue:

| Disposition | Rows | Meaning |
|---|---:|---|
| `WITNESSED_CLOSE` | 4,812 | The closing commit is DOS witnessed. |
| `POST_SNAPSHOT_WITNESSED_CLOSE` | 1 | #8416 gained a current-main DOS-witnessed closing commit after the audit snapshot. |
| `TYPED_EXCEPTION_LEGACY_PRE_ENFORCEMENT` | 1,247 | The close predates the first committed closure-audit surface. |
| `TYPED_EXCEPTION_NON_CODE_CLOSE` | 169 | GitHub records `NOT_PLANNED`; no code-ship claim is inferred. |
| `TYPED_EXCEPTION_DATA_WITNESS` | 16 | DOS records a data witness rather than a diff witness. |
| `NOT_CLOSURE_RESIDUAL_OPEN` | 2,204 | The issue is open and has no witnessed resolving commit. |
| `ACTION_OPEN_WITNESSED_COHORT` | 54 | The issue remains open despite a witnessed resolving commit; #8544 owns the exact cohort. |
| `ACTION_UNBOUND_CLOSED_ROW` | 2 | #8525 and #8534 were manually closed without a commit binding; #8545 and #8546 own the repairs. |

The enforcement boundary is commit
`9b7b2d9cabec6548e44407edcf587be1fcae9d28` at 2026-08-21 07:51:02 -07:00, the first
committed trunk surface containing `dispatch closure-audit`. #8416 closed seconds later, but
commit `756f7d854a0b272d81c5bc0b0b9fbdbdce4d1f4a` now independently grades `OK` /
`diff-witnessed` and is on current main. The only later unbound closed rows are #8525 and
#8534.

The original #8238 body retained only aggregate counts for its first truncated run, so its 84
row identities cannot be reconstructed without inventing evidence. The complete ledger instead
maps every extant repository issue, a strict superset of the original 1,000 fetched rows. Its
artifact test locks the row count, source bucket, receipt digest, disposition counts, and open
repair link for every action row so none can disappear silently.

Five copied GitHub issue titles contained strings rejected by the repository's public-leak gate.
The published receipt redacts only those title fields and records their issue numbers plus the
raw live-receipt digest in the reconciliation ledger; every grading field remains equivalent to
the live result.
