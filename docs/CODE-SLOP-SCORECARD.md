---
title: "fak Code-Slop Scorecard: the slop the compiler can't see"
description: "fak's code-slop scorecard grades the Go module on six deterministic slop axes into a slop-score (0-100, A-F) and a re-derivable slop-debt count — clones, vacuous tests, dead code, comment cruft."
---

# Code-slop scorecard

<!-- code-slop-scorecard: 2026-06-25 · process: tools/code_slop_scorecard.py -->

> Regenerate: `python tools/code_slop_scorecard.py --markdown --stamp DATE > docs/CODE-SLOP-SCORECARD.md`
> Verify snapshot freshness: `python tools/code_slop_scorecard.py --check-doc`

> The measuring stick for **slop the compiler can't see**: code that builds, vets clean, and has a test present, yet rots the kernel from the inside — copy-paste clones, tests that assert nothing, dead unexported symbols, commented-out code and tautological doc comments. Six deterministic axes (duplication · dead_code · comment_slop · vacuous_tests · stub_masquerade · churn_bloat), folded into a **slop-score** (0–100, A–F) and a **slop-debt** integer (the count of concrete, re-derivable slop defects). Every number below is re-derived from disk by `tools/code_slop_scorecard.py` — no hand-entry. Drive slop-debt to zero to make "less slop" provable.

## Corpus

| Metric | Value |
|---|---|
| Slop-score | 46.4/100 (grade F) |
| **Slop-debt (total HARD defects)** | **269** |
| Soft signals (advisory) | 0 |

## Per-KPI (worst-first)

| KPI | Score | Slop-debt | Detail |
|---|---:|---:|---|
| duplication | 0/100 | 238 | 238 duplicated block(s) (copy-pasted across 2+ sites) |
| dead_code | 0/100 | 27 | 27 dead unexported symbol(s) |
| vacuous_tests | 60/100 | 4 | 4 vacuous of 2989 Test func(s) |
| comment_slop | 100/100 | 0 | no comment slop |
| stub_masquerade | 100/100 | 0 | no exported stub-masquerade |
| churn_bloat | 100/100 | 0 | no commits in range (skipped) |

## What each axis catches

- **duplication** — a normalized 6-line window copy-pasted into 2+ places. [HARD]
- **dead_code** — an unexported symbol defined but referenced nowhere else. [HARD]
- **comment_slop** — tautological doc comments + commented-out code blocks. [HARD]
- **vacuous_tests** — a Test/Benchmark func that makes zero assertions. [HARD]
- **stub_masquerade** — an exported func with a trivial/panic body, not `[STUB]`. [SOFT]
- **churn_bloat** — recent commits adding `.go` files without retiring any. [SOFT]

> 269 unit(s) of slop-debt; score 46.4/100 (grade F); heaviest KPI: duplication (238 defect(s))

> next: retire slop-debt worst-first (see corpus.breakdown + per-KPI defects): de-duplicate clones, delete dead unexported symbols, drop commented-out code + tautological doc comments, add assertions to vacuous tests; re-run to prove the drop
