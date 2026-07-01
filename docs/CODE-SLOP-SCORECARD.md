---
title: "fak Code-Slop Scorecard: the slop the compiler can't see"
description: "fak's code-slop scorecard grades the Go module on six deterministic slop axes into a slop-score (0-100, A-F) and a re-derivable slop-debt count — clones, vacuous tests, dead code, comment cruft."
---

# Code-slop scorecard

<!-- code-slop-scorecard: 2026-06-30 · process: tools/code_slop_scorecard.py -->

> Regenerate: `python tools/code_slop_scorecard.py --markdown --stamp DATE > docs/CODE-SLOP-SCORECARD.md`
> Verify snapshot freshness: `python tools/code_slop_scorecard.py --check-doc`

> The measuring stick for **slop the compiler can't see**: code that builds, vets clean, and has a test present, yet rots the kernel from the inside — copy-paste clones, tests that assert nothing, dead unexported symbols, commented-out code and tautological doc comments. Six deterministic axes (duplication · dead_code · comment_slop · vacuous_tests · stub_masquerade · churn_bloat), folded into a **slop-score** (0–100, A–F) and a **slop-debt** integer (the count of concrete, re-derivable slop defects). Every number below is re-derived from disk by `tools/code_slop_scorecard.py` — no hand-entry. Drive slop-debt to zero to make "less slop" provable.

## Corpus

| Metric | Value |
|---|---|
| Slop-score | 51.4/100 (grade F) |
| **Slop-debt (total HARD defects)** | **735** |
| Soft signals (advisory) | 73 |

## Per-KPI (worst-first)

| KPI | Score | Slop-debt | Detail |
|---|---:|---:|---|
| duplication | 0/100 | 715 | 715 duplicated block(s) (copy-pasted across 2+ sites) |
| dead_code | 10/100 | 18 | 18 dead unexported symbol(s) |
| vacuous_tests | 80/100 | 2 | 2 vacuous of 7832 Test func(s) |
| comment_slop | 100/100 | 0 | no comment slop |
| stub_masquerade | 100/100 | 0 | no exported stub-masquerade |
| churn_bloat | 100/100 | 0 | no commits in range (skipped) |

## What each axis catches

- **duplication** — a normalized Go token-window copy-pasted into 2+ places. [HARD]
- **dead_code** — an unexported symbol defined but referenced nowhere else. [HARD]
- **comment_slop** — tautological doc comments + commented-out code blocks. [HARD]
- **vacuous_tests** — a Test/Benchmark func that makes zero assertions. [HARD]
- **stub_masquerade** — an exported func with a trivial/panic body, not `[STUB]`. [SOFT]
- **churn_bloat** — recent commits adding `.go` files without retiring any. [SOFT]

## stub_masquerade SOFT->HARD promotion (#781)

> Advisory readiness for promoting the `stub_masquerade` axis from SOFT (scores, never gates) to HARD (a gating defect). Re-derived from disk; the readout never performs the flip — moving the finding from `soft` to `defects` and bumping its weight stays a deliberate maintainer act, taken once the elapsed soak window is reviewed for zero false positives.

| Gate | State |
|---|---|
| symbol<->`[STUB]`-ledger link tight | yes |
| zero-FP soak (releases since 0.34.0) | 2/3 |
| promotable now | no |

> AWAITING SOAK: 2/3 release(s) since the detector shipped (0.34.0); stays SOFT (advisory)

> When `promotable` is yes: review the elapsed window for any false positive, then move the `stub_masquerade` finding from `soft` to `defects` and bump `KPI_WEIGHTS["stub_masquerade"]` in `tools/code_slop_scorecard.py` — the deliberate flip.

> 735 unit(s) of slop-debt; score 51.4/100 (grade F); heaviest KPI: duplication (715 defect(s))

> next: retire slop-debt worst-first (see corpus.breakdown + per-KPI defects): de-duplicate clones, delete dead unexported symbols, drop commented-out code + tautological doc comments, add assertions to vacuous tests; re-run to prove the drop
