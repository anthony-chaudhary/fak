---
title: "fak Code-Slop Scorecard: the slop the compiler can't see"
description: "fak's code-slop scorecard grades the Go module on six deterministic slop axes into a slop-score (ceiling 100, A-F, unbounded below) and a re-derivable slop-debt count — clones, vacuous tests, dead code, comment cruft."
---

# Code-slop scorecard

<!-- code-slop-scorecard: 2026-09-01 · process: tools/code_slop_scorecard.py -->

> Regenerate: `python tools/code_slop_scorecard.py --markdown --stamp DATE > docs/CODE-SLOP-SCORECARD.md`
> Verify snapshot freshness: `python tools/code_slop_scorecard.py --check-doc`

> The measuring stick for **slop the compiler can't see**: code that builds, vets clean, and has a test present, yet rots the kernel from the inside — copy-paste clones, tests that assert nothing, dead unexported symbols, commented-out code and tautological doc comments. Six deterministic axes (duplication · dead_code · comment_slop · vacuous_tests · stub_masquerade · churn_bloat), folded into a **slop-score** (ceiling 100 at zero defects, A–F, unbounded below so partial progress on a heavy-debt axis stays visible) and a **slop-debt** integer (the count of concrete, re-derivable slop defects). Every number below is re-derived from disk by `tools/code_slop_scorecard.py` — no hand-entry. Drive slop-debt to zero to make "less slop" provable.

## Corpus

**Slop-debt (total HARD defects): 107** — the primary, unbounded metric. Drive it to 0; every retired clone/dead symbol/vacuous test moves it. (The /100 score below saturates at zero defects and is NOT the driver.)

| Metric | Value |
|---|---|
| **Slop-debt (total HARD defects)** | **107** |
| Legacy bounded score (saturates; not the driver) | -5.6/100 (grade F) |
| Soft signals (advisory) | 2312 |

## Per-KPI (worst-first)

| KPI | Score | Slop-debt | Detail |
|---|---:|---:|---|
| duplication | -306/100 | 107 | 203 duplicated block(s): 11 extractable · 105 local · 87 pair (payoff-weighted debt 107.0) |
| dead_code | 100/100 | 0 | no dead unexported symbols |
| comment_slop | 100/100 | 0 | no comment slop |
| vacuous_tests | 100/100 | 0 | 32128 Test func(s), all assert |
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
| zero-FP soak (releases since 0.34.0) | 11/3 |
| promotable now | yes |

> READY: soak window elapsed and tree clean — a maintainer may review the window for zero FP, then move soft->defects + bump the weight

> When `promotable` is yes: review the elapsed window for any false positive, then move the `stub_masquerade` finding from `soft` to `defects` and bump `KPI_WEIGHTS["stub_masquerade"]` in `tools/code_slop_scorecard.py` — the deliberate flip.

> 107 unit(s) of slop-debt; score -5.6/100 (grade F); heaviest KPI: duplication (107 defect(s))

> next: retire slop-debt worst-first (see corpus.breakdown + per-KPI defects): de-duplicate clones, delete dead unexported symbols, drop commented-out code + tautological doc comments, add assertions to vacuous tests; re-run to prove the drop
