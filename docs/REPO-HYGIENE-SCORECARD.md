---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

<!-- repo-hygiene-scorecard: 2026-07-28 · process: tools/repo_hygiene_scorecard.py -->

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **16** |
| **a11y-debt (accessibility HARD defects)** | **2** |
| Composite score | 81.3/100 (grade B) |
| Advisory (soft) signals | 513 |
| Debt by group | verbosity:6 · organization:8 · indexing:0 · accessibility:2 |

## Per-KPI

Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| verbosity | `bloat` | 8 | 6 | 6 oversized, 15 long |
| organization | `placement` | 60 | 4 | 4 misplaced dated doc(s) |
| organization | `dir_discipline` | 64 | 3 | 3 near-duplicate dir group(s) |
| accessibility | `ai_tells` | 79 | 2 | 2 AI-tell phrase(s) across 431 doc(s) |
| organization | `root_hygiene` | 90 | 1 | 1 stray root doc(s), 0 stray root file(s) |
| accessibility | `plain_language` | 65 | 0 | 82 dense doc(s), 334 doc(s) with undefined acronyms, 20 literal-reader idiom(s) |
| verbosity | `redundancy` | 82 | 0 | 0 near-duplicate pair(s), 6 candidate(s) |
| accessibility | `jargon` | 83 | 0 | 210 naked first-screen jargon term(s) (0.4/doc) |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| indexing | `orphans` | 100 | 0 | 494/494 reader-facing docs reachable from an index (100.0%) |
| accessibility | `alt_text` | 100 | 0 | every doc image carries alt-text |

## Hygiene-debt work-list

### `bloat` (verbosity) — 6 defect(s), score 8
- oversized doc LEARNING-PATH.md (2489 lines > 1000) — split into sections or trim
- oversized doc docs/FAQ.md (2946 lines > 1000) — split into sections or trim
- oversized doc docs/blast-radius-containment-cohort.md (1018 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/INDEX.md (2878 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/README.md (2333 lines > 1000) — split into sections or trim
- oversized doc docs/fak/concept-glossary.md (2011 lines > 1000) — split into sections or trim

### `placement` (organization) — 4 defect(s), score 60
- dated/research doc outside docs/notes/: docs/archive/README-2026-06-25-before-fresh-start.md → move it and index it
- dated/research doc outside docs/notes/: docs/cache-frontier/reviews/2026-06-29.md → move it and index it
- dated/research doc outside docs/notes/: docs/cache-frontier/reviews/2026-07-04.md → move it and index it
- dated/research doc outside docs/notes/: tools/self_improve.runs/FIRST-CYCLE-2026-06-26.md → move it and index it

### `dir_discipline` (organization) — 3 defect(s), score 64
- near-duplicate sibling dirs: ['docs/benchmark', 'docs/benchmarking', 'docs/benchmarks'] — merge into one
- near-duplicate sibling dirs: ['internal/ctxplan', 'internal/ctxplans'] — merge into one
- near-duplicate sibling dirs: ['internal/market', 'internal/marketing'] — merge into one

### `ai_tells` (accessibility) — 2 defect(s), score 79
- AI-tell phrase in LEARNING-PATH.md: “at the heart of” — say it plainly
- AI-tell phrase in docs/concept-disambiguation-scorecard/README.md: “bespoke” — say it plainly

### `root_hygiene` (organization) — 1 defect(s), score 90
- non-front-door doc at root: MODEL-ARCH-SEAM.md → move to docs/notes/ (or add to the allowlist if genuinely a root doc)

