---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

<!-- repo-hygiene-scorecard: 2026-06-23 · process: tools/repo_hygiene_scorecard.py -->

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **3** |
| **a11y-debt (accessibility HARD defects)** | **0** |
| Composite score | 93.3/100 (grade A) |
| Advisory (soft) signals | 149 |
| Debt by group | verbosity:2 · organization:1 · indexing:0 · accessibility:0 |

## Per-KPI

Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| verbosity | `bloat` | 70 | 2 | 2 oversized, 3 long |
| organization | `dir_discipline` | 88 | 1 | 1 near-duplicate dir group(s) |
| accessibility | `plain_language` | 62 | 0 | 15 dense doc(s), 65 doc(s) with undefined acronyms, 7 literal-reader idiom(s) |
| accessibility | `jargon` | 74 | 0 | 59 naked first-screen jargon term(s) (0.6/doc) |
| accessibility | `ai_tells` | 85 | 0 | no AI-tell phrases |
| verbosity | `redundancy` | 100 | 0 | no near-duplicate docs |
| organization | `root_hygiene` | 100 | 0 | root holds only front-door / meta files |
| organization | `placement` | 100 | 0 | dated docs live under docs/notes/ |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| indexing | `orphans` | 100 | 0 | 88/88 reader-facing docs reachable from an index (100.0%) |
| accessibility | `alt_text` | 100 | 0 | every doc image carries alt-text |

## Hygiene-debt work-list

### `bloat` (verbosity) — 2 defect(s), score 70
- oversized doc LEARNING-PATH.md (2079 lines > 1000) — split into sections or trim
- oversized doc docs/FAQ.md (2662 lines > 1000) — split into sections or trim

### `dir_discipline` (organization) — 1 defect(s), score 88
- near-duplicate sibling dirs: ['docs/benchmark', 'docs/benchmarking', 'docs/benchmarks'] — merge into one

