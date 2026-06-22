---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

<!-- repo-hygiene-scorecard: 2026-06-22 · process: tools/repo_hygiene_scorecard.py -->

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **5** |
| Composite score | 89.3/100 (grade B) |
| Advisory (soft) signals | 145 |
| Debt by group | verbosity:0 · organization:5 · indexing:0 · accessibility:0 |

## Per-KPI

Eleven KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| organization | `placement` | 60 | 4 | 4 misplaced dated doc(s) |
| organization | `dir_discipline` | 88 | 1 | 1 near-duplicate dir group(s) |
| accessibility | `plain_language` | 64 | 0 | 15 dense doc(s), 61 doc(s) with undefined acronyms, 2 literal-reader idiom(s) |
| accessibility | `jargon` | 74 | 0 | 57 naked first-screen jargon term(s) (0.6/doc) |
| accessibility | `ai_tells` | 85 | 0 | no AI-tell phrases |
| verbosity | `bloat` | 94 | 0 | 0 oversized, 3 long |
| verbosity | `redundancy` | 100 | 0 | no near-duplicate docs |
| organization | `root_hygiene` | 100 | 0 | root holds only front-door / meta files |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| indexing | `orphans` | 100 | 0 | 85/85 reader-facing docs reachable from an index (100.0%) |

## Hygiene-debt work-list

### `placement` (organization) — 4 defect(s), score 60
- dated/research doc outside docs/notes/: docs/SCALING-LAWS-OF-AGENTS-2026-06-19.md → move it and index it
- dated/research doc outside docs/notes/: docs/gpu-parity-tracking-480.md → move it and index it
- dated/research doc outside docs/notes/: docs/model-arch-seam-status-487.md → move it and index it
- dated/research doc outside docs/notes/: docs/trust-floor-decomposition-492.md → move it and index it

### `dir_discipline` (organization) — 1 defect(s), score 88
- near-duplicate sibling dirs: ['docs/benchmark', 'docs/benchmarking', 'docs/benchmarks'] — merge into one

