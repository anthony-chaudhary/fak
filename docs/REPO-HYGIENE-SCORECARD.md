---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

<!-- repo-hygiene-scorecard: 2026-09-01 · process: tools/repo_hygiene_scorecard.py -->

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **13** |
| **a11y-debt (accessibility HARD defects)** | **0** |
| Composite score | 86.3/100 (grade B) |
| Advisory (soft) signals | 598 |
| Debt by group | verbosity:10 · organization:3 · indexing:0 · accessibility:0 |

## Per-KPI

Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| verbosity | `bloat` | 0 | 10 | 10 oversized, 14 long |
| organization | `dir_discipline` | 64 | 3 | 3 near-duplicate dir group(s) |
| accessibility | `plain_language` | 65 | 0 | 234 dense doc(s), 498 doc(s) with undefined acronyms, 23 literal-reader idiom(s) |
| verbosity | `redundancy` | 80 | 0 | 0 near-duplicate pair(s), 8 candidate(s) |
| accessibility | `ai_tells` | 85 | 0 | no AI-tell phrases |
| accessibility | `jargon` | 86 | 0 | 312 naked first-screen jargon term(s) (0.3/doc) |
| organization | `root_hygiene` | 100 | 0 | root holds only front-door / meta files |
| organization | `placement` | 100 | 0 | dated docs live under docs/notes/ |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| indexing | `orphans` | 100 | 0 | 844/844 reader-facing docs reachable from an index (100.0%) |
| accessibility | `alt_text` | 100 | 0 | every doc image carries alt-text |

## Hygiene-debt work-list

### `bloat` (verbosity) — 10 defect(s), score 0
- oversized doc INDEX.md (1091 lines > 1000) — split into sections or trim
- oversized doc LEARNING-PATH.md (2869 lines > 1000) — split into sections or trim
- oversized doc docs/FAQ.md (2947 lines > 1000) — split into sections or trim
- oversized doc docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.md (1031 lines > 1000) — split into sections or trim
- oversized doc docs/cli-reference.md (1739 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/INDEX.md (4068 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/README.md (3035 lines > 1000) — split into sections or trim
- oversized doc docs/fak/concept-glossary.md (2616 lines > 1000) — split into sections or trim
- oversized doc docs/generated/disambiguation/canonical-terms.md (1024 lines > 1000) — split into sections or trim
- oversized doc docs/generated/verb-surface.md (1073 lines > 1000) — split into sections or trim

### `dir_discipline` (organization) — 3 defect(s), score 64
- near-duplicate sibling dirs: ['config', 'configs'] — merge into one
- near-duplicate sibling dirs: ['internal/ctxplan', 'internal/ctxplans'] — merge into one
- near-duplicate sibling dirs: ['internal/market', 'internal/marketing'] — merge into one

