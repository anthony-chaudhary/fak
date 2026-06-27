---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

<!-- repo-hygiene-scorecard: 2026-06-27 · process: tools/repo_hygiene_scorecard.py -->

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **24** |
| **a11y-debt (accessibility HARD defects)** | **1** |
| Composite score | 87.4/100 (grade B) |
| Advisory (soft) signals | 234 |
| Debt by group | verbosity:2 · organization:5 · indexing:16 · accessibility:1 |

## Per-KPI

Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| indexing | `orphans` | 91 | 16 | 163/179 reader-facing docs reachable from an index (91.1%) |
| organization | `placement` | 60 | 4 | 4 misplaced dated doc(s) |
| verbosity | `bloat` | 60 | 2 | 2 oversized, 8 long |
| accessibility | `ai_tells` | 82 | 1 | 1 AI-tell phrase(s) across 164 doc(s) |
| organization | `dir_discipline` | 88 | 1 | 1 near-duplicate dir group(s) |
| accessibility | `plain_language` | 60 | 0 | 31 dense doc(s), 138 doc(s) with undefined acronyms, 11 literal-reader idiom(s) |
| accessibility | `jargon` | 77 | 0 | 105 naked first-screen jargon term(s) (0.5/doc) |
| verbosity | `redundancy` | 100 | 0 | no near-duplicate docs |
| organization | `root_hygiene` | 100 | 0 | root holds only front-door / meta files |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| accessibility | `alt_text` | 100 | 0 | every doc image carries alt-text |

## Hygiene-debt work-list

### `orphans` (indexing) — 16 defect(s), score 91
- orphan (reachable from no index/hub): BENCHMARK-GALLERY.md — index it or delete it
- orphan (reachable from no index/hub): docs/BENCH-DX-SCORECARD.md — index it or delete it
- orphan (reachable from no index/hub): docs/CODE-SLOP-SCORECARD.md — index it or delete it
- orphan (reachable from no index/hub): docs/MEMORY-ECC-INTEGRITY.md — index it or delete it
- orphan (reachable from no index/hub): docs/baselines/SYSTEMS-BASELINE-2026-06-26.md — index it or delete it
- orphan (reachable from no index/hub): docs/checkpoint-go-port-decision.md — index it or delete it
- orphan (reachable from no index/hub): docs/dispatch-status.md — index it or delete it
- orphan (reachable from no index/hub): docs/explainers/context-signal-to-noise.md — index it or delete it
- orphan (reachable from no index/hub): docs/fak/concept-glossary.md — index it or delete it
- orphan (reachable from no index/hub): docs/fak/dogfood-loop-scorecard.md — index it or delete it
- orphan (reachable from no index/hub): docs/fak/dojo.md — index it or delete it
- orphan (reachable from no index/hub): docs/fak/guard-verdict-rsi-loop.md — index it or delete it
- orphan (reachable from no index/hub): docs/fak/mcp-registry.md — index it or delete it
- orphan (reachable from no index/hub): docs/fak/qwen36-a100-gcp.md — index it or delete it
- orphan (reachable from no index/hub): docs/model-accounts.md — index it or delete it
- orphan (reachable from no index/hub): docs/region.md — index it or delete it

### `placement` (organization) — 4 defect(s), score 60
- dated/research doc outside docs/notes/: docs/archive/README-2026-06-25-before-fresh-start.md → move it and index it
- dated/research doc outside docs/notes/: docs/baselines/SYSTEMS-BASELINE-2026-06-26.md → move it and index it
- dated/research doc outside docs/notes/: docs/benchmarks/QWEN36-LOAD-PROFILE-440.md → move it and index it
- dated/research doc outside docs/notes/: tools/self_improve.runs/FIRST-CYCLE-2026-06-26.md → move it and index it

### `bloat` (verbosity) — 2 defect(s), score 60
- oversized doc LEARNING-PATH.md (2088 lines > 1000) — split into sections or trim
- oversized doc docs/FAQ.md (2676 lines > 1000) — split into sections or trim

### `dir_discipline` (organization) — 1 defect(s), score 88
- near-duplicate sibling dirs: ['docs/benchmark', 'docs/benchmarking', 'docs/benchmarks'] — merge into one

### `ai_tells` (accessibility) — 1 defect(s), score 82
- AI-tell phrase in docs/integrations/litellm.md: “bespoke” — say it plainly

