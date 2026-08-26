---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

<!-- repo-hygiene-scorecard: 2026-08-26 · process: tools/repo_hygiene_scorecard.py -->

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **48** |
| **a11y-debt (accessibility HARD defects)** | **1** |
| Composite score | 68.2/100 (grade D) |
| Advisory (soft) signals | 602 |
| Debt by group | verbosity:26 · organization:21 · indexing:0 · accessibility:1 |

## Per-KPI

Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| organization | `placement` | 0 | 18 | 18 misplaced dated doc(s) |
| verbosity | `redundancy` | 0 | 16 | 16 near-duplicate pair(s), 18 candidate(s) |
| verbosity | `bloat` | 0 | 10 | 10 oversized, 13 long |
| organization | `dir_discipline` | 64 | 3 | 3 near-duplicate dir group(s) |
| accessibility | `ai_tells` | 82 | 1 | 1 AI-tell phrase(s) across 510 doc(s) |
| accessibility | `plain_language` | 67 | 0 | 196 dense doc(s), 452 doc(s) with undefined acronyms, 23 literal-reader idiom(s) |
| accessibility | `jargon` | 86 | 0 | 288 naked first-screen jargon term(s) (0.3/doc) |
| organization | `root_hygiene` | 100 | 0 | root holds only front-door / meta files |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| indexing | `orphans` | 100 | 0 | 803/803 reader-facing docs reachable from an index (100.0%) |
| accessibility | `alt_text` | 100 | 0 | every doc image carries alt-text |

## Hygiene-debt work-list

### `placement` (organization) — 18 defect(s), score 0
- dated/research doc outside docs/notes/: docs/_witnesses/dispatch-thread-pressure-2026-08-14.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/end-to-end-value-chain-selfcheck-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/fleet-res-rollup-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/DOGFOOD-LIVE-WORK-2026-08-22.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/EDGE-ADVERSARIAL-2026-08-20.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/LIVE-CODING-GPT-5.6-SOL-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/LIVE-FAK-L4-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/LIVE-NATIVE-PROBE-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/OPERATOR-HOME-2026-08-20.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/SHIPPED-FAK-LAUNCH-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/manage-parity-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/manage-parity-hooks-2026-08-14.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/managed-tool-search-compat-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/micro-paired-value-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/microagent-real-kernel-2026-08-12.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/wip-ownership-seam-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/claims/fan-out-benchmark-fanbench-one-master-goal-n-sub-agents-n-1-1024.md → move it and index it

### `redundancy` (verbosity) — 16 defect(s), score 0
- near-duplicate (96%): docs/concepts/disambiguation-attention.md ≈ docs/concepts/disambiguation-decision.md — consolidate to one
- near-duplicate (96%): docs/concepts/disambiguation-attention.md ≈ docs/concepts/disambiguation-evict.md — consolidate to one
- near-duplicate (81%): docs/concepts/disambiguation-attention.md ≈ docs/concepts/disambiguation-layout.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-context-ctx.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-gateway-engine.md — consolidate to one
- near-duplicate (96%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-loop.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one
- near-duplicate (92%): docs/concepts/disambiguation-context-ctx.md ≈ docs/concepts/disambiguation-gateway-engine.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-context-ctx.md ≈ docs/concepts/disambiguation-loop.md — consolidate to one
- near-duplicate (92%): docs/concepts/disambiguation-context-ctx.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one
- near-duplicate (96%): docs/concepts/disambiguation-decision.md ≈ docs/concepts/disambiguation-evict.md — consolidate to one
- near-duplicate (81%): docs/concepts/disambiguation-evict.md ≈ docs/concepts/disambiguation-layout.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-gateway-engine.md ≈ docs/concepts/disambiguation-loop.md — consolidate to one
- near-duplicate (92%): docs/concepts/disambiguation-gateway-engine.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one
- near-duplicate (93%): docs/concepts/disambiguation-layout.md ≈ docs/concepts/disambiguation-score-debt.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-loop.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one

### `bloat` (verbosity) — 10 defect(s), score 0
- oversized doc INDEX.md (1005 lines > 1000) — split into sections or trim
- oversized doc LEARNING-PATH.md (2763 lines > 1000) — split into sections or trim
- oversized doc docs/FAQ.md (2947 lines > 1000) — split into sections or trim
- oversized doc docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.md (1031 lines > 1000) — split into sections or trim
- oversized doc docs/cli-reference.md (1461 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/INDEX.md (3674 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/README.md (2813 lines > 1000) — split into sections or trim
- oversized doc docs/fak/concept-glossary.md (2389 lines > 1000) — split into sections or trim
- oversized doc docs/generated/disambiguation/canonical-terms.md (1024 lines > 1000) — split into sections or trim
- oversized doc docs/generated/verb-surface.md (1102 lines > 1000) — split into sections or trim

### `dir_discipline` (organization) — 3 defect(s), score 64
- near-duplicate sibling dirs: ['docs/benchmark', 'docs/benchmarking', 'docs/benchmarks'] — merge into one
- near-duplicate sibling dirs: ['internal/ctxplan', 'internal/ctxplans'] — merge into one
- near-duplicate sibling dirs: ['internal/market', 'internal/marketing'] — merge into one

### `ai_tells` (accessibility) — 1 defect(s), score 82
- AI-tell phrase in docs/concept-disambiguation-scorecard/README.md: “bespoke” — say it plainly

