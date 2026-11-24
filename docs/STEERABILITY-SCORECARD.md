---
title: "fak steerability scorecard — the growth-invariant steering index"
description: "fak's deterministic steerability scorecard: eleven growth-invariant KPIs across modularity, coupling, navigability, and correction, folded into a 0-100 steerability index that stays flat as the repo grows — re-derived from the working tree."
---

# Steerability scorecard

<!-- steerability-scorecard: 2026-11-24 · process: tools/steerability_scorecard.py -->

This is the measuring stick for the question no other fak scorecard asks: as the repo doubles in size, does the **effort to steer, change, and navigate it stay roughly flat** — and if it drifts, do we know, and can we correct? Every number below is re-derived from the working tree by `tools/steerability_scorecard.py` — no hand-entry.

The headline is a 0–100 **steerability index**, not a debt count, because steerability is a property of *shape*, not *size*. Every KPI is **growth-invariant** — a ratio, density, or distribution percentile — so a 2×-larger repo with the same modular discipline scores *identically*. (A raw defect count, the kind every sibling scorecard uses, would climb just from getting bigger.)

> Regenerate: `python tools/steerability_scorecard.py --markdown --stamp DATE > docs/STEERABILITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Steerability index** | **79.3/100 (grade C)** |
| **Steering pressure** | **3131** (unbounded, lower is better) |
| Hard steerability-debt | 6 |
| Advisory drift signals | 4 |
| Index by group | modularity:68.2 · coupling:66.7 · navigability:84.0 · correction:100.0 |

## Per-KPI

Eleven KPIs, each 0–100, in four groups. `debt` = HARD steerability-debt (only `dispatch_god_file` and `ratchet_present` can emit it — everything else is advisory, because its cheapest fix would be gaming a detector). `god_file_rate` / `god_func_rate` SCORE the size rate but leave the raw count to `code_quality` (no portfolio double-count). `package_doc_frac` counts `// Package ...` docs for libraries and `// Command ...` docs for command packages. `churn_concentration` is HEAD-relative.

| Group | KPI | Score | Pressure | Debt | Clean-gain | Detail |
|---|---|---:|---:|:--:|---:|---|
| coupling | `dispatch_god_file` | 0 | 6 | 6 | +8.0 | 6 cmd dispatch god-file(s) |
| modularity | `func_size_dist` | 33 | 2607 | 0 | +6.7 | 2607/77299 functions over the soft length line (rate 3.37%) |
| modularity | `god_func_rate` | 69 | 237 | 0 | +2.2 | 237/77299 functions > 200 lines (rate 0.31%) |
| navigability | `package_doc_frac` | 84 | 184 | 0 | +1.6 | 977/1161 packages carry an orientation doc-comment (84%) |
| modularity | `file_size_dist` | 94 | 56 | 0 | +0.6 | file length p50=184 p90=576 (ref 520) over 8965 files |
| modularity | `god_file_rate` | 77 | 41 | 0 | +1.6 | 41/8965 files > 1500 lines (rate 0.46%) |
| coupling | `fan_in_gini` | 100 | 0 | 0 | +0.0 | fan-in Gini 0.56 over 819 packages (top abi:106, windowgate:101, model:54, scorecard:51, strmatch:46) |
| coupling | `hub_share` | 100 | 0 | 0 | +0.0 | top hub 'abi' imported by 106/1161 packages (9%) |
| correction | `ratchet_present` | 100 | 0 | 0 | +0.0 | control-pane ratchet present + this scorecard wired |
| correction | `worst_pkg_drift` | 100 | 0 | 0 | +0.0 | no package-LOC baseline pinned (informational) |
| correction | `churn_concentration` | 100 | 0 | 0 | +0.0 | churn Gini 0.57 over 62 files in HEAD~40..HEAD (HEAD-relative) |

## Hard-debt work-list

### `dispatch_god_file` (coupling) — 6 defect(s), score 0
- cmd dispatch god-file cmd/fak/guard.go (1553 lines > 1500) — split the verb table so a new command doesn't fight a monolith
- cmd dispatch god-file cmd/fak/release_ship.go (1586 lines > 1500) — split the verb table so a new command doesn't fight a monolith
- cmd dispatch god-file cmd/fak/serve.go (2203 lines > 1500) — split the verb table so a new command doesn't fight a monolith
- cmd dispatch god-file cmd/fak/serve_load_helpers.go (1511 lines > 1500) — split the verb table so a new command doesn't fight a monolith
- cmd dispatch god-file cmd/fak/serve_model_fit.go (1541 lines > 1500) — split the verb table so a new command doesn't fight a monolith
- cmd dispatch god-file cmd/fak/worktree_worker.go (1535 lines > 1500) — split the verb table so a new command doesn't fight a monolith

## Drift signals (advisory)

- **`file_size_dist`** — file-length p90 576 > ref 520 — the typical file is drifting large
- **`god_file_rate`** — 41 god-file(s) (rate 0.46%) — code_quality.architecture owns the count; split along seams (/modularize)
- **`god_func_rate`** — 237 god-function(s) (rate 0.31%) — code_quality owns the count
- **`package_doc_frac`** — 184 package(s) without a doc-comment header — a reader/agent has no one-line orientation

## Highest-index moves

| Gain if clean | KPI | Why this helps | Current detail |
|---:|---|---|---|
| +8.0 | `dispatch_god_file` | split command dispatch so new verbs avoid one monolith | 6 cmd dispatch god-file(s) |
| +6.7 | `func_size_dist` | split long routines at tested seams | 2607/77299 functions over the soft length line (rate 3.37%) |
| +2.2 | `god_func_rate` | split hard-to-review functions before they become shared chokepoints | 237/77299 functions > 200 lines (rate 0.31%) |
| +1.6 | `package_doc_frac` | add one useful orientation sentence where readers start | 977/1161 packages carry an orientation doc-comment (84%) |
| +1.6 | `god_file_rate` | split oversized files along ownership boundaries | 41/8965 files > 1500 lines (rate 0.46%) |

