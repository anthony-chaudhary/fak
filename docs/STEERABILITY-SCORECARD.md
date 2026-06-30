---
title: "fak steerability scorecard — the growth-invariant steering index"
description: "fak's deterministic steerability scorecard: eleven growth-invariant KPIs across modularity, coupling, navigability, and correction, folded into a 0-100 steerability index that stays flat as the repo grows — re-derived from the working tree."
---

# Steerability scorecard

<!-- steerability-scorecard: 2026-06-30 · process: tools/steerability_scorecard.py -->

This is the measuring stick for the question no other fak scorecard asks: as the repo doubles in size, does the **effort to steer, change, and navigate it stay roughly flat** — and if it drifts, do we know, and can we correct? Every number below is re-derived from the working tree by `tools/steerability_scorecard.py` — no hand-entry.

The headline is a 0–100 **steerability index**, not a debt count, because steerability is a property of *shape*, not *size*. Every KPI is **growth-invariant** — a ratio, density, or distribution percentile — so a 2×-larger repo with the same modular discipline scores *identically*. (A raw defect count, the kind every sibling scorecard uses, would climb just from getting bigger.)

> Regenerate: `python tools/steerability_scorecard.py --markdown --stamp DATE > docs/STEERABILITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Steerability index** | **94.0/100 (grade A)** |
| Hard steerability-debt | 0 |
| Advisory drift signals | 3 |
| Index by group | modularity:83.5 · coupling:100.0 · navigability:100.0 · correction:100.0 |

## Per-KPI

Eleven KPIs, each 0–100, in four groups. `debt` = HARD steerability-debt (only `dispatch_god_file` and `ratchet_present` can emit it — everything else is advisory, because its cheapest fix would be gaming a detector). `god_file_rate` / `god_func_rate` SCORE the size rate but leave the raw count to `code_quality` (no portfolio double-count). `package_doc_frac` counts `// Package ...` docs for libraries and `// Command ...` docs for command packages. `churn_concentration` is HEAD-relative.

| Group | KPI | Score | Debt | Clean-gain | Detail |
|---|---|---:|:--:|---:|---|
| modularity | `func_size_dist` | 56 | 0 | +4.4 | 295/13507 functions over the soft length line (rate 2.18%) |
| modularity | `god_file_rate` | 89 | 0 | +0.8 | 3/1421 files > 1500 lines (rate 0.21%) |
| modularity | `god_func_rate` | 92 | 0 | +0.6 | 11/13507 functions > 200 lines (rate 0.08%) |
| modularity | `file_size_dist` | 97 | 0 | +0.3 | file length p50=198 p90=546 (ref 520) over 1421 files |
| coupling | `fan_in_gini` | 100 | 0 | +0.0 | fan-in Gini 0.57 over 210 packages (top abi:72, model:38, windowgate:32, benchcli:26, cachemeta:25) |
| coupling | `hub_share` | 100 | 0 | +0.0 | top hub 'abi' imported by 72/324 packages (22%) |
| coupling | `dispatch_god_file` | 100 | 0 | +0.0 | no cmd dispatch god-file |
| navigability | `package_doc_frac` | 100 | 0 | +0.0 | 324/324 packages carry an orientation doc-comment (100%) |
| correction | `ratchet_present` | 100 | 0 | +0.0 | control-pane ratchet present + this scorecard wired |
| correction | `worst_pkg_drift` | 100 | 0 | +0.0 | no package-LOC baseline pinned (informational) |
| correction | `churn_concentration` | 100 | 0 | +0.0 | churn Gini 0.55 over 39 files in HEAD~40..HEAD (HEAD-relative) |

No hard steerability-debt: the index and the advisory drift signals below carry the story. 🎉

## Drift signals (advisory)

- **`file_size_dist`** — file-length p90 546 > ref 520 — the typical file is drifting large
- **`god_file_rate`** — 3 god-file(s) (rate 0.21%) — code_quality.architecture owns the count; split along seams (/modularize)
- **`god_func_rate`** — 11 god-function(s) (rate 0.08%) — code_quality owns the count

## Highest-index moves

| Gain if clean | KPI | Why this helps | Current detail |
|---:|---|---|---|
| +4.4 | `func_size_dist` | split long routines at tested seams | 295/13507 functions over the soft length line (rate 2.18%) |
| +0.8 | `god_file_rate` | split oversized files along ownership boundaries | 3/1421 files > 1500 lines (rate 0.21%) |
| +0.6 | `god_func_rate` | split hard-to-review functions before they become shared chokepoints | 11/13507 functions > 200 lines (rate 0.08%) |
| +0.3 | `file_size_dist` | keep typical files below the p90 size line | file length p50=198 p90=546 (ref 520) over 1421 files |

