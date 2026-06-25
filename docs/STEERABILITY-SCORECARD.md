---
title: "fak steerability scorecard — the growth-invariant steering index"
description: "fak's deterministic steerability scorecard: eleven growth-invariant KPIs across modularity, coupling, navigability, and correction, folded into a 0-100 steerability index that stays flat as the repo grows — re-derived from the working tree."
---

# Steerability scorecard

<!-- steerability-scorecard: 2026-06-25 · process: tools/steerability_scorecard.py -->

This is the measuring stick for the question no other fak scorecard asks: as the repo doubles in size, does the **effort to steer, change, and navigate it stay roughly flat** — and if it drifts, do we know, and can we correct? Every number below is re-derived from the working tree by `tools/steerability_scorecard.py` — no hand-entry.

The headline is a 0–100 **steerability index**, not a debt count, because steerability is a property of *shape*, not *size*. Every KPI is **growth-invariant** — a ratio, density, or distribution percentile — so a 2×-larger repo with the same modular discipline scores *identically*. (A raw defect count, the kind every sibling scorecard uses, would climb just from getting bigger.)

> Regenerate: `python tools/steerability_scorecard.py --markdown --stamp DATE > docs/STEERABILITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Steerability index** | **87.7/100 (grade B)** |
| Hard steerability-debt | 0 |
| Advisory drift signals | 4 |
| Index by group | modularity:82.5 · coupling:93.7 · navigability:65.0 · correction:100.0 |

## Per-KPI

Eleven KPIs, each 0–100, in four groups. `debt` = HARD steerability-debt (only `dispatch_god_file` and `ratchet_present` can emit it — everything else is advisory, because its cheapest fix would be gaming a detector). `god_file_rate` / `god_func_rate` SCORE the size rate but leave the raw count to `code_quality` (no portfolio double-count). `churn_concentration` is HEAD-relative.

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| modularity | `func_size_dist` | 45 | 0 | 140/5070 functions over the soft length line (rate 2.76%) |
| navigability | `package_doc_frac` | 65 | 0 | 103/159 packages carry a package doc-comment (65%) |
| coupling | `hub_share` | 81 | 0 | top hub 'abi' imported by 60/159 packages (38%) |
| modularity | `god_file_rate` | 91 | 0 | 1/586 files > 1500 lines (rate 0.17%) |
| modularity | `god_func_rate` | 94 | 0 | 3/5070 functions > 200 lines (rate 0.06%) |
| modularity | `file_size_dist` | 100 | 0 | file length p50=199 p90=519 (ref 520) over 586 files |
| coupling | `fan_in_gini` | 100 | 0 | fan-in Gini 0.59 over 78 packages (top abi:60, model:34, cachemeta:19, pathutil:17, appversion:16) |
| coupling | `dispatch_god_file` | 100 | 0 | no cmd dispatch god-file |
| correction | `ratchet_present` | 100 | 0 | control-pane ratchet present + this scorecard wired |
| correction | `worst_pkg_drift` | 100 | 0 | no package-LOC baseline pinned (informational) |
| correction | `churn_concentration` | 100 | 0 | churn Gini 0.55 over 34 files in HEAD~40..HEAD (HEAD-relative) |

No hard steerability-debt: the index and the advisory drift signals below carry the story. 🎉

## Drift signals (advisory)

- **`god_file_rate`** — 1 god-file(s) (rate 0.17%) — code_quality.architecture owns the count; split along seams (/modularize)
- **`god_func_rate`** — 3 god-function(s) (rate 0.06%) — code_quality owns the count
- **`hub_share`** — 'abi' is a chokepoint: imported by 38% of packages — a change to it ripples wide
- **`package_doc_frac`** — 56 package(s) without a doc-comment header — a reader/agent has no one-line orientation

