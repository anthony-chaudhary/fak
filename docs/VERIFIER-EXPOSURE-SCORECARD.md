---
title: "Verifier exposure scorecard"
description: "How far each of fak's nine ship gates leans on model judgment instead of a deterministic check, ranked by exposure against the debt threshold of 0.35."
---

# Verifier exposure scorecard

Schema: `fak-verifier-exposure/1`  
Baseline: `verifier_exposure_debt = 0` of 9 gates at threshold `0.35` (grade `A`).

| Rank | Gate | Kind | Exposure | Above debt floor |
|---:|---|---|---:|:---:|
| 1 | `trajctl-judge` | `llm_judge` | 0.25 | false |
| 2 | `policy-smart-approval` | `deterministic` | 0.20 | false |
| 3 | `ship-integrity` | `deterministic` | 0.20 | false |
| 4 | `antipattern` | `deterministic` | 0.00 | false |
| 5 | `dos-commit-audit` | `deterministic` | 0.00 | false |
| 6 | `dos-verify` | `deterministic` | 0.00 | false |
| 7 | `kpi-tests` | `deterministic` | 0.00 | false |
| 8 | `safecommit` | `deterministic` | 0.00 | false |
| 9 | `witness-rungs-w1-w3` | `deterministic` | 0.00 | false |

Exposure is a declared-signal heuristic, not an empirical exploit probability. Higher is hardened first; missing inventory sources fail the grade closed.
