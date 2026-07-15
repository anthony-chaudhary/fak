# Verifier exposure scorecard

Schema: `fak-verifier-exposure/1`  
Baseline: `verifier_exposure_debt = 1` of 9 gates at threshold `0.35` (grade `B`).

| Rank | Gate | Kind | Exposure | Above debt floor |
|---:|---|---|---:|:---:|
| 1 | `kpi-tests` | `self_report` | 0.80 | true |
| 2 | `trajctl-judge` | `llm_judge` | 0.25 | false |
| 3 | `antipattern` | `deterministic` | 0.00 | false |
| 4 | `dos-commit-audit` | `deterministic` | 0.00 | false |
| 5 | `dos-verify` | `deterministic` | 0.00 | false |
| 6 | `policy-smart-approval` | `deterministic` | 0.00 | false |
| 7 | `safecommit` | `deterministic` | 0.00 | false |
| 8 | `ship-integrity` | `deterministic` | 0.00 | false |
| 9 | `witness-rungs-w1-w3` | `deterministic` | 0.00 | false |

Exposure is a declared-signal heuristic, not an empirical exploit probability. Higher is hardened first; missing inventory sources fail the grade closed.
