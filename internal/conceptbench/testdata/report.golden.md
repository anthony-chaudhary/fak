# conceptbench leaderboard

- schema: fak.conceptbench.report.v1
- generated: 2026-07-08T00:00:00Z
- result_claim_allowed: true
- honesty_gate: 4 headline row(s), all referee-witnessed; 1 replay row(s) labeled and excluded — result claim allowed

## model × concept fidelity (pass@1)

| model | commit_stamp | lane | honesty | rollup |
| --- | --- | --- | --- | --- |
| claude-opus-4-8 | **1.00** | **1.00** | replay | 1.00 |
| glm-4-6 | 0.00 | 1.00 | — | 0.50 |

winner per concept: commit_stamp → claude-opus-4-8 · lane → claude-opus-4-8 · honesty → (no measured result)

## per-model rollup

| model | fidelity | pass/total | guard_refusal_rate | no_commit_reasons | tokens/turn | wall_clock_s |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-8 | 1.00 | 2/2 | 0.00 | — | 400.0 | 10.0 |
| glm-4-6 | 0.50 | 1/2 | 0.50 | OFF_TRUNK×1 | 450.0 | 22.5 |
