# Guarded Codex workflow-default uptake: first paired-study snapshot (2026-08-17)

## Verdict

The shipped default is reaching guarded sessions, but current evidence does **not** yet show an effectiveness gain. At the captured boundary, 15/15 workflow-default witnesses joined a guard witness, while only 1/11 `consider-workflow` decisions joined a downstream orchestration receipt and none proved a worker launch. Four short prompts were classified `likely-direct`; no orchestration receipt was observed for them, so the measured trivial-task false-positive rate is 0/4 at the receipt boundary. Fourteen outcomes remain unknown. These are uptake observations, not causal productivity evidence.

## Frozen paired task set

Each row must run twice through the same Codex build, model, reasoning setting, repository tip, sandbox, approval posture, and token/time ceiling. The guarded-default arm uses the installed `UserPromptSubmit` hook. The explicit-direct control arm sets the documented direct override and records that override. A run is complete only when both arms have terminal status and the listed evidence.

| ID | Shape | Frozen task | Expected default behavior | Required evidence |
|---|---|---|---|---|
| T1 | trivial | Read `go.mod` and report the declared Go version; do not edit. | direct | transcript, terminal status, elapsed time, token usage |
| T2 | multi-step serial | Audit one named CLI help surface against its implementation and tests; report mismatches only. | consider workflow, but one worker may remain optimal | decision witness, invocation/decline receipt, transcript, elapsed/tokens |
| T3 | parallelizable | Compare three named integration docs against their corresponding commands and produce one discrepancy table. | invoke an orchestration profile or emit a typed decline | decision witness, receipt, worker/lease rows, independent read-back |
| T4 | unattended | Run one bounded read-only issue-triage pass and emit the top five dispatchable agent-runtime issues. | invoke guarded workflow | decision witness, receipt, worker/lease rows, terminal status |
| T5 | implementation | Fix one preselected one-point agent-runtime issue with a failing-before/passing-after test. | invoke guarded workflow when safe | commit, test witness, receipt, worker/lease rows, elapsed/tokens |

The task text is frozen here before paired execution so later selection cannot favor either arm. T5 must name its issue before either arm starts; both arms operate from equivalent detached snapshots and only one result may land.

## Captured aggregate

Command:

```text
fak sessions workflow-default-report --codex-home C:\Users\USER\.codex-anthonydefault1 --json
```

Captured result (`_scratch/workflow-default-dogfood/report.json`, SHA-256 `ebbee7249f468a15454e2dd83ed10a3b0c82463eb591b38adbb7d457d8004858`):

| Signal | Count | Provenance |
|---|---:|---|
| witness files / valid decisions | 15 / 15 | FAK-authored local decision witnesses |
| joined guard witnesses | 15 | FAK-authored join evidence |
| `consider-workflow` | 11 | FAK-authored classification |
| `likely-direct` | 4 | FAK-authored classification |
| observed workflow use | 1 | observed joined orchestration receipt |
| observed direct declines | 0 | observed joined orchestration receipt |
| observed worker launches | 0 | observed receipt field |
| unknown outcome | 14 | no joined downstream receipt |

The local witness and receipt files are privacy-sensitive and remain outside git. Their digest inventory was captured during the run; the aggregate above is the public, provenance-bounded artifact. The report came from `cmd/fak` containing the #7112, #7135, and #7136 implementation lineage (at least `cmd/fak@r2876+g51d535c937`).

A contemporaneous `fak sessions codex-loop --recent --json` scan covered 20 sessions and reported 19 `provider=fak`, one `provider=openai`, and 11 unguarded sessions. That mixed observational window is unsuitable as a control arm: it lacks frozen tasks and equal budgets. It is therefore excluded from any effect estimate.

## Initial reads

- **Trivial false positives:** 0/4 at the only currently observable boundary (no joined orchestration receipt). This does not prove zero extra reasoning or latency inside the model.
- **Multi-step false negatives:** at least 10/11 `consider-workflow` rows have unknown downstream outcome. Unknown is not the same as a decline, bypass, or false negative.
- **Realized launches:** 0. The one observed invocation resolved to an ultracode profile, but the current receipt does not prove a worker started.
- **Cost and speed:** not comparable yet. The aggregate does not join per-run billed tokens or elapsed time, and no matched control pairs have completed.
- **Causality:** unclaimed. Injection counts prove operation of a FAK-authored default, not improved task outcomes.

## Next execution gate

Complete T1-T5 in paired arms and publish one row per arm with model, budget, tip, decision, observed invocation/decline, launch count, independent witness, terminal status, elapsed time, and billed/net tokens when available. Until all ten runs are captured, #7137 remains open and this note is a baseline rather than its completion witness.
