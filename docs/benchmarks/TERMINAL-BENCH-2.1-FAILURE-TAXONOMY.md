# Terminal-Bench 2.1 failure taxonomy and retry policy

Status: engine shipped, pending live-run wiring.
Date: 2026-06-26.
Issue: https://github.com/anthony-chaudhary/fak/issues/901
Parent epic: https://github.com/anthony-chaudhary/fak/issues/897

This note documents the general agent behavior that classifies a failed
Terminal-Bench task and decides a legal recovery. It is the engine half of
#901. It carries no task-specific answer knowledge, and it never bypasses a
command that fak correctly refused.

Code: `internal/terminalbench/failure.go` (tests in `failure_test.go`).

## Why a closed vocabulary

A free-text "why it failed" string cannot be grouped, counted, or audited. The
classifier emits one code from a fixed set, so the raw-vs-fak compare artifact
(#900) can tally failures by category and the go/no-go decision can see where
the pass rate is leaking. The closed set mirrors fak's refusal-reason doctrine:
a verifiable code from a known vocabulary beats a sentence.

## Failure categories

| Code | Meaning |
|---|---|
| `NONE` | Task passed; no recovery needed. |
| `CMD_ERROR` | A command exited non-zero with no more specific cause matched. |
| `ENV_SETUP_MISS` | Missing file, binary, permission, or unset variable the environment was expected to provide. |
| `TIMEOUT` | A command or the task exceeded its time budget. |
| `PKG_INSTALL` | A dependency or package install failed. |
| `BAD_FILE_EDIT` | An edit left a file syntactically broken. |
| `TEST_MISUNDERSTOOD` | The trace completed every milestone but the benchmark-native test oracle still failed: the agent solved the wrong thing. |
| `POLICY_BLOCK` | A fak verdict denied a command (see the split below). |
| `BUDGET_EXHAUSTED` | The task ran out of turn budget before reaching every required milestone. |
| `UNKNOWN` | Failure that no signal matched. It carries evidence so a human extends the taxonomy instead of hiding the gap. |

## Two signal sources

The classifier reads one `FailureSignals` per task. It works in two modes:

- Fixture replay. After `Run` replays a recorded trace, `ClassifyReport` walks
  the `Report` and classifies each failed fak-arm task from the milestone,
  test, and policy-block shape of the arm. No external harness is required.
- Live run. The Harbor/Codex rehearsal runner (#900) fills `CommandOutcome`
  rows (exit code, timeout flag, a short lower-cased stderr tail keyed by task
  command). Stderr is match material only; it must never carry a task's private
  answer.

## Branch priority

Branches are tried in a fixed order, so the same signals always produce the
same classification:

1. Policy block. A fak verdict that denied a command is the highest-leverage
   signal because it is the one cause fak itself controls.
2. Live command outcome. When per-command exit codes exist, the last failed
   command picks the most specific category.
3. Arm shape. With no exit codes, the milestone, test, and budget shape of the
   arm decides between `TEST_MISUNDERSTOOD`, `BUDGET_EXHAUSTED`, `CMD_ERROR`,
   and `UNKNOWN`.

## The policy-block split, and the safety invariant

`POLICY_BLOCK` has two sub-cases, and the difference is the whole point:

- Unnecessary block (a false positive). fak denied a benign command. Recovery
  is `REFINE_POLICY_FALSE_POSITIVE`: narrow the policy so the benign command is
  allowed. This *reduces* the unnecessary-block count. It is retryable.
- Dangerous block (a correct refusal). fak stopped a dangerous action. Recovery
  is `ESCALATE_FOR_REVIEW` with a safety hold. It is never retried and never
  auto-bypassed.

When both occur in one trace, the dangerous hold wins: the task is held for
review and is not auto-retried, and the benign false-positive is recorded for
later refinement but does not unlock a retry. Safety takes precedence over the
fixable cause.

The retry policy (`RetryDirectiveFor`) is conservative by construction. It
never returns a bypassing action for a dangerous block on any attempt, so
honoring its directives can only hold or lower the unnecessary-block count,
never raise it. That is the #901 acceptance line: a retry policy that improves
pass rate without increasing unnecessary blocks.

## Legal recovery actions

Every recovery is task-agnostic general behavior: command normalization,
failed-command repair, package-install retry, checkpoint restore, evidence-
guided resume, re-reading the public test oracle, policy false-positive
refinement, or escalation. None of them inject private answer knowledge and
none rewrite the task tests.

## Honesty boundary

This ships the classifier and the retry policy with unit tests. It does not
make any Terminal-Bench number claim. The live join - feeding real Harbor
command outcomes through `Classify` during a rehearsal - lands with the
rehearsal runner in #900. Until that exists, `result_claim_allowed` stays
false across the campaign artifacts.
