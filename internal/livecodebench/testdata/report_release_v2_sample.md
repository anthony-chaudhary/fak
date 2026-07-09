# LiveCodeBench Run Report

- Generated: `2026-07-09T00:00:00Z`
- Benchmark: `livecodebench`
- Release: `release_v2`
- Evidence class: `local-ungraded`
- Official harness: required=true available=false (this report covers local generations only; the official lcb_runner checker has not graded these generations, so no pass-rate is claimable)
- Result claim allowed: `false`
- Claim boundary: Local generations only: the same saved generations must be graded by the official lcb_runner checker before any pass-rate is claimable.

## Summary

| Metric | Value |
|---|---|
| Problems | 3 |
| Graded | 0 |

### Scenarios

| Scenario | Questions |
|---|---:|
| `codegeneration` | 3 |

## Per-Problem Verdicts

| question_id | scenario | arm | verdict | evidence_id |
|---|---|---|---|---|
| `lcb-sample-001` | codegeneration | - | ungraded | `local-ungraded:lcb-sample-001` |
| `lcb-sample-002` | codegeneration | - | ungraded | `local-ungraded:lcb-sample-002` |
| `lcb-sample-003` | codegeneration | - | ungraded | `local-ungraded:lcb-sample-003` |

## Promotion Requirements

- problem-ids-pinned-and-identical-across-arms
- release-version-and-date-window-recorded
- both-arms-generations-saved-with-digest
- official-lcb-runner-grader-output-recorded
- same-config-across-arms
