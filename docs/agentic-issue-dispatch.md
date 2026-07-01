---
title: "Issue-scoped headless worker dispatch"
description: "Manual runbook for launching one headless worker per GitHub issue with DOS arbitration, no-push commit-lane rules, and witnessed rollup."
---

# Issue-scoped headless worker dispatch

This runbook is the first-slice operator flow for issue #1790: turn a small set of
GitHub issues into headless workers without turning the shared trunk into a collision
domain. The flow is:

```
issue -> expected paths -> DOS arbitration -> wave launch -> witnessed rollup
```

Use this when an operator has a multi-issue commit-lane batch and wants one worker per
issue. For the always-on backlog driver, see [the issue-dispatch loop](dispatch-loop.md).
This page is the manual packet shape that a native command can later mechanize.
Use the [dispatch SLO glossary](dispatch-slo-glossary.md) for shared report and
status terms.

## 1. Select issues

Pick issues that are ready for a worker, not issues that still need product judgment.
Good candidates have a concrete effect, a bounded path scope, and no dependency on
another worker's output. Hold an issue for triage instead of dispatch when the expected
paths are unknown, the lane is private/exclusive, the acceptance witness is unclear, or
the issue is a sequenced epic rather than a leaf.

### Dependency markers

When an issue has an explicit dependency edge, put a `Dependencies` section in the issue
body. Each marker is one list item with the form `<relation>: #<issue>`. The supported
relations are:

- `after: #123`: this issue must wait until issue #123 is witnessed.
- `blocks: #456`: issue #456 must wait until this issue is witnessed.
- `related-only: #789`: issue #789 is context only and must not hold dispatch.

Example:

```markdown
## Dependencies
- after: #1756
- blocks: #1772
- related-only: #1706
```

Dispatch tooling treats `after` and `blocks` as blocking dependency edges. It carries
`related-only` as a non-blocking reference so operators can preserve context without
accidentally serializing independent workers.

Start from the issue metadata and record the selection in a gitignored run directory:

```bash
RUN_DIR=.dispatch-runs/issue-wave/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_DIR"
gh issue view <N> --json number,title,labels,body,url > "$RUN_DIR/issue-<N>.json"
```

The manifest is the operator's contract with the wave. Keep one JSONL row per proposed
worker:

```json
{"issue":1790,"worker_id":"w-001","lane":"docs","expected_paths":["docs/agentic-issue-dispatch.md"],"status":"planned"}
```

Each row must name:

- `issue`: the GitHub issue number.
- `worker_id`: stable local id for logs and retries.
- `lane`: the DOS lane that owns the expected tree.
- `expected_paths`: the narrowest write scope the worker is allowed to touch.
- `status`: `planned`, `deferred`, `running`, or a rollup status from section 6.

## 2. Derive expected paths

Expected paths are the safety boundary for both the prompt and `dos arbitrate`. Derive
them from the issue body, labels, linked files, and nearby code. Prefer a narrow set of
files or directories over a whole lane. If the issue needs discovery, give the worker a
read-only discovery task first and run a second dispatch after the path scope is known.

Examples:

| Issue shape | Expected path scope |
|---|---|
| Docs runbook | `docs/<name>.md` |
| CLI flag or command behavior | `cmd/fak/<command>.go`, focused tests |
| Internal leaf behavior | `internal/<leaf>/**`, focused `cmd/fak` wrapper if needed |
| Test-only proof | the exact `_test.go` file or `testdata/<case>/` tree |

Do not put `**` over the repo root, `cmd/**`, or `internal/**` into a same-wave worker
unless the issue is deliberately taking the whole lane. Broad scopes serialize the wave
because every overlapping worker should be refused.

Read the active lane taxonomy before assigning lanes:

```bash
dos doctor --workspace . --json > "$RUN_DIR/dos-doctor.json"
```

Use the lane and tree names from that output. Do not hardcode a lane name from memory
when the workspace declares the source of truth.

## 3. Arbitrate same-wave safety

Run DOS arbitration before launching the wave. The question is not "are these issues
different?" The question is "can these workers write their expected trees concurrently?"

For each manifest row:

```bash
dos arbitrate --workspace . --lane <lane> --tree "<expected path list>" --json
```

Route the verdict this way:

| Arbitration result | Action |
|---|---|
| `acquire` | Put the issue in the current wave and record the lease/region in the manifest. |
| `refuse` with `COLLISION_RISK` | Do not co-launch it. Move it to a later wave or narrow its expected paths and arbitrate again. |
| any contract/error result | Hold the issue; fix the lane/path manifest before launch. |

Same-wave collision refusal is a success condition. It means the arbiter prevented two
workers from mutating the same tree.

### Synthetic dry run

Use synthetic rows when validating the runbook or a future native command:

```json
{"issue":"SYN-1","worker_id":"w-syn-1","lane":"docs","expected_paths":["docs/agentic-issue-dispatch.md"],"status":"planned"}
{"issue":"SYN-2","worker_id":"w-syn-2","lane":"docs","expected_paths":["docs/agentic-issue-dispatch.md"],"status":"planned"}
```

The first row may acquire the `docs/agentic-issue-dispatch.md` region. The second row
must not be launched in the same wave because its expected path overlaps the live region.
The dry-run proof for #1790 is the manifest plus an arbitration transcript showing
`SYN-2` was deferred, not co-launched.

Captured dry-run table shape:

| Issue | Worker | Expected paths | Arbitration | Wave outcome |
|---|---|---|---|---|
| `SYN-1` | `w-syn-1` | `docs/agentic-issue-dispatch.md` | `ACQUIRE docs/agentic-issue-dispatch.md` | `running` |
| `SYN-2` | `w-syn-2` | `docs/agentic-issue-dispatch.md` | `REFUSE COLLISION_RISK overlaps w-syn-1` | `collision_deferred` |

The corresponding operator rollup row for the deferred worker is explicit and
non-successful:

```json
{"issue":"SYN-2","worker_id":"w-syn-2","lane":"docs","expected_paths":["docs/agentic-issue-dispatch.md"],"status":"collision_deferred","commit_sha":"","tests":[],"blocker_reason":"COLLISION_RISK overlaps w-syn-1 in same wave","witness":["dos arbitrate dry-run: REFUSE COLLISION_RISK"]}
```

This is a valid dry-run result: only `SYN-1` is launchable in the wave, and the
deferred row can be retried in a later wave without relaunching `SYN-1`.

## 4. Launch a wave

Wave size is bounded by three numbers:

```
wave = min(disjoint planned rows, serving worker seats, operator max concurrency)
```

Re-read the seat pool between waves if the host exposes one. If there is no seat-pool
reader, treat the host as `serving worker seats = 1` and run serially.

Create one log directory per worker:

```bash
mkdir -p "$RUN_DIR/w-001"
# host launch command writes stdout and stderr separately
# ... > "$RUN_DIR/w-001/run.log" 2> "$RUN_DIR/w-001/run.err"
```

The actual headless launch command is host-specific. The invariant is not. Each launch
must receive exactly one issue, one expected path scope, and one stop condition. Start all
workers in the wave as background tasks, wait for the wave to drain, then launch the next
deferred or remaining disjoint wave. Do not poll logs as the progress oracle; commits,
test exits, and witness reads are the oracle.

## 5. Worker prompt shape

The prompt should be boring and restrictive. Template:

```text
You are a headless worker in repo <workspace> for GitHub issue #<N>: <title/url>.

Goal:
- Make the smallest correct change for issue #<N>, or explicitly report why it cannot
  be produced.

Scope:
- You may write only these expected paths: <paths>.
- If the correct fix needs paths outside that list, stop and report the additional paths
  needed. Do not edit them.

Repository rules:
- Read AGENTS.md before editing.
- Work on the current trunk. Do not create a branch or worktree.
- If this is a commit-lane run, you may create a local signed commit by explicit path
  only, with #<N> in the subject/body and the correct fak stamp.
- Do not push, tag, force-push, rewrite history, reset hard, clean the tree, or restore
  tracked files.
- Do not edit unrelated files or revert other workers' changes.

Evidence:
- Run the lightest checks that prove the change.
- Final report must list: changed files, tests/checks run, local commit SHA if created,
  and an explicit blocker reason if blocked.
```

The no-push rule applies to every worker and to the operator running this packet. A worker
may produce a local path-scoped commit only when the packet says this is a commit-lane run.
Pushing, tagging, release work, and closing issues belong to the parent orchestrator after
the witnessed rollup.

## 6. Roll up outcomes

After a worker exits, update `rollup.jsonl` and a human `rollup.md`. Use a closed status
set so retries are mechanical:

| Status | Meaning | Retry? |
|---|---|---|
| `verified` | The effect was confirmed by an external witness. | No. |
| `blocked` | The worker named a concrete blocker and did not claim a ship. | Only after the blocker is cleared. |
| `auth_failed` | The worker never reached the model/tool session. | Yes, same issue and paths, new worker id, no relaunch of successful rows. |
| `failed` | The worker errored or timed out without a witnessed effect. | Yes after inspecting logs and tree state. |
| `unwitnessed` | The worker claimed success, but no accepted witness confirmed it. | Re-dispatch or manually witness; do not fold as done. |
| `collision_deferred` | DOS refused same-wave overlap. | Later wave only. |

Minimum rollup row:

```json
{"issue":1790,"worker_id":"w-001","lane":"docs","expected_paths":["docs/agentic-issue-dispatch.md"],"status":"verified","commit_sha":"<sha-or-empty>","tests":["<command>: pass"],"blocker_reason":"","witness":["dos commit-audit <sha>: OK diff-witnessed"]}
```

The human table should include the fields issue #1790 asks for:

| Issue | Worker | Status | Verified commit | Tests | Blocker |
|---|---|---|---|---|---|
| `#1790` | `w-001` | `verified` | `<sha>` | `markdown link check: pass` |  |

Retry only rows whose status calls for retry. Keep the successful rows fixed in the
rollup and launch a new worker id such as `w-001-r1` for the failed/auth-walled row.
This avoids re-running good workers and makes the denominator honest.

## 7. Witness rules

A worker final message is never proof of ship. Treat it as a claim to witness.

Accepted witnesses:

- **Commit effect:** verify the local commit exists and audit it with DOS:

  ```bash
  git show --name-only --oneline <sha>
  dos commit-audit --workspace . <sha>
  ```

  Fold the commit only if the audit is `OK` with a diff witness. A subject-only or
  missing-diff result is not enough for `verified`.

- **Issue binding:** the commit subject/body must cite `#<N>`. Without that binding, the
  parent close arm cannot witness-close the issue.
- **Tests:** rerun the relevant command from the parent/orchestrator, or use a captured
  non-model test exit code. A worker saying "tests passed" is not a test witness.
- **Created file:** read the file from the working tree or commit diff and confirm it is
  under the expected path scope. If a file was created outside scope, mark the row
  `failed` or `unwitnessed` until the operator adjudicates it.
- **Blocker:** a blocker is an honest non-ship outcome. It must name the missing decision,
  permission, dependency, or path expansion. Do not convert a blocker into success.

For a planned `(plan, phase)` ship, prefer:

```bash
dos verify --workspace . <PLAN> <PHASE> --json
```

For non-git effects that have no DOS CLI witness, gather an independent read-back from the
owning system and record `unwitnessed` if no such read-back exists.

## Definition of done for a dispatch run

- [ ] Every selected issue has a manifest row with issue, worker id, lane, and expected
      paths.
- [ ] `dos doctor --workspace . --json` was captured for the run.
- [ ] Same-wave candidates were checked with `dos arbitrate`; overlapping scopes were
      deferred rather than co-launched.
- [ ] Each launched worker prompt included the expected paths, AGENTS.md requirement,
      commit-by-path rule, and no-push rule.
- [ ] Worker stdout/stderr and metadata were captured under the gitignored run dir.
- [ ] `rollup.jsonl` and `rollup.md` list issue, worker id, status, verified commit SHA if
      any, tests, and blocker reason if blocked.
- [ ] Every `verified` row has a witness the worker did not author: `dos commit-audit`,
      `dos verify`, parent-rerun tests, or an independent created-file/effect read-back.
- [ ] Failed or auth-walled workers are marked for retry with new worker ids; successful
      rows are not relaunched.
- [ ] No worker or operator pushed, tagged, force-pushed, rewrote history, or closed issues
      as part of the wave.
- [ ] Any remaining `unwitnessed`, `failed`, or `blocked` row is explicitly carried into
      the parent orchestrator handoff.
