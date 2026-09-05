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

## 0. Plan the cohort at creation time

The steps below deconflict a wave *at launch time* with `dos arbitrate`. When you
are **creating** the batch â€” an agent emitting anywhere from 1 to 1000 issues in a
single run â€” plan the whole cohort *before* any issue is synced, so the collision
structure is visible while it is still cheap to fix:

```bash
fak issue cohort --from-plan candidates.json --json
```

The planner (`internal/issuecohort`, schema `fak.issue-cohort-plan.v1`) reviews
every candidate through the same `fak issue contract` spine, then folds the batch
into:

- **waves**: the dispatchable leaves partitioned into concurrency-safe sets using
  the *same disjoint-tree rule* `dos arbitrate` applies at launch (first-fit graph
  colouring over lane + path overlap). Every wave is safe to dispatch at once, and
  the number of waves is the number of sequential rounds the batch needs. This is
  the creation-time dual of section 3's same-wave arbitration â€” the collision that
  would otherwise be caught (and deferred) at launch is instead surfaced before the
  issues exist.
- **split-first**: rows declared (or detected) as an epic / non-leaf, or with an
  expected-step budget over the dispatch cap, each with a child-issue budget â€” so
  "the batch is 1000 issues" cannot hide "40 of them are really epics".
- **triage**: rows not yet scoped/routed/witnessed enough to dispatch.
- **duplicates**: marker keys that appear more than once, so a rerun that would
  *create* instead of *update* is visible as cleanup, not silent spam.

Cap a wave to the seat pool or an operator ceiling with `--max-wave N`. The plan is
advisory (exit 0 on a valid plan); it does not create anything.

To wave-partition the **existing** open backlog instead of a fresh candidate batch,
feed `gh issue list` output through the same planner:

```bash
gh issue list --state open --limit 500 --json number,title,body,labels > backlog.json
fak issue cohort --from-issues backlog.json --json
```

A cohort deconflicted here is deconflicted the same way the dispatcher re-checks it
below, so a clean plan turns section 3 into a confirmation instead of a surprise.

## 1. Select issues

Pick issues that are ready for a worker, not issues that still need product judgment.
Good candidates have a concrete effect, a bounded path scope, and no dependency on
another worker's output. Hold an issue for triage instead of dispatch when the expected
paths are unknown, the lane is private/exclusive, the acceptance witness is unclear, or
the issue is a sequenced epic rather than a leaf.

### Model capability and scope bounds

Match issue scope to worker model capability. When assigning work to smaller or
resource-constrained models (such as 7B/14B local models, fast/flash models, or worker agents):

- Restrict tickets strictly to S0/S1 leaves: single concern, 1–3 files in a single
  package, and exactly one unambiguous witness command.
- Mandate sequential subdivision: require multi-part tasks to split into reproduction,
  minimal implementation, and verification phases.
- Require scoped fail-to-abstain: workers must scope abstention strictly to isolated
  high-difficulty aspects (concurrency invariants, frozen ABI changes, low-level kernels,
  security policies), landing partial verified evidence for safe sub-components and emitting
  an explicit `ABSTAIN` verdict for the hard boundary rather than guessing or emitting unverified diffs.
- Enforce persistence through recoverable blockers: workers must not treat guard refusals or
  transient locks as terminal failures. They must query `fak recover <TOKEN>`, handle transient
  concurrency locks or pivot to sanctioned routes, and persist toward the goal without
  weakening safety gates.

### Required worker body sections

A worker-ready issue body must carry enough structure for a headless worker to act
without rediscovering the assignment. At minimum, require these sections:

- `Current state`: what is already true and what gap remains.
- `Scope`: either one `Scope` section or both `In scope` and `Out of scope`.
- `Done condition`: the observable end state for this one issue.
- `Witness`: the focused command, captured artifact, or read-back that proves done.
- `Likely files`: concrete path hints. Existing `Path hints`, `Paths`, and `Files`
  headings are accepted aliases.

Run `fak issue contract --from-issues <issues.json> --json` before launch. Each review
row reports `missing_sections` for these five body sections and `missing_fields` for the
stricter issue-contract fields. A row with missing required sections is triage debt, not
a worker launch candidate.


New shift-left briefs also carry one **problem frame** in the existing contract review;
there is no parallel priority system:

```markdown
- Centrality: Core | Enabling (<named Core outcome>) | Stewardship (<obligation>) | Peripheral
- P1: advanced | preserved | N/A - <concrete reason>
- P2: advanced | preserved | N/A - <concrete reason>
- P3: advanced | preserved | N/A - <concrete reason>
- P4: advanced | preserved | N/A - <concrete reason>
```

Each P1-P4 answer needs a short explanation; a bare label or bare `N/A` is ceremonial
and leaves the issue triage-only with a field-specific `problem_frame_repair` token.
Human output shows `centrality=<class>(<target>)`; JSON exposes the canonical
`problem_frame` object, including checks, reasons, and repairs. This preserves the shared
end-to-end goal while decoupling work by default: centrality explains connection to the
common outcomes, and the four checks prevent a locally convenient leaf from regressing
managed context, net-true value, bounded adaptation, or integrated operations.

**Migration boundary:** contracts using the new `Scope / tree`, `Witness / proof`, or
`Placement` vocabulary are gated now. Older issue bodies remain readable as
`centrality=unclassified` with `problem_frame.enforced=false` until deliberately migrated;
they are not silently reinterpreted or bulk-blocked.


`fak-dev issue cohort` carries that same canonical object into its portfolio rows and
dispatch-wave members. Both JSON surfaces include the complete `problem_frame`, including
the P1-P4 evidence; human portfolio and wave rows summarize each check beside centrality
and its target or obligation. Human output labels the fold `centrality (non-scoring)` and
keeps priority, readiness, dependencies, and expected effort beside it. Input order and wave
assignment do not sort by centrality. In particular, an urgent Stewardship obligation may
outrank ready Core work; Enabling rows retain their named Core target; Peripheral and
legacy `unclassified` rows remain visible. Directory, title, labels, and parent epic never
synthesize a class—the only source is `CandidateFromIssueDraft` / `ProblemFrame` from the
issue-contract seam.

`fak-dev issue fanout` follows the same rule for generated work. Every taxonomy child
declares its own ready canonical frame before it can enter cohort planning: QA, dogfood,
product, observability, integration, and docs children are Enabling toward the shipped
spine's observable Core outcome; release-claim children are Stewardship toward the honest
release obligation. These classes come from each child template's job, never from the parent
title, labels, directory, or centrality. The JSON fanout plan carries the frame unchanged
into cohort and dispatch output.


These five sections are the *structure* axis of ticket scope. The full scope
contract â€” structure, size (the S0â€“S4 ladder), atomicity, write-scope, cohort
placement, and work-class, each with the verb that measures it and the fix when
it fails â€” is [The scope of a GitHub ticket](ticket-scope.md); the repeatable
per-ticket pass is the [`/ticket-scope`](../.claude/skills/ticket-scope/SKILL.md)
skill.

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

### A GO is an admission-view answer, not an acquirability promise (#5056)

`dos arbitrate` decides **admission** â€” "may a worker with this tree be admitted,
given this view of the live-lease set." It takes no lock and journals nothing; the
grant itself happens only at `dos lease-lane acquire`, whose read-arbitrate-append
runs under the lane-lease mutex. So an `acquire` verdict (the `GO â€” you may take
lane â€¦` interpretation) is *not* a promise that the lease verb will grant the lane,
and a GO followed by an `acquire` refusal is not a bug. Three reasons, all by
design:

- **The kernel deliberately keeps two lease views.** The lock-held `acquire` read
  uses the structural fold (`live_leases(expire_dead=False)`): every un-RELEASEd
  booking stays visible, because a lease's *effect* (the booked tree) outlives the
  short-lived process that journaled it â€” the recorded pid of every **healthy**
  lease is dead within moments of acquisition, so a dead pid never means a free
  lane. The dead-eliding admission/contention view (`expire_dead=True`, the
  phantom-orphan self-heal used by the pre-admission sensor) exists for a
  different consumer. Forcing all readers to agree is a known regression: eliding
  a still-held region at acquire time double-books it â€” the exact TOCTOU the lease
  exists to prevent.
- **The two `arbitrate` surfaces do not agree, and the MCP one fails OPEN.** The
  `dos arbitrate` **CLI** reads the live lane-journal WAL by default â€” same set
  `acquire` folds â€” so it agrees with the authority; `--leases '[]'` opts out into
  an empty world. The **MCP tool** `dos_arbitrate` reads nothing: with
  `live_leases` omitted it arbitrates against *nothing live*, so every lane reads
  free (`dos_mcp/server.py`, `live_leases=list(live_leases or [])`). Observed live:
  MCP `dos_arbitrate` returned `"cluster lane 'tools' free â€” admitted."` with
  `GO â€” â€¦ disjoint from every live lease, so concurrent work is safe` for a lane
  `dos lease-lane acquire` refused the same second as `"already held by a live
  loop"` â€” [dos-kernel#246](https://github.com/anthony-chaudhary/dos-kernel/issues/246),
  which proposes defaulting to the WAL and failing *closed* when it is unreadable.
  Until then, use the CLI form or feed the real live set explicitly (`dos lease-lane
  live`, `fak leaseref live`). A FREE from a checker that cannot see the store is
  not evidence of a free lane. (The tool is still non-persisting â€” it journals
  nothing â€” but "no I/O" was never accurate: it already reads the workspace's
  `dos.toml`.)
- **Even a correct GO can be stale by acquire time** â€” arbitrate holds no lock, so
  a peer can take the lane between the two calls.

Route on the *lease* verb's verdict, not arbitrate's: GO means "admissible per the
admission view â€” proceed to `dos lease-lane acquire`", and a subsequent acquire
refusal is authoritative (wait, or pick a disjoint lane/tree). Never edit a tree on
the strength of a GO alone.

History: #5056 originally diagnosed a GO-then-refuse on lane `cmd` as a liveness
bug ("the holder's pid is dead, so the lease is stale and `acquire` is wrong") and
asked that all three readers (`arbitrate`, `top`, `lease-lane acquire`) be forced
to agree. That diagnosis was **retracted** by its author: the dead pid was the
expected steady state (it names the short-lived acquiring CLI process, not the
worker), and the make-the-readers-agree done condition would have reintroduced the
double-booking regression. Only this documentation clarification survives; the
lease-stranding it collided with is tracked separately as #4324.

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

## 8. Failed reverify reopen drill

If an issue was closed and a later reverify fails (`dos commit-audit`, `dos verify`,
parent-rerun tests, or an independent created-file read-back), reopen it as a new
recovery item. Do not edit away the original close comment; append the failed reverify
evidence so the close arm can learn from the false close.

Required reopen comment fields:

- `failed_sha`: the commit that originally justified the close.
- `failed_witness`: the command or read-back that failed.
- `failure`: the closed-vocabulary reason when one exists, or the shortest concrete
  reason from the verifier output.
- `requeue_as`: the new worker id or queue row that will retry the issue.
- `expected_paths`: the narrow path scope for the retry.
- `next_witness`: the witness required before the issue can close again.

Dry-run example for fake issue `#0000`:

```text
Reopening after failed reverify.

failed_sha: deadbee
failed_witness: dos commit-audit --workspace . deadbee
failure: missing diff witness for the claimed fix
requeue_as: w-0000-r1
expected_paths: docs/example.md
next_witness: commit-audit OK plus parent-rerun markdown check

The previous close is not accepted as witnessed. The retry keeps the issue open until
an independent witness corroborates the new claim.
```

Safe sequence:

```bash
gh issue reopen 0000 --comment "$(cat reopen-comment.txt)"
```

Then add a retry row to the dispatch manifest (`w-0000-r1`) and exclude the failed SHA
from the verified-close denominator. If the reverify failure was itself a false alarm,
add a follow-up comment with the recovered witness command and close only after the
recovered witness is independently reproducible.

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
