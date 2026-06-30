# Next-step scoping and issue-creation guard plan

Date: 2026-06-30

This plan hardens the path from "a task finished" to "the right next issue exists
and can be dispatched." The target is not more process around every agent turn.
The target is a small, enforceable contract for the contexts where fak or DOS
will create or sync GitHub issues by default.

## Current evidence

- `internal/taskmgr.ReviewHandoff` already refuses a completion handoff unless it
  carries `fak.task-handoff.v1`, `state: done`, `verified_done`, `current_state`,
  and either one or two `next_steps` or a `no_next_step_reason`.
- `internal/taskmgr.HandoffIssueBody` already renders a stable dedupe marker,
  task identity, completion witness, current state, "why next", and evidence.
- `fak guard` Stop hooks can enforce that handoff before a clean stop, and can
  optionally run `fak task handoff --live` before allowing the stop.
- `fak maturity route` turns maturity backlog rows into deduped public issues,
  but its body is narrower: lane, rung, gap, source, suggested action, witness.
- `score-signal`, `bench-signal`, and `gate-signal` all have plan-first issue
  feeders with self-tests, caps, dedupe discipline, and explicit live semantics.
  `score-signal` is already schedule-live; the others stay dry-run unless armed.
- `fak dispatch route/tick` routes existing issues by path, scope token, label,
  or keyword, then the worker prompt enforces `#N` in the commit subject, a
  `(fak <leaf>)` trailer, and a gate run before a done claim.
- `internal/hooks.LintCommitMessageWithOptions(..., requireIssue=true)` already
  has the author-time "this worker must bind to #N" check. It is not a general
  issue-quality check.

The weak point is producer-side scope. Once a vague issue exists, the dispatch
worker can only infer intent from prose and labels. That is where autonomous
issue creation needs a guard.

## Issue candidate contract

A generated GitHub issue is dispatchable only when the candidate carries these
fields, whether they are typed JSON before render or recognizable sections after
render:

1. Stable dedupe key: one source-owned marker so re-runs update, not duplicate.
2. Parent context: source task, issue, epic, milestone, scorecard row, or signal
   row, plus why this is the next useful leaf now.
3. Current state: what is already true, with witness provenance.
4. In scope: the smallest expected change surface and likely owning lane/path.
5. Out of scope: the nearby work this issue deliberately says no to.
6. Done condition: a concrete state change, not "improve" or "investigate".
7. Witness: command, file, metric, `dos verify`, `dos commit-audit`, or other
   read-back that can prove the done condition.
8. Acceptance gate: the package test, docs lint, workflow, smoke, or live probe
   the worker should run before committing.
9. Routing metadata: lane, labels, priority, and any path hints the router can
   confirm.
10. Boundary/risk notes: private-boundary, GPU/hardware, credentials, cost, human
    decision, or operator-only evidence.
11. Closure binding: worker-facing reminder that the resolving commit must cite
    `#N` and carry the matching `(fak <leaf>)` trailer.
12. Work-unit shape: `leaf` or `step` means worker-dispatchable; `epic`,
    `research`, `triage-only`, or `needs-triage` means decompose or review first.
13. Expected step budget: a small integer that tells the worker whether the issue
    is a one-turn leaf, a multi-step leaf, or accidentally an epic.
14. Assumptions and confusion risks: the assumptions a worker should verify, and
    nearby meanings it must not conflate.
15. Coordination notes: lane leases, dependency order, sibling-file risks, or
    "no special coordination" so the worker does not guess.
16. Trigger: the event or threshold that justifies creating/updating the issue.
17. Batch policy: the cap/grouping/update rule that prevents repeated signals
    from becoming issue spam.

## Default creation policy

Default behavior should be context-specific:

| Context | Default | Create/update only if |
|---|---|---|
| `fak guard` clean Stop handoff | validate by default; live GitHub sync only when `--task-handoff-live` is set | handoff is verified done, strict-scoped, public-routeable, and has 1-2 next steps |
| `fak loop drive` witnessed completion | validate handoff by default; issue sync remains explicit | loop witness passed and handoff strict-scope review passed |
| `fak task handoff` CLI | dry-run by default; `--live` opt-in | candidate contract passes and marker dedupe has been checked |
| `score-signal` | schedule-live allowed | self-test green, cap applied, dedupe/update path proven, regression is measured |
| `bench-signal` / `gate-signal` | dry-run scheduled; manual live arm | self-test green, cap applied, candidate contract passes |
| `fak maturity route` | dry-run by default; `--live` opt-in | public-routeable lane, strict issue body, and stable marker dedupe |
| `idea-scout` | dry-run scheduled until triaged | research/idea issues are labeled `needs-triage` and not dispatchable until scoped |

Refuse creation, or mark the candidate non-dispatchable, when any of these are
true:

- no independent completion or signal witness;
- no current-state summary;
- no explicit out-of-scope boundary;
- no done condition or witness command;
- no routeable lane/path signal;
- private/operator-only content would enter a public issue;
- more than two handoff next steps are proposed;
- an existing open issue carries the same marker and can be updated instead;
- the context is schedule-live without a proven dedupe cap;
- a live `gh` sync fails after the guard was configured to require live sync.

## Implementation phases

### Phase 1: shared issue-candidate review

Add a Go-owned review surface, preferably `internal/issuecontract`, with a small
JSON input and a CLI shell such as:

```bash
fak issue contract --file candidate.json
fak issue contract --from-plan issue-plan.json
```

The review returns a closed-vocabulary verdict, reasons, and whether the issue is
`dispatchable`, `triage_only`, `update_existing`, or `refused`. New tooling should
be Go; legacy Python feeders can shell to the CLI before live `gh issue create`
until they are ported.

Candidate reasons to add to `dos.toml`:

- `ISSUE_SCOPE_INCOMPLETE`: missing in-scope, out-of-scope, done condition, or
  witness.
- `ISSUE_UNROUTED`: no lane/path/scope signal strong enough for dispatch.
- `ISSUE_PRIVATE_BOUNDARY`: public issue would contain private or operator-only
  evidence.
- `ISSUE_LIVE_UNARMORED`: scheduled/live creation lacks dedupe cap or a soak
  exception.
- `ISSUE_NOT_DISPATCH_LEAF`: the row is explicitly an epic, research item,
  triage item, or otherwise a decomposition target rather than a worker leaf.
- `ISSUE_OVERSIZED_EXPECTED_STEPS`: the row declares a worker leaf, but the
  expected-step budget is above the dispatch threshold and should be split before
  sync or dispatch.
- `ISSUE_NOISE_CONTROL_INCOMPLETE`: live sync lacks a creation trigger or batch
  policy, or the batch policy does not name a concrete cap, grouping key,
  update/dedupe marker, or rerun policy, so repeats cannot be grouped into an
  existing work item intentionally.
- `ISSUE_AGENT_CONTEXT_INCOMPLETE`: live sync lacks worker-facing work-unit
  shape, expected-step budget, assumptions, confusion risks, or coordination
  notes, so the issue is not yet safe to hand to an autonomous worker.

### Phase 2: harden task handoff first

Extend `HandoffNextStep` with optional typed fields:

- `parent_ref`
- `lane`
- `paths`
- `in_scope`
- `out_of_scope`
- `done_condition`
- `witness`
- `acceptance_gate`
- `risk`

Keep backward compatibility by rendering legacy `body` text, but add a
`--strict-scope` mode that refuses missing typed scope. Then flip the guard Stop
hook from shadow to enforce for clean-stop handoffs after the tests and docs land.

Update `HandoffIssueBody` so the generated issue has stable sections:

- Current state
- Why this is next
- In scope
- Out of scope
- Done condition / witness
- Acceptance gate
- Boundary notes

### Phase 3: gate all automated issue producers

Apply the same review before live creation in:

- `fak maturity route`
- `score-signal`
- `bench-signal`
- `gate-signal`
- `idea-scout`
- any future `fak task handoff --live` call from `fak guard`

For Python-era feeders, the near-term gate is a `fak issue contract` subprocess
right before `gh issue create/edit`. The long-term direction is to port the core
issue planning/sync logic to Go subcommands.

### Phase 4: make dispatch refuse vague leaves

Teach the router/status surface to distinguish:

- `dispatchable`: scoped leaf, routeable, no human/private blocker;
- `triage_only`: useful issue, but missing scope or acceptance witness;
- `epic`: decomposition target, not worker target;
- `private_boundary`: visible to operators, not public dispatch;
- `blocked_by_human`: already recognized by the router.

Update `.github/issue-views.json` so default dispatch views exclude `triage_only`
and `needs-scope` labels once those labels exist.

### Phase 5: prove the loop end to end

Acceptance for "true end to end working":

1. A guarded worker finishes a witnessed task and writes a strict handoff.
2. The Stop hook validates the handoff and, in live mode, creates or updates one
   scoped issue.
3. `fak dispatch route` routes the issue to the intended lane without keyword
   guesswork.
4. `fak dispatch tick` renders a worker prompt that preserves the issue's scope,
   out-of-scope boundary, witness, and gate.
5. The worker commits on `main` with `#N` and `(fak <leaf>)`.
6. `dos commit-audit` grades the commit as diff-witnessed.
7. The close arm re-verifies the SHA and closes the issue.
8. `dispatch_status` reports the issue as witnessed/closed, not self-claimed.

## Current status

- Phase 1 is implemented: `internal/issuecontract` and `fak issue contract`
  provide the shared issue-candidate review surface.
- Phase 2 is implemented for task handoff: strict scoped next-step fields render
  into stable issue sections, `fak task handoff --live` implies strict scope, and
  the guard Stop hook fails closed when required live GitHub sync fails.
- The issue contract now carries an advisory `agent_context` score for work-unit
  shape, assumptions/confusion risks, coordination notes, trigger, and batch
  policy. Explicit epics/research/triage labels or work-unit values are
  `ISSUE_NOT_DISPATCH_LEAF`, and explicit high expected-step budgets are
  `ISSUE_OVERSIZED_EXPECTED_STEPS`, so they stay out of worker dispatch even if
  the prose is otherwise complete. Live sync also refuses
  `ISSUE_NOISE_CONTROL_INCOMPLETE` unless the issue names the trigger and batch
  policy that keep repeated signals organized. A non-empty but vague batch policy
  is still refused unless it names the concrete cap, grouping key,
  update/dedupe marker, or rerun behavior. Live sync also refuses
  `ISSUE_AGENT_CONTEXT_INCOMPLETE` unless the issue carries work-unit shape,
  expected-step budget, assumptions, confusion risks, and coordination notes.
- `fak issue contract --from-issues` now emits aggregate audit counts alongside
  per-issue reviews: dispatchability totals, reason buckets, and full/missing
  `agent_context` totals. That gives agents a high-volume repair queue by
  failure mode instead of requiring them to read every open issue row first. The
  same audit now also groups by lane, work-unit shape, expected-step bucket, and
  trigger/batch-policy key, with sample issue keys per group, so repeated signal
  floods can be repaired as one organized batch instead of becoming spam. It also
  emits explicit repair queues (`dispatch`, `split`, `scope`, `route`, `noise`,
  `private`) with next actions, missing-field counts, and sample keys, so agents
  can batch the correct repair across many rows before launching workers.
- The native dispatch prompt now parses the standard issue sections into an
  agent-first brief before the raw body, so the worker sees work-unit shape,
  assumptions, confusion risks, coordination notes, trigger, batch policy, scope,
  witness, and gate without mining the full issue text.
- The native issue router now treats unlabeled issue bodies with non-leaf
  `Work unit` values (`epic`, `research`, `triage-only`, `decompose`, etc.) as
  non-dispatchable, and also skips `research` / `needs-scope` labels. That makes
  the GitHub issue body itself a dispatch contract, not just a human-facing label
  convention. It also treats an explicit `Expected steps` value above the
  current dispatch threshold (`8`) as a split target, so a nominal leaf that
  declares a large step budget does not enter worker dispatch as one oversized
  task. End-to-end `RouteIssues` admission now also consumes the shared
  `issuecontract` review, so a label/path-routable issue with missing scope,
  done condition, witness, route metadata, or private-boundary safety is skipped
  into the same closed-vocabulary reason buckets the audit command reports.
- Skipped router rows now carry `reason`, `next_action`, `work_unit`, and
  `expected_steps`, and the router counts skips by reason. That preserves the
  legacy skip bucket while giving agents separate queues for human blockers,
  scope gaps, non-leaf decomposition, and oversized leaf splitting. The native
  router payload and `fak dispatch route` text output also expose
  `repair_queues` (`dispatch`, `split`, `scope`, `route`, `noise`, `private`,
  `human`) with step budgets, split child-issue budgets, sample issue numbers,
  reason buckets, and next actions, so supervisors can batch issue-flood cleanup
  directly from the dispatch surface and forecast how many child work-unit issues
  decomposition will create before syncing a large batch.
- Routed rows now preserve `work_unit`, `expected_steps`, `trigger`, and
  `batch_policy`, and lane groups expose a `step_budget` plus per-issue step
  metadata. That lets dispatch supervisors balance lanes by expected work units
  instead of treating a lane with ten one-step leaves the same as ten larger
  multi-step leaves. The native dispatch tick and wave-pricing planner now use
  lane `step_budget` for default lane picks, falling back to one step per issue
  when old payloads do not provide explicit step metadata.
- Phase 3 is partially implemented: the Go issue producers now use the shared
  contract where they create scoped work, while research/complaint/legacy feeders
  stay `triage_only` until a human or later port supplies dispatch scope.
- Phase 4 is partially implemented as default-routing policy: dispatch surfaces
  can carry path-scoped launch plans, default issue views exclude triage-only
  labels, and dispatch admission consumes the shared review result instead of
  relying only on labels. The remaining gap is to audit every non-dispatch issue
  picker/launcher path for the same review boundary.
- Phase 5 remains open: no single witnessed run has yet proven create/update,
  route, dispatch, commit, audit, close, and status read-back end to end.

Verified on 2026-06-30 with:

- `FAK_FAST=0 ./test.ps1 ./cmd/fak -run TestTaskHandoff -count=1`
- `FAK_FAST=0 ./test.ps1 ./cmd/fak -run TestRunGuardStopHook -count=1`
- `FAK_FAST=0 ./test.ps1 ./cmd/fak -run TestDispatchPrice -count=1`
- `FAK_FAST=0 ./test.ps1 ./cmd/fak -run TestDispatchScorecard -count=1`
- `./test.ps1 ./cmd/fak -run TestIssueContract -count=1`
- `./test.ps1 ./internal/issuecontract -count=1`
- `./test.ps1 ./internal/dispatchtick -count=1`
- `./test.ps1 ./cmd/fak -run TestDispatch -count=1`

## Test plan

- Unit-test issue-contract review reasons and dispatchability classes.
- Unit-test `ReviewHandoff --strict-scope` rejects missing `out_of_scope`,
  `done_condition`, `witness`, or lane/path.
- Unit-test issue body rendering includes the stable sections above.
- Unit-test marker dedupe still updates an existing issue instead of creating a
  duplicate.
- CLI-test dry-run modes do not call `gh`.
- CLI-test live modes with injected fake `gh` call create/update only after the
  contract passes.
- Stop-hook test: shadow reports would-block; enforce blocks; live enforce fails
  closed when `gh` sync fails.
- Router fixture test: scoped generated issue routes by exact scope/path; vague
  issue becomes `triage_only` or `unrouted`.
- Commit-lint test: spawned worker path passes `requireIssue=true` only with a
  bindable `#N`.

## Non-goals

- Do not make every human-authored issue satisfy the strict machine contract on
  day one. Start with generated issues and dispatch defaults.
- Do not force all exploratory research into dispatchable leaves. Research can
  exist, but it should be labeled and excluded from default dispatch until scoped.
- Do not create public issues for private GPU-server, Slack-control, credential,
  account, or operator-only evidence.
- Do not replace agent judgment inside the worker. The guard should constrain
  issue creation and closure proof; the worker still chooses the smallest correct
  implementation.
- Do not turn epics into worker tasks. Epics decompose; leaves dispatch.
