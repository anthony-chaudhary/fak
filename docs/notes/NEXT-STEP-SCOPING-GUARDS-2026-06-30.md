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

