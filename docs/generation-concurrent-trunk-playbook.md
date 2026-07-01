---
title: "Generation Concurrent-Trunk Playbook"
description: "How now, next, second-next, and future work streams share one trunk without feature branches, stale evidence, or runtime exposure confusion."
---

# Generation Concurrent-Trunk Playbook

**Issue:** #1651.
**Stream:** `gen/next`.
**Status:** operator playbook for branchless concurrent teams.

This playbook is the handoff a future agent can use without rereading the full
generation epic. It explains how generation-labeled teams work concurrently on
one shared trunk while keeping feature exposure and promotion evidence separate.

## Core Rule

Generation is a planning horizon, not an isolation boundary.

- Priority still answers "how urgent or valuable is this?"
- The shared trunk rule still answers "where does the commit land?"
- Runtime feature gates still answer "who can execute or see this?"
- Generation answers "what evidence moves this work closer to now, farther from
  now, or out of the portfolio?"

No generation label authorizes a feature branch, side worktree, private
integration lane, broad `git add -A`, force-push, or stale-trunk escape.

## Team Model

| Stream | What the team may do on `main` | Exposure gate | Promotion evidence |
|---|---|---|---|
| `gen/now` | Ship product, operator, or hygiene changes with the normal focused witness. | Usually none, or a normal default-on feature path. | Focused test, command output, release/readiness evidence, or issue close witness. |
| `gen/next` | Ship near-term foundation behind a command flag, schema, doc contract, or dogfood-only path. | Default-off flag, dry-run mode, hidden report field, or explicit dogfood route. | A focused gate plus an operator readout showing less ambiguity, contention, or manual work. |
| `gen/second-next` | Ship compatibility policy, interfaces, simulations, or prototypes that cannot surprise current users. | Compile-time seam, fixture-only path, non-default command, or documented no-op scaffold. | Simulation, compatibility test, dependency edge, or kill criteria that names what would make it promotable. |
| `gen/future` | Ship research, standards analogues, decision models, or option ledgers. | Documentation and issue/project metadata only unless separately gated. | Sourced memo, assumption ledger, market/standard signal, or decision model a later stream can consume. |

The streams can run concurrently when their file trees are disjoint and the lane
lease admits them. They do not run concurrently merely because their generation
labels differ.

## Admission Checklist

Before launching workers for generation work:

1. Confirm the issue has `generation` plus exactly one stream label.
2. Confirm it has a concrete work unit, expected step budget, done condition,
   witness, and acceptance gate.
3. Derive the narrow expected paths and run `dos arbitrate` for the lane/tree.
4. If two issues overlap the same tree, serialize or split them into path-scoped
   child issues. Do not solve overlap by creating branches.
5. If the work is not current-product-safe, require a runtime gate or inert
   artifact before allowing code to land.

## Commit And Handoff Rules

All streams use the same shared-trunk ship discipline:

- Commit on the configured development branch by explicit path.
- Include the issue number and `(fak <leaf>)` stamp in the subject.
- Run the smallest sufficient gate before claiming done.
- Treat a self-authored final message as a claim, not a witness.
- Record generation-specific evidence in the issue comment, doc, report, commit
  body sidecar, or release note rather than inventing a generation branch.

If a worker discovers the expected path scope is wrong, it should stop with the
additional paths it needs. It should not edit outside scope and then rely on a
later merge to reconcile the surprise.

## Promotion

Promote work toward `gen/now` only when evidence retires the blocker that kept it
later-horizon. Examples:

- A `gen/next` dry-run report becomes a default-on command after CI, operator
  readout, and backout evidence are green.
- A `gen/second-next` compatibility seam becomes `gen/next` after a fixture
  proves the current API can carry it without breaking existing callers.
- A `gen/future` research memo becomes `gen/second-next` after it names a
  concrete interface, benchmark, or decision model.

Promotion evidence must name the witness and the retired assumption. A label-only
move is not evidence.

## Demotion And Retirement

Demote, park, or retire work when live evidence says the current stream is wrong:

- The witness is stale and no cheap re-witness path exists.
- The feature gate leaks behavior into current users before promotion.
- The path scope causes repeated lane collisions or branch/worktree pressure.
- A stronger shipped design makes the option redundant.
- The invalidating assumption below fails.

Retirement evidence should be as concrete as promotion evidence: issue link,
commit, test, report, benchmark, or operator readout.

## Invalidating Assumptions

This playbook depends on these assumptions:

- Lane leases and path-scoped commits are granular enough for useful concurrency.
- Runtime feature gates can keep later-horizon code inert until promotion.
- Operators can read generation labels, issue metadata, and witnesses cheaply
  enough to avoid stale stream assignments.
- The shared trunk remains enforceable by hooks, `fak commit`, and dispatch
  prompts.

If those assumptions fail, do not create generation branches as a workaround.
File or update the smallest issue that fixes the failed assumption, demote the
affected stream if needed, and keep the public evidence with the work item.

## Fast Continuation Path

A future agent continuing generation concurrency work should read, in order:

1. This playbook.
2. [`docs/generation.md`](generation.md) for stream definitions and sidecars.
3. [`docs/dispatch-loop.md`](dispatch-loop.md) for worker routing and prompt
   behavior.
4. [`docs/agentic-issue-dispatch.md`](agentic-issue-dispatch.md) for manual
   wave launch and witnessed rollup.

Then inspect the target issue, run the relevant route/arbitration command, and
ship only the smallest witnessed increment.
