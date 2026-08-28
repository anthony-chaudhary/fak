---
title: "Release branch-regime status"
description: "How fak release status reports two separate release facts, and what the branch-regime status means for cutting and tracking a release."
---

# Release Branch-Regime Status

`fak release status` reports two separate release facts:

- `rolling` / `latest_tag`: whether the public release tag is current.
- `branch_regime`: whether the development branch has drifted from the release branch, and whether a promotion is currently blocked.

The branch-regime fields are additive JSON:

- `development_branch`, `development_head`: the hot integration role and resolved commit.
- `release_branch`, `release_head`: the public/front-door role and resolved commit.
- `development_ahead`, `release_ahead`, `drift`: git-derived branch distance.
- `promotion_blocked`, `promotion_blockers`: why a dev -> main promotion should hold.
- `release_lock_held`: whether another release writer owns the single-writer lock.

Operator actions:

| Status | Action |
|---|---|
| `drift=no_drift` | Hold; development and release heads match. |
| `drift=development_ahead` and no blockers | Promote with `fak release ship --execute --open-pr --source-branch <development> --trunk <release> --base origin/<development>`; the command opens an exact-SHA promotion PR instead of pushing the live development branch straight to the release branch. |
| `RELEASE_AHEAD` | Stop and inspect; the release/front-door branch has commits not in development. |
| `DEVELOPMENT_CI_RED` | Fix or confirm CI on the development head before promotion. |
| `RELEASE_LOCK_HELD` | Wait for the active release writer or inspect `python tools/release_lock.py status`. |
| `*_HEAD_UNKNOWN` or `BRANCH_ROLE_CONFIG` | Refresh refs or fix `dos.toml [branch_roles]` before trusting the status. |

Do not read a quiet `main` as "nothing shipped" once `development_branch` is `dev`; read `branch_regime.development_ahead` and `promotion_blockers` instead.

## Actionable CI base-red diagnosis

Run `fak release status --json` and read `rolling.ci_diagnosis` when the exact release candidate is red. A diagnosis is actionable only when its `run_id` and `head_sha` bind to that exact candidate; do not substitute a newer run, a different branch head, or a locally reproduced approximation.

The diagnosis keeps `CI_BASE_RED` as the compatibility umbrella while reporting a typed cause from the committed release-status vocabulary. The classifier covers the declared workflow families and retains `unknown` as the fail-closed fallback: an unrecognized or incomplete failure must not be presented as a known cause.

When a cause maps to repository work, `work_units` identifies the affected package-scoped units rather than treating the whole tree as one repair. Empty work units are evidence too: they mean the diagnosis has not yet located a dispatchable repository change and must not be invented from adjacent failures.

Provider and runner billing remain operator-owned. The status report may attribute observed jobs and durations, but it does not authorize retries, cancel unrelated runs, or claim provider charges it cannot witness. The operator chooses whether to retry, repair a named work unit, or hold the release.
