---
title: "On-demand releases"
description: "Use the existing release cadence workflow through the native front door:"
---
# On-demand releases

Use the existing release cadence workflow through the native front door:

```bash
fak release dispatch --execute --wait --json
```

## What the dispatch sends

The command sends an exact, visible input set to `.github/workflows/release-cadence.yml` on
`main`: `dry_run=false`, `require_ci_green=true`, and `force=false`. The workflow owns the
release lock, CI-readiness gate, version decision, cut, tag, publish, and artifact verification.
The local peer-dirty checkout is not copied into the release: GitHub checks out the selected
ref, while the local `fak release ship` path independently cuts in a transient detached
worktree.

## Preview without touching GitHub

Preview the dispatch without touching GitHub:

```bash
fak release dispatch --json
```

## Explicit overrides

Useful overrides remain explicit:

```bash
fak release dispatch --execute --plan-only  # decide and plan; do not cut
fak release dispatch --execute --ref main   # select the workflow ref explicitly
fak release dispatch --execute --force      # bypass only the substantive-commit floor
```

`--require-ci-green` defaults to true. Disable it only for a deliberate recovery release with
`--require-ci-green=false`; drift, parse, locking, and publication gates still apply.

## Waiting for the run witness

Use `--wait` with `--execute` when the caller needs a final witness. GitHub returns the new run ID
as part of that dispatch, and fak polls only that ID; concurrent dispatches therefore cannot be
mistaken for this one. JSON includes `run_id`, `run_url`, `status`, `conclusion`, and a typed
`verdict` (`passed`, `failed`, `cancelled`, `timed_out`, or `github_refused`). The default timeout
is 30 minutes and can be bounded explicitly, for example `--timeout 10m`. A failed, cancelled,
timed-out, malformed, or refused run exits nonzero.

Without `--wait`, successful execution preserves the enqueue-only behavior: it returns a queued
run URL when GitHub exposes one and the workflow Actions URL in JSON. Enqueueing does not prove
publication; the release is complete only after publication verification is green.

## The local operator-driven cut

For a local operator-driven cut, use `fak release ship --execute --json`. That command keeps the
shared checkout untouched by working from a detached clean worktree and uses the same release
substrate and single-writer lock.
