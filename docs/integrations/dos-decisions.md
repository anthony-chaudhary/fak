---
title: "fak-DOS writable decisions adapter"
description: "DOS 0.29.0's native dos decisions --all --json surface is read-only. The public fak-dos adapter adds the missing host-write boundary without modifying the..."
---
# fak-DOS writable decisions adapter

DOS 0.29.0's native `dos decisions --all --json` surface is read-only. The public
`fak-dos` adapter adds the missing host-write boundary without modifying the cached
plugin or a global Python installation.

## The three commands

```text
fak-dos decisions add --workspace . --key bench-6349-blank --action OPEN_ISSUE --severity P1 --payload '{"case":"blank-reason"}' --json
fak-dos decisions list --workspace . --json
fak-dos decisions remove --workspace . --key bench-6349-blank --json
```

## Add and list semantics

`add` is idempotent by key: the same structured action, severity, and payload returns
`created:false`; conflicting reuse is rejected. `list` delegates to
`dos decisions --all --json`, appends host rows, and revalidates lease-dependent
`ARBITER_REFUSE` rows against the authoritative `dos lease-lane live` WAL projection.
A refusal whose blocker has been released disappears from the active output; use
`list --all` to retain it as resolved history with `resolution=LEASE_RELEASED`, or
`list --summary` to emit the measurable `cleared` count beside active and superseded
rows. An unreadable live set is fail-closed and clears nothing.

## The read boundary for automation

This adapter is the fak-side decisions read boundary; automation that needs a current
queue should consume it rather than invoking the native read-only DOS renderer directly:

```text
fak-dos decisions list --workspace . --json           # active rows only
fak-dos decisions list --workspace . --all --json     # active + resolved history
fak-dos decisions list --workspace . --summary --json # cleanup count and both sets
```

## Remove is idempotent cleanup

`remove` is an idempotent cleanup operation. Host events live under the workspace's
ignored DOS state at `.dos/decisions/host.jsonl`; they never edit plans, plugin caches,
or global installs.

## Install

Install with `go install github.com/anthony-chaudhary/fak/cmd/fak-dos@latest`, or run
from a checkout with `go run ./cmd/fak-dos ...`.
