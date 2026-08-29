---
title: "runtime - runtime:worker-execution identity"
description: "The dispatch worker process that selects a backend, optionally wraps it with fak guard, and executes one lane-scoped work packet. Scope: runtime:worker-execution. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# runtime

**Meaning:** The dispatch worker process that selects a backend, optionally wraps it with fak guard, and executes one lane-scoped work packet.

## Do not conflate with

- **DOS decision kind:** The worker runtime executes admitted work; a DOS decision kind classifies persisted arbitration or resolver state about whether work may proceed.

## Query this identity

```console
$ fak disambiguation query "runtime" --scope-kind "runtime" --scope-value "worker-execution" --json
```

## Identity

- **Scope:** `runtime:worker-execution`
- **Aliases:** worker execution runtime
- **Owner:** `dispatchworker / dispatch`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `runtime-source/1`

## Source witnesses

- `cmd/dispatchworker/main.go` (go-source, revision `runtime-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-runtime-source`)

[Back to the disambiguation index](../INDEX.md)
