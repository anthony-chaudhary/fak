---
title: "runtime"
description: "Canonical fak documentation for runtime, including its scope, nearest contrasts, and operational use."
---

# runtime

**Meaning:** The wrapper process that launches a guest command under fak policy, hook, capability, and stop-gate enforcement.

## Do not conflate with

- **agent session:** The guard runtime enforces a launched process; an agent session is the durable execution identity and state pointers that may outlive one wrapper process.

## Query this identity

```console
$ fak disambiguation query "runtime" --scope-kind "runtime" --scope-value "guard-enforcement" --json
```

## Identity

- **Scope:** `runtime:guard-enforcement`
- **Aliases:** guard enforcement runtime
- **Owner:** `guard / guard`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `runtime-source/1`

## Source witnesses

- `cmd/fak/guard.go` (go-source, revision `runtime-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-runtime-source`)

[Back to the disambiguation index](../INDEX.md)
