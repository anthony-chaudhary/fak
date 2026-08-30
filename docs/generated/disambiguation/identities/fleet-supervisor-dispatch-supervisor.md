---
title: "fleet supervisor - dispatch:supervisor identity"
description: "A decision layer whose input is witnessed liveness, worker verdicts, escalations, and leases; missing witnesses cause escalation rather than inference. Scope: dispatch:supervisor. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# fleet supervisor

**Meaning:** A decision layer whose input is witnessed liveness, worker verdicts, escalations, and leases; missing witnesses cause escalation rather than inference.

## Do not conflate with

- **dispatch loop:** The supervisor reasons over witnessed fleet state; a loop is one recurring execution state machine it may observe or act on.

## Query this identity

```console
$ fak disambiguation query "fleet supervisor" --scope-kind "dispatch" --scope-value "supervisor" --json
```

## Identity

- **Scope:** `dispatch:supervisor`
- **Aliases:** supervisor
- **Owner:** `supervisoragent / fleet`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/supervisoragent/input.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)
