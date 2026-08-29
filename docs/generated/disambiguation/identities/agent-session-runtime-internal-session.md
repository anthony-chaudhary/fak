---
title: "agent session - runtime:internal/session identity"
description: "A durable, addressable agent execution record carrying drive state and pointers without storing the provider transcript. Scope: runtime:internal/session. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# agent session

**Meaning:** A durable, addressable agent execution record carrying drive state and pointers without storing the provider transcript.

## Do not conflate with

- **session resume:** The session is the durable execution identity; resume is one transition that re-admits a paused session.

## Query this identity

```console
$ fak disambiguation query "agent session" --scope-kind "runtime" --scope-value "internal/session" --json
```

## Identity

- **Scope:** `runtime:internal/session`
- **Aliases:** session
- **Owner:** `session / session`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `session-source/1`

## Source witnesses

- `internal/session/descriptor.go` (go-source, revision `session-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-session-source`)

[Back to the disambiguation index](../INDEX.md)
