---
title: "session recovery - runtime:internal/session identity"
description: "A bounded repair or reroute response when persisted or cumulative session state cannot safely continue unchanged. Scope: runtime:internal/session. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# session recovery

**Meaning:** A bounded repair or reroute response when persisted or cumulative session state cannot safely continue unchanged.

## Do not conflate with

- **session resume:** Recovery responds to corrupt or over-envelope state; resume merely re-admits a valid paused session.
- **recovery checkpoint:** Recovery is the repair action; a recovery checkpoint is the structured continuation state handed to that action.

## Query this identity

```console
$ fak disambiguation query "session recovery" --scope-kind "runtime" --scope-value "internal/session" --json
```

## Identity

- **Scope:** `runtime:internal/session`
- **Aliases:** recovery
- **Owner:** `session / session`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `session-source/1`

## Source witnesses

- `internal/session/quarantine.go` (go-source, revision `session-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-session-source`)

[Back to the disambiguation index](../INDEX.md)
