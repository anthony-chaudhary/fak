---
title: "recovery checkpoint - runtime:internal/session identity"
description: "A typed snapshot of goal, pending turn, continuation, generation, and state revision emitted when session recovery is requested. Scope: runtime:internal/session. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# recovery checkpoint

**Meaning:** A typed snapshot of goal, pending turn, continuation, generation, and state revision emitted when session recovery is requested.

## Do not conflate with

- **context compaction:** The checkpoint preserves control-plane continuation state; compaction rewrites model-visible history to reduce resident context.
- **session recovery:** The checkpoint is evidence and continuation data for recovery, not the repair or reroute action itself.

## Query this identity

```console
$ fak disambiguation query "recovery checkpoint" --scope-kind "runtime" --scope-value "internal/session" --json
```

## Identity

- **Scope:** `runtime:internal/session`
- **Aliases:** checkpoint
- **Owner:** `session / session`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `session-source/1`

## Source witnesses

- `internal/session/cumulative_envelope.go` (go-source, revision `session-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-session-source`)

[Back to the disambiguation index](../INDEX.md)
