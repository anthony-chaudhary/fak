---
title: "session resume - runtime:internal/session identity"
description: "The paused-to-running boundary that re-admits an existing session using warm KV when available or a safe cold re-prefill. Scope: runtime:internal/session. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# session resume

**Meaning:** The paused-to-running boundary that re-admits an existing session using warm KV when available or a safe cold re-prefill.

## Do not conflate with

- **agent session:** Resume changes the run state of an existing session; it is not the session identity or transcript.
- **session recovery:** Resume continues a valid paused session; recovery repairs or reroutes state that cannot safely continue as-is.

## Query this identity

```console
$ fak disambiguation query "session resume" --scope-kind "runtime" --scope-value "internal/session" --json
```

## Identity

- **Scope:** `runtime:internal/session`
- **Aliases:** resume
- **Owner:** `session / session`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `session-source/1`

## Source witnesses

- `internal/session/resume.go` (go-source, revision `session-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-session-source`)

[Back to the disambiguation index](../INDEX.md)
