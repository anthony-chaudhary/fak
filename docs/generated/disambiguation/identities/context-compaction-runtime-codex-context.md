---
title: "context compaction - runtime:codex-context identity"
description: "A context-window event that replaces prior history so resident input falls while cumulative usage and transcript bytes may continue rising. Scope: runtime:codex-context. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# context compaction

**Meaning:** A context-window event that replaces prior history so resident input falls while cumulative usage and transcript bytes may continue rising.

## Do not conflate with

- **recovery checkpoint:** Compaction reduces resident model context; a recovery checkpoint preserves typed continuation state for rerouting or repair.

## Query this identity

```console
$ fak disambiguation query "context compaction" --scope-kind "runtime" --scope-value "codex-context" --json
```

## Identity

- **Scope:** `runtime:codex-context`
- **Aliases:** compaction
- **Owner:** `session / session`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `session-source/1`

## Source witnesses

- `internal/session/compactaudit.go` (go-source, revision `session-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-session-source`)

[Back to the disambiguation index](../INDEX.md)
