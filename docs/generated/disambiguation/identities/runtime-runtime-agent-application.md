---
title: "runtime - runtime:agent-application identity"
description: "The host-side agent application loop that turns model completions into tool calls and final answers through the Planner seam. Scope: runtime:agent-application. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# runtime

**Meaning:** The host-side agent application loop that turns model completions into tool calls and final answers through the Planner seam.

## Do not conflate with

- **agent kernel:** The agent application runtime drives the task loop; the agent kernel mediates its model and tool effects at the enforcement boundary.

## Query this identity

```console
$ fak disambiguation query "runtime" --scope-kind "runtime" --scope-value "agent-application" --json
```

## Identity

- **Scope:** `runtime:agent-application`
- **Aliases:** agent application runtime
- **Owner:** `agent / agent`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `runtime-source/1`

## Source witnesses

- `internal/agent/chat.go` (go-source, revision `runtime-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-runtime-source`)

[Back to the disambiguation index](../INDEX.md)
