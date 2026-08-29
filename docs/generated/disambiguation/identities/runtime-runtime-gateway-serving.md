---
title: "runtime - runtime:gateway-serving identity"
description: "The configured gateway server that exposes HTTP or MCP transport, authentication, routing, kernel mediation, and observability. Scope: runtime:gateway-serving. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# runtime

**Meaning:** The configured gateway server that exposes HTTP or MCP transport, authentication, routing, kernel mediation, and observability.

## Do not conflate with

- **fak CLI kernel:** The gateway runtime is a long-lived transport server; the fak CLI kernel is the command surface used to configure and launch it.

## Query this identity

```console
$ fak disambiguation query "runtime" --scope-kind "runtime" --scope-value "gateway-serving" --json
```

## Identity

- **Scope:** `runtime:gateway-serving`
- **Aliases:** gateway serving runtime
- **Owner:** `gateway / gateway`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `runtime-source/1`

## Source witnesses

- `internal/gateway/gateway.go` (go-source, revision `runtime-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-runtime-source`)

[Back to the disambiguation index](../INDEX.md)
