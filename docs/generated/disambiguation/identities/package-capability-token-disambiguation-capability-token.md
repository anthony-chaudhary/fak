---
title: "package capability token - disambiguation:capability-token identity"
description: "A string explicitly registered by a public package as a negotiated ABI capability; it is inventory evidence, not automatically a canonical term or an authorization verdict. Scope: disambiguation:capability-token. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# package capability token

**Meaning:** A string explicitly registered by a public package as a negotiated ABI capability; it is inventory evidence, not automatically a canonical term or an authorization verdict.

## Do not conflate with

- **exported Go symbol candidate:** A capability token comes from explicit RegisterCapability declarations; an exported symbol candidate comes from the package's reviewed public identifier surface.

## Query this identity

```console
$ fak disambiguation query "package capability token" --scope-kind "disambiguation" --scope-value "capability-token" --json
```

## Identity

- **Scope:** `disambiguation:capability-token`
- **Aliases:** capability manifest token
- **Owner:** `abi / abi`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `go-source/1`

## Source witnesses

- `internal/abi/registry.go` (go-source, revision `go-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-go-source`)

[Back to the disambiguation index](../INDEX.md)
