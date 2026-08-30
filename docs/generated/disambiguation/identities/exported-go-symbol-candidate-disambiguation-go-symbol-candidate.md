---
title: "exported Go symbol candidate - disambiguation:go-symbol-candidate identity"
description: "A reviewed exported type, function, variable, or constant from non-test, non-generated Go source that may warrant canonical or incidental terminology classification. Scope: disambiguation:go-symbol-candidate. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# exported Go symbol candidate

**Meaning:** A reviewed exported type, function, variable, or constant from non-test, non-generated Go source that may warrant canonical or incidental terminology classification.

## Do not conflate with

- **package capability token:** An exported symbol is discovered from a public Go declaration; a capability token is an explicit runtime negotiation declaration and need not be an exported identifier.

## Query this identity

```console
$ fak disambiguation query "exported Go symbol candidate" --scope-kind "disambiguation" --scope-value "go-symbol-candidate" --json
```

## Identity

- **Scope:** `disambiguation:go-symbol-candidate`
- **Aliases:** exported symbol candidate
- **Owner:** `disambiguation / disambiguation`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `go-source/1`

## Source witnesses

- `internal/disambiguation/go_source.go` (go-source, revision `go-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-go-source`)

[Back to the disambiguation index](../INDEX.md)
