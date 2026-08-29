---
title: "index lifecycle class - disambiguation:entry-authority identity"
description: "The authority status of a disambiguation entry: current, versioned, research, or archived; it says what role the source may play, not whether a feature is enabled. Scope: disambiguation:entry-authority. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# index lifecycle class

**Meaning:** The authority status of a disambiguation entry: current, versioned, research, or archived; it says what role the source may play, not whether a feature is enabled.

## Do not conflate with

- **activation posture:** Lifecycle class states an entry's authority and historical role; activation posture states whether behavior is off, shadowing, or on.
- **capability maturity rung:** Index lifecycle class governs terminology evidence; a capability maturity rung describes how operationally mature a product capability is in its own domain.

## Query this identity

```console
$ fak disambiguation query "index lifecycle class" --scope-kind "disambiguation" --scope-value "entry-authority" --json
```

## Identity

- **Scope:** `disambiguation:entry-authority`
- **Aliases:** lifecycle class
- **Owner:** `disambiguation / disambiguation`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `lifecycle-source/1`

## Source witnesses

- `internal/disambiguation/entry.go` (go-source, revision `lifecycle-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-lifecycle-source`)

[Back to the disambiguation index](../INDEX.md)
