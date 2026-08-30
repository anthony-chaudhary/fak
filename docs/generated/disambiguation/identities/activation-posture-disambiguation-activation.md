---
title: "activation posture - disambiguation:activation identity"
description: "The normalized behavior state off, shadow, or on; it is always interpreted within the owning domain and does not imply lifecycle authority or maturity. Scope: disambiguation:activation. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# activation posture

**Meaning:** The normalized behavior state off, shadow, or on; it is always interpreted within the owning domain and does not imply lifecycle authority or maturity.

## Do not conflate with

- **index lifecycle class:** Activation posture states whether behavior runs; lifecycle class states the authority role of the indexed definition.
- **capability maturity rung:** Activation posture is a three-state operating switch; maturity is a domain-specific readiness ladder and may advance independently.

## Query this identity

```console
$ fak disambiguation query "activation posture" --scope-kind "disambiguation" --scope-value "activation" --json
```

## Identity

- **Scope:** `disambiguation:activation`
- **Aliases:** rollout posture
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
