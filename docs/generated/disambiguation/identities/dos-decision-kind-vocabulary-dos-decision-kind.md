---
title: "DOS decision kind - vocabulary:dos-decision-kind identity"
description: "A persistent DOS row category identifying arbitration refusal work whose resolution depends on the current lane-lease state. Scope: vocabulary:dos-decision-kind. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# DOS decision kind

**Meaning:** A persistent DOS row category identifying arbitration refusal work whose resolution depends on the current lane-lease state.

## Do not conflate with

- **ABI refusal reason:** ARBITER_REFUSE classifies a DOS decision row; it is not a ReasonCode in the kernel adjudication ABI.
- **policy posture verdict:** A DOS decision kind drives resolver lifecycle and history, not an ALLOW/DENY policy amendment result.
- **hook gate class:** A DOS decision kind persists arbitration state; a hook gate class controls execution isolation for a checker.

## Query this identity

```console
$ fak disambiguation query "DOS decision kind" --scope-kind "vocabulary" --scope-value "dos-decision-kind" --json
```

## Identity

- **Scope:** `vocabulary:dos-decision-kind`
- **Aliases:** ARBITER_REFUSE
- **Owner:** `dosdecision / dos`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `reason-source/1`

## Source witnesses

- `internal/dosdecision/revalidate.go` (go-source, revision `reason-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-reason-source`)

[Back to the disambiguation index](../INDEX.md)
