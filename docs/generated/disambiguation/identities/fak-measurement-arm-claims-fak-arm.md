---
title: "fak measurement arm - claims:fak-arm identity"
description: "The fak-enabled treatment measured against a declared alternative; calling it a baseline obscures which arm is the comparator. Scope: claims:fak-arm. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# fak measurement arm

**Meaning:** The fak-enabled treatment measured against a declared alternative; calling it a baseline obscures which arm is the comparator.

## Do not conflate with

- **tuned baseline:** The fak arm is the treatment; the tuned baseline is the credible comparator.
- **naive baseline:** The fak arm is the treatment; the naive baseline is only contextual floor evidence.

## Query this identity

```console
$ fak disambiguation query "fak measurement arm" --scope-kind "claims" --scope-value "fak-arm" --json
```

## Identity

- **Scope:** `claims:fak-arm`
- **Aliases:** fak baseline
- **Owner:** `claimcheck / claims`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `claims-source/1`

## Source witnesses

- `internal/claimcheck/claimcheck.go` (go-source, revision `claims-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-claims-source`)

[Back to the disambiguation index](../INDEX.md)
