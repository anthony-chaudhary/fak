---
title: "witness provenance - claims:provenance identity"
description: "The closed label stating how a reported value was obtained: witnessed, observed, modeled, or simulated; it does not replace a reproduction witness. Scope: claims:provenance. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# witness provenance

**Meaning:** The closed label stating how a reported value was obtained: witnessed, observed, modeled, or simulated; it does not replace a reproduction witness.

## Do not conflate with

- **simulated evidence:** Provenance is the classification carried by evidence; simulated evidence is one explicitly labeled provenance class.
- **net-true claim:** Provenance answers how a number was obtained; net-true grading also requires baseline, net cost, scope, witness, and realization.

## Query this identity

```console
$ fak disambiguation query "witness provenance" --scope-kind "claims" --scope-value "provenance" --json
```

## Identity

- **Scope:** `claims:provenance`
- **Aliases:** provenance
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
