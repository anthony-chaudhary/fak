---
title: "simulated evidence - claims:simulated identity"
description: "Stand-in data explicitly labeled SIMULATED; it can test a path but cannot be narrated as a witnessed real-world measurement. Scope: claims:simulated. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# simulated evidence

**Meaning:** Stand-in data explicitly labeled SIMULATED; it can test a path but cannot be narrated as a witnessed real-world measurement.

## Do not conflate with

- **witness provenance:** SIMULATED is one provenance value; witness provenance is the full closed classification field.
- **net-true claim:** Simulation status says how evidence was produced; net-true status grades the entire scoped claim.

## Query this identity

```console
$ fak disambiguation query "simulated evidence" --scope-kind "claims" --scope-value "simulated" --json
```

## Identity

- **Scope:** `claims:simulated`
- **Aliases:** SIMULATED
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
