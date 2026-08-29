---
title: "compute kernel - computing:processor identity"
description: "An arithmetic routine executed by a processor. Scope: computing:processor. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# compute kernel

**Meaning:** An arithmetic routine executed by a processor.

## Do not conflate with

- **agent kernel:** The fak management boundary governs agent behavior; it is not a processor arithmetic routine.

## Query this identity

```console
$ fak disambiguation query "compute kernel" --scope-kind "computing" --scope-value "processor" --json
```

## Identity

- **Scope:** `computing:processor`
- **Aliases:** none
- **Owner:** `kernel / kernel`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-11T00:00:00Z`
- **Probe:** `public-seed/1`

## Source witnesses

- `README.md#how-it-works` (document, revision `692e4b57d0`, checked `2026-08-11T00:00:00Z`; probe `fak-disambiguation-seed`)

[Back to the disambiguation index](../INDEX.md)
