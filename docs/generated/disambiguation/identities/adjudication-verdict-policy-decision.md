---
title: "adjudication verdict - policy:decision identity"
description: "The typed per-call result emitted by the adjudication fold: allow, deny, transform, quarantine, require-witness, defer, or indeterminate, with a reason and deciding rung. Scope: policy:decision. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# adjudication verdict

**Meaning:** The typed per-call result emitted by the adjudication fold: allow, deny, transform, quarantine, require-witness, defer, or indeterminate, with a reason and deciding rung.

## Do not conflate with

- **policy declaration:** A verdict is produced for one call; a declaration is reusable configuration consulted while producing it.
- **capability floor:** A verdict records the result for one call; the capability floor bounds authority before that decision is made.
- **ABI refusal reason:** The verdict states what happens; the reason code explains why that outcome was selected.

## Query this identity

```console
$ fak disambiguation query "adjudication verdict" --scope-kind "policy" --scope-value "decision" --json
```

## Identity

- **Scope:** `policy:decision`
- **Aliases:** adjudication decision
- **Owner:** `abi / abi`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `policy-source/1`

## Source witnesses

- `internal/abi/types.go` (go-source, revision `policy-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-policy-source`)

[Back to the disambiguation index](../INDEX.md)
