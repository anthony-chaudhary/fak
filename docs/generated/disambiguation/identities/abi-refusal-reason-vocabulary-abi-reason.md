---
title: "ABI refusal reason - vocabulary:abi-reason identity"
description: "A closed trainable ReasonCode explaining why an adjudication refused a call; POLICY_BLOCK means an explicit policy rule denied it. Scope: vocabulary:abi-reason. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# ABI refusal reason

**Meaning:** A closed trainable ReasonCode explaining why an adjudication refused a call; POLICY_BLOCK means an explicit policy rule denied it.

## Do not conflate with

- **policy posture verdict:** A refusal reason explains a tool-call decision; a policy posture verdict is the ALLOW/DENY result of folding organization amendment authority.
- **hook gate class:** ReasonCode labels adjudication semantics; a hook gate class declares which tree surface a hook may mutate.
- **DOS decision kind:** A refusal reason is a closed ABI label; a DOS decision kind classifies an operator queue row and its resolver lifecycle.

## Query this identity

```console
$ fak disambiguation query "ABI refusal reason" --scope-kind "vocabulary" --scope-value "abi-reason" --json
```

## Identity

- **Scope:** `vocabulary:abi-reason`
- **Aliases:** POLICY_BLOCK, refusal reason
- **Owner:** `abi / abi`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `reason-source/1`

## Source witnesses

- `internal/abi/reasons.go` (go-source, revision `reason-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-reason-source`)

[Back to the disambiguation index](../INDEX.md)
