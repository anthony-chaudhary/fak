---
title: "dispatch worker - dispatch:worker identity"
description: "One executing worker record with structured issue, lane, backend, and witnessed-result fields; its free-form output is untrusted narration. Scope: dispatch:worker. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# dispatch worker

**Meaning:** One executing worker record with structured issue, lane, backend, and witnessed-result fields; its free-form output is untrusted narration.

## Do not conflate with

- **account seat:** A worker is one execution process; a seat is provider-account capacity that may host multiple worker sessions.

## Query this identity

```console
$ fak disambiguation query "dispatch worker" --scope-kind "dispatch" --scope-value "worker" --json
```

## Identity

- **Scope:** `dispatch:worker`
- **Aliases:** worker process
- **Owner:** `dispatchaudit / dispatch`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/dispatchaudit/dispatchaudit.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)
