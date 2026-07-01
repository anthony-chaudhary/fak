---
title: "Safe-to-raise-cap checklist"
description: "Operator checklist for raising issue-dispatch max workers without outrunning seats, host capacity, leases, rate budget, or closure honesty."
---

# Safe-to-raise-cap checklist

Use this before increasing the issue-dispatch loop's `--max-workers` value or a
scheduled task's `-MaxWorkers` value. Raising the static ceiling is allowed only
when every gate below is green. A hold is a good outcome when one gate says the
larger fleet would only create noise, double-book an account, or hide unwitnessed
work.

The decision is:

```text
allowed_cap = min(proposed_max_workers, green_seats, host_cap, lease_headroom, rate_budget_workers)
raise iff allowed_cap >= proposed_max_workers and closure_honesty is green
```

## Inputs

Capture these before deciding:

- Operator status card: `python tools/dispatch_status.py --fast`
- Spawn dry run: `go run ./cmd/fak dispatch tick`
- Seat pool or routing summary from the active account-seat source.
- Host cap from the dispatch preflight or host-cap status row.
- Lease state from the current DOS or dispatch lease ledger.
- Closure honesty from the witnessed close/audit surface.

## Gates

| Gate | Green to raise | Hold instead |
|---|---|---|
| Seats | `green_seats >= proposed_max_workers`; every seat is routable and no account is in auth failure. | `REFUSE_NO_SEAT`, auth failures, account-pool skew, or cooldowns would make the extra workers idle or double-booked. |
| Host cap | `host_cap >= proposed_max_workers`; CPU, memory, and process headroom recovered after the last wave. | Host cap is below the proposal, load is still clearing, or process cleanup is stale. |
| Lease health | Live leases plus planned workers fit under the cap; stale leases are scavenged; in-flight issue de-dup is clean. | A stale lease, orphaned worker, or duplicated issue would make the live count ambiguous. |
| Rate budget | Remaining provider or account budget covers the proposed wave plus retry overhead. | Active throttle cooldowns, low remaining quota, or recent rate-limit errors would turn capacity into retries. |
| Closure honesty | Recent closes are `TRUE_RESOLVED`/witnessed; `CLAIMED_CLOSED` and `unwitnessed` rows are not being counted as throughput. | The close arm has unresolved claimed-closed drift, failed reverify rows, or unwitnessed worker claims. |

## Decision Rows

### Raise

```json
{
  "current_max_workers": 2,
  "proposed_max_workers": 4,
  "green_seats": 5,
  "host_cap": 6,
  "lease_headroom": 4,
  "rate_budget_workers": 4,
  "closure_honesty": "green",
  "decision": "raise",
  "reason": "all gates cover proposed cap 4"
}
```

Operator note: raise to `4`, keep the next scheduled run dry-run unless the spawn
arm is already live, and recheck the status card after one wave drains.

### Hold

```json
{
  "current_max_workers": 4,
  "proposed_max_workers": 8,
  "green_seats": 8,
  "host_cap": 8,
  "lease_headroom": 7,
  "rate_budget_workers": 3,
  "closure_honesty": "watch",
  "decision": "hold",
  "reason": "rate budget only covers 3 workers and closure audit still has unwitnessed rows"
}
```

Operator note: do not raise. Clear the throttle cooldown and witness or reopen the
unwitnessed close rows first; then rerun the checklist with fresh inputs.

## Definition of Done

- [ ] The proposed cap is no higher than every green capacity gate.
- [ ] No gate is using a worker's self-report as evidence.
- [ ] Any hold names the exact failing gate and the next witness to refresh.
- [ ] The decision row is saved with the operator run notes or status handoff.
