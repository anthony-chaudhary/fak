---
name: stale-work-loop
description: Turn a fak stale-work packet into dedicated, contract-valid issue units, collision-safe dispatch waves, and witness-only reconciliation. PLAN by default; use when an operator asks to adjudicate stale-work candidates, file the dedicated issues, or launch fresh workers for already-filed stale-work issues.
allowed-tools: Read, Bash
metadata:
  opencode: claude-only
---

# /stale-work-loop

This skill operates the two-stage contract in [`docs/stale-work.md`](../../../docs/stale-work.md).
The discovery session never edits a candidate. A dedicated issue is the only authority for a
fresh worker to retain, update, or delete that candidate.

## 1. Plan first

Capture the #6613 packet and an issue read-back containing open plus recently closed rows, then
run the loop without a live flag:

```bash
fak stale-work --json --limit 20 > packet.json
fak stale-work loop --packet packet.json --issues open-and-recent.json --json
```

Inspect every unit:

- `issue.contract.ok` must be true before filing a generated body.
- `dispatch.status` must remain `REFUSE` while no dedicated issue exists.
- every ready issue has a distinct `worker_id`;
- overlapping paths occupy different waves; and
- `counts.launches` is zero.

Do not edit candidate content during this phase.

## 2. Explicit live gates

No live action is implied by invoking this skill. Use a live flag only when the operator
explicitly asked for that exact effect:

```bash
# GitHub mutation only. Requires the dedupe snapshot.
fak stale-work loop --packet packet.json --issues open-and-recent.json --live-issues --json

# Worker launch only. Re-read issues after filing and inspect contracts first.
fak stale-work loop --packet packet.json --issues refreshed-issues.json --live-launch --json
```

Never combine discovery and candidate mutation in one worker identity. The loop routes every
issue through `fak dispatch tick` with its own issue-derived identity and lease tree.

## 3. Reconcile from witnesses, not narration

Feed the dispatch/issue read-backs through `--witnesses`. `SHIPPED` requires all four:

1. independent issue state is `CLOSED`;
2. `dos commit-audit` says `OK` with `diff-witnessed`;
3. the focused acceptance read-back is `CLAIM_TEST_GREEN`; and
4. an independent source records `retained`, `updated`, or `deleted`.

Anything less is `STILL_OPEN` or `ABSTAIN`. A worker log or final message is never closure
evidence.

Persist only when requested:

```bash
fak stale-work loop --packet packet.json --issues refreshed-issues.json \
  --witnesses witnesses.json --state prior.json --state-out next.json --json
```

The evidence digest invalidates a cached adjudication whenever the candidate evidence changes.
