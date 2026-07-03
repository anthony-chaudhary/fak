

---

<!-- Dispatch-witness overlay recorded 2026-07-03 by the fleet worker routed to
this issue (route: lane=docs target=#2272). Recorded here because the worker's
capability floor refuses outward `gh issue comment` (preview-confirm gate; the
confirm-token re-propose is stripped by the harness) — an operator or
labeled-token pass may copy the evidence block below onto the issue. No code
change was made; this is a blocked-report, not a partial ship. -->

## Dispatch witness (2026-07-03): blocked on #2271 (R2) — the ledgers this fold reads do not exist at HEAD

R3's own scope line is the blocker: "The queue is a **fold over existing
ledgers (packets + acks), not a second store**." At current `main`
(HEAD `2bc81478`) there is no packet ledger and no ack ledger to fold:

- `dos verify` plan=`no-babysit` phase=`R2` → `shipped: false, source: "none"`
  (no registry row, no matching ship commit).
- `git log --grep=2271` and `--grep=2272` on `main` → zero commits.
- No `fak.escalation.v1` type anywhere in `cmd/` or `internal/`
  (`rg "escalation.v1|EscalationPacket"` → no hits).
- `cmd/fak/notify.go` is the SIGCHLD-style push notifier (#761): it fans a
  StopEvent out to native/webhook/Slack sinks, observes only, and records **no
  ack row**. `internal/blockerpost` is outbound Slack posting, also ack-less.
- R1's shipped package says it itself: `internal/operatortouches/
  operatortouches.go` package doc — "escalation_handling_p50 reports not_yet
  **until the R2 escalation packet's ack row exists (#2271)**".

Why no honest increment was landable under this dispatch:

1. **Dependency**: building `fak waiting` without R2 means either inventing the
   packet/ack schema inside R3 (that is #2271's scope, and any commit here must
   bind `#2272`, mis-binding the closure audit) or shipping a verb that can
   never show a row and whose witness ("no queue row exists without a packet
   row behind it") is vacuously green — a fabricated pass.
2. **Misroute**: the issue header says **Lane: `cmd`** (a Go verb + kernel
   object), but this dispatch routed it to the **`docs` lane** and requires a
   `(fak docs)` commit trailer. A genuine fix touches `cmd/fak/` +
   `internal/`, outside the docs file-tree; the required trailer could never
   match the changed lane, so the closure auditor would refuse the very commit
   that resolved it.

## Generation stream

gen/now. P1 rung of a live operator-attention epic; its dependency gate (#2271)
is checkable today with `dos verify`, and the misroute costs a wasted worker
per dispatch tick until corrected. Labels already on the issue: `enhancement`,
`operator`, `priority/P1`.

## Parent context

Parent: epic #2269 (spine `docs/notes/CONCEPT-NO-BABYSITTING-2026-07-01.md`,
rung R3). Sibling dependency: #2271 (R2 escalation packet — the packet + ack
ledgers this fold reads). Adjacent shipped rung: R1 `internal/operatortouches`
(#2270).

## Current state

Blocked, no code change. Depends on #2271 (R2 escalation packet + ack ledger),
which `dos verify` confirms unshipped. R1 (#2270) has landed as
`internal/operatortouches` (issue still open); its honesty fence independently
witnesses the missing ack row.

## Why now

(gen/now) Every dispatch tick that re-routes this issue as-is burns a worker
on the same wall: the fold's input ledgers don't exist and the required
`(fak docs)` binding can't match a `cmd`-lane fix. Until #2271 ships and the
route is corrected, this P1 sits unbuildable while looking dispatchable.

## Working spine

Ship R2 (#2271) first: the typed `fak.escalation.v1` packet + ack rows emitted
by `fak notify`, the refusal ESCALATE disposition, and the operator brief.
Then R3 becomes the small fold the issue describes: `internal/waiting` (pure
fold over packet+ack ledger rows → rows with age, held resources, deadline,
expiry action, ranked by cost-of-delay) + a thin `cmd/fak/waiting.go` shell
(`fak waiting [--json]`), expiry executing the safe default through the normal
admission path, closing rows with reason `EXPIRED_DEFAULT`.

## In scope

Once unblocked: a pure fold package `internal/waiting` (packet+ack ledger rows
→ queue rows with age, held resources, deadline, expiry action, ranked by
cost-of-delay), a thin `cmd/fak/waiting.go` shell (`fak waiting [--json]`),
expiry firing the safe default through the normal admission path, row closure
reason `EXPIRED_DEFAULT`, and the ledger fixture test.

## Out of scope

Defining the `fak.escalation.v1` packet/ack schema (that is #2271); any second
store or new ledger (the issue forbids it); Slack/notify transport changes;
docs-lane edits standing in for the verb (a note cannot satisfy the witness).

## Done condition

Unchanged from the issue. Additionally: re-route the dispatch to the `cmd`
lane before the next attempt — the closure binding for a real fix is
`(fak waiting)`/`(fak cmd)`-shaped, not `(fak docs)`.

## Witness

Unchanged from the issue (ledger fixture: packet in → row; ack in → row gone;
expiry → default fired + closed row). The fixture becomes constructible only
once #2271 defines the packet/ack rows.

## Acceptance gate

`go test ./internal/waiting ./cmd/fak -count=1` green (run under WSL
`./test.ps1` on the Windows host), including the ledger fixture proving
packet in → row, ack in → row gone, expiry → default fired + row closed
`EXPIRED_DEFAULT`, and no row without a backing packet row.

## Closure binding

Resolving commit cites #2272 in the subject and carries a `cmd`-lane stamp
(`(fak waiting)` per the cmd/<dir>-leaf convention, or the lane the router
assigns after correction) — NOT `(fak docs)`; the current docs route makes the
binding unsatisfiable for a real fix.

## Lane

cmd (per the issue header). The 2026-07-03 dispatch routed lane=docs; that
route is the misroute this overlay records — correct it before re-dispatch.

## Work unit

leaf

## Expected steps

6 — ship #2271 (or confirm its packet+ack ledger landed), build the
`internal/waiting` fold, add `cmd/fak/waiting.go`, write the ledger fixture
test, run the acceptance gate under WSL, commit citing #2272 with the
cmd-lane stamp.

## Assumptions

- #2271 lands packet + ack rows on an append-only ledger readable without a
  live daemon (the fold is offline, like `operatortouches`).
- "Held resources" (idle workers, leases, reserved budget) are resolvable from
  existing lease/ledger state at fold time — no new bookkeeping store.

## Confusion risks

- Do not build the packet schema here — that is #2271; a #2272-bound commit
  defining `fak.escalation.v1` mis-binds the closure audit.
- Do not ship the verb with no ledger behind it — an always-empty queue passes
  a vacuous witness and is exactly the fabricated pass the dispatch forbids.
- Expiry must go through the normal admission path, never a bypass.

## Trigger

2026-07-03 dispatch (route lane=docs target=#2272) hit the dependency +
misroute wall; this overlay is that worker's blocked-report witness.

## Batch policy

One overlay per groom; re-grooms update this file in place rather than filing
duplicates. The dependency stays tracked on #2271 — no new issue filed.

## Likely files

- `internal/waiting/waiting.go` (+ `waiting_test.go`, the ledger fixture)
- `cmd/fak/waiting.go`
- the #2271 packet/ack ledger package it folds over

## Coordination

- Suggest labeling this issue blocked-by #2271 (the worker's floor refuses
  `gh issue edit`; an operator pass can add it).
- The `docs`-lane route for this issue should be corrected to `cmd` in the
  dispatcher; otherwise every future worker hits the same trailer/lane
  impossibility before writing a line.
