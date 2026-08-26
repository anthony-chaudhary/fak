---
title: "Blast-radius containment W5 - elect fixer - #2717"
description: "Ticket body for issue #2717 of the blast-radius containment cohort (#2712-#2720): elect exactly one fixer through an exclusive lease (`fak knownbad claim`)"
---

# W5 — elect fixer — #2717

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(knownbad): elect exactly one fixer per known-bad via an exclusive lease (W5, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, hot-tree, priority/P1
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2717

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713. Uses the dos exclusive-lease seam (`internal/leaseref` / `dos arbitrate`).

## Current state

When a shared bug is found, several agents may each decide to fix the same shared file — N racing edits/PRs to one tree, the `hot-tree` collision at fleet scale. `dos arbitrate` can keep trees disjoint, but nothing declares "exactly one agent owns fixing THIS known-bad tree"; the affected agents have no pointer to a fixer, so the default is to collide or to each wait blindly.

## Why this is next

This is the **elect one fixer** step. Scope-hold (W4) parks the affected agents; without a single elected fixer, either nobody fixes it (everyone parked) or everybody does (collision). Exactly-one is the invariant that makes parking safe.

## Working spine

A known-bad signature can be **claimed** by exactly one fixer:

- `fak knownbad claim <signature>` acquires an **exclusive dos lease** on the signature's broken tree and records the claimant on the ledger record.
- A second `claim` on an already-claimed signature is REFUSED (structured reason) and returns the current fixer's identity — so the loser gets a pointer, not a collision.
- The claim is the thing W4's skip card and W7's operator card point at ("parked — @fixer owns the fix").

Exactly-one is enforced by the exclusive lease (the arbiter already guarantees a single exclusive holder over an intersecting tree), not by ledger bookkeeping alone.

## In scope

- `cmd/fak/knownbad.go`: the `claim` subcommand over the exclusive-lease seam.
- `internal/knownbad`: record the claimant + claim time on the signature.
- Tests: two concurrent claims -> exactly one wins, the other is refused with the winner's identity.

## Out of scope

Releasing the claim / resolving the signature (W6), the dispatcher skip (W4), the operator card (W7). This only elects and records the single fixer.

## Done condition

Two agents race `fak knownbad claim <sig>`: exactly one acquires the exclusive lease and is recorded as fixer; the other exits refused, printing the winner's identity. A claim on an unclaimed signature succeeds and stamps the claimant.

## Witness

`go test ./internal/knownbad/... ./cmd/fak/...` green including the race case (two claims, one winner); a captured transcript of the second claim's refusal naming the winner. Cite in the commit body.

## Acceptance gate

`make ci` green; the two-claims-one-winner test green; the refusal transcript captured.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` landing the `claim` verb + the exclusive-lease wiring + the race test.

## Dependencies

- after: #2713 (ledger)
- related: #2712, W4 (#scope-hold points at the claimant), W6 (#auto-release drops the claim)

## Likely files

- `cmd/fak/knownbad.go`
- `internal/knownbad/knownbad.go`
- `internal/leaseref/`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- The dos exclusive lease over an intersecting tree is a sufficient exactly-one mutex (the arbiter refuses a second exclusive holder), so no extra distributed-lock code is needed.
- A dead claimant's lease is reaped by the existing lease-liveness path; a stale claim then becomes re-claimable (see the shared-trunk lease-reap gotcha).

## Confusion risks

- Exactly-one is enforced by the LEASE, not by the ledger write — do not "claim" by only appending a record (two agents could both append). The lease acquisition is the gate.
- A refused claim must return the WINNER (so the loser has a pointer), not just "refused."

## Coordination

New subcommand + lease integration; touches `cmd/fak/knownbad.go` (owned by W1) — sequence after #2713 lands. Low tree-collision risk otherwise.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per fixer-election seam; deduped by the `fak knownbad claim` verb.
