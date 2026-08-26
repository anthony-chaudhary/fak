---
title: "Blast-radius containment W6 - auto-release - #2718"
description: "Ticket body for issue #2718 of the blast-radius containment cohort (#2712-#2720): witness-gated auto-release of the held agents (`fak knownbad resolve`)"
---

# W6 — auto-release — #2718

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(knownbad): witness-gated auto-release of held agents on a proven fix (W6, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P1
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713, #2716, #2717
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2718

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713, the scope-hold W4, and the fixer election W5. Reuses `internal/affectedtests` + `dos verify`.

## Current state

A held/parked agent (W4) has no automatic way back. Once the elected fixer (W5) lands the fix, the known-bad signature stays `open`, so the affected agents stay parked forever unless something clears it. Clearing it on a fixer's **self-report** ("I fixed it") is exactly the failure this repo's witness discipline forbids — an unproven claim must not release the fleet.

## Why this is next

This is the **auto-release** step — the one that makes parking safe to enter, because it is guaranteed to end on real evidence. It is the closing arm of the loop: without it, scope-hold is a one-way door.

## Working spine

`fak knownbad resolve <signature>` flips a signature `open -> resolved` **only when the fix is witnessed**:

- Witness = a green `fak affected` over the broken package AND/OR `dos verify` binding the fixer's commit to the signature's tree. No witness -> stays `open`, reported as `not yet` with the missing witness.
- On resolve: drop the fixer's exclusive lease (W5) and clear the scope-hold so W4 stops skipping the previously-blocked issues — they become dispatchable again on the next tick.

The witness is the gate; `resolve` is refused (structured reason) without it.

## In scope

- `cmd/fak/knownbad.go`: the `resolve` subcommand, witness-gated.
- `internal/knownbad`: the `open -> resolved` transition + release of the hold/lease.
- Reuse `internal/affectedtests` (green over the tree) and the `dos verify` seam for the witness.
- Tests: resolve refused with no witness; resolve succeeds + releases with a witnessed green over the tree.

## Out of scope

Detecting the shared failure (W2), estimating blast radius (W3), the skip itself (W4), the operator card (W7), TTL/expiry (W8). This only closes a signature on evidence.

## Done condition

`fak knownbad resolve <sig>` with no witness leaves the signature `open` and prints the missing witness; with a witnessed green `fak affected` over the tree it flips to `resolved`, drops the fixer's lease, and the previously `BLOCKED_BY_KNOWN_BAD` issues route as dispatchable on the next `fak dispatch route`.

## Witness

`go test ./internal/knownbad/... ./cmd/fak/...` green including both the refused-without-witness and released-with-witness cases; a captured transcript showing a skipped issue becoming dispatchable after resolve. Cite in the commit body.

## Acceptance gate

`make ci` green; the two resolve cases green; the skip->dispatchable transcript captured.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` landing the witness-gated `resolve` + the hold/lease release + tests.

## Dependencies

- after: #2713 (ledger), W4 (the hold to release), W5 (the lease to drop)
- related: #2712

## Likely files

- `cmd/fak/knownbad.go`
- `internal/knownbad/knownbad.go`
- `internal/affectedtests/`

## Lane

dispatch

## Work unit

leaf

## Expected steps

7

## Assumptions

- A green `fak affected` over the broken package is an acceptable machine witness that the shared bug is gone for v0; `dos verify` on the fixer commit strengthens it.
- Releasing the hold is a ledger status flip the dispatcher already re-reads each tick (no push needed).

## Confusion risks

- Never resolve on a self-report or a bare commit subject — the witness must be an independent green/verify. Absent it, report `not yet`, not `done`.
- Resolve must release BOTH the scope-hold (W4) and the fixer's lease (W5); dropping one without the other leaves the fleet half-stuck.

## Coordination

Extends `cmd/fak/knownbad.go` (owned by W1/W5) — sequence after those land. Reuses existing affected/verify seams.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per auto-release seam; deduped by the `fak knownbad resolve` verb.
