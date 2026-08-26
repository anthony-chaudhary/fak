---
title: "Blast-radius containment W8 - bound - #2720"
description: "Ticket body for issue #2720 of the blast-radius containment cohort (#2712-#2720): TTL and revoke so a stale known-bad entry cannot wedge the fleet (`internal/knownbad`)"
---

# W8 — bound — #2720

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(knownbad): bounded TTL + revoke so a stale known-bad can't wedge the fleet (W8, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, security, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2720

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713. Hardens the liveness check W1 stubs. This is the safety valve for the whole epic.

## Current state

W1 (#2713) records a known-bad signature and matches against *live* ones, but a v0 liveness check is just "status == open." A signature that never expires — or was **misattributed** (a flaky test read as a shared bug, a fix that landed but was never resolved) — would scope-hold the affected agents **forever**. That is the exact inverse of today's blindness: instead of nobody noticing the bug, nobody can escape a phantom one.

## Why this is next

Every other child makes the hold *tighter*; this one makes it *bounded*. It is what lets an operator (and the fleet) trust scope-hold — a hold that cannot self-expire is a liability, not a feature. Ship it alongside the spine's consumers so no signature can wedge the fleet.

## Working spine

Two bounded escape hatches on every signature:

- **TTL**: each record carries a bounded `ttl_seconds` (default e.g. 30-60m). `fak knownbad match` treats a record past `discovered_at + ttl` as expired -> `matched:false`, so the scope-hold auto-lifts even if nobody resolves it. A live shared bug re-fires and re-records; a phantom one just ages out.
- **Revoke**: `fak knownbad revoke <signature> --reason <class>` falsifies an open record immediately (e.g. "it was flaky, not shared") -> stops matching now. A structured refuse reason is emitted if a worker tries to `claim`/`resolve` an already expired/revoked signature.

Both paths release the scope-hold (W4) the same way `resolve` (W6) does — the difference is `resolve` needs a witness, TTL/revoke are the *unproven-so-release* safety valves.

## In scope

- `internal/knownbad`: TTL-aware liveness in the `Match` fold (`now` supplied as data); the `revoke` transition.
- `cmd/fak/knownbad.go`: the `revoke` subcommand + a default TTL on `record`.
- A closed refuse reason for acting on an expired/revoked signature.
- Tests: past-TTL record does not match; `revoke` stops matching immediately; both lift the hold.

## Out of scope

Recognition (W2), estimate (W3), skip (W4), election (W5), witnessed resolve (W6), the card (W7). This only bounds and falsifies signatures.

## Done condition

A record past its `ttl_seconds` reports `matched:false` (hold auto-lifts); `fak knownbad revoke <sig>` flips an open record to `revoked` so it stops matching immediately; a `claim`/`resolve` against an expired/revoked signature returns a structured refuse reason.

## Witness

`go test ./internal/knownbad/... ./cmd/fak/...` green including the past-TTL non-match, the revoke, and the refuse-on-expired cases; a captured transcript of a match flipping to `false` once `now` passes the TTL. Cite in the commit body.

## Acceptance gate

`make ci` green; the TTL + revoke cases green; the refuse reason resolves in the closed vocabulary (`dos man wedge <TOKEN> --explain`).

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` landing the TTL-aware liveness + `revoke` + the refuse reason + tests.

## Dependencies

- after: #2713 (ledger)
- related: #2712, W4 (the hold both paths lift), W6 (the witnessed-resolve sibling)

## Likely files

- `internal/knownbad/knownbad.go`
- `internal/knownbad/knownbad_test.go`
- `cmd/fak/knownbad.go`
- `dos.toml`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- A bounded default TTL is safe because a still-live shared bug re-fires and re-records; the cost of a too-short TTL is one extra rediscovery, the cost of no TTL is a permanent wedge.
- The `Match` fold already takes `now` as data (W1), so TTL is a pure comparison, not a clock read.

## Confusion risks

- TTL/revoke are the UNWITNESSED release valves; `resolve` (W6) is the WITNESSED close. Keep them distinct — a revoke is "this was never really a shared bug," a resolve is "the shared bug is proven gone."
- A too-long TTL re-creates the wedge this issue exists to prevent; pick a bounded default and make it overridable per record.

## Coordination

Extends `internal/knownbad` + `cmd/fak/knownbad.go` (owned by W1) and `dos.toml` — sequence after #2713; narrow pathspec on `dos.toml`.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per TTL/falsifiability seam; deduped by the `fak knownbad revoke` verb.
