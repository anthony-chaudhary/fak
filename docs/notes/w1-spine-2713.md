---
title: "Blast-radius containment W1 - spine - #2713"
description: "Ticket body for issue #2713 of the blast-radius containment cohort (#2712-#2720): fleet-wide known-bad ledger with record and match (`internal/knownbad`) - the ledger every other ticket reads"
---

# W1 — spine — #2713

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(knownbad): fleet-wide known-bad signature ledger + `fak knownbad record/match` (blast-radius spine, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P1
- **milestone:** 6
- **parent:** #2712
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2713

### Body

## Parent context

Parent: blast-radius containment epic #2712. This is the epic's **load-bearing spine** — the fleet-wide substrate every other child (W2..W8) reads and writes. Nothing else in the epic works until this exists.

## Current state

The fleet has no shared place to record "this failure is known-bad and shared." Every containment surface today is siloed: `internal/guardrsi/livelock.go` is per-trace, `internal/attemptbudget` is per-issue-id, `internal/blockerpost` is a human Slack post. So when agent-1 discovers a shared bug, agents 2..N have no signal to read before they burn a full cycle rediscovering it.

## Why this is next

This is the smallest thing that turns "N agents each rediscover the bug" into "1 agent records it, the rest read it." Every other child (scope-hold, single-fixer, auto-release, operator card) is a consumer of this ledger; it must land first.

## Working spine

A new pure package `internal/knownbad` plus a `fak knownbad` verb:

- **Signature**: a stable, content-free id derived from `(failure_class, normalized tree globs, optional guardrsi FailureHash)` — same shape as `guardrsi.ArgsDigest`/`failureHash` (sha256 over a canonical key). Two agents hitting the same shared cause produce the same signature.
- **Record**: `fak knownbad record --tree <globs> --reason <class> --note "..."` appends one JSONL record `{schema:"fak.known-bad.v1", signature, reason_class, tree_globs, discovered_by, discovered_at_unix, ttl_seconds, status:"open"}` to a fleet-visible ledger (journal-style append, same idiom as the other fak ledgers).
- **Match/query**: `fak knownbad match --tree <globs> [--json]` returns whether the requested tree intersects any *live* (open, unexpired) known-bad signature, with the matching record(s). Exit non-zero (or a JSON `matched:true`) so a worker OR the dispatcher can short-circuit before burning a cycle.

Pure fold core (signature derivation, tree-glob intersection, liveness by `now` supplied as data) in `internal/knownbad`; impure shell (ledger read/write, clock, flags) in `cmd/fak/knownbad.go`.

## In scope

- `internal/knownbad`: signature derivation, the record shape, tree-glob intersection (reuse the same glob semantics `dos arbitrate`/lease trees use), and the pure `Match(records, req, nowUnix) -> matches` fold.
- `cmd/fak/knownbad.go`: `record` and `match` subcommands over a JSONL ledger; `--json`; deterministic, clock-injected.
- Unit tests for signature stability, intersection, and liveness.

## Out of scope

Cross-trace auto-promotion (that is W2), the blast-radius agent set (W3), dispatcher hold wiring (W4), fixer election (W5), auto-release (W6), the operator card (W7), and TTL/GC policy beyond a plain unexpired check (W8). W1 ships the substrate and the two verbs; the consumers fan out.

## Done condition

`fak knownbad record --tree internal/foo/** --reason build` writes a record; a second shell `fak knownbad match --tree internal/foo/bar.go` reports `matched:true` with that record; `fak knownbad match --tree internal/other/**` reports `matched:false`. The pure core has passing tests for signature stability and intersection.

## Witness

Captured terminal transcript of the record-then-match sequence above (both the intersecting and disjoint match), plus `go test ./internal/knownbad/...` green. Record the transcript in the commit body or a focused test.

## Acceptance gate

`make ci` green; `go test ./internal/knownbad/... ./cmd/fak/...` green; the record→match transcript captured.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` that lands `internal/knownbad` + `cmd/fak/knownbad.go` + tests with the transcript witness.

## Dependencies

- blocks: #2712 (epic spine — the other children depend on this)

## Likely files

- `internal/knownbad/knownbad.go`
- `internal/knownbad/knownbad_test.go`
- `cmd/fak/knownbad.go`
- `cmd/fak/main.go`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- The JSONL append-ledger idiom (used by the other fak ledgers) is an acceptable fleet-visible store for v0; a shared path can be resolved the same way `internal/journal` resolves its store.
- Tree-glob intersection can reuse the same semantics the dos lease/arbitrate path already uses, so a known-bad tree and a lease tree compare apples to apples.

## Confusion risks

- Do NOT reuse `guardrsi.LivelockDetector` state (that is per-trace and in-memory); this is a durable, cross-trace ledger. W2 bridges the two.
- `knownbad` is about a *runtime-discovered shared failure*, distinct from the router's static `BLOCKED_BY_HUMAN` per-issue label (`fak dispatch skipped`).

## Coordination

New package + new file; low collision risk. Touches `cmd/fak/main.go` (dispatch table) — a hot file; commit by narrow pathspec.

## Trigger

Filed once at epic-spine creation.

## Batch policy

One issue per spine; deduped by the `fak knownbad` verb key.
