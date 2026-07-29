---
title: "Blast-radius containment W2 - recognize - #2714"
description: "Ticket body for issue #2714 of the blast-radius containment cohort (#2712-#2720): cross-trace `FailureHash` correlation so a shared root cause is recognized once (`internal/guardrsi`)"
---

# W2 — recognize — #2714

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(guardrsi): promote a cross-trace repeated FailureHash to a fleet known-bad candidate (blast-radius W2, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2714

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713 (this feeds it). Extends `internal/guardrsi/livelock.go`.

## Current state

`internal/guardrsi/livelock.go` already computes a content-free `FailureHash` for a repeated failing tool call — but the `LivelockDetector` is keyed **per trace** (`byTrace map[string]livelockRun`), so it only ever notices one agent looping on its own. When 9 agents each hit the *same* shared failure once, no counter crosses a threshold: each trace sees a single failure, and nothing joins the nine identical `FailureHash` values into "this is shared."

## Why this is next

This is the automatic **recognize** step. Without it, a shared failure only becomes known-bad if a worker manually declares it (the W1 record path). Cross-trace correlation makes the fleet notice the shared cause on its own, from signals guardrsi already emits.

## Working spine

A fleet-wide correlator (new `internal/guardrsi/fleetcorrelate.go`, pure) keyed by `FailureHash`, counting **distinct TraceIDs** within a rolling window. When a `FailureHash` is observed from >= K distinct traces inside the window, it emits a known-bad candidate that the gateway forwards to `fak knownbad record` (#2713) — carrying the failure's reason class and the union of the traces' declared trees as the signature tree. Same emissions from a single trace never promote (that stays the existing per-trace livelock nudge).

Pure fold: `Correlate(observations []{TraceID, FailureHash, Reason, TreeGlobs, TS}, k, windowSecs, now) -> []Candidate`. The gateway holds the observation buffer under its existing server mutex (same discipline the `LivelockDetector` uses today).

## In scope

- `internal/guardrsi/fleetcorrelate.go`: the distinct-trace-over-window fold + `Candidate` shape.
- Wiring the gateway's existing livelock observation point to also feed the correlator and, on promotion, call the W1 record seam.
- Tests: K distinct traces promote; K emissions from one trace do not; observations outside the window age out.

## Out of scope

The ledger itself (W1 #2713), the blast-radius agent set (W3), any dispatcher behavior (W4). This only *produces* candidates.

## Done condition

The fold, given 3 distinct traces emitting the same `FailureHash` within the window, returns exactly one candidate; given 3 emissions from a single trace, returns none; observations older than the window do not count toward K.

## Witness

`go test ./internal/guardrsi/...` green with the three cases above; a captured JSON candidate from a table test. Cite the test name in the commit body.

## Acceptance gate

`make ci` green; `go test ./internal/guardrsi/...` green.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak guardrsi)` that lands `internal/guardrsi/fleetcorrelate.go` + tests + the gateway wiring.

## Dependencies

- after: #2713 (needs the knownbad record seam to forward candidates to)
- related: #2712

## Likely files

- `internal/guardrsi/livelock.go`
- `internal/guardrsi/fleetcorrelate.go`
- `internal/gateway/gateway.go`

## Lane

guardrsi

## Work unit

leaf

## Expected steps

6

## Assumptions

- The gateway is the single place all traces' livelock observations already pass through, so one correlator instance sees the whole fleet's failures.
- The union of the correlated traces' declared trees is a good-enough signature tree for v0; W3 refines the true blast radius.

## Confusion risks

- Keep the per-trace `LivelockDetector` untouched — this is an *additional* cross-trace aggregator, not a replacement. One nudges a single stuck agent; the other promotes a shared cause.
- Distinct-trace count, NOT total-emission count: 5 loops in one session must not look like 5 agents.

## Coordination

Touches `internal/gateway/gateway.go` (contended) — narrow pathspec commit; arbitrate the gateway keyword lane.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per correlation seam; deduped by the `(fak guardrsi)` cross-trace-correlator work.
