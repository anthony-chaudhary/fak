---
title: "Blast-radius containment W4 - scope-hold - #2716"
description: "Ticket body for issue #2716 of the blast-radius containment cohort (#2712-#2720): hold only the intersecting issues, with the `BLOCKED_BY_KNOWN_BAD` reason (`internal/dispatchtick`)"
---

# W4 — scope-hold — #2716

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(dispatch): scope-hold only the blast-radius-intersecting issues via BLOCKED_BY_KNOWN_BAD (W4, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P1
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713, #2715
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2716

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713 and the blast-radius estimator W3. Extends `internal/dispatchtick` and `cmd/fak/dispatch_skipped.go`.

## Current state

`internal/attemptbudget` holds a *single* issue after N failed attempts; `fak dispatch skipped` surfaces issues the router statically labelled `BLOCKED_BY_HUMAN`. Neither reacts to a *live, runtime-discovered* shared blocker: an issue whose tree intersects a known-bad signature is dispatched anyway, so the worker walks straight into the shared bug. The naive fix — pause the whole fleet — is the opposite failure (global stall); we want to hold **only** the affected agents.

## Why this is next

This is the **scope-hold** step and the whole point of "progress, not stall": disjoint agents keep shipping while only the blast-radius-intersecting work waits. Without it the ledger is informational only.

## Working spine

The dispatch router consults the known-bad ledger (#2713) and the blast estimate (W3) before offering an issue:

- An issue whose declared paths intersect a **live** known-bad signature is skipped with a NEW closed-vocabulary reason `BLOCKED_BY_KNOWN_BAD`, carrying the signature id and the elected fixer (W5) as the "next action."
- An issue disjoint from every live signature dispatches normally.

Add `BLOCKED_BY_KNOWN_BAD` to the router's skip-reason set and to the `fak dispatch skipped` card (a distinct row from the human-blocked set), and register it in the dos refuse-reason vocabulary so the skip is a structured, verifiable refusal.

## In scope

- `internal/dispatchtick`: the known-bad intersection check + the new skip reason.
- `cmd/fak/dispatch_skipped.go`: render the known-bad-blocked rows (separate from `BLOCKED_BY_HUMAN`).
- Register `BLOCKED_BY_KNOWN_BAD` in the closed refuse vocabulary (`dos.toml [reasons]` / the reason set).
- Tests: intersecting issue skipped with the reason; disjoint issue still dispatchable.

## Out of scope

Electing the fixer (W5), releasing the hold (W6), the operator card (W7). This only decides skip-vs-dispatch per issue against the live ledger.

## Done condition

With one live known-bad over tree T, `fak dispatch route` skips exactly the issues whose paths intersect T (reason `BLOCKED_BY_KNOWN_BAD`) and still routes issues disjoint from T for dispatch.

## Witness

`go test ./internal/dispatchtick/... ./cmd/fak/...` green; a captured `fak dispatch route --json` (or the skipped card) over a fixture with one live signature, showing the scoped skip + a disjoint dispatchable issue. `dos man wedge BLOCKED_BY_KNOWN_BAD --explain` reports a valid, refusable reason. Cite in the commit body.

## Acceptance gate

`make ci` green; the scoped-skip fixture transcript captured; the reason resolves in the closed vocabulary.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak dispatch)` landing the router check + the skip reason + the card + the vocabulary entry + tests.

## Dependencies

- after: #2713 (ledger), W3 (estimate)
- related: #2712

## Likely files

- `internal/dispatchtick/dispatchtick.go`
- `cmd/fak/dispatch_skipped.go`
- `dos.toml`

## Lane

dispatch

## Work unit

leaf

## Expected steps

7

## Assumptions

- The router already has each issue's declared paths (it routes by lane/paths today), so intersection needs no new issue metadata.
- Adding a reason to the closed vocabulary is the accepted way to make a new structured skip verifiable (the refuse-reason discipline).

## Confusion risks

- Hold ONLY the intersecting subset — never all issues. A global pause is the failure mode this issue exists to avoid.
- `BLOCKED_BY_KNOWN_BAD` (dynamic, runtime) is distinct from `BLOCKED_BY_HUMAN` (static, router-labelled) — different rows, different next-actions.

## Coordination

Touches `internal/dispatchtick/dispatchtick.go` and `dos.toml` (both contended) — arbitrate the dispatch keyword lane; narrow pathspec commit.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per scope-hold seam; deduped by the `BLOCKED_BY_KNOWN_BAD` reason.
