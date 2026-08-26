---
title: "Blast-radius containment W3 - estimate - #2715"
description: "Ticket body for issue #2715 of the blast-radius containment cohort (#2712-#2720): blast radius as the import graph intersected with live leases (`internal/blastradius`)"
---

# W3 — estimate — #2715

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(blast): blast-radius estimator — join the affectedtests import graph with live leases (W3, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2715

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713. Joins `internal/affectedtests` (dependency graph) with the live lease set.

## Current state

`fak affected` (`internal/affectedtests`) already computes the **dependency blast radius of a change** — the changed packages plus every package that transitively imports one — but only to select tests. The dos lease ledger (`internal/leaseref`, `dos lease-lane live`) knows which trees live workers hold. **Nothing joins them**: given a broken package/tree, the fleet cannot answer "which in-flight agents are actually affected?" So today a hold is all-or-nothing guesswork.

## Why this is next

Scope-hold (W4), single-fixer (W5), and the operator card (W7) all need the *affected set*. This is the **estimate** step: turn a broken tree into the concrete list of live leases and queued issues that intersect its dependency blast radius, so the fleet can hold precisely those and let the rest run.

## Working spine

`fak blast estimate <path|package> [--json]`:

1. Expand the broken package to its dependents via the `internal/affectedtests` import graph (the dependency blast radius).
2. Read the live lease set (`internal/leaseref` / the dos lease ledger).
3. Return the leases whose tree intersects the blast radius, plus the queued issues whose declared paths intersect — the **affected set**.

Pure join core in a new `internal/blastradius` (or under `internal/knownbad`): `Estimate(graph, brokenPkg, leases, issues) -> AffectedSet`. Impure shell gathers the graph, the leases, and the issue paths.

## In scope

- `internal/blastradius`: the pure `Estimate` join + `AffectedSet` shape.
- `cmd/fak/blast.go`: `estimate` subcommand, `--json`.
- Reuse `internal/affectedtests` for the graph and `internal/leaseref` for the leases — no new graph/lease code.
- Tests over a synthetic graph + lease set.

## Out of scope

Acting on the affected set — holding (W4), electing a fixer (W5), rendering the operator card (W7). This verb only *reports* who is affected.

## Done condition

Given a synthetic import graph, a broken package P, and a set of live leases, `Estimate` returns exactly the leases whose tree intersects P's dependents and excludes the disjoint ones; the queued-issue intersection behaves the same.

## Witness

`go test ./internal/blastradius/...` green; a captured `fak blast estimate --json` over a fixture showing the affected leases and the excluded disjoint ones. Cite in the commit body.

## Acceptance gate

`make ci` green; `go test ./internal/blastradius/... ./cmd/fak/...` green.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak blast)` landing `internal/blastradius` + `cmd/fak/blast.go` + tests.

## Dependencies

- after: #2713
- related: #2712, W4 (#scope-hold consumes this)

## Likely files

- `internal/blastradius/blastradius.go`
- `internal/blastradius/blastradius_test.go`
- `cmd/fak/blast.go`
- `internal/affectedtests/`
- `internal/leaseref/`

## Lane

dispatch

## Work unit

leaf

## Expected steps

7

## Assumptions

- The `affectedtests` import graph is reusable outside the test-selection path (it already returns a package->dependents structure).
- Lease trees and package import paths can be reconciled to a common tree/glob form for intersection.

## Confusion risks

- Blast radius here is the *dependency* blast radius (who imports the broken package), NOT the changed-file set of a diff. Reuse the dependents direction of the graph.
- A lease tree may be globs; a package is a path — intersect at the tree/glob level, the same way W1 does.

## Coordination

Mostly new package + new verb; touches `cmd/fak/main.go` (dispatch table) — narrow pathspec.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per estimator seam; deduped by the `(fak blast)` verb.
