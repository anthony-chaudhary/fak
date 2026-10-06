# perf(engine): reach the continuous batcher from the serving decode path

## Parent context

Coordinates with parent #8395 (https://github.com/anthony-chaudhary/fak/issues/8395).

<!-- fak-engine-key: oss-port-p0-batching-wire -->
<!-- fak-cross-key: oss-port-p0-batching-wire -->
<!-- fak-public-issue: anthony-chaudhary/fak#8395 -->

```routing
lane: engine/batching
paths: ["internal/engine/continuous_batcher.go", "docs/tickets/oss-port-p0/TICKET-01-batching-wire.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#8395
cross_key: oss-port-p0-batching-wire
```

Process cause: verification-gap

## Scope class

single-package

## Current state

`internal/engine/continuous_batcher.go` is 28.5 KB and exports `Step`, `Submit`
and `YieldSlot`. A repository-wide search for callers finds references only inside
that same file: no production line outside it submits work.

A 28.5 KB component that no production line calls is present in the tree and absent
from the serving decode path. The implementation exists; the call path does not.

## For

Local serving clients using the native engine decode path on a Strix Halo appliance.

## Problem

Continuous batching capability cannot be observed or measured, because the ordinary
decode path never submits a request into a batch slot.

## Today

The batcher is exported and unreachable. A serving request runs strictly one at a time.

## Better

The ordinary serving decode path submits into the existing batcher, so a request
occupies a batch slot and slot concurrency becomes measurable on a live engine.

## Why this is next

This is a wiring task, not a port: the mechanism already exists in this tree. It is
the cheapest unit that converts a dormant component into reachable capability, and it
is a prerequisite for any later claim that a port closed batching.

## Working spine

Serving decode request enters -> submits to the continuous batcher -> occupies a
batch slot across decode steps -> yields its slot -> a native verb reports measured
slot concurrency for the live engine.

## Core through-line

Reach the existing batcher from the ordinary serving decode path, and prove the
reachability with a test that names the production caller rather than exercising
the batcher in isolation.

## Gold-plating boundary

No new scheduling policy, no replacement of the existing batcher, no multi-request
batching redesign, no configuration flag, and no speed number is predicted here.

## Verifiable Witness

```bash
go test ./internal/engine/... -run 'TestContinuousBatch' -count=1
```

The witness must include a test that fails when the production caller is removed. A
test that only drives the batcher directly does not satisfy this ticket, because it
proves the component works rather than that it is reached.

## Done condition / witness

Witness: `go test ./internal/engine/... -run 'TestContinuousBatch' -count=1`.

- [ ] [SW-VERIFIED] At least one production caller outside `continuous_batcher.go`
  calls `Submit` on the ordinary serving decode path.
- [ ] [SW-VERIFIED] A test asserts that call path by name, so deleting the caller
  turns the test red.
- [ ] [HW-WITNESSED] A native verb reports batch-slot concurrency for a live Fak
  engine. `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue
or the resolving commit.

## Acceptance gate

The focused engine test is green, it fails against the parent tree, and the routing
fence and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the
`engine` leaf in its commit subject.

## Witness

The focused engine test above is the completion witness for the call path. The
slot-concurrency measurement is a separate hardware read against a live engine and
must cite its own receipt; a green software test is not that measurement.

## Likely files

- `internal/engine/continuous_batcher.go` — the existing batcher to be reached.
- the serving decode caller that must submit to it.
- `docs/tickets/oss-port-p0/TICKET-01-batching-wire.md` — routing and acceptance contract.

## Lane

engine/batching

## Blast radius and affected lanes

One package: `internal/engine`. Model backends, wire protocols and non-serving
callers remain outside the leaf.

## Quarantined fallback mechanism

The current strictly serial decode path continues unchanged while this leaf is
absent. If the batcher cannot be reached without changing token semantics, keep the
serial path and leave this issue open rather than weakening that invariant.

- Centrality: Enabling (native multi-request batching is reachable from the serving path)
- P1 Context: advanced — a named baseline capability gains a reachable call path.
- P2 Net value: advanced — converts a dormant 28.5 KB component into real capability.
- P3 Adaptation: preserved — existing batcher internals remain unchanged.
- P4 Operations: advanced — yields one bounded dispatchable P0 unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #8395.

## Completion standard

development

## Target operating envelope

- batch slot concurrency: not applicable (this leaf proves the call path; the slot measurement is a separate hardware receipt)

## Witnessed operating envelope

- batch slot concurrency: not applicable (no production caller exists yet, so no concurrency can be witnessed)