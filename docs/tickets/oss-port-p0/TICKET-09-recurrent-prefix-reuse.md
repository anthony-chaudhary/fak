# perf(engine): extend-only reuse of recurrent state, never an arbitrary prefix replay

## Parent context

Coordinates with parent #10994 (https://github.com/anthony-chaudhary/fak/issues/10994).

Direct port/adapt of the extend-only recurrent prefix reuse rule studied in slotstream at
revision `53028cf` (MIT). A Gated DeltaNet recurrent state cannot be reused the way an
attention KV prefix is: a recurrent state is produced by folding tokens *in order*, so
there is no sound way to graft an arbitrary cached prefix onto the front of a new prompt.
The only sound reuse is extend-only — take a state checkpoint at a shared boundary and
append the new suffix.

<!-- fak-engine-key: oss-port-p0-recurrent-prefix-reuse -->
<!-- fak-cross-key: oss-port-p0-recurrent-prefix-reuse -->
<!-- fak-public-issue: anthony-chaudhary/fak#10994 -->

```routing
lane: engine/recurrent-prefix
paths: ["internal/engine/continuous_batcher.go", "internal/engine/recurrent_handoff.go", "docs/tickets/oss-port-p0/TICKET-09-recurrent-prefix-reuse.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#10994
cross_key: oss-port-p0-recurrent-prefix-reuse
```

Process cause: verification-gap

## Scope class

single-package

## Current state

The recurrent path of `internal/engine/continuous_batcher.go` has no prefix reuse at all.
`Slot` carries `ExecutionDepth`, `CurrentDepth` and `RecurrentPasses` (lines 95-97);
`initSlot` (line 421) copies the request's full `PromptTokens` into the slot and
initialises `CurrentDepth` and `RecurrentPasses` to zero; `Step` (line 650) increments
`CurrentDepth` and `RecurrentPasses` on every pass. A recurrent request therefore always
begins at depth zero from a full prompt with an empty recurrent state, and there is no
seam on this path that could seed a slot from a prior state.

The admission contract for that seam exists but is untethered.
`internal/engine/recurrent_handoff.go` defines `RecurrentStateCheckpoint` (line 8),
`NewRecurrentStateCheckpoint` (line 17), `RecurrentHandoffRequest` (line 31) and
`AdmitRecurrentHandoff` (line 68), which admits KV prefix cloning and recurrent-state
handoff together, all-or-cold, behind a closed refusal vocabulary
(`RecurrentHandoffPrefixRefused`, `RecurrentHandoffRequestMalformed`,
`RecurrentHandoffCheckpointMismatch`) and rejecting any model, prefix or boundary
mismatch. A repository-wide search finds those constructors and that admission function
called only from `internal/engine/recurrent_handoff_test.go`. No production line in the
module calls them, and nothing connects them to the batcher's recurrent path.

The distinction this ticket must hold is the whole point: a full prompt in `initSlot`
folds from scratch and is always correct, while an extend-only reuse is only correct when
the suffix genuinely extends the checkpoint's boundary.

## For

Hybrid recurrent models on the continuous batcher, where the same conversation is submitted
turn after turn to the same batcher.

## Problem

A recurrent request cannot reuse an arbitrary cached prefix the way an attention KV prefix
is reused, because the state is an ordered fold and no suffix can be grafted onto a
boundary it did not extend. Without an explicit extend-only seam, there is nothing that
could safely express reuse, and nothing that would refuse a misuse if one were added by
accident.

## Today

Every recurrent pass folds the entire prompt from zero. The extend-only admission
primitive exists, is fail-closed, is fully covered by its own test, and has no caller.

## Better

The recurrent path can adopt a checkpoint at a shared prefix boundary and prefill only the
suffix that extends it, and it refuses — through the existing closed refusal vocabulary —
any request whose boundary, model or prefix identity does not match.

## Why this is next

The concept is ported directly from a permissive upstream (slotstream @ `53028cf`, MIT)
and the admission contract is already written and already fail-closed in this package, so
this leaf is the bounded step of connecting a proven primitive to the recurrent path
rather than designing one. It is the cheapest unit that makes the hybrid reuse contract
reachable, and it is the only leaf in this wave that touches recurrent state.

## Working spine

A conversation is submitted recurrently -> the batcher captures a recurrent state
checkpoint at the shared prefix boundary -> a later turn of that conversation supplies the
boundary and the checkpoint -> the admission runs cold and all-or-cold -> on admission the
slot folds only the extending suffix from the handed-off state, and on any mismatch the
slot folds the full prompt from zero.

## Core through-line

Make extend-only recurrent-state reuse reachable from the batcher's recurrent path and
prove the refusal side as loudly as the reuse side: a non-extending request must fold in
full, not graft.

## Gold-plating boundary

No arbitrary-prefix recurrent reuse — the extend-only rule is the invariant, not a
starting point. No recurrent state stored to or loaded from disk, no cross-process or
cross-node checkpoint transfer, no partial or approximate state handoff, no concurrent
reuse of one checkpoint by two slots, no change to the KV prefix clone contract, and no
reclaim or eviction policy for checkpoints. On any admission refusal the slot folds the
full prompt from zero; it never degrades to a partial reuse.

## Verifiable Witness

```bash
go test ./internal/engine/... -run 'TestRecurrentExtendOnlyPrefixReuse' -count=1
```

The test must fail against the parent tree. It asserts both directions: an admitted
extend-only handoff prefills only the suffix and matches a from-zero fold of the same
tokens, and a non-extending or mismatched boundary is refused and folds in full.

## Done condition / witness

Witness: `go test ./internal/engine/... -run 'TestRecurrentExtendOnlyPrefixReuse' -count=1`.

- [ ] [SW-VERIFIED] The recurrent path of `internal/engine/continuous_batcher.go` reaches
  `AdmitRecurrentHandoff` from a production line, so the primitive is no longer
  test-only.
- [ ] [SW-VERIFIED] A test asserts an admitted extend-only handoff produces the same
  tokens as a from-zero fold while prefilling only the suffix, and the test FAILS against
  the parent tree and PASSES after the change.
- [ ] [SW-VERIFIED] A test asserts the refusal side: a boundary, model or prefix mismatch
  yields a typed refusal from the existing closed vocabulary and a full from-zero fold, so
  no partial reuse is reachable.
- [ ] [HW-WITNESSED] The saved recurrent prefill is measured on a live engine.
  `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue or
the resolving commit. The test named above does not exist yet; adding it is part of this
ticket's work.

## Acceptance gate

The focused engine test is green, it fails against the parent tree, and the routing fence
and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `engine` leaf in
its commit subject.

## Witness

The focused engine test above is the completion witness for the extend-only reuse seam and
its refusal path. The recurrent prefill saving measurement is a separate hardware read
against a live engine and must cite its own receipt; a green software test is not that
measurement.

## Likely files

- `internal/engine/continuous_batcher.go` — the recurrent path that must seed a slot from a
  handed-off state instead of always folding from zero.
- `internal/engine/recurrent_handoff.go` — the existing fail-closed admission contract this
  leaf reaches; its vocabulary is reused unchanged.
- `docs/tickets/oss-port-p0/TICKET-09-recurrent-prefix-reuse.md` — routing and acceptance
  contract.

## Lane

engine/recurrent-prefix

## Blast radius and affected lanes

One package: `internal/engine`, and within it one concern: the recurrent state reuse path.

`internal/engine/continuous_batcher.go` is also the seam of the sibling ticket
`oss-port-p0-batching-wire` (TICKET-01). The two are disjoint concerns in the same file and
are coordinated **by scope, not by editing the sibling ticket**: this leaf touches only the
recurrent state reuse path — how a slot is seeded with a handed-off state and how a
mismatch refuses. It must not touch batch admission, the `Step`/`Submit`/`YieldSlot` wiring,
or slot scheduling. If a change to this leaf requires altering any of those, that part
belongs to the sibling leaf and this ticket stops at the boundary.

Model backends, the wire protocols and the KV prefix clone contract remain outside the leaf.
`internal/engine` is a serialized lane; a concurrent change in that package must land first
or this leaf waits for it.

## Quarantined fallback mechanism

The current from-zero recurrent fold continues unchanged while this leaf is absent, and it
is the fallback for every refusal: a slot with no checkpoint, a mismatched boundary or a
refused admission folds the full prompt from zero. If a correct extend-only fold cannot be
proven equal to a from-zero fold in the test, the slot folds in full and this issue stays
open rather than the equality check being weakened.

- Centrality: Enabling (the recurrent path reuses a proven checkpoint instead of folding every prompt from zero)
- P1 Context: advanced — an existing fail-closed contract becomes reachable from production.
- P2 Net value: advanced — extend-only suffix prefill removes the repeated recurrent fold cost.
- P3 Adaptation: preserved — the existing refusal vocabulary and KV prefix clone contract are reused unchanged.
- P4 Operations: advanced — yields one bounded dispatchable P0 unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #10994.

## Completion standard

development

## Target operating envelope

- recurrent prefill tokens per reused turn: not applicable (development leaf; the prefill saving is a separate hardware receipt)

## Witnessed operating envelope

- recurrent prefill tokens per reused turn: not applicable (no production line reaches the recurrent admission today, so no reuse can be witnessed)
