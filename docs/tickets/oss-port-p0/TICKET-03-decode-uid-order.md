# fix(engine): order decode batch work by a stable request UID

## Parent context

Coordinates with parent #8395 (https://github.com/anthony-chaudhary/fak/issues/8395).

Port of the stable-UID decode ordering convention from mini-sglang at pinned
revision `9a91cfa` (MIT licensed), referenced there at `decode.py:32-35`. Ported by
adapting the convention to this tree's scheduler; the upstream file is not vendored.

<!-- fak-engine-key: oss-port-p0-decode-uid-order -->
<!-- fak-cross-key: oss-port-p0-decode-uid-order -->
<!-- fak-public-issue: anthony-chaudhary/fak#8395 -->

```routing
lane: modelengine/decode
paths: ["internal/modelengine/nativesched.go", "docs/tickets/oss-port-p0/TICKET-03-decode-uid-order.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#8395
cross_key: oss-port-p0-decode-uid-order
```

Process cause: verification-gap

## Scope class

single-package

## Current state

`internal/modelengine/nativesched.go` already mints a stable per-request identifier
and already uses it for victim ordering: `admitPrepared` (lines 446-447) increments
`NativeScheduler.seqNo` and stores it on the lane's `schedLane.seqNo` field
(declared at line 1082). That field is the lane's admission-order identity and is
read by `nativesched_preempt.go` at line 313 to keep the preempted queue ordered,
at line 464 to pick the preemption victim, at line 526 as the KV `LastUsed` stamp,
and at lines 582 and 598 as the GPU-direct swap label.

The decode batch itself is assembled by position, not by that identifier.
`runIteration` (line 633) rebuilds the running set from `s.lanes` after dropping
terminal lanes, readmitting preempted lanes and promoting waiting lanes, and copies
`active` in `s.lanes` slice order. `stepOnce` (line 850) then walks `active` in that
order to build `seqs` and `ids`, and hands them to `model.BatchSession.StepBatch`
(`internal/model/batch_step.go:7`), where index `b` of `bs.Seqs` IS the batch
position.

So the batch position a request occupies is a function of mutable slice position,
not of `seqNo`. Slice position changes whenever a lane is prepended by readmission,
appended by promotion, or removed by preemption. Nothing asserts that batch
composition is a function of `seqNo` alone.

## For

Native model-engine serving, and specifically any deployment whose model selection
is MoE under tensor parallelism.

## Problem

Decode batch position is derived from mutable scheduler slice order. If two ranks of
a tensor-parallel deployment disagree about which request occupies which batch slot
in a given step — because they observed different promotion or preemption history —
then MoE expert and route selection diverge across ranks, and a batched decode step
that disagrees with itself produces wrong output.

## Today

The stable UID exists and is used for preemption. The decode batch ignores it, and
the disagreement it permits is silent.

## Better

Decode batch composition is ordered by the stable request UID, so every rank derives
the same slot assignment from the same immutable identity regardless of local
promotion or preemption history.

## Why this is next

This is a correctness item, not a speed item, and is stated as such. No latency,
throughput or token-rate improvement is claimed or predicted by this ticket: a faster
batch that is internally inconsistent is worse than a slower consistent one. The only
outcome claimed is that two ranks deriving slot assignment from the same identity
cannot disagree.

Among the scheduling units in this cohort, ordering correctness is the one whose
absence is invisible on a single-rank box and catastrophic on a multi-rank one.
It is the cheapest possible mitigation because the identity field already exists —
the work is to make the decode batch read it, and to prove it with a test that
fails when the ordering is removed.

## Working spine

Serving request admitted -> scheduler mints a stable request UID -> decode batch
composition sorts lanes by that UID -> every rank derives the same slot assignment
from the same identity -> batched decode is internally consistent across ranks.

## Core through-line

Order the decode batch by the stable request UID and prove the ordering survives
lane churn: promotion, readmission and preemption must not move a request to a
different batch slot than its UID dictates.

## Gold-plating boundary

No expert-parallel or tensor-parallel transport, no collective implementation, no
routing table, no new preemption policy, no change to the existing `seqNo` preemption
ordering, no reordering of the KV allocator, no latency or throughput target, and no
performance claim of any kind.

## Verifiable Witness

```bash
go test ./internal/modelengine/ -run 'TestNativeSchedulerDecodeStableUIDOrder' -count=1
```

`TestNativeSchedulerDecodeStableUIDOrder` is added by this ticket; it does not exist
on the parent tree.

## Done condition / witness

Witness: `go test ./internal/modelengine/ -run 'TestNativeSchedulerDecodeStableUIDOrder' -count=1`.

- [ ] [SW-VERIFIED] The new test FAILS on the parent tree and PASSES after the
  change. A test that passes before the change proves nothing.
- [ ] [SW-VERIFIED] The test forces lane churn — promotion of a waiting lane,
  readmission of a preempted lane, and preemption-driven removal — and asserts the
  decode batch composition afterwards is still ascending in the stable request UID.
- [ ] [SW-VERIFIED] Batch composition is asserted through the position handed to
  `StepBatch`, not through an internal slice, so a reordering anywhere between
  admission and the batch call turns the test red.
- [ ] [SW-VERIFIED] The existing `seqNo` preemption ordering in
  `nativesched_preempt.go` still passes unchanged.
- [ ] [HW-WITNESSED] A multi-rank expert-parallel decode run agrees across ranks.
  `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue
or the resolving commit.

## Acceptance gate

The focused model-engine test is green, it fails against the parent tree, and the
routing fence and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the
`modelengine` leaf in its commit subject.

## Witness

The focused model-engine test above is the completion witness for batch ordering.
Cross-rank agreement on live hardware is a separate measurement and must cite its
own receipt; a green software test is not that measurement.

## Likely files

- `internal/modelengine/nativesched.go` — `runIteration` and `stepOnce`, where the
  decode batch is composed.
- `internal/modelengine/nativesched_test.go` — gains the new ordering test.
- `docs/tickets/oss-port-p0/TICKET-03-decode-uid-order.md` — routing and acceptance
  contract.

## Lane

modelengine/decode

## Blast radius and affected lanes

One package: `internal/modelengine`. `internal/model` is read as the callee, not
modified. Weight loading, wire protocols and non-serving callers remain outside the
leaf. Ordering changes slot position only; the per-request token stream, the KV
accounting and the preemption policy are unchanged.

## Quarantined fallback mechanism

The current slice-order batch composition continues unchanged while this leaf is
absent. If ordering by `seqNo` cannot be made stable across churn without changing
the promotion or preemption contract, leave batch composition as it is and keep this
issue open rather than trading a known deterministic order for an unstable one.

- Centrality: Core (decode batch composition correctness)
- P1 Context: advanced — an existing stable identity becomes load-bearing for decode ordering.
- P2 Net value: advanced — removes a silent cross-rank divergence path for MoE decode.
- P3 Adaptation: preserved — the existing UID field and preemption ordering are reused unchanged.
- P4 Operations: advanced — yields one bounded dispatchable correctness unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #8395.

## Completion standard

development

## Target operating envelope

- decode batch position stability: not applicable (this is an ordering correctness invariant proven in software, not a hardware throughput figure)

## Witnessed operating envelope

- decode batch position stability: not applicable (no multi-rank decode run is driven by this leaf, so cross-rank agreement is unobserved and recorded as a gap)