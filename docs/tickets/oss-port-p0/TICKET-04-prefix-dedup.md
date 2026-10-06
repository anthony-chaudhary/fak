# fix(engine): compute a shared cold prefix once per admitted batch

## Parent context

Coordinates with parent #1914 (https://github.com/anthony-chaudhary/fak/issues/1914).

Port of the in-batch cold-prefix dedup / twin-defer mechanism from sglang at pinned
revision `b8ec5449` (Apache-2.0 licensed), referenced there at
`schedule_policy.py:271-309`. Ported by adapting the mechanism to this tree's batch
admission; the upstream file is not vendored.

<!-- fak-engine-key: oss-port-p0-prefix-dedup -->
<!-- fak-cross-key: oss-port-p0-prefix-dedup -->
<!-- fak-public-issue: anthony-chaudhary/fak#1914 -->

```routing
lane: modelengine/batch-admission
paths: ["internal/modelengine/nativesched.go", "internal/modelengine/nativesched_prefill.go", "docs/tickets/oss-port-p0/TICKET-04-prefix-dedup.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#1914
cross_key: oss-port-p0-prefix-dedup
```

Process cause: verification-gap

## Scope class

single-package

## Current state

Batch admission in `internal/modelengine/nativesched.go` treats admitted requests as
independent. `admitPrepared` mints a `seqNo`, appends the lane to `s.waiting`
(lines 446-448) and never inspects the prompt again. `runIteration` (line 633)
promotes waiting lanes into `s.lanes` purely by position — FIFO by append order, or
through the injected `PromotionPicker` — with no comparison of prompts between the
lanes it promotes together.

Prefix reuse exists, but only against ALREADY-COMPUTED history. `SetRadixKV`
(line 124 of `nativesched_prefill.go`), `SetPrefixTree` (line 146) and `PrefixStats`
(line 168) describe a prefix TREE: `qwenPrefillChunkBudget` calls `lookupPrefix`
(line 326) to find how much of an incoming prompt a previous request already
covered. A tree hit can only ever match tokens some earlier request already wrote.
Two requests admitted in the SAME batch, sharing a cold prefix that neither has
computed, both miss the tree, and both prefill the entire shared prefix.

Nothing in the tree, and nothing in admission, compares one batch member's prompt
against another batch member's prompt. The twin case is undetectable today.

## For

Native model-engine serving under concurrent load, where several in-flight requests
share a long system or repository prefix that is cold on this engine.

## Problem

A shared cold prefix is prefilled once per request instead of once per batch, so N
concurrent requests that differ only after a long common head pay N times the same
prefill work and hold the scheduler N times as long for it.

## Today

Prefix reuse is history-scoped, not batch-scoped. Concurrent requests sharing a cold
prefix each recompute it in full.

## Better

Batch admission recognises that two requests in the same admitted batch share a cold
prefix, computes that prefix once, and defers the twins until it completes — so the
shared work is done once per batch instead of once per request.

## Why this is next

The reuse machinery is already in the tree but is blind to exactly the case the port
addresses, so the change is a comparison at admission plus a defer, not a new cache.
It is the one unit here that removes duplicated compute rather than bounding or
ordering it, and it is separable from the prefill-budget and decode-ordering leaves:
each touches a different decision point.

## Working spine

Concurrent requests admitted into one batch -> admission compares their prompts and
finds a shared cold prefix -> the leader request prefills that prefix -> the twins are
deferred until the leader completes -> the shared prefix is computed once.

## Core through-line

Recognise a shared cold prefix inside one admitted batch, compute it once, and prove
the twins consumed the leader's result rather than recomputing it.

## Gold-plating boundary

No cross-batch or global prefix cache, no radix tree redesign or node format change,
no prefix eviction policy, no change to `lookupPrefix` semantics for already-computed
history, no speculative or partial prefix publication, no request reordering across
the promotion policy, and no throughput figure claimed from a software test.

## Verifiable Witness

```bash
go test ./internal/modelengine/ -run 'TestNativeSchedulerBatchColdPrefixTwinDefer' -count=1
```

`TestNativeSchedulerBatchColdPrefixTwinDefer` is added by this ticket; it does not
exist on the parent tree.

## Done condition / witness

Witness: `go test ./internal/modelengine/ -run 'TestNativeSchedulerBatchColdPrefixTwinDefer' -count=1`.

- [ ] [SW-VERIFIED] The new test FAILS on the parent tree and PASSES after the
  change. A test that passes before the change proves nothing.
- [ ] [SW-VERIFIED] The test admits two requests into one batch that share a long
  cold prefix, and asserts the shared prefix is prefilled ONCE — by counting the
  actual prefill work for the shared span, not by asserting on an internal flag.
- [ ] [SW-VERIFIED] The twins are DEFERRED, not merely reordered: the test asserts
  they do not begin the shared span until the leader has completed it, and that they
  resume correctly afterwards.
- [ ] [SW-VERIFIED] Requests that share NO prefix, or that share only an already-cached
  prefix, take the unchanged path and the test asserts the tree-hit behaviour is
  preserved.
- [ ] [HW-WITNESSED] Prefill MAC count or time-to-first-token under a twin-heavy
  workload on a live appliance. `unobserved` is recorded as a gap, never as a pass.

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

The focused model-engine test above is the completion witness for the dedup
contract. A MAC-count or latency read against a live appliance is a separate
hardware measurement and must cite its own receipt; a green software test is not that
measurement.

## Likely files

- `internal/modelengine/nativesched.go` — `admitPrepared` and `runIteration`, where
  batch composition and promotion happen.
- `internal/modelengine/nativesched_prefill.go` — the prefix and prefill-chunk seam
  the dedup consumes.
- `internal/modelengine/nativesched_test.go` — gains the new twin-defer test.
- `docs/tickets/oss-port-p0/TICKET-04-prefix-dedup.md` — routing and acceptance
  contract.

## Lane

modelengine/batch-admission

## Blast radius and affected lanes

One package: `internal/modelengine`. The radix-KV and prefix-tree packages are read
as existing collaborators, not modified. Weight loading, wire protocols and
non-serving callers remain outside the leaf. The defer changes WHEN a twin starts,
not what it computes: the twin's token stream and final result are unchanged.

## Quarantined fallback mechanism

Today's per-request prefill of a shared cold prefix continues unchanged while this
leaf is absent, and remains the path for every batch that does not match. If twin
deferral cannot be completed without weakening the existing preemption protection
for partially-prefilled lanes, or if a twin cannot be resumed from the leader's
result, leave admission unchanged and keep this issue open rather than shipping a
dedup that can strand a deferred request.

- Centrality: Enabling (a cold prefix shared inside one batch is computed once, not once per request)
- P1 Context: advanced — existing prefix reuse gains the batch-scoped case it is blind to.
- P2 Net value: advanced — removes duplicated prefill compute under concurrent load.
- P3 Adaptation: preserved — the radix prefix tree and chunked prefill are reused unchanged.
- P4 Operations: advanced — yields one bounded dispatchable unit with a dedup witness.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #1914.

## Completion standard

development

## Target operating envelope

- shared-prefix prefill count per admitted batch: not applicable (this leaf proves the dedup decision, not a throughput figure on live hardware)

## Witnessed operating envelope

- shared-prefix prefill count per admitted batch: not applicable (batch admission performs no cold-prefix comparison today, so no dedup can be witnessed on the serving path)