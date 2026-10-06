# perf(gateway): route by uncached prefill token volume, not total prompt length

## Parent context

Coordinates with parent #10853 (https://github.com/anthony-chaudhary/fak/issues/10853).

Direct port/adapt of the uncached-volume routing rule studied in vidur at revision
`abae7f6` (MIT). A replica whose prefix cache already holds the prompt should not be
charged the full prompt again, so the placement cost of a candidate must be weighted by
the *uncached remainder* rather than by total prompt length.

<!-- fak-gateway-key: oss-port-p0-route-uncached-volume -->
<!-- fak-cross-key: oss-port-p0-route-uncached-volume -->
<!-- fak-public-issue: anthony-chaudhary/fak#10853 -->

```routing
lane: gateway/route-uncached-volume
paths: ["internal/gateway/residency_router.go", "internal/gateway/replica_router.go", "docs/tickets/oss-port-p0/TICKET-07-route-uncached-volume.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#10853
cross_key: oss-port-p0-route-uncached-volume
```

Process cause: verification-gap

## Scope class

single-package

## Current state

The placement overlap signal in this tree is a *message-segment count*, not a token
volume. `internal/gateway/replica_router.go` lowers a request into one digest per message
via `prefixSegments` (line 809), and `internal/gateway/residency_router.go` scores
locality from the longest common run of those digests: `PrefixResidencyIndex.Overlap`
(line 379) returns an `int` segment count, `TierOverlap` (line 392) and
`TierRouteContribution` (line 700) add tier weights over that same count, and
`CacheAwarePolicy.score` (line 742) folds the result with live in-flight load.

Total prompt tokens do reach the gateway, but only as an admission bound: `Classify`
(`routing.go`, line 394) fills `RequestClass.PromptTokens` and `TierPolicy.Route`
(line 318) rejects a tier whose `MaxPromptTokens` cannot hold it. No routing score
anywhere in the package knows how many tokens of a request a candidate replica would
actually have to prefill.

The consequence is concrete: two replicas can have identical segment overlap and very
different uncached token volume — a replica holding the 4-message scaffold of a 4-turn
conversation is scored the same as a replica holding nothing, and the router cannot tell
them apart.

## For

Multi-replica gateway fleets serving long-prompt agent traffic where prefix residency
differs between replicas.

## Problem

Replica placement is scored on a segment count, so a replica that already holds most of a
long prompt is charged as if it had to prefill all of it, and a replica that holds
nothing is not penalised by the tokens it will actually pay for.

## Today

The routing score has no token-volume term. A candidate's local prefix overlap is
counted in messages, and the same overlap yields the same score regardless of how many
tokens remain uncached.

## Better

Placement weights each candidate by the uncached prefill volume it would incur — the
prompt's token total minus the token volume its prefix cache already holds — so a
replica holding a long prefix outranks an empty replica that looks equivalent on segment
count.

## Why this is next

The concept is ported directly from a permissive upstream (vidur @ `abae7f6`, MIT) and
it upgrades an existing, already-reached scoring function rather than adding a new one.
It is a strictly local change to two functions in one package and composes with the
sticky-prefix sibling without overlapping it: this ticket changes what a candidate
*costs*, the sibling changes which candidate a session *returns to*.

## Working spine

A request arrives -> the gateway lowers it to a prefix run and a token volume -> each
admissible candidate reports the token volume its prefix cache already holds -> the
routing score weights each candidate by its uncached remainder -> the lowest-uncached
candidate wins, and the decision remains replayable.

## Core through-line

Weight replica placement by the uncached prefill volume a candidate would incur, and prove
it by showing that a candidate holding most of a long prompt outranks an equivalent-on-
segment-count candidate that holds nothing.

## Gold-plating boundary

No tier capacity policy change, no rewrite of the existing segment-count overlap signal
(which stays as the locality term), no new admission bounds on `MaxPromptTokens`, no
tokenizer added to the routing hot path — volume must be carried on the request, not
computed there — no cost model beyond the uncached remainder, and no throughput or
latency number is predicted here.

## Verifiable Witness

```bash
go test ./internal/gateway/... -run 'TestRouteUncachedVolume' -count=1
```

The test must fail against the parent tree. It places two candidates that score
identically on segment overlap, gives one a large held prefix volume, and asserts the
volume-weighted candidate wins while the old segment-only score calls it a tie.

## Done condition / witness

Witness: `go test ./internal/gateway/... -run 'TestRouteUncachedVolume' -count=1`.

- [ ] [SW-VERIFIED] The routing score at `internal/gateway` carries an uncached-token-volume
  term derived from a held-prefix volume the request supplies, with no tokenizer call on
  the routing path.
- [ ] [SW-VERIFIED] A test asserts that on an equal-segment-overlap tie the candidate with
  the larger held prefix volume wins, and the test FAILS against the parent tree and
  PASSES after the change.
- [ ] [SW-VERIFIED] The placement decision stays a pure function of its inputs, so replaying
  the same request yields the same winner.
- [ ] [HW-WITNESSED] The uncached-token saving is measured on a live multi-replica gateway.
  `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue or
the resolving commit. The test named above does not exist yet; adding it is part of this
ticket's work.

## Acceptance gate

The focused gateway test is green, it fails against the parent tree, and the routing fence
and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `gateway` leaf
in its commit subject.

## Witness

The focused gateway test above is the completion witness for the uncached-volume term. The
token-saving measurement is a separate hardware read against a live gateway and must cite
its own receipt; a green software test is not that measurement.

## Likely files

- `internal/gateway/residency_router.go` — `PrefixResidencyIndex` and
  `CacheAwarePolicy.score`, the scoring path that must carry the volume term.
- `internal/gateway/replica_router.go` — `prefixSegments`, where the request's token
  volume is carried alongside the segment run.
- `docs/tickets/oss-port-p0/TICKET-07-route-uncached-volume.md` — routing and acceptance
  contract.

## Lane

gateway/route-uncached-volume

## Blast radius and affected lanes

One package: `internal/gateway`. `internal/gateway` is a serialized lane, which is
expected for this wave; a concurrent change in that package must land first or this leaf
waits for it. The sibling lane `gateway/sticky-prefix-route` touches the same package and
is coordinated by scope — this leaf owns the candidate *cost* term and must not change
which replica a session is returned to.

## Quarantined fallback mechanism

The current segment-only overlap score continues unchanged while this leaf is absent. A
request that carries no volume signal must fall back to the existing segment-count
ranking rather than be treated as zero-volume on every candidate, which would collapse
the ranking; if that fallback cannot be kept sound, leave the score alone and leave this
issue open.

- Centrality: Enabling (replica placement is ranked by the uncached prefill volume a candidate actually pays)
- P1 Context: advanced — a reachable routing signal gains the term that reflects real prefill cost.
- P2 Net value: advanced — a cache-warm replica is no longer scored as if it held nothing.
- P3 Adaptation: preserved — the existing segment-overlap locality term and tier weights are unchanged.
- P4 Operations: advanced — yields one bounded dispatchable P0 unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #10853.

## Completion standard

development

## Target operating envelope

- uncached prefill tokens per routed request: not applicable (development leaf; the token saving is a separate hardware receipt)

## Witnessed operating envelope

- uncached prefill tokens per routed request: not applicable (no uncached-volume term exists in the routing score yet, so no saving can be witnessed)
