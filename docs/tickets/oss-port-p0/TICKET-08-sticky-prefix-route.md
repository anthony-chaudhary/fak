# perf(gateway): keep a conversation on the replica that holds its prefix cache

## Parent context

Coordinates with parent #10852 (https://github.com/anthony-chaudhary/fak/issues/10852).

Direct port/adapt of the sticky prefix routing rule studied in vidur at revision
`abae7f6` (MIT). Once a replica holds a hot prefix, load-balancing the next request of
that conversation onto a different replica forfeits the cache; the conversation should
stay where its prefix already lives.

<!-- fak-gateway-key: oss-port-p0-sticky-prefix-route -->
<!-- fak-cross-key: oss-port-p0-sticky-prefix-route -->
<!-- fak-public-issue: anthony-chaudhary/fak#10852 -->

```routing
lane: gateway/sticky-prefix-route
paths: ["internal/gateway/residency_router.go", "internal/gateway/replica_router.go", "docs/tickets/oss-port-p0/TICKET-08-sticky-prefix-route.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#10852
cross_key: oss-port-p0-sticky-prefix-route
```

Process cause: verification-gap

## Scope class

single-package

## Current state

The stickiness primitive exists in this tree and is inert on the production path.
`internal/gateway/residency_router.go` defines `AffinityPicker` (line 829) and
`CacheAwarePolicy.PickWithAffinity` (line 838), backed by a bounded affinity map with
`SetAffinity` (line 633), `GetAffinity` (656), `ClearAffinity` (678),
`WithAffinityTTL` (552) and `WithAffinityMax` (562). A repository-wide search finds
`PickWithAffinity` and `SetAffinity` called only from `residency_router_test.go` and
`residency_router_affinity_test.go` — no production line in the module calls either.

The production placement path never reaches them. `ReplicaDispatch.pickByPolicy`
(`internal/gateway/replica_router.go`, line 736) calls the three-argument
`r.policy.Pick(candidates, prefix, load)`, and that `Pick` (residency_router.go line 899)
delegates to `PickWithAffinity(candidates, prefix, "", load)` — with an empty affinity
key. No production line derives a conversation identity to supply one.

The fallback is not sticky either. `ReplicaDispatch.pickKeyed` (line 693) is a stateless
rendezvous over `replicaRendezvousKey(prefix)`: it is pure and replayable by design, and
it never consults which replica holds the prefix. Because the rendezvous runs over the
admissible set, a membership change that removes the current winner moves the winner to
a cold replica and takes the warm prefix with it.

## For

Multi-turn agent conversations routed across a replica fleet, where the conversation's
prefix is already resident on one replica.

## Problem

A conversation's next turn is free on the replica that holds its prefix and expensive on
every other replica, yet nothing routes it back to the replica that holds it, so each turn
re-pays the whole prefix.

## Today

The production pick supplies an empty affinity key, so the sticky branch never engages,
and the rendezvous fallback ignores prefix residency entirely. A warm prefix can be
forfeited when the admissible set changes.

## Better

The production placement path derives a conversation-scoped affinity key and honours the
replica that holds that conversation's prefix, falling back to the current ranking
whenever the sticky replica is no longer admissible.

## Why this is next

The concept is ported directly from a permissive upstream (vidur @ `abae7f6`, MIT) and the
bounded map, the TTL and the capacity bound already exist and are already covered by
tests; the missing piece is the production key. It is the cheapest unit that makes
existing, tested stickiness actually reachable, and it is disjoint from the sibling
uncached-volume leaf, which changes what a candidate costs rather than which candidate a
session returns to.

## Working spine

A conversation's first turn is placed by the existing ranking -> the placement records an
affinity from the conversation identity to the winning replica -> the conversation's next
turn consults that affinity and stays on the replica holding its prefix -> when that
replica is no longer admissible the placement falls back to the existing ranking and
re-records the new affinity.

## Core through-line

Route a conversation back to the replica that already holds its prefix, and prove the
stickiness by the production placement entry point — a test that drives
`PickWithAffinity` directly proves the component, not that it is reached.

## Gold-plating boundary

No consistency or rebalancing protocol for the affinity map: no cluster-wide affinity
replication, no gossip, no ownership migration, no load-balance re-optimisation pass, no
scheduler feedback loop that unsticks an overloaded holder, and no sticky TTL or capacity
tuning. The bounded map, its TTL and its capacity bound already exist and are reused
unchanged. This leaf wires the existing primitive to the production pick and nothing
more. Note the tradeoff plainly: stickiness buys cache locality with load balance, and
this leaf deliberately accepts an imbalanced holder rather than paying for a rebalancer.

## Verifiable Witness

```bash
go test ./internal/gateway/... -run 'TestStickyPrefixRoute' -count=1
```

The test must fail against the parent tree. It places the first turn through the production
placement entry point, removes or degrades the winner so a non-sticky pick would move, and
asserts the next turn of the same conversation returns to the replica that holds its
prefix.

## Done condition / witness

Witness: `go test ./internal/gateway/... -run 'TestStickyPrefixRoute' -count=1`.

- [ ] [SW-VERIFIED] The production placement entry point derives a conversation-scoped
  affinity key and records it; no production call still passes an empty affinity key.
- [ ] [SW-VERIFIED] A test drives the production entry point, not `PickWithAffinity`
  directly, and asserts the second turn returns to the replica holding the prefix; the
  test FAILS against the parent tree and PASSES after the change.
- [ ] [SW-VERIFIED] A test asserts the honest fallback: when the sticky replica is no
  longer admissible, placement falls back to the existing ranking and re-records affinity
  rather than routing to a dead or wrong upstream.
- [ ] [HW-WITNESSED] The per-turn prefix re-prefill saving is measured on a live gateway.
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

The focused gateway test above is the completion witness for the production stickiness
wiring. The prefix re-prefill saving measurement is a separate hardware read against a
live gateway and must cite its own receipt; a green software test is not that measurement.

## Likely files

- `internal/gateway/residency_router.go` — `AffinityPicker`, `PickWithAffinity` and the
  bounded affinity map whose production wiring is missing.
- `internal/gateway/replica_router.go` — `pickByPolicy`, the production placement entry
  point that currently supplies no affinity key.
- `docs/tickets/oss-port-p0/TICKET-08-sticky-prefix-route.md` — routing and acceptance
  contract.

## Lane

gateway/sticky-prefix-route

## Blast radius and affected lanes

One package: `internal/gateway`. `internal/gateway` is a serialized lane, which is
expected for this wave; a concurrent change in that package must land first or this leaf
waits for it. The sibling lane `gateway/route-uncached-volume` touches the same package and
is coordinated by scope — that leaf owns the candidate cost term, this leaf owns the
conversation affinity key. The harness version router's own session stickiness is a
separate concern and stays outside the leaf.

## Quarantined fallback mechanism

The current stateless rendezvous placement continues unchanged while this leaf is absent,
and it remains the fallback whenever no affinity key can be derived or the sticky replica
has left the admissible set. If honouring stickiness would route to a replica that no
longer holds the prefix or no longer carries the model, the placement must fall back to
the existing ranking rather than honour a stale affinity.

- Centrality: Enabling (a multi-turn conversation returns to the replica that holds its prefix cache)
- P1 Context: advanced — existing, already-tested stickiness becomes reachable from the production pick.
- P2 Net value: advanced — a conversation stops re-paying its prefix on every turn.
- P3 Adaptation: preserved — the bounded affinity map, its TTL and its capacity bound are reused unchanged.
- P4 Operations: advanced — yields one bounded dispatchable P0 unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #10852.

## Completion standard

development

## Target operating envelope

- prefix re-prefill tokens per conversation turn: not applicable (development leaf; the prefix saving is a separate hardware receipt)

## Witnessed operating envelope

- prefix re-prefill tokens per conversation turn: not applicable (no production line supplies an affinity key yet, so no sticky turn can be witnessed)
