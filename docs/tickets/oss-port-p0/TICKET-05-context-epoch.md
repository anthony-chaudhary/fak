# fix(gateway): isolate provider cache prefixes per context epoch

## Parent context

Coordinates with parent #10702 (https://github.com/anthony-chaudhary/fak/issues/10702).

Port of the context-epoch concept from opencode at pinned revision `4eb29a6` (MIT
licensed). Ported by adapting the concept to this tree's gateway request path; the
upstream file is not vendored.

<!-- fak-gateway-key: oss-port-p0-context-epoch -->
<!-- fak-cross-key: oss-port-p0-context-epoch -->
<!-- fak-public-issue: anthony-chaudhary/fak#10702 -->

```routing
lane: gateway/context-epoch
paths: ["internal/gateway/gateway.go", "docs/tickets/oss-port-p0/TICKET-05-context-epoch.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#10702
cross_key: oss-port-p0-context-epoch
```

Process cause: verification-gap

## Scope class

single-package

## Current state

`internal/gateway/gateway.go:867` constructs the in-kernel planner through
`agent.NewInKernelPlannerWithConfig`, passing the model ID as the planner's identity.
`modelID` is the only per-planner identity established at that seam today; nothing
carries a turn-scoped or session-scoped generation marker alongside it.

Two neighbouring surfaces in this package already know the shape of the problem, and
neither closes it:

- `internal/gateway/harness_coherence.go:42` computes a content-free SHA-256 digest
  of the inbound protected prefix — the bytes of the raw request body from the start
  through the first `cache_control` marker object. That file's own header records the
  hazard: two managers ride the same wire blind to each other, one that preserves the
  cached head and one that rewrites its own history and bursts the provider cache.
- `internal/gateway/prompt_order.go:1016-1027` clears any existing `CacheControl` on
  every message and then places exactly one `ephemeral` breakpoint at the prefix
  boundary, so there is a single deterministic cacheable head per request.

So the gateway can tell that a protected prefix CHANGED between two turns, and it
always advertises one stable cacheable head. What it cannot do is name which epoch
that head belongs to, and nothing fences a provider-side cache read on the epoch that
produced it. A provider-side cache hit is therefore admitted on prefix bytes alone: if
the current epoch invalidated state that those bytes previously described, the hit
still returns it.

## For

Served gateway requests that carry a provider-side cacheable prefix, on any Anthropic
passthrough or cached-provider route.

## Problem

A provider-side cache hit is trusted on prefix content alone. Once the current epoch
has invalidated state that a cached prefix describes, the gateway still accepts that
hit and serves the stale state, because nothing on the request names the epoch that
would let the stale read be refused.

## Today

Provider cache reads are admitted on a content match with no epoch identity. A stale
provider-side hit is undetectable and silently served.

## Better

Every served request carries a context epoch, and the provider-side cache prefix a
request may hit is isolated to that epoch — so a cached prefix can never return state
the current epoch invalidated.

## Why this is next

This is a cache-correctness seam and it is ranked above the performance work in this
cohort, and the reason is simple: a faster wrong cache is worse than a slower right
one. Every other unit here makes a computation cheaper or better ordered while
holding the result correct; this one closes a path where a cheap cached result can be
WRONG, and wrongness compounds silently because the provider cache lives outside this
process and outside any test. A performance unit that leaves a stale-read path open
would be optimising a system whose answers are not trusted. It is also cheap: the
prefix digest and the single deterministic breakpoint that this needs already exist in
this package.

## Working spine

Serving request arrives -> the gateway assigns a context epoch -> the provider-side
cache prefix that request may hit is scoped to that epoch -> an epoch that invalidated
state cannot be served from a prefix cached under an earlier epoch -> the stale read
is refused rather than returned.

## Core through-line

Carry a context epoch on the served request and fence the provider-side cache prefix
read on it, so a prefix cached under a superseded epoch is refused rather than
returned.

## Gold-plating boundary

No prefix-cache eviction or cost policy, no change to cache breakpoint placement or
tiering, no compaction or message-pruning redesign, no new provider protocol
negotiation, no prompt-cache pricing or savings accounting, no change to the
content-free digest's privacy property, and no latency or cost figure claimed from a
software test.

## Verifiable Witness

```bash
go test ./internal/gateway/ -run 'TestGatewayContextEpochCachePrefixIsolation' -count=1
```

`TestGatewayContextEpochCachePrefixIsolation` is added by this ticket; it does not
exist on the parent tree.

## Done condition / witness

Witness: `go test ./internal/gateway/ -run 'TestGatewayContextEpochCachePrefixIsolation' -count=1`.

- [ ] [SW-VERIFIED] The new test FAILS on the parent tree and PASSES after the
  change. A test that passes before the change proves nothing.
- [ ] [SW-VERIFIED] The test drives two epochs over one session: the first populates a
  provider-side cached prefix, the second epoch invalidates state that prefix
  described, and the test asserts the cached prefix is REFUSED rather than returned.
- [ ] [SW-VERIFIED] The test asserts the isolation holds across a path that preserves
  the protected prefix verbatim, so the refusal is attributed to the epoch and not to
  an incidental digest change.
- [ ] [SW-VERIFIED] A request whose epoch matches the cached prefix still hits, so the
  fence is an epoch fence and not a blanket cache disable.
- [ ] [SW-VERIFIED] The content-free property of the protected-prefix digest is
  preserved: the epoch and the fence carry no prompt bytes.
- [ ] [HW-WITNESSED] A provider-cache read on a live appliance under an epoch bump.
  `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue
or the resolving commit.

## Acceptance gate

The focused gateway test is green, it fails against the parent tree, and the routing
fence and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `gateway`
leaf in its commit subject.

## Witness

The focused gateway test above is the completion witness for epoch isolation. A
provider-cache read against a live appliance is a separate hardware measurement and
must cite its own receipt; a green software test is not that measurement.

## Likely files

- `internal/gateway/gateway.go` — the planner and request-path seam at line 867
  where the per-request identity is established.
- `internal/gateway/harness_coherence_test.go` — gains the new epoch-isolation test.
- `docs/tickets/oss-port-p0/TICKET-05-context-epoch.md` — routing and acceptance
  contract.

## Lane

gateway/context-epoch

## Blast radius and affected lanes

One package: `internal/gateway`. That package is large and serialized — concurrent
work in it contends — so this leaf is deliberately narrow and touches one decision
point. Compaction, tool routing, session lifecycle and the model-engine leaves are
outside it. The fence narrows which cache reads are admitted; it does not change the
content of any response for an epoch that is current.

## Quarantined fallback mechanism

The current prefix-content-matched provider cache behaviour continues unchanged while
this leaf is absent. If epoch assignment cannot be threaded onto the served request
without changing the wire shape a downstream client depends on, fall back to refusing
the provider-side cached read for the ambiguous case and record the resulting cost as
a gap — never to serving a stale prefix. If neither is possible, leave the seam
unchanged and keep this issue open rather than shipping an isolation that can be
bypassed.

- Centrality: Core (provider cache correctness on the served request path)
- P1 Context: advanced — an existing prefix-digest surface gains the epoch identity it lacks.
- P2 Net value: advanced — closes a stale provider-cache read that returns invalidated state.
- P3 Adaptation: preserved — the content-free digest and deterministic breakpoint are reused unchanged.
- P4 Operations: advanced — yields one bounded dispatchable correctness unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #10702.

## Completion standard

development

## Target operating envelope

- stale provider-cache reads admitted per context epoch: not applicable (this leaf proves the epoch fence in software, not a provider-side hit-rate figure on live hardware)

## Witnessed operating envelope

- stale provider-cache reads admitted per context epoch: not applicable (no context epoch is carried on the served request today, so no epoch-scoped cache admission can be witnessed)