<!-- fak-cross-key: oss-refresh-20260929-cache-pin-owner -->
# Audit shared page pin ownership and prove a consuming path

```routing
lane: ctxmmu
paths: ["docs/tickets/oss-refresh-20260929/TICKET-cache-pin-owner.md"]
expected_steps: 3
```

## Target repository

Public `anthony-chaudhary/fak`; core runtime mechanism. Dispatch through `fak-flow spawn oss-refresh-20260929-cache-pin-owner --route-repo public`. The specification belongs to the public runtime backlog.

## Working spine

Read source-confirmed invariant -> reproduce through the public runtime seam -> bounded correction -> independent regression -> durable green acceptance receipt.

## Current state

Pin and Unpin are digest-counted without owner identity. A duplicate release can consume another owner's count in a constructed API example. Coordinator readback found no production call to PagedStore.Pin/Unpin in the public tree; the MMU stages and reads pages without using those methods. This is a source-level concern, not a demonstrated runtime defect. The first deliverable is a consuming-path audit and an explicit disposition.

## Source and license

LMCache proposed PR #5098, head dc0939588e3dac720b6fefe125d91d1c613f407e; lmcache/v1/cache_engine.py:941. Source: https://github.com/LMCache/LMCache/pull/5098. Observation 2026-09-29; upstream tracker states are dated. Apache-2.0 (vLLM/LMCache) or MIT (llama.cpp), ADAPT with source attribution retained. No source code is copied by this ticket. The cited upstream PR and revision are the source anchor; this is a bounded source assessment.

## Core through-line

Trace PagedStore construction, its interfaces and all Pin/Unpin consumers at the claimed public revision. Record file:line evidence and an executable reachability witness when a production consumer exists. If no consumer exists, close as WATCH with a concrete future trigger; do not add unused receipt machinery. If a consuming defect is reproduced, create a separately scoped implementation contract covering the actual caller and independent regression.

## Gold-plating boundary

No new inference backend, cache rewrite, rollout or performance claim. No hardware qualification is needed for the CPU correctness invariant. Keep the repair bounded to this outcome and its consuming call path.

## File:Line seam

- `internal/ctxmmu/paged_store.go:163` at public baseline `365167aa5c8b5f3252221594eb5cf2a38379f16c`.

## Dedupe and routing

Dedupe key `oss-refresh-20260929-cache-pin-owner`. Both-repo capability queries and nearby issues are in the cohort note. The initial 56-contract native census contains no contract with this same outcome. Related tickets remain separate owners; refresh before claim because peer backlog can advance.

## Done condition / witness

All unchecked criteria below must be satisfied through the real consuming entrypoint; preserve correct existing behavior and reject the adversarial case.

## Definition of done

- [ ] Audit the current public tree for PagedStore construction, interfaces, Pin/Unpin call sites and production entrypoint reachability; retain commands and file:line evidence in this specification.
- [ ] Record WATCH if no consuming path exists, with a trigger when a real caller begins pinning; otherwise reproduce the actual caller failure and register a distinct implementation contract with that caller in its scope.
- [ ] Run `go test ./internal/ctxmmu -count=1`; retain a public signed commit containing the audit receipt. No runtime repair or hardware claim is required for this audit.

## Witness

```bash
go test ./internal/ctxmmu -count=1
```

## Process cause

Process cause: verification-gap

Source readback exposes a possible invariant gap; production reachability must be resolved before implementation is justified.

## Acceptance gate

```bash
go test ./internal/ctxmmu -count=1
```

Acceptance is a future repair gate; this study has not shipped the repair. An unexecuted future test must not be reported green.

## Parent context

OSS refresh campaign GOAL-oss-refresh-20260929. This bounded leaf is grounded by the upstream issue/PR cited above.

## Why this is next

This bounded audit prevents a source-only concern from becoming unused runtime code and preserves the upstream ownership lesson for a future real consumer.

## Closure binding

Close this audit only after its consuming-path disposition and acceptance receipt are committed, with any proved implementation outcome routed to a distinct native contract.
