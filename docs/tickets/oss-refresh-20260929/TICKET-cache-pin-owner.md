<!-- fak-cross-key: oss-refresh-20260929-cache-pin-owner -->
# Make shared page pin release ownership-safe

```routing
lane: ctxmmu
paths: ["internal/ctxmmu/paged_store.go", "internal/ctxmmu/paged_store_test.go"]
expected_steps: 5
```

## Target repository

Public `anthony-chaudhary/fak`; core runtime mechanism. Dispatch through `fak-flow spawn oss-refresh-20260929-cache-pin-owner --route-repo public`. The specification belongs to the public runtime backlog.

## Working spine

Read source-confirmed invariant -> reproduce through the public runtime seam -> bounded correction -> independent regression -> durable green acceptance receipt.

## Current state

Pin and Unpin are digest-counted without owner identity. A duplicate release by one caller can consume another caller's pin. This is an API safety gap; the study does not prove a live duplicate-release incident.

## Source and license

LMCache proposed PR #5098, head dc0939588e3dac720b6fefe125d91d1c613f407e; lmcache/v1/cache_engine.py:941. Source: https://github.com/LMCache/LMCache/pull/5098. Observation 2026-09-29; upstream tracker states are dated. Apache-2.0 (vLLM/LMCache) or MIT (llama.cpp), ADAPT with source attribution retained. No source code is copied by this ticket. The cited upstream PR and revision are the source anchor; this is a bounded source assessment.

## Core through-line

Introduce an additive owner/generation pin receipt with idempotent release, then connect one real consuming path and explicitly bound compatibility of legacy callers. Expand the scope fence only through a witnessed consuming caller, not an unused helper.

## Gold-plating boundary

No new inference backend, cache rewrite, rollout or performance claim. No hardware qualification is needed for the CPU correctness invariant. Keep the repair bounded to this outcome and its consuming call path.

## File:Line seam

- `internal/ctxmmu/paged_store.go:163` at public baseline `365167aa5c8b5f3252221594eb5cf2a38379f16c`.

## Dedupe and routing

Dedupe key `oss-refresh-20260929-cache-pin-owner`. Both-repo capability queries and nearby issues are in the cohort note. The initial 56-contract native census contains no contract with this same outcome. Related tickets remain separate owners; refresh before claim because peer backlog can advance.

## Done condition / witness

All unchecked criteria below must be satisfied through the real consuming entrypoint; preserve correct existing behavior and reject the adversarial case.

## Definition of done

- [ ] Reproduce the current gap through the real target entrypoint.
- [ ] Implement the bounded change and independent edge-case regression.
- [ ] With owners A and B on one digest, releasing A twice must leave B protected under eviction pressure; stale generation and cancellation cases fail safely; a real caller obtains and releases the receipt.
- [ ] Verify `go test ./internal/ctxmmu -count=1`, public leak/boundary checks, and commit/land through the public native workflow.

## Witness

```bash
go test ./internal/ctxmmu -count=1
```

## Process cause

Process cause: verification-gap

Source readback exposes an unguarded invariant; a real-entrypoint regression is required before claiming the repair.

## Acceptance gate

```bash
go test ./internal/ctxmmu -count=1
```

Acceptance is a future repair gate; this study has not shipped the repair. An unexecuted future test must not be reported green.

## Parent context

OSS refresh campaign GOAL-oss-refresh-20260929. This bounded leaf is grounded by the upstream issue/PR cited above.

## Why this is next

The source-confirmed invariant gap affects trustworthy cache reuse or loading in the public native runtime. Add the smallest independent correctness regression before expanding features.

## Closure binding

Close this native ticket only after its reproduction and acceptance command pass on the resolving signed public commit, with the ticket ID and source attribution recorded.
