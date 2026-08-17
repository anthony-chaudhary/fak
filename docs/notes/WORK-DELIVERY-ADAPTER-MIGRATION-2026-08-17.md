# Work-delivery adapter migration — 2026-08-17

## What changed

`internal/workdelivery` now exposes a single `fak.work-delivery-adapter/v1` envelope for the existing commit, build/CI, dispatch/fleet, integration, and release seams. Every envelope names a canonical `internal/deliverystages` stage and bottleneck. A receipt changes exactly one delivery axis.

## Consumers migrated

The adapter API is additive: callers can emit recording-only commit evidence, verification-only build evidence, exact dispatch/fleet blocker identity, integration-only push evidence, and explicit release-readiness evidence. Existing seam schemas remain accepted; their local words resolve through the canonical stage crosswalk while callers migrate.

## Impact, cutover, rollback

Impact: committed or pushed work can no longer be represented by this API as implicitly compile-admitted or release-ready. Cutover: seam owners call the typed constructors and serialize the returned envelope beside their existing payload. Rollback: stop emitting the additive envelope; no existing JSON field or consumer is removed in this change.
