# feat(agent): expose actual sequence-prefill route in native receipts

<!-- fak-agent-key: native-sequence-prefill-route-receipt -->

GitHub issue: [#12698](https://github.com/anthony-chaudhary/fak/issues/12698).

```routing
lane: agent
paths: ["internal/model/native_inference_receipt.go", "internal/agent/inkernel_decode.go", "internal/agent/inkernel_planner.go", "internal/agent/inkernel_prefill_route.go", "internal/agent/inkernel_prefill_route_test.go", "docs/tickets/native-prefill-route/TICKET-01-actual-route-receipt.md"]
expected_steps: 5
```

## Parent context

Coordinates with: #12695 (bounded embedding rows for sequence prefill).
This leaf exposes existing model-session evidence through the native request receipt.

## Current state

`internal/model/qwen35_hal.go:81` defines `Qwen35SequencePrefillRouteStatus` and
`:93` returns the session's latest actual route decision. The native receipt in
`internal/model/native_inference_receipt.go:7` does not export that decision.
`internal/agent/inkernel_planner.go:1625` builds a receipt with a generic decode
forward path, fixed top-level `fallback_active:false`, and configured chunk size.
Those fields do not prove that this request actually executed whole-sequence prefill.
The session may have declined sequence prefill and continued by token replay.

## Why now

Serving qualification consumes native receipts. A model-session decline can be
lost before those receipts reach the caller, so qualification cannot distinguish
actual whole-sequence execution from eligibility or an alternate prefill route.

## Problem frame

- Centrality: Core.
- P1 Context: advanced - expose the actual per-request route already known by the engine.
- P2 Net value: advanced - prevent tuning and qualification decisions based on a missing route.
- P3 Adaptation: preserved - add an optional native-receipt field; existing readers remain compatible.
- P4 Operations: advanced - retain decline reasons and mixed-chunk uncertainty in durable evidence.

For native serving clients / Problem: route decisions are absent from returned evidence /
Today: chunk eligibility and decode identity are the available hints / Better because:
the receipt names actual executed or declined prefill / Witness: deterministic request
measurement and JSON tests plus a service receipt from the hardware validation owner.

## Working spine

1. Add an explicit per-request sequence-prefill route receipt shape.
2. Capture status after every real prefill call before the session can close.
3. Aggregate conservatively across chunks and keep no-prefill requests unmeasured.
4. Attach the snapshot to the normal native completion receipt.
5. Verify independently and ship the scoped public change.

## Core through-line

Actual prefill invocation -> session route status -> request-local aggregation ->
native inference receipt JSON -> a caller can inspect effective path, fallback,
decline reason and whole-sequence qualification without guessing.

The additive `qwen35_sequence_prefill_route` object counts executed prefill calls
and freshly observed decisions. Its `complete` and outer
`native_performance_qualifying` fields describe the request aggregate; nested
`status` preserves a representative actual model decision, prioritizing an
observed nonqualifying route. Missing decisions remain unobserved. The field is
absent when the request executed no prefill calls. Route qualification does not
certify numerical parity or physical performance.

## Gold-plating boundary

No kernel optimization, backend replacement, new model eligibility policy,
private parser requirement, hardware deployment, global fallback reinterpretation,
or performance speedup claim. Coordinate the small decode measurement seam with
the concurrently landing cancellation repair before editing that file.

## Done condition

- [x] [SW-VERIFIED] The normal native receipt carries actual observed sequence route status.
- [x] [SW-VERIFIED] Every executed prefill chunk contributes; later success cannot erase earlier fallback or unknown evidence.
- [x] [SW-VERIFIED] A cache-only request with no prefill call is explicitly unmeasured and nonqualifying.
- [x] [SW-VERIFIED] Missing/unsupported status never fabricates a successful route.
- [x] [SW-VERIFIED] A final one-token call cannot reuse a preceding eligible chunk's stale sequence status.
- [x] [SW-VERIFIED] Capture precedes session close; reset/reuse cannot leak an earlier request's status.
- [x] [SW-VERIFIED] Independent tests cover JSON output, decline, mixed chunks, success, no-prefill and request reset.

## Definition of done

The scoped receipt and real call-path wiring land with independent software tests.
Physical service qualification is witnessed separately by the hardware validation
owner; synthetic tests do not count as hardware evidence.

## Witness

The deterministic source witness proves the actual session status has no request
observer; behavioral tests then exercise the native receipt path.

### Verifiable Witness

Pre-fix deterministic source witness:

```text
git grep -n 'Qwen35SequencePrefillRouteStatus' 3b930c397 -- internal/agent internal/model/native_inference_receipt.go
```

Observed on public revision `3b930c397`: exit 1, no receipt or request observer.
Post-fix behavioral witness:

```text
go test ./internal/agent -run 'TestNativePrefillRoute' -count=1
go test ./internal/model ./internal/agent -short -count=1
```

Validation on the isolated candidate based on `0d69e314ca2c9919e99568eff1bcc8c32d390aad`
with Go 1.26.7, Windows/amd64, `GOWORK=off`:

- Nine independently authored `TestNativePrefillRoute` cases passed, including
  real prefill-helper call-path assertions and receipt JSON. The coordinator
  independently reran the focused command successfully.
- The full `internal/agent -short` package passed (144.525 seconds).
- A duplicate package run under concurrent load tripped the timing-sensitive
  `TestBlackboard_ZeroCopyVsJSONLatency` threshold at 1.327 microseconds per
  operation (limit: below 1 microsecond). Its isolated rerun passed in 0.045
  seconds after the competing load ended; no receipt source or test was changed.
- `go vet ./internal/agent ./internal/model`, `gofmt` and `git diff --check` passed.
- The full model package ran and failed in four CPU numerical tests:
  `TestQwen35PrefillAndStepCPUContinuation` (NaN prefill logits),
  `TestQwen35LinearAttnBatchedResumesState`,
  `TestQwen35LinearAttnBatchedMatchesScalar`, and
  `TestQwen35ChunkedBatchedMatchesScalar` (float rounding mismatches).
  Independent baseline runs at public revision
  `816506b53c263edd44e4b2a151ac80cac3b44031` reproduced the same NaN and exact
  mismatched float values in these four tests, after resolving that baseline's
  module checksums with `-mod=mod`. The existing failures are tracked by #12433
  and #12038. This is not a green full-model-suite claim.

No physical execution or numerical-parity qualification is claimed by these
software witnesses. The hardware validation owner captures the deployed receipt.

### File:Line seams

- `internal/model/qwen35_hal.go:81` (`Qwen35SequencePrefillRouteStatus`)
- `internal/model/native_inference_receipt.go:7` (`NativeInferenceReceipt`)
- `internal/agent/inkernel_decode.go:459` (`nativeInferenceMeasurement`)
- `internal/agent/inkernel_decode.go:363` (prefill calls)
- `internal/agent/inkernel_planner.go:1598` (`buildNativeInferenceReceipt`)

### Blast radius and fallback

Two public packages: internal/model and internal/agent. Existing JSON fields and
native inference behavior remain compatible. Unobserved status is nonqualifying;
the additive field does not force runtime route selection. Private factory and
hardware validation continue independently.

## Likely files

- `internal/model/native_inference_receipt.go`
- `internal/agent/inkernel_decode.go`
- `internal/agent/inkernel_planner.go`
- `internal/agent/inkernel_prefill_route.go`
- `internal/agent/inkernel_prefill_route_test.go`

## Lane

agent; two packages; five steps.

## Acceptance gate

Scoped tests and boundary/leak/provenance checks pass. A captured real service
receipt remains a separate physical qualification criterion and is not fabricated.

## Closure binding

One signed public resolving commit with the issue reference, independent test
context and software witness. Completion standard: integrated.
