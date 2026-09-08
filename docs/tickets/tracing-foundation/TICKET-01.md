# feat(trajectory): define the typed causal envelope shared by agent-facing tracers

```routing
lane: trajectory
paths:
  - pkg/traceabi/**
  - internal/trajectory/**
expected_steps: 8
```

<!-- fak-trace-foundation-key: typed-causal-envelope-v1 -->

## Parent context

#6053. Foundation companion that extracts the reusable identity contract needed by that coordination-specific trace/explain/replay issue. It follows completed #5629 and #8460 without reopening their gateway W3C propagation work. The timing/identifier inventory in #10621 should be consumed as evidence, but this leaf does not wait on unrelated time-accounting design.

## Why this is next

Every proposed domain tracer otherwise invents its own identity fields first, creating migration work before useful traces can compose. This contract is the smallest serial foundation.

## Problem

For agent, serving, and kernel developers, one logical operation currently changes identity shape as it crosses trace producers. A gateway span, runtime event, trajectory event, and compute event can describe the same work but cannot be joined without producer-specific guesses. Today the join is ad hoc. Better because one typed envelope makes causal navigation deterministic while preserving native identifiers rather than replacing them. Witness: two asynchronous runtime events and one joined child normalize into one trace graph with stable IDs before and after JSON round-trip.

## Centrality and P1-P4

**Centrality: Enabling.** This is the narrow shared contract that makes the higher-value tensor, context, turn, and agent-flow tracers composable.

- P1 Context: advanced - a compact identity envelope lets an agent request one relevant causal slice instead of replaying full logs.
- P2 Net value: preserved - correlation removes manual joins, but no latency or token-saving claim is made until measured by a consumer.
- P3 Adaptation: advanced - identifiers, parentage, ordering, and correlation kinds are closed and validated; unknown mappings remain explicit.
- P4 Operations: advanced - the same envelope can cross harness, serving, memory, and compute adapters without making any sink authoritative.

## Current state

At `f21eb416d92f03286747f84e888fb7dd1038c7a9`:

- `internal/trajectory/runtime_event.go:33-44 (RuntimeEvent)` has `event_id`, `session_id`, `turn_id`, a flat `trace_id`, and sequence, but no span identity, parent span, or typed aliases for producer-native correlation IDs.
- `internal/trajectory/runtime_event.go:139-150 (AsTrajectoryEvents)` embeds the runtime row in payload and projects `SessionID` to `ConversationID`; the flat trace identity is no longer a first-class causal field on `trajectory.Event`.
- `internal/trajectory/event.go:72-85 (Event)` has many event `ParentIDs`, provenance, visibility, and payload, but no W3C span context or explicit mapping between run/request/session/turn identifiers.
- `internal/gateway/tracecontext.go:17-20 (traceContext)` owns W3C `TraceID`, `ParentID`, and flags as a gateway-private type.
- `internal/computetrace/trace.go:22-47 (Event)` independently uses `RunID` and `RequestID` without the trajectory trace/span vocabulary.

The capability is **PARTIAL**: strong local identities exist, but there is no shared causal envelope.

## Core through-line

Add one versioned, data-only `CausalEnvelope` to public `pkg/traceabi` with:

- `trace_id`, `span_id`, optional `parent_span_id`, and monotonic `sequence`;
- `logical_id`, `operation`, and `phase` for domain-stable identity;
- typed `correlations[]` entries (`session`, `turn`, `request`, `run`, `tool_call`, `artifact`, `producer_native`) whose values remain opaque;
- source schema/version and explicit `mapping_status` (`exact`, `derived`, `unknown`);
- deterministic validation, canonical ordering, and JSON round-trip.

Wire the first tracer bullet inside `internal/trajectory`: map `RuntimeEvent` to the public envelope without inventing a span or parent when the source lacks one. The exported package is the stable cross-repository seam; domain adapters for gateway, compute, tensors, context, and private platform flows are follow-ons.

## Working spine

1. Define `fak-causal-envelope/1` and closed correlation/mapping vocabularies in `pkg/traceabi`.
2. Validate required IDs, W3C-sized IDs when marked W3C, nonzero sequence, duplicate correlations, and self-parenting.
3. Canonicalize correlation ordering without changing opaque values.
4. Add an additive internal envelope projection for `RuntimeEvent` that imports only the public package.
5. Preserve the envelope through `AsTrajectoryEvents` as typed metadata rather than requiring payload parsing.
6. Add a golden fork/join fixture and strict JSON decode.
7. Document compatibility: legacy rows map to `mapping_status=unknown`, never to fabricated causality.

## Gold-plating boundary

Do not add an OpenTelemetry SDK, collector, storage engine, UI, clock-synchronization protocol, or instrumentation in every producer. Do not replace `X-Trace-Id`, session IDs, run IDs, or request IDs. Keep `pkg/traceabi` data-only and stdlib-only: no private knowledge, recorder, storage, policy engine, or internal imports. This issue defines the shared envelope plus one internal adapter.

## Dedupe

Search across open and closed public issues on 2026-09-08 found:

- #5629 and completed children #8107/#8109: W3C propagation and bounded OTLP for served gateway requests; they do not define a cross-producer semantic envelope.
- #8460: preserves opaque gateway correlation IDs beside W3C context; this proposal generalizes that separation without changing gateway code.
- #6053: records global coordination decisions; this proposal is its reusable identity dependency, not the coordination record.
- #10621: inventories timing sources and IDs; it is research input, not the implementation contract.
- #6801: asks for vendor-neutral telemetry/eval/artifact sinks; it consumes this envelope but owns sink conformance.

No issue titled or bodied as a typed shared causal envelope was found. Keep the dedupe key above stable.

## Prior-art ledger

- **Source:** OpenTelemetry Trace API `specification/trace/api.md:219-280,311-443@ce9394dc3dd53bb182611d2f3ba1e8f53975e905`.
- **Source event:** commit `ce9394dc3dd53bb182611d2f3ba1e8f53975e905`, 2026-09-08T13:54:14Z; source state: shipped specification on `main`.
- **Observed at:** 2026-09-08T16:10:19Z.
- **License:** Apache-2.0.
- **Disposition:** **ADAPT / DEFAULT**. Reuse immutable trace/span context and parent semantics; retain fak-native typed correlations and stdlib-only implementation. No upstream code copied.
- **Refresh trigger:** OpenTelemetry trace API semantic revision or W3C Trace Context revision.
- **Spirit extension:** immutable transport context should not erase richer local identity; preserve both and make the mapping evidence explicit.

## Done condition

The public trace ABI owns one versioned, strict causal envelope, and the trajectory adapter can project current runtime events into it without losing or inventing identity.

## Definition of done

- [ ] Public `pkg/traceabi.CausalEnvelope` and its schema descriptor define every required field and closed vocabulary without importing `internal/*`.
- [ ] Validation rejects empty IDs, invalid marked-W3C IDs, duplicate correlation pairs, zero sequence, and self-parenting.
- [ ] Correlations serialize in deterministic kind/value order.
- [ ] `RuntimeEvent` projection preserves event/session/turn/trace/sequence exactly.
- [ ] Missing span/parent information is labeled unknown, not synthesized.
- [ ] `AsTrajectoryEvents` preserves the envelope as first-class typed data across encode/decode.
- [ ] Legacy trajectory fixtures remain readable with explicit unknown mapping status.
- [ ] `pkg/traceabi` stays stdlib-only and adds no hot-path recorder; an external-package test proves it is consumable across the public/private boundary.

## Verifiable Witness

```bash
go test ./pkg/traceabi ./internal/trajectory -run 'TestCausalEnvelope|TestRuntimeEventCausalProjection|TestCausalEnvelopeForkJoin|TestExternalConsumer' -count=1
```

The golden fixture must prove one trace ID survives runtime-event projection and JSON round-trip; two parents join one child in stable order; malformed or guessed mappings are refused or marked unknown.

## Witness

Run the deterministic `go test ./pkg/traceabi ./internal/trajectory -run 'TestCausalEnvelope|TestRuntimeEventCausalProjection|TestCausalEnvelopeForkJoin|TestExternalConsumer' -count=1` gate above.

## Acceptance gate

A fixture containing a root, two asynchronous children, and one fan-in event decodes to the same canonical envelope bytes twice, retains all native correlations, and reports zero fabricated identities.

## Likely files

- `pkg/traceabi/envelope.go`
- `pkg/traceabi/envelope_external_test.go`
- `internal/trajectory/runtime_event.go`
- `internal/trajectory/runtime_event_test.go`

## Lane

trajectory

## Expected steps

8

## Closure binding

The resolving commit cites this issue and uses `(fak traceabi)`. Report `pkg/traceabi@rev`, the golden envelope digest, and the exact cross-package witness command above.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12258
