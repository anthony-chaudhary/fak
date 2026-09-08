# feat(traceabi): query, explain, and export one causal trace through a stable agent-facing surface

```routing
lane: traceabi
paths:
  - pkg/traceabi/**
  - cmd/fak/**
expected_steps: 8
```

<!-- fak-trace-foundation-key: causal-trace-query-explain-export-v1 -->

## Parent context

#6053 and #6801. This is the public read-surface companion to `typed-causal-envelope-v1`, `many-to-many-value-lineage-v1`, and `bounded-private-evidence-policy-v1`. All three are hard prerequisites: this issue queries their `pkg/traceabi` contracts and must not define a second envelope, lineage graph, capture policy, evidence vocabulary, or private-only query model. It is not #4262's Perfetto projection: that issue remains the owner of Perfetto encoding for `MicroSpan`.

## Why this is next

Once canonical envelope and policy rows exist, an agent needs a bounded read path immediately; otherwise useful traces remain expert-only artifacts and invite ad hoc full-file reads.

## Problem

For an agent debugging one outcome, trace artifacts are useful only if it can ask bounded causal questions without hand-parsing producer schemas. Today `trajquery` filters flat rows, while each trace command owns separate summaries. Better because one stable surface answers “what happened,” “why/what contributed,” and “give this slice to another tool,” with privacy/loss evidence preserved. Witness: the same fixture queried by trace/span/logical ID yields stable human and JSON explanations and a canonical export whose re-import digest matches.

## Centrality and P1-P4

**Centrality: Core agent usability.** This is the read path that turns tracing data into usable agent evidence.

- P1 Context: advanced - bounded slices prevent whole-trace context dumps.
- P2 Net value: preserved - query cost, rows scanned/returned, bytes, and truncation are receipted; no efficiency claim is inferred.
- P3 Adaptation: advanced - scope rewrite, depth/row/byte limits, visibility, and loss propagation are mandatory.
- P4 Operations: advanced - one CLI/API separates semantic query/explain from sink-specific exporters.

## Current state

At `f21eb416d92f03286747f84e888fb7dd1038c7a9`:

- `internal/trajquery/query.go:41-48,79-96` defines flat SELECT/projection/AND-filter/LIMIT execution over `Row map[string]any`.
- `internal/trajquery/parser.go:10-17` intentionally excludes joins, functions, and graph traversal.
- `cmd/fak/trajquery.go:12-40` exposes only `run|validate`; `:72-104` validates/re-writes scope then streams matching rows.
- `cmd/fak/computetrace.go:55-74` has a separate aggregate-only summary.
- #4262 remains open for a Perfetto/Chrome-trace exporter over `MicroSpan`; #8109 already owns OTLP gateway export.

The capability is **PARTIAL**: secure flat query exists, but its internal package cannot be consumed across the public/private boundary, and no public causal explain/canonical slice export contract exists.

## Dependencies and schedule

- Hard prerequisite: `typed-causal-envelope-v1` supplies public causal identity.
- Hard prerequisite: `many-to-many-value-lineage-v1` supplies public predecessor/contributor traversal data.
- Hard prerequisite: `bounded-private-evidence-policy-v1` supplies public privacy, evidence, loss, and capture receipts.
- Schedule after TICKET-01, TICKET-02, and TICKET-03 because all four touch `pkg/traceabi/**`.
- The `cmd/fak/**` work is a thin consumer only; coordinate that path at dispatch. Existing `internal/trajquery` remains unchanged and is not an implementation dependency.

## Core through-line

Add the canonical query request/result types, deterministic indexes, bounded `Ancestors`, `Descendants`, `Contributors`, and `Explain` helpers, and one shared explanation model to public `pkg/traceabi`. They operate only on the prerequisite envelope, lineage, capture, evidence, and loss contracts. Add `fak trace query|explain|export` as a thin adapter over that public API. Export canonical NDJSON first; named adapters are registered capabilities, so Perfetto delegates to #4262 and OTLP remains the existing gateway exporter rather than being reimplemented. Do not route private consumers through `internal/trajquery`.

## Working spine

1. In `pkg/traceabi`, index trace/span/logical IDs and validate duplicate/dangling identities.
2. Add bounded ancestor/descendant/contributor traversal.
3. Render one shared explanation model as concise text and JSON.
4. Preserve visibility, evidence class, privacy receipt, and loss/drop metadata.
5. Add explicit max depth, rows, bytes, and scanned rows.
6. Export canonical NDJSON with schema and source digest.
7. Refuse unknown/unavailable exporter names with discoverable capability output.
8. Wire `fak trace` as a thin adapter and prove an external-package consumer plus golden round-trip fixture.

## Gold-plating boundary

No graph database, GUI, SQL joins, live tail service, arbitrary plugins, OTLP rewrite, Perfetto encoder, or migration of the existing flat `internal/trajquery` language. Keep `pkg/traceabi` stdlib-only, data-oriented, and free of private knowledge. The first slice is offline canonical artifacts, four causal operations, one NDJSON export, bounded output, and a thin CLI adapter.

## Dedupe

- #6053 owns coordination decision explanation/replay; this supplies the general read surface.
- #4262 owns Perfetto encoding for micro traces; this only exposes adapter registration/delegation.
- #6801 owns sink/evaluator conformance; it consumes canonical slices.
- #5776 owns token-class-specific trace/query fields.
- #6528 owns derived-context explanation.
- #10621 inventories identifier/timing producers.

Searches for `trace query`, `trace explain`, and `trace export` across open/closed public issues on 2026-09-08 found no generic bounded causal surface matching this scope.

## Prior-art ledger

- **Source:** Perfetto `docs/analysis/trace-processor.md:3-7,106-123,129-152,170-203,263-321@2566e6091d0b6cd2d0d7aadd6ad54f8227c737fd`.
- **Source event/state:** commit `2566e6091d0b6cd2d0d7aadd6ad54f8227c737fd`, 2026-09-08T15:17:16Z; shipped docs on `main`.
- **Observed at:** 2026-09-08T16:10:19Z. **License:** Apache-2.0.
- **Disposition:** **ADAPT / DEFAULT** — reuse query/summarize/export separation and bounded command output; keep fak's causal/visibility semantics and stdlib-only code. No code copied.
- **Refresh trigger:** Perfetto trace-processor interface change or a second canonical fak trace schema.

## Done condition

An agent can query, explain, and canonically export one bounded causal trace slice without producer-specific parsing or privacy/evidence loss.

## Definition of done

- [ ] Public `pkg/traceabi` request/result types and four typed causal operations return stable ordered results without importing `internal/*`.
- [ ] Human and JSON explanation derive from one model and agree.
- [ ] Limits apply to depth, returned rows, bytes, and scanned rows.
- [ ] Truncation/drops and missing lineage are explicit, never silent success.
- [ ] Visibility/privacy/evidence fields cannot be removed by projection.
- [ ] Canonical NDJSON export re-imports to the same semantic digest.
- [ ] Unknown exporters refuse and list available adapters.
- [ ] Existing scoped SQL query behavior remains compatible.
- [ ] An external-package test proves public/private consumers can use the same query and explanation model.

## Verifiable Witness

```bash
go test ./pkg/traceabi ./cmd/fak -run 'TestCausal|TestTraceExplain|TestTraceExportRoundTrip|TestExternalQueryConsumer' -count=1
```

## Witness

Run the deterministic `go test ./pkg/traceabi ./cmd/fak -run 'TestCausal|TestTraceExplain|TestTraceExportRoundTrip|TestExternalQueryConsumer' -count=1` gate above.

## Acceptance gate

One fork/fan-in/value-contraction fixture produces identical bounded explanations by trace, span, and logical output lookup, and its canonical export re-imports to the same digest with all privacy/evidence/loss metadata intact.

## Likely files

- `pkg/traceabi/query.go`
- `pkg/traceabi/query_test.go`
- `pkg/traceabi/query_external_test.go`
- `cmd/fak/trace.go`

## Lane

traceabi

Schedule after the three prerequisite `pkg/traceabi` tickets and coordinate the thin `cmd/fak/**` shim under its declared path before dispatch.

## Expected steps

8

## Closure binding

The resolving commit cites this issue and uses `(fak traceabi)`. Report `pkg/traceabi@rev`, canonical export digest, external-consumer result, and witness output.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12261
