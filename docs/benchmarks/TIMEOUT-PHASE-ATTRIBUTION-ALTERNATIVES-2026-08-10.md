# Timeout-phase attribution alternatives — 2026-08-10

Issue: [#6133](https://github.com/anthony-chaudhary/fak/issues/6133)

## Contract

Every arm receives the same timeout-at-stage trace, instrumentation points, process lifecycle, and sampling policy. An independent oracle grades phase-attribution precision and recall. Report dropped traces, ingest/query latency, CPU, peak RSS, network and storage bytes, and total cost.

Required arms:

1. fak native closed-vocabulary timeout classifier;
2. one undifferentiated timeout-bucket tuned baseline;
3. OpenTelemetry spans;
4. Datadog APM;
5. AWS X-Ray.

No equivalent first-class fak integration is declared today. If one ships, add a separate `fak + integration` arm.

## Local witness

`internal/timeoutphase/compare.go` classifies six attempts spanning unknown, before-startup, edit, test, commit, and push. fak emits all six expected phases; the undifferentiated baseline is marked incorrect. Every external tracing pipeline remains unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkClassifyTimeoutPhase-32  9.469..10.44 ns/op  0 B/op  0 allocs/op
median: 9.825 ns/classification
```

This is the pure in-process classification path. It is not an instrumentation, exporter, collector, backend-query, resource, or billing witness.

## Honest status

The contract and local fixture are present, but the comparison is incomplete. Issue #6133 remains open until all five same-trace arms have independent correctness, latency, resource, and total-cost witnesses.
