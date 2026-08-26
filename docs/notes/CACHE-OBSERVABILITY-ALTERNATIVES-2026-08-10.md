# Cache-observability alternatives — 2026-08-10

Issue: [#6122](https://github.com/anthony-chaudhary/fak/issues/6122)

## Contract

Every arm receives the same cache-observation trace, label/cardinality policy, process lifetime, and aggregation interval. An independent oracle checks counters, ratios, dropped events, and cardinality. Report ingestion and query latency, CPU, peak RSS, network and storage bytes, and total cost.

Required arms:

1. fak native cache observer;
2. no-telemetry tuned baseline;
3. Prometheus client plus scrape;
4. OpenTelemetry metrics SDK/exporter plus collector;
5. Datadog DogStatsD plus agent intake.

No equivalent first-class fak integration is declared today. If one ships, add a separate `fak + integration` arm.

## Local witness

`internal/cacheobs/compare.go` records 1,000 observations of a 128-token prompt with 96 reused prefix tokens. fak reports 1,000 turns, 128,000 prompt tokens, and 96,000 reused tokens; the zero-work baseline is marked incorrect because it records no evidence. Every external pipeline remains unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkObserveCacheHit-32  20.91..23.80 ns/op  0 B/op  0 allocs/op
median: 22.21 ns/observation
```

This is the in-process mutex-protected counter path. It is not an exporter, collector, scrape, query, storage, or billing result.

## Honest status

The contract and local fixture are present, but the comparison is incomplete. Issue #6122 remains open until all five same-trace arms have independent correctness, latency, resource, and total-cost witnesses.
