# Cache reuse-divergence detection alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's cross-provenance reuse-divergence fold and a raw-counter/no-detector baseline. Prometheus, OpenTelemetry, Prometheus rules, and Datadog retain zero measurements until real ingestion, storage, queries, and alert paths process the common trace; [#6155](https://github.com/anthony-chaudhary/fak/issues/6155) tracks those witnesses.

## Same-workload contract

Every arm receives three cache records at the same 0.15 absolute-ratio tolerance: one stable witnessed/observed pair (0.80 versus 0.75), one divergent pair (0.80 versus 0.30), and one witnessed-only record. The oracle requires two comparable records, one single-class record, and exactly one alert naming the divergent source. Complete runs report true/false/missed alerts, ingest and alert latency, records/second, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native reuse-divergence fold | native | yes | exact one-alert oracle |
| raw reuse counters without divergence detection | tuned baseline | yes | stores the axes but emits no alert |
| fak + Prometheus | first-class integration | no | real scrape/rule/alert path required |
| fak + OpenTelemetry | first-class integration | no | real export/collector path required |
| Prometheus recording and alerting rules | external | no | zero measurements |
| Datadog anomaly monitor | external | no | zero measurements |

Adapters and local query-shaped functions are not monitoring-system witnesses.

## Local native witness

```text
go test ./internal/cachewitness -bench BenchmarkFoldReuseDivergence -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 556.0, 542.2, 615.7, 589.4, 593.4 ns/op. Median: **589.4 ns/three-record fold**, **496 B/op**, **9 allocs/op**. This is in-process folding cost, not scrape-to-alert latency.

## Reproduce

```text
go test ./internal/cachewitness -run TestCompareLocalKeepsDivergenceAlternativesExplicit
go test ./internal/cachewitness -bench BenchmarkFoldReuseDivergence -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
