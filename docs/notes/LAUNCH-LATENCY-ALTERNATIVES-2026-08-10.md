# Worker launch-latency alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's summary and a raw-event/no-summary baseline only. Prometheus, OpenTelemetry, and Datadog have zero measurements until real ingestion and queries run; [#6137](https://github.com/anthony-chaudhary/fak/issues/6137) tracks those witnesses.

## Same-workload contract

Every arm consumes the same six dispatch/heartbeat observations spanning fixed `[1,2,5,10,30]` second edges, including one heartbeat preceding dispatch. Every backend must use the same lower-closed bucket convention, nearest-rank p50/p95 convention, and independent oracle. Full runs report bucket/quantile quality, dropped observations, ingestion/query latency, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native launch-latency summary | native | yes | correct six-sample histogram, p50=2s, p95=30s, one negative |
| raw launch events without summary | tuned no-feature baseline | yes | incorrect: no buckets or percentiles |
| Prometheus histogram | external | no | zero measurements |
| OpenTelemetry metrics | external | no | zero measurements |
| Datadog distribution metric | external | no | zero measurements |

No equivalent first-class fak backend integration was identified, so none is fabricated. Adapters and mocks are not backend witnesses.

## Local native witness

```text
go test ./internal/launchlatency -bench BenchmarkLaunchLatencySummary -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 159.0, 159.8, 203.8, 209.2, 198.7 ns/op. Median: **198.7 ns/summary**, **528 B/op**, **6 allocs/op**. This measures in-process folding only, not telemetry ingestion/query latency or cost.

## Reproduce

```text
go test ./internal/launchlatency -run TestCompareLocalKeepsTelemetryAlternativesExplicit
go test ./internal/launchlatency -bench BenchmarkLaunchLatencySummary -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
