# Cache reuse trend-gate alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's trailing-window cache-reuse regression fold and a raw-JSONL/no-gate baseline. Prometheus, OpenTelemetry, Prometheus rules, and Datadog retain zero measurements until real ingestion, query, storage, and alert paths process the common ledger; [#6157](https://github.com/anthony-chaudhary/fak/issues/6157) tracks those witnesses.

## Same-workload contract

Every arm receives five chronological rows: two stable four-turn sessions at 0.80 reuse, one ignored single-turn cold row, and two recent four-turn sessions totaling 0.35 reuse. Both windows meet the eight-turn minimum; the -0.45 change exceeds the 0.05 regression tolerance. The oracle requires exactly one `REGRESSED` alert, with the single-turn row excluded from both windows. The rows represent 1,700 prompt tokens.

Complete monitoring runs report true/false/missed regressions, query and scrape-to-alert latency, rows/second, represented tokens, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native trailing-window trend gate | native | yes | exact one-regression oracle |
| raw JSONL ledger without trend gate | tuned baseline | yes | preserves rows but emits no verdict |
| fak + Prometheus | first-class integration | no | real scrape/query/alert path required |
| fak + OpenTelemetry | first-class integration | no | real export/collector path required |
| Prometheus recording and alerting rules | external | no | zero measurements |
| Datadog change and anomaly monitor | external | no | zero measurements |

Adapters and local query-shaped functions are not monitoring-system witnesses.

## Local native witness

```text
go test ./internal/cachevalueledger -bench BenchmarkFoldTrendGate -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 723.3, 693.3, 877.4, 851.8, 909.8 ns/op. Median: **851.8 ns/five-row fold**, **2,644 B/op**, **8 allocs/op**. This is in-process fold cost, not ledger-ingest or scrape-to-alert latency.

## Reproduce

```text
go test ./internal/cachevalueledger -run TestCompareLocalKeepsTrendAlternativesExplicit
go test ./internal/cachevalueledger -bench BenchmarkFoldTrendGate -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
