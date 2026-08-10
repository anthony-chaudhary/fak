# Bounded cache-snapshot alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's bounded fsynced JSONL snapshot/replay and an unbounded append-only JSONL baseline. Prometheus, OpenTelemetry, SQLite, Prometheus TSDB, and ClickHouse retain zero measurements until their real stores execute the common retention workload; [#6165](https://github.com/anthony-chaudhary/fak/issues/6165) tracks those witnesses.

## Same-workload contract

Every arm receives five ordered provider-cache turns representing 1,500 input tokens under a three-row retention requirement. The native oracle requires the latest three rows in order, two dropped rows, successful replay, and non-empty durable storage. The append-only baseline preserves all five rows and therefore fails the bounded-retention oracle.

The native implementation truncates, writes, flushes, and `fsync`s the target. It is **not atomic replacement**; this packet does not claim otherwise. Complete store runs must additionally measure malformed/partial tails and crash-loss behavior, write/read latency and throughput, CPU/RSS/network/storage, represented tokens, and total cost.

| Arm | Class | Local availability |
|---|---|---:|
| fak native bounded fsynced JSONL snapshot | native | yes |
| unbounded append-only JSONL | tuned baseline | yes |
| fak + Prometheus | first-class integration | no |
| fak + OpenTelemetry | first-class integration | no |
| SQLite WAL | external | no |
| Prometheus TSDB | external | no |
| ClickHouse | external | no |

Adapters and in-memory substitutes are not storage-system witnesses.

## Local native witness

```text
go test ./internal/vcachesnapshot -bench BenchmarkWriteReadBoundedSnapshot -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 1,997,710; 1,962,634; 1,961,638; 1,966,473; 1,975,365 ns/op. Median: **1,966,473 ns/write+read cycle**, **74,207 B/op**, **39 allocs/op**. This includes local `fsync`, not remote-store ingestion.

## Reproduce

```text
go test ./internal/vcachesnapshot -run TestCompareLocalKeepsSnapshotAlternativesExplicit
go test ./internal/vcachesnapshot -bench BenchmarkWriteReadBoundedSnapshot -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
