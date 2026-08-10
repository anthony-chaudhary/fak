# Retry-attempt budgeting alternatives — 2026-08-10

Issue: [#6127](https://github.com/anthony-chaudhary/fak/issues/6127)

## Contract

Every arm receives the same upstream failure trace, retryable-status classes, attempt and backoff policy, process lifetime, and concurrency. An independent oracle checks each retry/stop decision, successful recovery, and request amplification. Report latency, upstream requests, CPU, peak RSS, network bytes, and total cost.

Required arms:

1. fak native failure-class-aware attempt budget;
2. unlimited-retry tuned baseline;
3. Envoy retry budget;
4. gRPC retry policy;
5. AWS SDK adaptive retry.

No equivalent first-class fak integration is declared today. If one ships, add a separate `fak + integration` arm.

## Local witness

`internal/attemptbudget/compare.go` classifies three repeated structural failures at a three-attempt ceiling. fak emits a held structural-block verdict after three attempts; the unlimited baseline schedules a fourth attempt and is marked incorrect. Every external runtime remains unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkDecideExhausted-32  604.1..819.3 ns/op  0 B/op  0 allocs/op
median: 744.3 ns/decision
```

This is the pure in-process classification fold. It is not an upstream request, recovery, concurrency, network, or billing witness.

## Honest status

The contract and local fixture are present, but the comparison is incomplete. Issue #6127 remains open until all five same-trace arms have independent correctness, latency, resource, and total-cost witnesses.
