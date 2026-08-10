# Tool-call rate-limiting alternatives — 2026-08-09

Issue: [#6116](https://github.com/anthony-chaudhary/fak/issues/6116)

## Contract

Every arm receives the same request-arrival trace, call and cost caps, key dimension, window semantics, warmup, and concurrency. The independent oracle checks each admission decision and overshoot. Report decision equivalence, overshoot, latency and throughput, state and network bytes, peak RSS, and total cost.

Required arms:

1. fak native per-trace/per-tool/global limiter;
2. no-limiter tuned baseline;
3. Envoy local rate limit;
4. Kong rate limiting;
5. Redis-cell.

No equivalent first-class fak integration is declared today. If one ships, add it as a separate `fak + integration` arm rather than replacing its standalone external arm.

## Local witness

`internal/ratelimit/compare.go` executes 1,000 sequential calls for one trace with a 750-call cap. fak admits 750 and denies 250 with `RATE_LIMITED`; the zero-work baseline admits all 1,000 and is correctly marked incorrect. Envoy, Kong, and Redis-cell remain unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkNativeRateLimitCall-32  31.43..42.22 ns/op  24 B/op  1 alloc/op
median: 35.95 ns/call
```

This is the in-process admitted-call path with one mutex-protected key. It is not a distributed, concurrent, or wall-clock-window result and has no external cost witness.

## Honest status

The contract and local fixture are present, but the comparison is incomplete. Issue #6116 remains open until all five arms run with equivalent semantics and independent correctness, latency, resource, and cost witnesses.
