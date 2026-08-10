# Cache-cost accounting alternatives — 2026-08-09

Issue: [#6119](https://github.com/anthony-chaudhary/fak/issues/6119)

## Contract

Every arm receives the same provider-observed prompt and resident-prefix trace, service SKU, rates, and billing period. An independent bill reconciliation checks admission-token equivalence and billed-unit error. Report those quality measures plus latency, bytes processed, peak RSS, and total cost.

Required arms:

1. fak native resident-prefix admission-token accounting;
2. charge-the-full-prompt tuned baseline;
3. AWS Pricing Calculator;
4. Google Cloud Pricing Calculator;
5. Azure Pricing Calculator.

No equivalent first-class fak integration is declared today. If one ships, add a separate `fak + integration` arm.

## Local witness

`internal/cacheprice/compare.go` prices a 4,096-token prompt with 3,072 provider-observed resident prefix tokens. fak reports 1,024 admission tokens; the full-prompt baseline reports 4,096 and is marked incorrect for this fixture. Every cloud calculator remains unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkAdmissionTokens-32  0.1839..0.1998 ns/op  0 B/op  0 allocs/op
median: 0.1920 ns/call
```

This sub-nanosecond result is compiler-optimized integer arithmetic, not end-to-end billing latency. It establishes no cloud-price or total-cost claim.

## Honest status

The contract and arithmetic fixture are present, but the comparison is incomplete. Issue #6119 remains open until all five same-trace arms have independent bill, latency, resource, and total-cost witnesses.
