# KV-memory budget-model alternatives — 2026-08-10

Issue: [#6131](https://github.com/anthony-chaudhary/fak/issues/6131)

## Contract

Every arm receives the same model shape, precision, context lengths, batch/concurrency, and serving lifecycle. An independent GPU allocation oracle reconciles KV bytes per token, peak allocation, and fit/concurrency correctness. Report latency, throughput, GPU and host memory, and total cost.

Required arms:

1. fak native MHA/MLA/DSA KV budget model;
2. full-MHA closed-form tuned baseline;
3. vLLM memory profiler;
4. SGLang memory-pool telemetry;
5. NVIDIA GenAI-Perf.

No equivalent first-class fak integration is declared for this budgeting capability. If one ships, add a separate `fak + integration` arm.

## Local witness

`internal/kvbudget/compare.go` computes FP16 KV bytes per token for the declared `GLM52DSA` shape. fak reports 129,536 bytes/token; the full-MHA baseline intentionally models different conventional architecture semantics and is marked incorrect for this fixture. Every runtime/GPU arm remains unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkKVBytesPerTokenModel-32  3.910..4.196 ns/op  0 B/op  0 allocs/op
median: 4.133 ns/model evaluation
```

This is closed-form CPU arithmetic. It is not a GPU allocation, serving fit, throughput, resource, or billing witness.

## Honest status

The contract and local fixture are present, but the comparison is incomplete. Issue #6131 remains open until all five same-shape arms have independent correctness, latency, resource, and total-cost witnesses. The required GPU work must run on a sanctioned compute node per `docs/fleet-compute-nodes.md`.
