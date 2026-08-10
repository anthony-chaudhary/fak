# Engine-cache invalidation alternatives — 2026-08-09

Issue: [#6114](https://github.com/anthony-chaudhary/fak/issues/6114)

## Contract

The comparison workload is one quarantined model-KV span and its dependent attention-index entry. Every arm must start from equivalent reusable engine state and prove that neither poisoned object can be reused after invalidation. Report correctness, invalidated objects, latency, control requests and bytes, peak resource use, and total cost.

Required separate arms:

1. fak native governance and engine-cache invalidation adapter;
2. no-invalidation tuned baseline;
3. standalone vLLM;
4. standalone SGLang;
5. standalone LMCache;
6. `fak + vLLM` through the first-class `--engine-cache-engine=vllm` path;
7. `fak + SGLang` through the first-class `--engine-cache-engine=sglang` path.

LMCache is an external arm, not an integration arm: no first-class LMCache fak integration is declared today. If one ships, add `fak + LMCache` separately rather than relabeling the standalone result.

## Local witness

`internal/enginecache/compare.go` executes the native governance/adapter path against a loopback HTTP endpoint and keeps every real engine arm unavailable. It proves request generation and result accounting only; it does **not** prove any serving engine actually evicted KV state. The no-invalidation baseline is correctly marked incorrect because it preserves both poisoned objects.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkCompareNativeInvalidation-32  33739..60728 ns/op  8687..8828 B/op  82 allocs/op
median: 53183 ns/invalidation
```

This loopback number includes the local HTTP round trip. It is not a latency claim against vLLM, SGLang, or LMCache and has no total-cost witness.

## Honest status

The comparison contract is present, but the benchmark is incomplete. No real vLLM, SGLang, or LMCache state was inspected, and neither first-class integrated engine arm has run. Issue #6114 remains open until all seven same-workload arms have independent correctness, latency, resource, and cost witnesses.
