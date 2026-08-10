# Prefix KV reuse comparison contract

Status: **INCOMPLETE**. Issue: #6036.

The `internal/radixkv` native prefix index must be judged on the same model and shared-prefix request trace as the strongest practical alternatives—not only by an in-memory lookup microbenchmark.

## Required arms

1. **No reuse:** the tuned serving stack with provider/vLLM prefix caching disabled.
2. **fak native:** the same stack with fak `radixkv` prefix KV reuse enabled.
3. **SGLang RadixAttention:** the next-best external implementation, using equivalent cache budget, concurrency, warmup, and request order.
4. **fak + llm-d:** the repository's first-class `llm-d` integration with prefix-cache-aware routing enabled.

`docs/benchmarks/RADIXATTENTION-RESULTS.md` is prior local evidence about fak's radix data structure, but it does not satisfy this contract: it lacks a same-model SGLang arm and the equivalent `fak + llm-d` arm.

## Shared witness schema

Every arm must report:

- output equivalence/correctness;
- prefix hit rate;
- TTFT distribution (at least p50/p95);
- decoded throughput in tokens/second;
- KV memory bytes and peak accelerator memory; and
- total cost for the frozen trace.

The witness must pin model/weights, serving versions, accelerator, cache budget, request trace digest, concurrency, warmup, and cache state. Any arm that changes those conditions is not comparable. Until all four arms are present, `fak native-benchmarks --check` remains non-zero and no net-true prefix-reuse winner is claimable.
