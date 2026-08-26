# Tool-result cache native-vs-alternative comparison witness

Date: 2026-08-09  
Capability: `tool_result_caching`  
Status: **INCOMPLETE**

## Arms

- `fak native`: the three-tier tool vDSO in `internal/vdso`.
- tuned no-feature baseline: an optimized uncached upstream call.
- strongest practical external alternatives: Redis client-side/server-assisted caching and Momento Cache.

No Redis or Momento first-class fak integration is declared in the current integration catalog, so these are standalone external arms. If one becomes first-class, add a separate `fak + integration` arm.

## Same-workload local runner

`vdso.CompareLocal` emits `fak-tool-result-cache-comparison/1`. It runs identical deterministic tool-call/result bytes through the native static tier and an in-process uncached baseline. Both produce exact output equivalence; native records zero upstream calls while the baseline records one per request. Redis and Momento remain `available=false` until real services execute the same workload.

## Local hit overhead

Host: Windows amd64, AMD Ryzen 9 9950X. Command:

```text
go test ./internal/vdso -run '^$' -bench '^BenchmarkToolResultCacheComparison$' -benchmem -benchtime=500000x -count=5
```

Median across five runs: **303.5 ns/hit, 600 B/op, 6 allocs/op**. This is a static-tier local hit and does not include network, serialization, fill, eviction, invalidation, or upstream latency.

## Missing live witness

Completion requires a real deterministic upstream and identical request/invalidation traces for all four arms, under fixed cache budgets, warmup, concurrency, and process lifetime. Capture byte-equivalent outputs, hit rate, p50/p95 latency, upstream calls avoided, peak RSS or service resource consumption, and total cost. Include stale-read checks after every mutation.

## Honest verdict

No net-true cache winner is established. The native local hit is observed and avoids the synthetic upstream call, but Redis and Momento have not run and no networked end-to-end value has been measured.
