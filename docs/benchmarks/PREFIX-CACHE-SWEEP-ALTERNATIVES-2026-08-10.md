# Prefix-cache budget sweep alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** This packet executes fak's prefix-cache sweep and a no-cache baseline. libCacheSim, Caffeine, and Redis/Valkey have zero measurements until their real simulators/servers consume the same trace; [#6146](https://github.com/anthony-chaudhary/fak/issues/6146) tracks those witnesses.

## Same-workload contract

Every arm receives five timestamped accesses containing exact repeated prefixes and divergent children, the same four token budgets, an unbounded ceiling pass, and the same write-delay and token-cost assumptions. Full witnesses report hit correctness, reuse ratio/tokens, evictions, ROI-knee error, simulation/runtime latency and throughput, CPU/RSS/storage/network, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native prefix-cache sweep | native | yes | valid four-point curve, positive ceiling reuse, finite knee |
| no prefix cache | tuned baseline | yes | zero saved tokens; no ROI knee |
| libCacheSim | external | no | zero measurements |
| Caffeine simulator | external | no | zero measurements |
| Redis or Valkey maxmemory policies | external | no | zero measurements |

Repository inspection found no equivalent first-class fak integration distinct from the native radix-backed sweep. Adapters and in-memory mocks are not external cache witnesses.

## Local native witness

```text
go test ./internal/cachesweep -bench BenchmarkSweepPrefixCacheBudgets -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 22,746; 24,891; 28,420; 30,736; 29,412 ns/op. Median: **28,420 ns/five-pass sweep**, **82,136 B/op**, **120 allocs/op**. This measures local simulation only, not external cache latency or infrastructure cost.

## Reproduce

```text
go test ./internal/cachesweep -run TestCompareLocalKeepsCacheSimulatorAlternativesExplicit
go test ./internal/cachesweep -bench BenchmarkSweepPrefixCacheBudgets -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
