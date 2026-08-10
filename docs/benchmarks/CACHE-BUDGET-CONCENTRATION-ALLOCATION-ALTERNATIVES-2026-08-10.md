# Cache-budget concentration allocation alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's concentration-weighted allocator, equal-share allocation, and request-volume proportional allocation. LMCache, Mooncake, vLLM, and SGLang retain zero measurements until real shared cache pools execute the common workload; [#6159](https://github.com/anthony-chaudhary/fak/issues/6159) tracks those witnesses.

## Same-workload contract

Every arm allocates 1,200 cache bytes at top-K one across three tenants: a concentrated 9:1 workload, a flat 1:1 workload, and an unmeasured workload. The oracle requires exact budget conservation, concentration preference over the flat tenant, and a protected 400-byte flat share for the unmeasured tenant. Equal-share ignores concentration; volume proportional allocation can starve the unmeasured tenant.

Complete engine runs replay an equivalent request trace and report captured cache value, budget conservation, starved tenants, allocation and request latency, throughput, cache bytes, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native concentration-weighted allocation | native | yes | conserves 1,200 bytes and protects unmeasured share |
| equal-share cache allocation | tuned baseline | yes | conservative but ignores concentration |
| request-volume proportional allocation | tuned baseline | yes | demand-aware but can starve unmeasured tenant |
| fak + LMCache | first-class integration | no | real shared cache runtime required |
| fak + Mooncake | first-class integration | no | real shared cache runtime required |
| vLLM cache-aware routing | external | no | zero measurements |
| SGLang HiCache and cache-aware scheduling | external | no | zero measurements |

Adapters and allocator-shaped mocks are not cache-engine witnesses.

## Local native witness

```text
go test ./internal/vcachecal -bench BenchmarkAllocateByConcentration -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 439.0, 472.8, 471.5, 534.3, 567.7 ns/op. Median: **472.8 ns/three-bucket allocation**, **712 B/op**, **10 allocs/op**. This is allocator overhead, not cache-engine request latency.

## Reproduce

```text
go test ./internal/vcachecal -run TestCompareLocalKeepsAllocationAlternativesExplicit
go test ./internal/vcachecal -bench BenchmarkAllocateByConcentration -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
