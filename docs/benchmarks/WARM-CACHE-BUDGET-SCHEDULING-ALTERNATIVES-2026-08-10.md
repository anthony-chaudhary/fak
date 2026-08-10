# Warm-cache budget scheduling alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's rate-limit-aware warm-budget planner and ranked scheduler plus a demand-only/no-proactive-warming baseline. LMCache, Mooncake, NIXL, vLLM, and SGLang retain zero measurements until their real cache engines run the common request trace; [#6152](https://github.com/anthony-chaudhary/fak/issues/6152) tracks those witnesses.

## Same-workload contract

Every arm receives the same provider rate-limit snapshot (10 RPM/20k TPM ceilings, 8 RPM/10k TPM real traffic), 1,000-token anchor, one-minute TTL, and six candidates spanning hot/large, dense, frequent/small, cold/large, regulated/hot, and invalid-negative demand. Fak's sustainable set is two. The independent quality oracle expects `hot-large` then `dense`, excludes regulated content, captures 89,000 score units, and consumes 2,000 planned warm tokens.

Complete engine runs replay the same arrival trace and report useful warm hits, captured value, wasted writes, quota violations, request/planning latency and throughput, input/write/read tokens, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native warm-budget scheduler | native | yes | correct two-prefix set; 89,000 value units |
| demand-only fills without proactive warming | tuned baseline | yes | zero warm quota and zero pre-arrival value |
| fak + LMCache | first-class integration | no | real transfer and cache runtime required |
| fak + Mooncake | first-class integration | no | real transfer and cache runtime required |
| fak + NIXL | first-class integration | no | real lease/transfer runtime required |
| vLLM automatic prefix caching | external | no | zero measurements |
| SGLang HiCache and cache-aware scheduling | external | no | zero measurements |

Adapters and planner-shaped mocks are not cache-engine witnesses.

## Local native witness

```text
go test ./internal/vcachegov -bench BenchmarkWarmBudgetSchedule -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 158.3, 139.7, 177.4, 182.6, 189.6 ns/op. Median: **177.4 ns/six-candidate plan**, **488 B/op**, **4 allocs/op**. This is scheduler overhead, not cache transfer, model latency, or end-to-end cost.

## Reproduce

```text
go test ./internal/vcachegov -run TestCompareLocalKeepsWarmSchedulingAlternativesExplicit
go test ./internal/vcachegov -bench BenchmarkWarmBudgetSchedule -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
