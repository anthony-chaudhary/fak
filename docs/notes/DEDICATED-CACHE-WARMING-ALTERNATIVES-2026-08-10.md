# Dedicated cache-warming alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's dedicated-warm planner/readback accounting and a demand-only baseline. Anthropic, Gemini, OpenAI, LMCache, Mooncake, vLLM, and SGLang retain zero measurements until their real APIs and cache stores run the common fanout; [#6162](https://github.com/anthony-chaudhary/fak/issues/6162) tracks those witnesses.

## Same-workload contract

Every arm receives five cases over the same 1,024-token/4,096-byte fingerprint: profitable Anthropic explicit warm, fingerprint mismatch, below-break-even fanout, OpenAI automatic-cache ordering, and unsupported active capability. Fak also reconciles one closed two-call fanout with 2,048 observed cache-read tokens. The oracle requires one dedicated confirmed warm, correct refusal reasons, and no fabricated provider result.

Complete provider/engine runs replay the same continuation fanout and report decision quality, useful and wasted warms, TTFT and planning latency, cache-write/read/input tokens, bytes, CPU/RSS/network/storage, throughput, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native dedicated warm planner and accounting | native | yes | exact five-case oracle; one confirmed warm |
| demand-only fills without dedicated warming | tuned baseline | yes | no warm spend and no pre-fanout cache entry |
| fak + Anthropic prompt caching | first-class integration | no | real API usage/readback required |
| fak + Gemini CachedContent | first-class integration | no | real explicit cache lifecycle required |
| fak + OpenAI automatic prefix caching | first-class integration | no | real API usage/readback required |
| fak + LMCache | first-class integration | no | real engine required |
| fak + Mooncake | first-class integration | no | real engine required |
| vLLM automatic prefix caching | external | no | zero measurements |
| SGLang HiCache | external | no | zero measurements |

Adapters, fake provider responses, and local cache-shaped stores are not provider/engine witnesses.

## Local native witness

```text
go test ./internal/vcachewarm -bench BenchmarkPlanAndReconcileWarm -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 96.47, 95.07, 100.4, 106.6, 107.9 ns/op. Median: **100.4 ns/five-case planning plus reconciliation**, **0 B/op**, **0 allocs/op**. This is pure planner overhead, not provider TTFT or cache-fill latency.

## Reproduce

```text
go test ./internal/vcachewarm -run TestCompareLocalKeepsWarmingAlternativesExplicit
go test ./internal/vcachewarm -bench BenchmarkPlanAndReconcileWarm -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
