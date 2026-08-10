# Provider-cache economics alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's provider-cache economics/value fold and raw provider usage. Prometheus, OpenTelemetry, Anthropic, OpenAI, Datadog, and LangSmith retain zero measurements until their real telemetry paths process the common trace; [#6164](https://github.com/anthony-chaudhary/fak/issues/6164) tracks those witnesses.

## Same-workload contract

Every arm receives four chronological turns across two prefix families: cold input, a 900-token cache write, a 900-token cache read, and one context-shed row. Raw totals are 1,700 input, 900 write, and 900 read tokens. The independent oracle requires exact counters, two families, four turns, and a non-zero net token-equivalent result under the same multipliers.

Complete systems report counter/value accuracy, hit rate, fold/query latency, input/write/read/context tokens, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability |
|---|---|---:|
| fak native provider-cache economics fold | native | yes |
| raw provider usage without economics fold | tuned baseline | yes |
| fak + Prometheus | first-class integration | no |
| fak + OpenTelemetry | first-class integration | no |
| Anthropic usage and cost reporting | external | no |
| OpenAI usage and cost reporting | external | no |
| Datadog LLM Observability | external | no |
| LangSmith | external | no |

Adapters and synthetic dashboards are not telemetry-system witnesses.

## Local native witness

```text
go test ./internal/vcacheobserve -bench BenchmarkObserveEconomics -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 355988, 308389, 361503, 377797, 385603 ns/op. Median: **361,503 ns/four-turn fold**, **217,966 B/op**, **3,612 allocs/op**. This includes all native report panels, not remote query latency.

## Reproduce

```text
go test ./internal/vcacheobserve -run TestCompareLocalKeepsEconomicsAlternativesExplicit
go test ./internal/vcacheobserve -bench BenchmarkObserveEconomics -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
