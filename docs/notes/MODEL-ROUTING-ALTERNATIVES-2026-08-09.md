# Model routing native-vs-alternative comparison witness

Date: 2026-08-09  
Capability: `model_routing`  
Status: **INCOMPLETE**

## Arms

- `fak native`: per-aspect ordered manifest routing in `internal/modelroute`.
- tuned no-feature baseline: send every item to the fixed strongest candidate model.
- strongest practical external alternative: RouteLLM, with a frozen checkpoint.
- equivalent first-class integrations, each measured separately: `fak + LiteLLM Router`, `fak + OpenRouter routing`, and `fak + Portkey router`.

The separate integration arms are intentional. Compatibility or adapter registration is not a benchmark result, and the integrations must not be folded into one synthetic “external router” row.

## Same-workload local runner

`modelroute.CompareLocal` emits `fak-model-routing-comparison/1`. It sends a frozen six-case decision corpus through the native manifest and fixed-model baseline, recording expected-selection accuracy and decision overhead. RouteLLM and all three integration arms remain present with `available=false` until real processes and model responses are supplied.

The corpus covers aspect, tool wildcard, prompt length, complexity, latency, and default matching. Native fixture selection is expected to be 6/6; the fixed strongest-model baseline is expected to select the fixture's designated model for only the two cases that need it. These are policy-selection checks, **not** end-task quality scores.

Observed local decision overhead on Windows amd64, AMD Ryzen 9 9950X, five runs at 100,000 six-case corpus iterations:

| Arm | Median ns / six decisions | Approx. ns / decision | B/op | allocs/op |
|---|---:|---:|---:|---:|
| fak-native manifest | 148.1 | 24.7 | 0 | 0 |

This measurement excludes inference and therefore cannot establish an end-to-end latency or cost advantage.

Reproduce:

```text
go test ./internal/modelroute -run 'TestCompareLocal|TestComparisonCorpus' -count=1
go test ./internal/modelroute -run '^$' -bench '^BenchmarkModelRoutingComparison$' -benchmem -count=5
```

## Missing live witness

Completion requires one frozen prompt/task corpus and identical candidate models through all six arms. An independent grader must capture task success and route quality; each process must report p50/p95 latency, input/output tokens, peak RSS or equivalent resource consumption, and total cost. Provider state, concurrency, retries, warmup, and checkpoint versions must be fixed and recorded.

## Honest verdict

No net-true routing winner is established. The local runner proves only deterministic selection behavior and local decision overhead. It does not substitute manifest accuracy for response quality, treat a mock server as an external router, or infer performance from first-class integration support.
