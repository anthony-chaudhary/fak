# Deadline-aware admission alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** This packet establishes only the local fak-native arm and an executable no-feature FIFO baseline. It does not claim results for Mooncake, standalone vLLM, standalone SGLang, or either first-class fak integration. Issue [#6135](https://github.com/anthony-chaudhary/fak/issues/6135) remains open for those independent runtime witnesses.

## Same-workload contract

Every arm receives the same four-request queue: two tied deadlines, one degradable request predicted to miss by the fixed threshold, and one non-degradable predicted miss. The oracle requires deterministic deadline/ID order, shedding only the degradable miss, and retaining the required request. Full witnesses must also report admission precision/recall, deadline-miss rate, queue latency and throughput, CPU/RSS/accelerator use, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native EDF admission | native | yes | correct: 3 admitted, 1 shed |
| FIFO without predicted-miss shedding | tuned no-feature baseline | yes | incorrect: admits all 4 and preserves arrival order |
| Mooncake deadline-aware admission | external | no | zero measurements; real scheduler required |
| vLLM priority scheduling | external | no | zero measurements; real server required |
| SGLang priority scheduling | external | no | zero measurements; real server required |
| fak + vLLM priority scheduling | first-class integration | no | separate real integration run required |
| fak + SGLang priority scheduling | first-class integration | no | separate real integration run required |

Adapters, mocks, package discovery, and local availability do not establish an external or integration result.

## Local native witness

Command:

```text
go test ./internal/deadlineadmit -bench BenchmarkAdmitPredictedMiss -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X, five runs:

```text
158.9, 161.7, 206.8, 203.9, 204.8 ns/op
416 B/op
8 allocs/op
```

Median: **203.9 ns/admission plan**, **416 B/op**, **8 allocs/op**. This microbenchmark measures only in-process Go policy execution; it is not end-to-end scheduler latency, throughput, GPU utilization, or cost.

## Reproduce

```text
go test ./internal/deadlineadmit -run TestCompareLocalKeepsSchedulerAndIntegrationArmsExplicit
go test ./internal/deadlineadmit -bench BenchmarkAdmitPredictedMiss -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
