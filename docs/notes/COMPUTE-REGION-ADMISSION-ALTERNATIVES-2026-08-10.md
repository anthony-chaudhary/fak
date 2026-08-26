# Compute-region admission alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** This packet executes fak's region admission and an unbounded/no-admission baseline. Kubernetes, Slurm, Ray, and AWS Batch have zero measurements until real schedulers enforce the same region inventory; [#6139](https://github.com/anthony-chaudhary/fak/issues/6139) tracks those witnesses.

## Same-workload contract

Every arm receives one exclusive live lease over device region `0-1`, a declared device address space `0-7`, and four claims: overlapping `1-2`, disjoint `2-3`, out-of-taxonomy `8`, and different-class KV tier `0`. The oracle requires collision refusal, disjoint admission, taxonomy refusal, and cross-class admission. Full witnesses report admission quality, constraint violations, scheduling latency/throughput, CPU/RSS/control-plane bytes, accelerator idle time, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native compute-region admission | native | yes | correct four-decision sequence |
| dispatch without region admission | tuned no-feature baseline | yes | incorrect: admits collision and taxonomy violation |
| Kubernetes scheduler | external | no | zero measurements |
| Slurm scheduler | external | no | zero measurements |
| Ray scheduler | external | no | zero measurements |
| AWS Batch | external | no | zero measurements |

Repository inspection found no equivalent first-class fak integration distinct from this native admission seam. Configuration adapters and mocks are not scheduler witnesses.

## Local native witness

```text
go test ./internal/computeadmit -bench BenchmarkDecideComputeRegionAdmission -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 1107, 1120, 1854, 1757, 1696 ns/op. Median: **1696 ns/four decisions** (about **424 ns/decision**), **1105 B/op**, **36 allocs/op**. This is pure policy execution, not end-to-end scheduler latency or accelerator cost.

## Reproduce

```text
go test ./internal/computeadmit -run TestCompareLocalKeepsSchedulerAlternativesExplicit
go test ./internal/computeadmit -bench BenchmarkDecideComputeRegionAdmission -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
