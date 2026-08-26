# Lane/tree collision admission alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's in-process admission and a geometry-only baseline. DOS arbitration, GitHub Actions, Kubernetes Lease coordination, and etcd have zero measurements until their real stores/runtimes enforce the same lock state; [#6144](https://github.com/anthony-chaudhary/fak/issues/6144) tracks those witnesses. Mis-scoped capacity issue #6143 was closed and replaced.

## Same-workload contract

Every arm receives two live leases and five requests: same-lane/disjoint-tree, cross-lane/overlapping-tree, exclusive lane, declared read-only, and self-lease renewal. The same taxonomy supplies canonical trees and exclusivity. Full witnesses report false allows/denies, admission quality, decision/acquisition latency and throughput, CPU/RSS/network/storage, contention, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native lane and tree admission | native | yes | correct five-verdict sequence |
| geometry-only tree overlap | tuned baseline | yes | incomplete semantics: misses lane and exclusivity policy |
| DOS arbitrate | first-class integration | no | separate real lease-store run required |
| GitHub Actions concurrency groups | external | no | zero measurements |
| Kubernetes Lease coordination | external | no | zero measurements |
| etcd concurrency mutex | external | no | zero measurements |

Mocks, parsers, and adapters are not distributed-lock witnesses.

## Local native witness

```text
go test ./internal/laneadmit -bench BenchmarkDecideLaneTreeAdmission -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 5626, 6056, 6488, 7209, 9005 ns/op. Median: **6488 ns/five decisions** (about **1298 ns/decision**), **4551 B/op**, **196 allocs/op**. This is in-process policy cost, not distributed acquisition latency.

## Reproduce

```text
go test ./internal/laneadmit -run TestCompareLocalKeepsLockAlternativesExplicit
go test ./internal/laneadmit -bench BenchmarkDecideLaneTreeAdmission -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
