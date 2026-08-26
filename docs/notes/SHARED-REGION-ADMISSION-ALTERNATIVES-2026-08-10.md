# Shared region admission alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's shared execution-surface admission and a tuned geometry-only incumbent. DOS arbitration, Git-ref lease acquisition, Kubernetes, etcd, and GitHub Actions retain zero measurements until their real stores and runtimes enforce the same region state; [#6149](https://github.com/anthony-chaudhary/fak/issues/6149) tracks those witnesses.

`internal/regionadmit` is distinct from the lower-level `internal/laneadmit` contract: it resolves or infers regions at the execution-surface seam, permits two strictly narrowed disjoint sub-regions, and consumes lease projections from multiple acquisition systems. Those acquisition systems remain separate integration arms rather than being counted as native evidence.

## Same-workload contract

Every arm receives two live leases, one taxonomy, and six requests: narrowed same-lane/disjoint-region, cross-lane/overlapping-region, exclusive lane, declared read-only, self-renewal, and hierarchical sub-lane. Full witnesses report false allows/denies, admission precision/recall, decision and acquisition latency, throughput, CPU/RSS/network/storage, and total cost.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native shared region admission | native | yes | correct six-verdict sequence |
| geometry-only region overlap | tuned baseline | yes | incomplete: misses exclusivity and hierarchical lane serialization |
| fak + DOS arbitrate | first-class integration | no | real arbiter and lease-store run required |
| fak + Git-ref leases | first-class integration | no | real cross-process acquisition and visibility run required |
| Kubernetes Lease coordination | external | no | zero measurements |
| etcd concurrency mutex | external | no | zero measurements |
| GitHub Actions concurrency groups | external | no | zero measurements |

Adapters, parsers, and in-memory stand-ins are not distributed acquisition witnesses.

## Local native witness

```text
go test ./internal/regionadmit -bench BenchmarkDecideSharedRegionAdmission -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 8068, 8089, 10218, 9977, 12329 ns/op. Median: **9977 ns/six decisions** (about **1663 ns/decision**), **6090 B/op**, **259 allocs/op**. This measures in-process admission only, not lease acquisition or distributed coordination.

## Reproduce

```text
go test ./internal/regionadmit -run TestCompareLocalKeepsRegionAlternativesExplicit
go test ./internal/regionadmit -bench BenchmarkDecideSharedRegionAdmission -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
