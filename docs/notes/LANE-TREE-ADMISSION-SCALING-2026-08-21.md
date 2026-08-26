# Lane/tree admission scaling — 2026-08-21

## Verdict

Pure in-process admission scales with the number of live leases for every write-capable
scenario measured. At 4,096 leases, the median disjoint decision was **4.97 ms/op** with
**1.70 MB/op** and **106,499 allocs/op**. The declared read-only path stayed flat at
**65.09 ns/op**, **48 B/op**, and **3 allocs/op** because it returns before scanning the
lease view.

These are policy-decision costs. They are not distributed lease acquisition, git-ref
synchronization, process launch, or end-to-end agent latency.

## Provenance

- Base commit: `720cf16d35018b6fd29b88f4a8e6f650dc3d4565`.
- Benchmark source: `internal/laneadmit/compare_test.go`, SHA-256
  `ec4fc02956664a743a04e6aac60610e2fc1ae8956d232a3055a541e5ac642bbb`.
- Go: `go1.26.6 linux/amd64`.
- CPU: AMD Ryzen 9 9950X 16-Core Processor.
- Environment: WSL Linux/amd64 on the Windows development host, following the repository's
  test-execution rule.
- Samples: five Go benchmark samples per scenario and lease count; the table reports medians.

The base commit plus the benchmark-source digest identifies the exact measured source even
though this artifact and the benchmark land together in the resolving commit.

## Workload

Each sub-benchmark constructs a deterministic live-lease slice at
`N=1,8,64,512,4096`:

- `disjoint`: every lease has a different lane and non-overlapping tree.
- `first_conflict`: only the first lease overlaps the requested tree.
- `last_conflict`: only the last lease overlaps the requested tree.
- `all_conflict`: every lease uses the requested lane; lease IDs are reverse ordered so the
  conflict-evidence sort cannot inherit already-sorted input.
- `self_lease`: the first lease is the requester's own overlapping lease and is skipped; all
  remaining leases are disjoint.
- `read_only`: every lease would conflict, but `ReadOnly=true` proves the empty-write-footprint
  fast return does not inspect them.

`TestDecideScalingFixtures` checks the expected admit/refuse result and conflict count for all
30 matrix cells before benchmark timing is trusted.

## Median results

The growth column is median time divided by that scenario's `N=1` median. It is useful for
shape within a scenario, not for comparing scenarios whose fixed work differs.

| Scenario | N | median ns/op | B/op | allocs/op | time vs N=1 |
|---|---:|---:|---:|---:|---:|
| disjoint | 1 | 1,479.00 | 464 | 29 | 1.00x |
| disjoint | 8 | 9,116.00 | 3,376 | 211 | 6.16x |
| disjoint | 64 | 65,855.00 | 26,672 | 1,667 | 44.53x |
| disjoint | 512 | 689,137.00 | 213,041 | 13,315 | 465.95x |
| disjoint | 4096 | 4,974,545.00 | 1,703,995 | 106,499 | 3,363.45x |
| first_conflict | 1 | 1,287.00 | 984 | 45 | 1.00x |
| first_conflict | 8 | 6,168.00 | 3,899 | 227 | 4.79x |
| first_conflict | 64 | 44,401.00 | 27,214 | 1,683 | 34.50x |
| first_conflict | 512 | 368,905.00 | 213,742 | 13,331 | 286.64x |
| first_conflict | 4096 | 4,977,597.00 | 1,706,192 | 106,519 | 3,867.60x |
| last_conflict | 1 | 2,473.00 | 984 | 45 | 1.00x |
| last_conflict | 8 | 5,990.00 | 3,899 | 227 | 2.42x |
| last_conflict | 64 | 43,056.00 | 27,214 | 1,683 | 17.41x |
| last_conflict | 512 | 340,795.00 | 213,742 | 13,331 | 137.81x |
| last_conflict | 4096 | 6,143,232.00 | 1,706,221 | 106,519 | 2,484.12x |
| all_conflict | 1 | 1,664.00 | 816 | 33 | 1.00x |
| all_conflict | 8 | 4,211.00 | 4,163 | 157 | 2.53x |
| all_conflict | 64 | 26,557.00 | 30,170 | 1,112 | 15.96x |
| all_conflict | 512 | 427,917.00 | 238,233 | 8,731 | 257.16x |
| all_conflict | 4096 | 2,932,319.00 | 2,567,313 | 69,671 | 1,762.21x |
| self_lease | 1 | 94.95 | 48 | 3 | 1.00x |
| self_lease | 8 | 4,604.00 | 2,960 | 185 | 48.49x |
| self_lease | 64 | 41,016.00 | 26,256 | 1,641 | 431.97x |
| self_lease | 512 | 330,072.00 | 212,625 | 13,289 | 3,476.27x |
| self_lease | 4096 | 2,707,865.00 | 1,703,577 | 106,473 | 28,518.85x |
| read_only | 1 | 63.40 | 48 | 3 | 1.00x |
| read_only | 8 | 80.95 | 48 | 3 | 1.28x |
| read_only | 64 | 66.79 | 48 | 3 | 1.05x |
| read_only | 512 | 64.01 | 48 | 3 | 1.01x |
| read_only | 4096 | 65.09 | 48 | 3 | 1.03x |

## Interpretation

- Disjoint, first-conflict, last-conflict, and multi-peer self-renewal all scan the live-lease
  slice. Conflict position therefore does not create an early exit.
- A self-renewal is nearly free only when its own lease is the sole live lease. With peers
  present, it skips itself and still pays for the remaining scan.
- The saturated path appends one evidence row per conflict and sorts the evidence by lease ID.
  That path is asymptotically **O(K log K)** in the number of conflicts, while its retained
  conflict slice is **O(K)**. The measured medians are noisy and do not establish a clean
  superlinear curve across every interval, so this artifact names the code path without claiming
  a fitted exponent.
- The all-conflict case can be faster than disjoint scanning because same-lane detection avoids
  the per-lease tree-overlap work; its larger retained evidence still raises bytes/op above the
  disjoint case at high N.
- Read-only admission is the only measured path whose time, bytes, and allocations remain flat
  as the supplied lease view grows.

## Reproduce

```bash
go test ./internal/laneadmit -run '^TestDecideScalingFixtures$' -count=1
go test ./internal/laneadmit -run '^$' -bench BenchmarkDecideScaling -benchmem -count=5
```

The raw five-sample run completed successfully in 254.945 seconds. Host noise is visible in
some timing samples; allocation counts are stable and the table uses medians rather than the
best sample.
