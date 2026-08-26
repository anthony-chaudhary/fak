# Lease-ref lifecycle burst scaling — 2026-08-21

## Verdict

The production `internal/leaseref` lifecycle stays cleanup-safe in this two-clone,
bare-origin harness: every measured cell finished with **zero residual lease refs**, including
the injected cancellation and crash paths.

Useful throughput stops improving before raw child-launch throughput does. At `N=128`,
disjoint admission still accepted all 128 children, but median convergence throughput was
**31.91/s**, versus **63.83/s** admitted and **63.33/s** released, while convergence p95 reached
**1,341.00 ms**. Under a full hotspot, attempted throughput rose to **581.90/s**, but only two
leases could be admitted across the two independently visible clones; useful convergence and
release throughput fell to **4.55/s**.

This is a visibility-and-convergence benchmark, not proof of cross-machine mutual exclusion.
`git update-ref` compare-and-swap is atomic within one clone, but two clones can each admit the
same lease ID before either synchronizes with the origin. The hotspot's two admissions make that
boundary observable rather than hiding it.

## Provenance

- Base commit: `a3d44ad07b5c092a9ce50cfc94805be5f1b83281`.
- Benchmark source: `internal/leaseref/lifecycle_burst_test.go`, SHA-256
  `44293501acbacff1fe2fc8dcadd9655994dd594063472860d67e59d192806149`.
- Source status at capture: `?? internal/leaseref/lifecycle_burst_test.go`.
- Go: `go1.26.6 linux/amd64`.
- Git: `git version 2.43.0`, SHA-1 object format.
- CPU: AMD Ryzen 9 9950X 16-Core Processor.
- Storage: WSL local temporary filesystem, with two working clones and one bare origin over
  Git file transport.
- Samples: five Go benchmark samples per matrix cell; latency and throughput tables report
  medians of those samples.

The base commit plus source digest identifies the measured working overlay before the benchmark
and this artifact land together.

## Workload

Each benchmark cell constructs a real bare origin and two real clones, then drives the production
`AcquireFenced`, `Sync`, `ReleaseFenced`, and `Reap` functions concurrently:

- Widths: `N=1,8,32,128`.
- `disjoint`: every child uses a unique lease ID.
- `overlap25`: 25% of children contend for repeated IDs in one clone; the remainder are
  disjoint.
- `hotspot`: every child uses one lease ID, split across both clones to expose the
  visibility-only boundary.
- Cancellation: when at least three leases are admitted, one child is cancelled after acquire.
- Cancellation cleanup: the cancelled child's lifecycle context is cancelled, then
  `ReleaseFenced` runs from a detached cleanup context.
- Crash: when at least two leases are admitted, one child omits explicit release.
- Cleanup: logical time advances beyond TTL without a wall-clock sleep, then both clones and
  the origin are reaped and scanned for residual `refs/fak/leases/**`.

`TestLeaseLifecycleBurstMatrix` locks the 12-cell matrix. `TestLeaseLifecycleBurstSpine` drives
the production two-clone path and requires admitted and refused acquisitions, cancellation,
crash cleanup, typed failures, Git subprocess activity, and zero residual refs before benchmark
numbers are trusted.

## Median lifecycle latency

### Acquire

| N | Pattern | p50 ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|
| 1 | disjoint | 6.38 | 8.17 | 8.76 |
| 1 | overlap25 | 5.33 | 6.96 | 8.07 |
| 1 | hotspot | 5.29 | 6.57 | 6.85 |
| 8 | disjoint | 7.49 | 27.44 | 36.95 |
| 8 | overlap25 | 6.78 | 17.32 | 29.95 |
| 8 | hotspot | 13.20 | 44.67 | 48.41 |
| 32 | disjoint | 49.89 | 71.05 | 82.33 |
| 32 | overlap25 | 28.10 | 65.34 | 77.34 |
| 32 | hotspot | 41.37 | 75.64 | 92.72 |
| 128 | disjoint | 113.30 | 191.50 | 201.00 |
| 128 | overlap25 | 119.40 | 186.90 | 208.00 |
| 128 | hotspot | 80.13 | 180.50 | 206.40 |

### Converge

| N | Pattern | p50 ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|
| 1 | disjoint | 25.80 | 29.36 | 29.80 |
| 1 | overlap25 | 20.52 | 25.78 | 29.02 |
| 1 | hotspot | 20.38 | 26.00 | 29.94 |
| 8 | disjoint | 38.46 | 61.34 | 78.36 |
| 8 | overlap25 | 24.97 | 46.53 | 54.39 |
| 8 | hotspot | 29.27 | 48.11 | 65.46 |
| 32 | disjoint | 118.60 | 240.00 | 267.20 |
| 32 | overlap25 | 96.30 | 184.80 | 187.80 |
| 32 | hotspot | 24.59 | 42.11 | 44.94 |
| 128 | disjoint | 680.00 | 1,341.00 | 1,347.00 |
| 128 | overlap25 | 690.40 | 888.40 | 896.50 |
| 128 | hotspot | 27.85 | 43.59 | 43.59 |

### Release

| N | Pattern | p50 ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|
| 1 | disjoint | 4.70 | 5.57 | 6.04 |
| 1 | overlap25 | 3.87 | 4.94 | 5.36 |
| 1 | hotspot | 3.82 | 4.95 | 5.34 |
| 8 | disjoint | 5.18 | 18.08 | 20.03 |
| 8 | overlap25 | 4.78 | 16.98 | 25.88 |
| 8 | hotspot | 5.01 | 19.33 | 19.50 |
| 32 | disjoint | 36.75 | 61.98 | 64.73 |
| 32 | overlap25 | 21.56 | 55.79 | 63.45 |
| 32 | hotspot | 4.66 | 8.79 | 8.79 |
| 128 | disjoint | 116.60 | 160.10 | 175.30 |
| 128 | overlap25 | 81.86 | 138.30 | 141.10 |
| 128 | hotspot | 3.92 | 4.91 | 4.91 |

### Reap

| N | Pattern | p50 ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|
| 1 | disjoint | 1.56 | 4.73 | 5.38 |
| 1 | overlap25 | 1.36 | 4.14 | 4.58 |
| 1 | hotspot | 1.41 | 4.38 | 5.17 |
| 8 | disjoint | 11.19 | 24.01 | 32.31 |
| 8 | overlap25 | 8.84 | 14.61 | 23.14 |
| 8 | hotspot | 4.30 | 14.65 | 25.53 |
| 32 | disjoint | 55.37 | 91.94 | 91.94 |
| 32 | overlap25 | 30.53 | 60.76 | 60.76 |
| 32 | hotspot | 4.25 | 5.82 | 7.29 |
| 128 | disjoint | 153.00 | 248.20 | 248.20 |
| 128 | overlap25 | 111.30 | 189.70 | 189.70 |
| 128 | hotspot | 3.77 | 5.10 | 5.10 |

## Median throughput

`attempts/s` is launch throughput and includes refused children. The admitted, converged, and
released columns are the useful-work rates.

| N | Pattern | attempts/s | admitted/s | converged/s | released/s |
|---:|---|---:|---:|---:|---:|
| 1 | disjoint | 21.82 | 21.82 | 21.82 | 21.82 |
| 1 | overlap25 | 27.12 | 27.12 | 27.12 | 27.12 |
| 1 | hotspot | 26.44 | 26.44 | 26.44 | 26.44 |
| 8 | disjoint | 80.96 | 80.96 | 19.06 | 70.84 |
| 8 | overlap25 | 115.30 | 100.80 | 30.87 | 86.44 |
| 8 | hotspot | 95.61 | 23.90 | 12.55 | 11.95 |
| 32 | disjoint | 81.83 | 81.83 | 38.32 | 79.28 |
| 32 | overlap25 | 115.70 | 90.38 | 22.98 | 86.77 |
| 32 | hotspot | 302.00 | 18.87 | 9.44 | 9.44 |
| 128 | disjoint | 63.83 | 63.83 | 31.91 | 63.33 |
| 128 | overlap25 | 89.62 | 67.91 | 27.66 | 67.21 |
| 128 | hotspot | 581.90 | 9.09 | 4.55 | 4.55 |

## Typed outcomes and observable resource counts

This table is the deterministic one-iteration artifact captured with
`FAK_LEASEREF_BURST_JSON=1`. Reaped refs are successful deletions summed across both clones and
the bare origin; they are not unique lease IDs.

| N | Pattern | admitted / refused | Refusal reasons | converge ok / errors | Sync reasons | release ok | cancel / crash | reaped / residual | Git processes | stdout / stdin / record bytes |
|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | disjoint | 1 / 0 | — | 1 / 0 | — | 1 | 0 / 0 | 1 / 0 | 19 | 560 / 35 / 161 |
| 1 | overlap25 | 1 / 0 | — | 1 / 0 | — | 1 | 0 / 0 | 1 / 0 | 19 | 588 / 38 / 172 |
| 1 | hotspot | 1 / 0 | — | 1 / 0 | — | 1 | 0 / 0 | 1 / 0 | 19 | 543 / 30 / 157 |
| 8 | disjoint | 8 / 0 | — | 2 / 5 | `SYNC_PUSH_ERROR=5` | 7 | 1 / 1 | 17 / 0 | 102 | 6,103 / 595 / 1,288 |
| 8 | overlap25 | 7 / 1 | `LEASE_CONTENDED=1` | 2 / 4 | `SYNC_PUSH_ERROR=4` | 6 | 1 / 1 | 15 / 0 | 94 | 5,710 / 635 / 1,358 |
| 8 | hotspot | 2 / 6 | `LEASE_CONTENDED=6` | 1 / 1 | `SYNC_PUSH_ERROR=1` | 1 | 0 / 1 | 2 / 0 | 52 | 1,138 / 60 / 1,256 |
| 32 | disjoint | 32 / 0 | — | 10 / 21 | `SYNC_PUSH_ERROR=21` | 31 | 1 / 1 | 65 / 0 | 398 | 24,648 / 2,275 / 5,152 |
| 32 | overlap25 | 25 / 7 | `LEASE_CONTENDED=5`, `LEASE_HELD=2` | 0 / 24 | `SYNC_PUSH_ERROR=24` | 24 | 1 / 1 | 26 / 0 | 302 | 15,441 / 1,113 / 5,432 |
| 32 | hotspot | 2 / 30 | `LEASE_CONTENDED=25`, `LEASE_HELD=5` | 1 / 1 | `SYNC_PUSH_ERROR=1` | 1 | 0 / 1 | 2 / 0 | 138 | 3,124 / 60 / 5,024 |
| 128 | disjoint | 128 / 0 | — | 13 / 114 | `SYNC_PUSH_ERROR=114` | 127 | 1 / 1 | 257 / 0 | 1,553 | 98,856 / 8,995 / 20,608 |
| 128 | overlap25 | 97 / 31 | `LEASE_CONTENDED=21`, `LEASE_HELD=10` | 37 / 59 | `SYNC_PUSH_ERROR=59` | 96 | 1 / 1 | 195 / 0 | 1,309 | 82,980 / 8,375 / 21,728 |
| 128 | hotspot | 2 / 126 | `LEASE_CONTENDED=50`, `LEASE_HELD=76` | 1 / 1 | `SYNC_PUSH_ERROR=1` | 1 | 0 / 1 | 2 / 0 | 380 | 20,036 / 60 / 20,096 |

The production runner exposes Git command stdout and stdin, not file-transport payload bytes.
The artifact therefore reports process I/O and serialized lease-record bytes, leaves transport
bytes null, and does not invent a network-byte estimate.

## Interpretation

- The first material tail jump appears at `N=32`. Disjoint acquire p95 grew from **27.44 ms**
  at `N=8` to **71.05 ms**, while converge p95 grew from **61.34 ms** to **240.00 ms**.
- Disjoint admitted throughput peaked at **80.96/s** at `N=8`, remained **81.83/s** at
  `N=32`, then fell to **63.83/s** at `N=128`. Release throughput followed the same shape:
  **70.84/s**, **79.28/s**, then **63.33/s**.
- Wildcard ref synchronization is the dominant high-width limiter. The single-shot `N=128`
  disjoint cell admitted every child, but 114 of 127 synchronization attempts returned typed
  `SYNC_PUSH_ERROR`; release still succeeded for all 127 non-crashed children, including the
  cancelled child's detached cleanup.
- The overlap workload reduces admitted work through typed `LEASE_CONTENDED` and `LEASE_HELD`
  refusals, then pays the same synchronization pressure. At `N=128`, median converged throughput
  was only **27.66/s** despite **67.91/s** admitted.
- Hotspot attempt throughput is intentionally misleading. Two clones can each admit the same
  ID before synchronization, after which nearly all children are refused. The useful released
  rate declines from **11.95/s** at `N=8` to **4.55/s** at `N=128`.
- Cleanup remained complete in all 12 cells. Cancellation used normal cleanup, crash simulation
  relied on TTL expiry plus reap, and the final raw ref scan found zero residual refs.

## Reproduce

```bash
go test ./internal/leaseref -run '^TestLeaseLifecycleBurst(Matrix|Spine)$' -count=1
go test ./internal/leaseref -run '^$' -bench BenchmarkLeaseLifecycleBurst -benchmem -count=5
FAK_LEASEREF_BURST_JSON=1 go test ./internal/leaseref -run '^$' -bench '^BenchmarkLeaseLifecycleBurst$' -benchmem -benchtime=1x -count=1
```

The exact five-sample command completed successfully in **257.558 seconds**. Timing noise from
process launch and the shared development host is visible; the tables use medians rather than
the best sample.
