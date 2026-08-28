---
title: "Issue #9519 — live host-memory saturation receipt"
description: "Immutable raw receipt and independent recalculation for the portable host-memory saturation sweep."
---

# Issue #9519 — live host-memory saturation receipt

## Verdict

This sanctioned Linux capture supplies the previously missing immutable live artifact for the host-memory saturation sweep. It measures **sustained host memory-copy throughput**, not hardware-counter DRAM traffic. It does not measure device HBM, infer bandwidth from utilization, or turn capacity, pressure, paging, or process/storage I/O into bandwidth.

The byte-for-byte raw collector output is [`capture.json`](capture.json), SHA-256 `c5aa21b93baf3470b5b94772786a84fee43f79877668d9b0b4a8922b99ab13e2`.

## Provenance and envelope

- Source tree: `32b9d39886feb6eeb5f50ab1f30f0cf5bead1b4a`, clean for `cmd/fak` and `internal/modelperfobs` before cross-compilation.
- Linux/amd64 binary SHA-256: `dd6d0705e486729b0e1b897cb35d467d295c0176a44c7a5519e02d6bc4bc4e1b`.
- Toolchain: Go 1.26.7. The binary lacked an embedded VCS stamp, so provenance is bound by the clean source-tree check plus the binary hash; it did not self-report the source commit.
- UTC capture window: `2026-08-28T00:29:49Z` through `2026-08-28T00:29:57Z`.
- Scrubbed sanctioned GCP envelope: Linux 6.6.137+, Intel Xeon CPU at 2.20 GHz, 8 logical CPUs, 1 NUMA node, `32,872,532 kB` MemTotal.

Public-safe reproduction command:

```text
fak model-observe bandwidth collect --count 1 --interval 10ms --phase other --shape small --nvidia-device __fak_no_device__ --measure-host-roofline --roofline-sweep --roofline-bytes 268435456 --roofline-trials 5 --roofline-duration 100ms --roofline-threads 8 --output capture.json --pretty=true
```

## Independent recalculation

The nested roofline schema is `fak-host-memory-roofline-sweep/1`. All 4 requested points are present; 4 are valid and 0 are omitted. Every point preserves five raw trials, and every trial has positive `traffic_bytes` and `duration_ms`. Recomputed medians are:

| Workers | Median GB/s |
|---:|---:|
| 1 | 7.316334 |
| 2 | 14.477725 |
| 4 | 28.250186 |
| 8 | 23.442987 |

The raw observed maximum is 28.250186 GB/s at 4 workers. Applying the recorded 0.9 threshold makes 4 workers the first saturation knee. The stored sustainable roof is the conservative plateau value, 23.442987 GB/s, with plateau worker counts `[4, 8]`; it is intentionally distinct from the raw observed maximum.

The receipt retains `dram_isolation=not-proven` and `interpretation=sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth`. `device_counters` is false, so no device/HBM observation is populated. The capacity and Linux pressure/reclaim/swap fields remain separate context, not bandwidth measurements.

## Remaining scope

This artifact closes only #9519's immutable-live-artifact gap. It does not retire the genuine Apple package/unified-memory capture (#9427), dual-memory-node local/remote capture (#9619), or production-pattern roof matrix (#9147).
