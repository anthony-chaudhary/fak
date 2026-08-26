---
title: "System baselines for performance evidence: sampled non-SUT load, explicit unknowns, and contamination gates"
description: "Dated field-borrow pass for issue #9116. It witnesses fak's partial host/process sensors, records the primary-source basis for windowed CPU/pressure sampling and repeated/interleaved benchmarks, and defines the capture-to-nativeperf spine without pretending ambient load can be subtracted from throughput."
---

# System baselines for performance evidence

> Observed at: `2026-08-26T15:04:52Z`. Source event times and refresh triggers are in the
> ledger below. Status: implementation spine tracked by
> [#9116](https://github.com/anthony-chaudhary/fak/issues/9116). Companions:
> [ghost-slowdown baseline](RESEARCH-GHOST-SLOWDOWN-BASELINE-ABLATION-2026-08-05.md),
> [native-performance hill climb](../benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md), and
> [`internal/bench` host snapshots](../../internal/bench/README.md).

## Outcome

Performance receipts need a sampled frame around the measurement: what the host did, which
process tree was the system under test (SUT), what remained non-SUT, which axes were actually
read, and whether the run is eligible to support a claim. The report must preserve raw facts
separately from policy. It may mark a run clean, investigate, or invalid; it must never
"correct" throughput by subtracting ambient CPU because contention, cache eviction, memory
pressure, throttling, and scheduler delay do not combine linearly.

This is **Enabling** work for the Core P2 native-inference hill climb. It makes a fast result
trustworthy enough to act on; it does not itself claim a speedup.

## Self-query witness: PARTIAL

The required dogfood query was:

```text
fak capabilities "ambient host load background processes benchmark interference system baseline"
```

It returned `no matching capability` on 2026-08-26. The dev index executable was unavailable
on this workstation, so that source class is recorded as unavailable rather than as an empty
result. Raw `rg` cross-checks found strong pieces but no join:

- `internal/bench/cpumemstress.go:141-170,263-279` records Linux load/memory before and after
  one stress run and can refuse an already-busy host. It is point-in-time, Linux-centric, and
  not attached to ordinary benchmark repetitions.
- `internal/harnessres/fleet.go:143,205,281` selects and samples fak-owned process trees with
  unreadable processes kept explicit. Production host wiring does not compute whole-host CPU
  or the host-minus-SUT ambient residual.
- `internal/stallscan/stallscan.go:33-105` has host CPU, queue, faults, context switches,
  syscalls, memory, process churn, and top-consumer shapes. Its committed 2026-08-05 study
  measured that one-second PID polling caught only 10 of 200 short-lived injected processes,
  so process births are corroborating rather than a complete attribution source.
- `internal/nativeperf/receipt.go:18-31,221-238` binds machine, workload, engine, quality,
  memory, and repetition results. `internal/nativeperf/gate.go:94-151` gates output noise but
  carries no ambient evidence, so it cannot distinguish SUT regression from host contention.

Verdict: **PARTIAL**. Sensors, process-tree ownership, and a native performance gate are
present; a sampled, portable baseline artifact and its nativeperf consumer are absent.

## Dated primary-source ledger

| Source | State and event time | Immutable/current anchor | What changed in the design | License disposition | Refresh trigger |
|---|---|---|---|---|---|
| Linux Pressure Stall Information | shipped kernel interface; documented April 2018; Linux `45c13f3f9e3bb15fd89ff2864c6f627a3b4b4229`; observed 2026-08-26 | [`psi.rst@45c13f3`](https://github.com/torvalds/linux/blob/45c13f3f9e3bb15fd89ff2864c6f627a3b4b4229/Documentation/accounting/psi.rst#L7-L65) | CPU, memory, and I/O contention are best represented by cumulative time stalled, including `some` versus system-wide `full`, not inferred only from load average. Bracket each repetition; system and cgroup PSI are separate scopes and are not subtracted. | **INSPIRE-ONLY**; GPL-2.0 WITH Linux-syscall-note tree, no code copied | kernel PSI ABI/doc revision |
| Linux `/proc/stat` and cgroup v2 | shipped kernel interfaces at Linux `45c13f3`; observed 2026-08-26 | [`proc.rst@45c13f3`](https://github.com/torvalds/linux/blob/45c13f3f9e3bb15fd89ff2864c6f627a3b4b4229/Documentation/filesystems/proc.rst#L1595-L1640), [`cgroup-v2.rst@45c13f3`](https://github.com/torvalds/linux/blob/45c13f3f9e3bb15fd89ff2864c6f627a3b4b4229/Documentation/admin-guide/cgroup-v2.rst#L1147-L1152) | Use cumulative host CPU, steal, context-switch, runnable, and SUT-cgroup CPU deltas. A dedicated SUT cgroup makes host-minus-SUT CPU a bounded residual; mismatched/reset counters become unknown. Kernel docs warn that `iowait` is unreliable, so it is not a validity gate. | **INSPIRE-ONLY**; same license boundary | kernel proc/cgroup ABI change |
| Microsoft Windows Performance Counters | shipped platform guidance; page last updated 2025-07-14; observed 2026-08-26 | [Collecting Performance Data](https://learn.microsoft.com/en-us/windows/win32/perfctrs/collecting-performance-data) | Rate counters require two timestamped samples. Legacy process-instance names can be reassigned after exit; Process V2 includes PID in the instance identity. Carry actual windows and key deltas by PID/start identity, never image name alone. | **INSPIRE-ONLY**; proprietary documentation, use platform APIs only | Windows counter/API guidance update |
| Google Benchmark v1.9.5 | released 2026-01-21; pinned at `192ef10025eb2c4cdd392bc502f0c852196baa48`; observed 2026-08-26 | [random interleaving](https://github.com/google/benchmark/blob/192ef10025eb2c4cdd392bc502f0c852196baa48/docs/random_interleaving.md), [variance guidance](https://github.com/google/benchmark/blob/192ef10025eb2c4cdd392bc502f0c852196baa48/docs/reducing_variance.md#L77-L120), [repetition statistics](https://github.com/google/benchmark/blob/192ef10025eb2c4cdd392bc502f0c852196baa48/docs/user_guide.md#L1148-L1156) | Repetitions expose mean, median, standard deviation, and CV; random interleaving reduces bias from changing system state. Ambient evidence complements those controls and does not replace them. | **ADAPT**, Apache-2.0 repository; no implementation copied | release or benchmark methodology change |
| Criterion.rs 0.7.0 | released; tag commit `567405d25363804dd1e6d440a0c9d6612c4cecd8`; analysis doc last changed 2024-07-10; observed 2026-08-26 | [`analysis.md@567405d`](https://github.com/bheisler/criterion.rs/blob/567405d25363804dd1e6d440a0c9d6612c4cecd8/book/src/analysis.md#L12-L56) | Warmup, outlier classification, retained samples, and a configurable statistical noise threshold are separate concerns. Mark contaminated repetitions but never silently delete them. | **ADAPT**, Apache-2.0 OR MIT | release/methodology change |
| OpenTelemetry System semantic conventions v1.44.0 | released 2026-08-04 at `e10a930844c6951757a43b849d364f7d056ac32b`; observed 2026-08-26 | [`system-metrics.md@e10a930`](https://github.com/open-telemetry/semantic-conventions/blob/e10a930844c6951757a43b849d364f7d056ac32b/docs/system/system-metrics.md#L149-L232), [attribute requirement levels](https://github.com/open-telemetry/semantic-conventions/blob/e10a930844c6951757a43b849d364f7d056ac32b/docs/general/attribute-requirement-level.md#L28-L33) | Raw cumulative CPU time is canonical; utilization is delta-derived. Host and process scopes stay separate. High-cardinality or identifying attributes are opt-in, so default receipts carry aggregates rather than raw process identity. System metrics remain Development, so exact exported names are WATCH. | **ADAPT**, Apache-2.0; schema vocabulary only | semantic-conventions release/stability change |
| SPECjbb2015 Run and Reporting Rules | published benchmark standard; observed 2026-08-26 | [run rules PDF](https://www.spec.org/jbb2015/docs/runrules.pdf) | Other applications on the SUT and the system configuration are part of run validity/reporting, not invisible background detail. fak should bind the observed environment to each receipt. | **INSPIRE-ONLY**, copyrighted standard; no text or implementation copied | rules revision |
| NVIDIA NVML vR610 | shipped API, updated 2026-05-26; observed 2026-08-26 | [device utilization](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html), [process utilization](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) | CPU-idle does not imply GPU-uncontended. A later hardware adapter should record device-wide utilization plus aggregate non-SUT process presence/memory at the same repetition boundary. | **INSPIRE-ONLY**; call supported APIs through fak's GPU adapter, copy no documentation/code | NVML API revision |

Coverage checked: normative platform documentation, benchmark methodology, schema conventions,
run/reporting rules, and fak's own code/tests/issues. No external implementation is ported.
Google Benchmark issues/history beyond the pinned random-interleaving mechanism and vendor
roadmaps were not needed for the v1 capture contract; they remain an explicit omission rather
than an empty result.

## Candidate matrix

| Candidate | Fact or inference | Portfolio disposition | fak seam and falsifier |
|---|---|---|---|
| Per-repetition windowed raw deltas with actual elapsed time | source fact from Windows/Linux counters; corroborated by fak's spawn-window defect | **DEFAULT** | `internal/systembaseline`: every count/rate carries samples and window, aligned 1:1 with performance repetitions. Falsified if a serialized rate lacks its measured interval or cannot be tied to one repetition. |
| Host pressure consequences (Linux PSI; Windows scheduler/fault counters when available) | source fact plus platform adaptation | **DEFAULT when available** | Platform reader with typed presence. Falsified if absence becomes zero/healthy or if sampler overhead materially perturbs the benchmark. |
| SUT process-tree versus non-SUT attribution | fak-present process-tree mechanism plus inference to performance evidence | **DEFAULT** | Capture names the root identity and sample coverage; aggregate/scrubbed output only. Falsified by a fixture where an owned child is charged ambient or an unrelated process is charged to SUT. |
| Repetitions, dispersion, and randomized/interleaved A/B order | shipped Google Benchmark mechanism | **DEFAULT, complementary** | Existing nativeperf repetitions/noise gate. Falsified if host evidence is used to waive inadequate repetitions or excessive CV. |
| Subtract ambient CPU from latency or throughput to produce a corrected score | unsupported inference | **EXCLUDE** | No field in the artifact or gate may emit corrected performance. Reopen only with a validated causal resource model that predicts held-out interference regimes. |
| cgroup/job-object isolation and per-cgroup PSI | shipped OS mechanisms, not needed for the portable v1 | **OPTIONAL-MODULE / follow-on** | Linux cgroup and Windows Job Object adapters. Falsified as a default if setup privileges or containment change the workload/operator cost. |
| Raw command lines, hostnames, PIDs, image names, and complete process lists in public receipts | operationally available but unnecessary/high-cardinality | **EXCLUDE by default** | Emit aggregate counters publicly; make scrubbed top consumers opt-in/local diagnostic data. Reopen only for access-controlled artifacts with a named need. |
| GPU clocks, power, thermals, and per-process device memory | campaign requirements already tracked, not covered by CPU/process sources | **WATCH / follow-on** | Hardware adapter under the same presence/verdict schema. Trigger: #4367/#8129/#8486 capture contract consolidation. |

## Default and coverage frontiers

The **default frontier** is a bounded pre-run baseline plus a bracketed observation for every
real benchmark repetition, explicit host/SUT/non-SUT frames, presence bits, sampler cost, a
digest, and a closed claim-eligibility verdict. Nativeperf consumes the 1:1 evidence slice
together with engine, quality, memory, repetitions, and throughput; contaminated evidence
becomes investigate/invalid, not pass. Reports retain all repetitions and may show both all-run
and clean-only statistics when enough clean samples exist; they never silently drop an outlier.

The **coverage frontier** is richer isolation and attribution: Linux per-cgroup PSI, Windows Job
Objects/Process V2 or ETW for short-lived descendants, GPU/process telemetry, frequency and
thermal state, disk/network pressure, automatic paired retry, and fleet storage/query. These
are valuable, but omitting them still leaves the v1 user able to capture a truthful supported
subset with unknown axes named. They fan out after the working spine.

## Core through-line and safe-consumer contract

Shortest path: `fak bench system-baseline -- <real benchmark command>` -> bounded baseline and
during-run samples -> `fak.system-baseline/v1` JSON -> digest verification and
`clean|investigate|invalid` -> native-performance receipt/gate readback.

Safe consumers must:

1. check schema and digest before reading values;
2. check per-axis availability and sample coverage before using a metric;
3. use the artifact's measured window and denominator;
4. treat `investigate`/`invalid`, required-unknown axes, and insufficient coverage as ineligible
   for a passing performance claim;
5. retain the measured throughput/latency unchanged and use ambient data only to stratify,
   annotate, retry, investigate, or exclude.

The gold-plating boundary excludes device telemetry, privileged event streams, storage/UI,
automatic remediation, and universal thresholds from #9116's first witness.

## P1-P4 check

- **P1 — preserved:** one compact digest-bound artifact; no provider/model context growth.
- **P2 — advanced:** ambient work and sampler overhead are counted; no false corrected gain.
- **P3 — advanced:** facts and policy are separate, platform axes are optional-but-explicit,
  and threshold changes do not rewrite captured evidence.
- **P4 — advanced:** capture, verify, render, and native gate consumption share the real CLI
  path and an actionable verdict.

## Durable tracker

[#9116](https://github.com/anthony-chaudhary/fak/issues/9116) contains the spine and witness.
The issue was deduplicated against open and closed system-baseline/ambient-load work before
creation. Closest predecessors are #8757 (native receipts) and #8759 (native gate); #8760 and
#9020 remain separate SUT-profiler work.
