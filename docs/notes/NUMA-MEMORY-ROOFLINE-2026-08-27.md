# NUMA host-memory roofline capture — 2026-08-27

Issue: #9619. Centrality: Enabling.

## Result

`fak model-observe bandwidth numa-topology` records Linux NUMA topology from `/sys/devices/system/node` plus the calling process's `Cpus_allowed_list`. It preserves sparse node IDs, online state, memory-bearing status, CPU membership, allowed CPUs, and explicit omission reasons.

`fak model-observe bandwidth numa-import --input CAPTURE.json` normalizes an externally captured `fak-numa-roofline-capture/1` matrix. A pair is reported only when the artifact independently verifies both CPU placement (`sched_getcpu` plus sysfs CPU/node membership) and resident-page memory placement (`/proc/self/numa_maps`). Requested `numactl --cpunodebind=N --membind=M` arguments alone are not proof.

The portable aggregate copy benchmark remains unchanged. This slice intentionally does not add an in-process "NUMA local" label: first-touch without independently verified CPU and resident-page placement is ambiguous.

## Capture contract

For every requested CPU-node/memory-node pair, an external runner must preserve:

- the exact requested command and raw trials;
- verified CPU and memory node plus verifier evidence;
- explicit omissions for memoryless/offline/cpuset-ineligible nodes and permission or placement failures;
- bounded working-set, peak-buffer, target-duration, and runtime-budget fields;
- `dram_isolation: "not-proven"` unless hardware counters establish DRAM traffic isolation.

The importer rejects malformed topology, unsupported verifier labels, requested/verified mismatches, invalid trials, duplicate pairs, unbounded captures, and missing local baselines. Local/remote ratios are computed only against a verified local pair for the same CPU node. Host ratios and rooflines never become device/HBM observations.

## Witness

Deterministic tests cover sparse sysfs IDs, memoryless nodes, process cpuset restriction, verified local/remote ratio construction, unverified placement rejection, and bounded-capture rejection. A genuine dual-memory-node sanctioned Linux capture remains required before #9619 can close.
