# Capacity-Valid Serving Sweeps (`fak.serving-sweep.v1`)

Multi-concurrency serving sweeps evaluate model serving throughput, saturation dynamics, and tail latency (p99 SLA) boundaries across concurrency levels. This document defines the capacity-valid serving sweep contract governed by the `fak.serving-sweep.v1` schema and executed through `fak webbench serving`.

## 1. The Serving Capacity Problem

Conventional serving benchmarks frequently report single-concurrency throughput or run unconstrained load against an endpoint without tracking server batch capacity. This causes two primary distortions:
1. **Unconstrained Queue Overflow**: Submitting requests far beyond the backend's admitted batch capacity measures HTTP queue buffering and connection queuing latency, rather than actual GPU compute performance.
2. **Silent Engine Substitution**: Multi-turn benchmarks can drift between native and fallback runtime paths across points without recording engine provenance.

To ensure benchmark integrity, the `fak.serving-sweep.v1` contract requires explicit binding of:
- **Workload digest**: Cryptographic SHA-256 hash of the request dataset, prompts, and decoding parameters.
- **Engine receipt digest**: Cryptographic SHA-256 hash verifying the specific backend engine identity (`fak-native`), preventing silent fallback substitution.
- **Declared batch capacity**: Maximum concurrent in-flight requests supported before queue saturation (e.g. 32 on lab GPU compute).
- **Capacity source**: Formal provenance asserting how capacity was determined (e.g. `declared-manifest`, `probe-bench`).

## 2. Invalidation and Refusal Rules

Under `EvaluateServingSweep` (`internal/webbench/serving_sweep.go`), the following invariants are strictly enforced:

| Failure Condition | Status Code | Policy Action |
|---|---|---|
| **Above-Capacity Load** | `capacity_exceeded` | Points where `concurrency > batch_capacity` are invalidated. They cannot be selected as peak throughput or SLA knee. |
| **Engine Drift** | `engine_identity_mismatch` | Any change in backend engine or engine receipt digest between points marks the point invalid. |
| **Workload Drift** | `workload_identity_mismatch` | Any change in dataset or request parameters between points invalidates the point. |
| **Unknown Capacity** | `capacity_unknown` | Undeclared or unmeasured batch capacity (`capacity <= 0` or missing source) refuses peak and knee claims fail-closed. |
| **Sparse Points** | `insufficient_points` | Fewer than two valid points cannot establish a saturation curve or support a peak claim. |

## 3. Peak Throughput and SLA Knee Selection

1. **Capacity-Valid Peak**:
   The highest aggregate output token throughput (`ThroughputTokensS`) observed strictly among **valid, in-capacity** concurrency points. Monotonic terminal points prior to capacity limits remain censored from claiming global saturation unless range closure is proven.
2. **SLA Knee**:
   The highest concurrency point that simultaneously satisfies configured latency budgets:
   - `--ttft-p99-budget-ms`: Maximum allowed 99th-percentile Time-To-First-Token.
   - `--itl-p99-budget-ms`: Maximum allowed 99th-percentile Inter-Token Latency.

## 4. Benchmark Execution Command

The canonical command to capture a certified capacity sweep on lab GPU compute:

```bash
fak webbench serving \
  --dataset testdata/webbench/sample-tasks.jsonl \
  --endpoints ours=http://127.0.0.1:8000/v1 \
  --concurrencies 1,2,4,8,16 \
  --batch-capacities ours=32 \
  --capacity-sources ours=declared-manifest \
  --engines ours=fak-native \
  --engine-receipts ours=sha256:c3b5dc1b4e0fb9682547890ef93c5d8869f0ab59218d6e32bc502f9e4210d7a4 \
  --ttft-p99-budget-ms 2000 \
  --itl-p99-budget-ms 100 \
  --out docs/_witnesses/issue-10078-webbench-qwen38-capacity-sweep/receipt.json
```

## 5. Witness Evidence

A captured witness receipt is recorded in `docs/_witnesses/issue-10078-webbench-qwen38-capacity-sweep/receipt.json`, certifying Qwen 3.8 27B serving performance across concurrency levels 1 through 16 under declared batch capacity of 32 on lab GPU hardware.
