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

The canonical command to capture a capacity-valid sweep must use the verified model, endpoint, batch capacity, and engine receipt from the run being claimed. Do not reuse historical digests or capacity values unless the current setup, model, quality envelope, and provenance receipt prove they still apply.

```bash
fak webbench serving \
  --dataset testdata/webbench/sample-tasks.jsonl \
  --tracks ours \
  --endpoints ours="${VERIFIED_ENDPOINT_URL}" \
  --concurrencies "${VERIFIED_CONCURRENCY_RANGE}" \
  --batch-capacities ours="${VERIFIED_BATCH_CAPACITY}" \
  --capacity-sources ours="${VERIFIED_CAPACITY_SOURCE}" \
  --engines ours=fak-native \
  --model "${VERIFIED_MODEL}" \
  --engine-receipts ours="sha256:${VERIFIED_ENGINE_RECEIPT_SHA256}" \
  --ttft-p99-budget-ms 2000 \
  --itl-p99-budget-ms 100 \
  --out "${VERIFIED_RECEIPT_PATH}"
```

## 5. Witness Evidence

The historical receipt at `docs/_witnesses/issue-10078-webbench-qwen38-capacity-sweep/receipt.json` is retained as an unmodified historical artifact. It does not currently certify Qwen 3.8 serving performance for `fak#10078`: the current parity gate rejects the artifact with `fewer than two comparable valid points: positive measured token throughput is required`. The factual readback for that refusal is recorded in `experiments/benchmark/runs/by-machine/modular-10078-readback/validation.json`.

`fak#10078` remains open until a new or revalidated receipt passes the current gate with hardware witnessed execution, at least two comparable in-capacity points with positive measured output-token throughput, an identity-stable workload and engine receipt, quality/setup provenance, setup and measurement overhead accounting, an attached capacity manifest or probe, independent remote commit and artifact ancestry, and range evidence showing rise, plateau, and bounded tail behavior for the claimed capacity envelope.
