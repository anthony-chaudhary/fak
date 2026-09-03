---
title: "Native-performance Grafana panel proof and local Metal observability"
description: "Reproduce the deterministic panel coverage matrix, operate the local Qwen3.8 Metal observability pipeline, query Grafana dashboards and Prometheus metrics, and validate the execution receipt contract."
---

# Native-performance Grafana panel proof and local Metal observability

Issue #9817 binds the four native-performance dashboards to their contracts,
controlled Prometheus fixtures, deterministic visual witnesses, and a separate live
fak-native Qwen3.8 receipt:

- `fak-native-kernel-performance.json`
- `fak-native-backends.json`
- `fak-native-artifacts.json`
- `fak-native-slo.json`

The deterministic fixture proof is complete only when every query-bearing panel appears
in the matrix and every static row/text panel appears as a non-query surface. Fixture
success is **not** a live performance claim. Live completion additionally requires one
fresh Qwen3.8 execution whose scrubbed receipt names `fak-native`, the `inkernel` runtime,
an explicit native forward path, and zero fallback.

Issue #9898 establishes the local Apple Silicon Qwen3.8 Metal observability proof bundle,
providing an executable observer harness, Prometheus scraping checks, dashboard query
validation, and a structured, zero-fallback execution receipt.

## Local Qwen3.8 Metal observability pipeline

The local observability pipeline enables operators on Apple Silicon hardware (such as an
Apple M3 Pro with 36 GiB unified memory) to execute, scrape, query, and verify fak-native
Qwen3.8 inference through Grafana without relying on remote or containerized dependencies:

```
+--------------------------+      Prometheus Scrape      +--------------------------+
|  fak serve               | -------------------------> |  Prometheus 3.2.1        |
|  --engine inkernel       |        (:18085/metrics     |  (127.0.0.1:9091)        |
|  --qwen38-runtime native |         or :8080/metrics)  +--------------------------+
|  --metal                 |                                      |
|  --model qwen38:27b      |                                      | Datasource Query
+--------------------------+                                      v
             |                                          +--------------------------+
             | Chat /v1/chat/completions                |  Grafana 12.4.2          |
             v                                          |  (127.0.0.1:3000)        |
+--------------------------+                            |  4 Native Dashboards     |
| scripts/local-qwen38-    |                            +--------------------------+
| metal-observe.sh         |                                      |
+--------------------------+                                      | Query Validation
             |                                                    v
             +--------------------------------------> [ Public Structured Receipt ]
                                                      local-qwen38-metal-live-proof.json
```

### Pipeline components

1. **In-Kernel Model Execution (`fak serve`)**:
   - Invoked with `--engine inkernel --qwen38-runtime native --metal --model qwen38:27b`.
   - Uses streamed Q4_K quantization (`FAK_METAL_STREAM_Q4K=1 FAK_Q4K_FREE_CPU=1 FAK_Q4K=1`) to fit within the 36 GiB unified memory envelope while releasing retained CPU backing memory.
   - Enforces fail-closed execution: any silent fallback to `llama.cpp` or CPU emulation immediately fails.

2. **Metrics Exposition and Prometheus Scraping**:
   - `fak serve` exposes native telemetry via `/metrics`.
   - Prometheus (either Homebrew-native at `127.0.0.1:9091` or containerized via `tools/grafana/up.sh`) scrapes the target endpoint.
   - Scrape health is verified via `/api/v1/targets` and direct endpoint polling.

3. **Dashboard Query Evaluation and Validation**:
   - Queries corresponding to all four native-performance dashboards are evaluated against the scraped metrics or Prometheus API.
   - Sentinel values and anti-coercion checks ensure missing metrics are never silently coerced to zero.

4. **Public Scrubbed Receipt Generation**:
   - Produces a machine-readable JSON witness (`tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json`) conforming to `fak-native-qwen38-metal-observation/v1`.
   - Scrubbed of all private hostnames, paths, credentials, and raw logs.

## Grafana dashboards breakdown

The observability stack provisions four dedicated native-performance dashboards under `tools/grafana/dashboards/`:

### 1. `fak-native-kernel-performance` (UID: `fak-native-kernel-performance`)
Tracks in-kernel latency breakdown, throughput, and hardware efficiency.
- **Request Counter**: `count(fak_native_receipt_requests_total{engine="inkernel",backend=~"$backend",forward_path=~"$forward_path"}) or vector(-1)`
- **Request Rate**: `sum(rate(fak_native_receipt_requests_total{engine="inkernel",backend=~"$backend",forward_path=~"$forward_path"}[$__rate_interval])) or vector(-1)`
- **Latency by Phase**: Computes average seconds per request spent in `queue`, `prefill`, `decode`, and `kernel` phases using `fak_native_receipt_phase_seconds_total`.
- **KV & Transfer Byte Rates**: Tracks memory bandwidth consumption via `fak_native_receipt_bytes_total`.
- **Evidence Freshness**: Monitored via `fak_native_receipt_latest_age_seconds` and `fak_native_receipt_latest_stale`.

### 2. `fak-native-backends` (UID: `fak-native-backends`)
Displays backend-specific operational parameters and device isolation.
- **Backend Identity**: `fak_native_runtime_info{engine="inkernel",backend="metal",model="qwen3.8",planner="inkernel",owner="fak"}`.
- **Backend Request Rate**: `sum by (backend, forward_path) (rate(fak_native_receipt_requests_total{engine="inkernel",backend=~"$backend"}[5m]))`.
- **Device Utilization & Pressure**: Tracks memory pressure, backend execution delays, and hardware sync events.
- **Fallback Exclusion**: Verifies `rate(fak_native_receipt_unsupported_total[5m]) == 0`.

### 3. `fak-native-artifacts` (UID: `fak-native-artifacts`)
Catalogs durable artifacts produced by native runs.
- **Artifact Indexing**: Tracks `benchmark_receipt`, `metal_profile_bundle`, `kernel_trace`, and `comparison_report` states via `fak_native_artifact_info`.
- **Correlation Key Mapping**: Binds artifacts to the immutable run identifier (`npc1_<hex32>`).
- **Artifact Freshness**: Validates `fak_native_receipt_latest_stale == 0`.

### 4. `fak-native-slo` (UID: `fak-native-slo`)
Tracks 10 formal SLO objectives across time-to-first-token (TTFT), inter-token latency (TPOT), throughput, queue delay, cache efficiency, transfer share, kernel share, memory pressure, evidence freshness, and receipt coverage.
- **SLO State**: Evaluates regression or missing evidence via `fak_native_slo_state{engine="fak-native",backend=~"$backend"}`.
- **Objective Values**: Evaluates normalized ratios via `fak_native_slo_value`.
- **Violation Monitoring**: Alerts on `fak_native_slo_violation == 1`.

## Prometheus metrics contract

The `/metrics` endpoint exposes the following metric families for native execution:

| Metric Name | Type | Key Labels | Purpose |
|---|---|---|---|
| `fak_native_runtime_info` | Gauge | `engine`, `backend`, `forward_path`, `model`, `planner`, `owner` | Proves the exact runtime identity (`engine="inkernel"`, `backend="metal"`, `model="qwen3.8"`). Value is always `1`. |
| `fak_native_receipt_requests_total` | Counter | `engine`, `backend`, `forward_path` | Total count of completed, verified native execution receipts. |
| `fak_native_receipt_phase_seconds_total` | Counter | `engine`, `backend`, `forward_path`, `phase` | Cumulative seconds spent in `queue`, `prefill`, `decode`, and `kernel`. |
| `fak_native_phase_seconds_total` | Counter | `engine`, `backend`, `forward_path`, `phase`, `kind` | Wall, active, and wait durations per phase. |
| `fak_native_receipt_bytes_total` | Counter | `engine`, `backend`, `forward_path`, `kind` | Cumulative bytes transferred for `kv` cache and device `transfer`. |
| `fak_native_receipt_signal_supported` | Gauge | `signal` | Indicates authoritative signal support (`queue`, `kernel`, `transfer`). |
| `fak_native_receipt_latest_age_seconds` | Gauge | None | Seconds elapsed since the most recent native execution observation. |
| `fak_native_receipt_latest_stale` | Gauge | None | `0` when the latest observation is fresh (within 900s), `1` when stale or absent. |
| `fak_native_receipt_unsupported_total` | Counter | None | Incremented when an execution fails or uses an unsupported path. |

### Anti-coercion rule
To preserve evidentiary integrity, PromQL panel targets in the native dashboards must **never** coerce missing metrics to zero using `or vector(0)` or similar constructs. An unexecuted model or broken scrape must render an empty series or trigger explicit unavailable sentinels (`or vector(-1)`), ensuring that absence of evidence is never displayed as green zero-latency or zero-error performance.

## Execution receipt contract

Scrubbed live receipts follow schema `fak-native-qwen38-metal-observation/v1` and are stored at `tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json`. Every receipt must satisfy:

1. **Runtime & Engine Identity**:
   - `engine`: `"inkernel"` / `"fak-native"`
   - `model`: `"qwen38:27b"` (family `Qwen3.8`)
   - `runtime`: `"native"`
   - `qwen38_runtime`: `"native"`
   - `backend`: `"metal"`
   - `device`: Non-empty sanitized string (e.g., `"Apple M3 Pro"`)
   - `required_execution` and `observed_execution` blocks must both specify `engine: "fak-native"`, `runtime_engine: "inkernel"`, `planner: "inkernel"`, `model_owner: "fak"`.

2. **Zero Fallback Invariant**:
   - `fallback_count`: `0`
   - `fallback_active`: `false`
   - `llama_cpp_used`: `false`
   - All nested execution blocks must enforce the same zero-fallback constants.

3. **Observed Completion**:
   - `live_execution_obtained`: `true`
   - `observed_execution.completed`: `true`
   - `observed_execution.output_tokens`: `> 0`
   - `run_id`: Format `npc1_<32-hex-characters>` matched by `observed_execution.correlation_key`.
   - `native_receipt.forward_path`: Prefix `metal/`
   - `native_receipt.q4k`: `true`
   - `native_receipt.sha256`: 64-character lowercase SHA-256 hash of response.

4. **Live Dashboard Query Validation**:
   - `dashboard_queries.status`: `"valid"`
   - `dashboard_queries.queries_passed`: `> 0`
   - `dashboard_queries.failed_queries`: `0`
   - `dashboard_queries.zero_coerced`: `false`

5. **Privacy & Freshness**:
   - `raw_logs_committed`: `false`
   - `private_identifiers_committed`: `false`
   - `completed_at_utc`: ISO 8601 / RFC 3339 UTC timestamp.
   - Must be within `MAX_AGE_SECONDS` (900 seconds / 15 minutes) of validation time.

## Verification runbook

### 1. Run regression tests
Validate that the observer script and receipt reject invalid engines, backends, runtimes, fallbacks, stale timestamps, and failed dashboard queries:

```bash
bash scripts/local-qwen38-metal-observe_test.sh
```

### 2. Validate receipt file
Validate an existing receipt against the contract schema:

```bash
bash scripts/local-qwen38-metal-observe.sh --validate tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json --now <RFC3339_TIMESTAMP>
```

### 3. Validate live dashboard queries
Verify that a Prometheus metrics exposition or receipt satisfies dashboard query requirements:

```bash
bash scripts/local-qwen38-metal-observe.sh --validate-dashboard-queries tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json
```

### 4. Execute a local observation run
On Apple Silicon hardware with the cached `qwen38:27b` artifact, execute a live observation pass:

```bash
bash scripts/local-qwen38-metal-observe.sh --output tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json
```

---

## Reproduce the matrix

Run from the repository root:

```bash
go test ./internal/nativeperfcoverage -count=1 -v
```

The test parses all four dashboard and contract JSON files, extracts each panel target,
parses the controlled Prometheus fixtures, validates every metric and bounded label, and
prints a stable dashboard-by-dashboard and panel-by-panel matrix. The suite then evaluates
all 54 panel targets, both annotations, and all five query variables through `promtool test
rules` over multi-point controlled data. It also audits annotations and dashboard
variables so a query can neither hide outside a panel nor drift from the same metric
contract.

The package test suite includes negative controls for:

- a renamed or unknown metric;
- a missing supervised fixture job;
- invalid or unsupported PromQL;
- stale evidence;
- `vector(0)` or another missing-as-zero coercion;
- an unavailable panel without an honest `UNAVAILABLE`/reason state;
- a non-native or fallback engine; and
- the complete successful matrix.

PromQL is parsed with `promtool`; the controlled fixture query check uses a temporary
Prometheus data directory and never contacts an external service. Validate the committed
fixture and alert files separately as part of the repository gate:

```bash
for fixture in tools/grafana/provisioning/fixtures/fak-native-*.prom; do
  promtool check metrics < "$fixture"
done
promtool check rules tools/grafana/native-performance-alerts.yml
promtool test rules internal/nativeperfslo/testdata/native-performance-alerts.test.yml
```

## What the visual witnesses prove

Existing Grafana captures remain the detailed dashboard witnesses for kernel, backend,
and SLO states. The aggregate populated and unavailable PNGs cover the complete final set,
including the artifacts dashboard:

- `tools/grafana/provisioning/witnesses/fak-native-panel-coverage-populated.png`
- `tools/grafana/provisioning/witnesses/fak-native-panel-coverage-unavailable.png`
- `tools/grafana/provisioning/witnesses/fak-native-panel-coverage-manifest.json`

The populated image is a visual index of every panel whose query produced a real
fixture-derived series in the controlled Prometheus evaluation; `or vector(-1)` is removed
for that populated assertion so the sentinel cannot make a row green. The unavailable
image is the matching coverage index for each panel's required unavailable path. The
detailed Grafana captures prove kernel, backend, and SLO UI states; the aggregate index
adds the artifacts dashboard and binds the complete four-dashboard set to the matrix. The
aggregate images are not pixel captures of Grafana, not screenshots of a live model run,
and not benchmark evidence.

## Fixture versus live evidence

| Evidence | Proves | Does not prove |
|---|---|---|
| Contract + fixture matrix | Query syntax, metric names, supervised jobs, bounded labels, panel coverage, and missing-data semantics | A model executed or a device produced the displayed values |
| Deterministic PNGs | Populated and explicit-unavailable presentation for the complete dashboard set | Grafana availability on a production host or current performance |
| Scrubbed live receipt | One exact Qwen3.8 request executed through fak-native without fallback | A comparative gain, SLO promotion, or general hardware capacity |

The live receipt location is
`tools/grafana/provisioning/witnesses/fak-native-qwen38-live-proof-unavailable.json`.
As of **August 28, 2026**, it honestly records `status: unavailable`: the local native
Metal route refused before load because its current streamed-Q4_K admission requirement
exceeded host memory, the sanctioned GCP probe lacked a refreshable credential, and the
private-lab bridge had no authorized control channel. No live Qwen3.8 result is claimed
from those attempts, and no private hostname, channel, credential, path, or raw log is
committed.

## Complete the live half on sanctioned compute

First select an eligible route using `docs/fleet-compute-nodes.md`. For the public cloud
route:

```bash
python3 tools/gcp_gpu_probe.py --all-tiers
python3 tools/gcp_bench.py \
  --dry-run \
  --tier <ELIGIBLE_TIER> \
  --engine fak-cuda-q8 \
  --hf-repo unsloth/Qwen3.8-27B-GGUF \
  --hf-file Qwen3.8-27B-Q4_K_M.gguf \
  --fak-ref <ISSUE_9817_COMMIT>
```

Continue only when the probe reports an authenticated tier with quota and the dry run
still names Qwen3.8 plus the fak CUDA engine. On an authorized private-lab route, follow
`docs/private-comms-channel.md` to reach the private command authority, run
`dgxbridge doctor`, then `dgxbridge doctor -probe`, and require a fixed-token readback
before dispatching the same native-only workload.

The compute-side serve and bounded request are:

```bash
FAK_Q4K=1 fak serve \
  --addr 127.0.0.1:8137 \
  --engine inkernel \
  --gguf qwen38:27b \
  --model qwen38:27b \
  --backend cuda \
  --context-budget-tokens 4096

curl -fsS http://127.0.0.1:8137/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen38:27b","max_tokens":8,"messages":[{"role":"user","content":"Reply with exactly: Q38"}]}'
```

Add the repository's normal authentication flag/header when the listener is not isolated
to loopback. Accept the live half only when readiness and `/v1/models` identify the exact
model, the response passes the fixed-output check, and the scrubbed receipt records:

- immutable Qwen3.8 artifact revision and SHA-256;
- `execution.engine: "fak-native"`;
- runtime/planner `inkernel`, model owner `fak`, and a native CUDA or Metal forward path;
- `fallback_count: 0`, `fallback_active: false`, and `llama_cpp_used: false`;
- bounded controls, positive wall time, token counts, and exact-output quality;
- unavailable memory/energy as `N/A` with a reason, never numeric zero; and
- no endpoint, hostname, credential, private path, prompt text, or raw log.

Replace the unavailable receipt with that scrubbed artifact and rerun the full matrix.
Until then, the deterministic Grafana contract is proven and the live dependency remains
explicitly pending.
