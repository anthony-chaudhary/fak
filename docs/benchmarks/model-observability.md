# Model inference observability spine

`modelperfobs` turns any OpenAI-compatible model server into request-level,
queryable evidence without requiring a patched backend. It is the shortest path
from an agent workload to the three latency dimensions that distinguish a slow
prompt/queue from slow token generation:

- end-to-end latency;
- time to first streamed token (TTFT);
- time per output token (TPOT), output tok/s, and inter-chunk p50/p95.

It also records prompt/completion token counts, model, HTTP status, errors, and a
correlation ID in append-only JSONL. The proxy sends that ID upstream and returns
it as `X-Fak-Observation-ID`, so backend logs and agent outcomes can join to the
same request.

## Working spine

Start an OpenAI-compatible backend such as Qwen3.8-27B, then place the proxy in
front of it:

```bash
fak model-observe proxy \
  --backend http://127.0.0.1:8000 \
  --listen 127.0.0.1:8091 \
  --ledger _scratch/qwen38/model-perf.jsonl
```

Point the harness's OpenAI base URL at `http://127.0.0.1:8091`. Preserve
`stream: true`; for exact token rates, ask the backend for streaming usage
(`stream_options: {"include_usage": true}`). Then rank the likely bottleneck:

```bash
fak model-observe report \
  --input _scratch/qwen38/model-perf.jsonl \
  --format md
```

## Cache-state transition benchmark

Run the hermetic cache-state spine before accepting a cold/warm comparison:

```bash
fak model-observe cache-state-bench \
  --output _scratch/modelperfobs/cache-state.json
fak model-observe cache-state-bench \
  --verify _scratch/modelperfobs/cache-state.json
```

The report keeps each arm result beside its transition receipt. A receipt names
the backend identity, target layer, mechanism, start/end time, pre/post metric
snapshots, pinned-prefix probes, and the proved, unproved, failed, or unsupported
result. The runner excludes any arm whose reset exits successfully but still
reuses the pinned prefix. It also rejects stale samples, counter resets, backend
identity changes, and request-count or overlap evidence of concurrent traffic.

The built-in backend is deliberately narrow: it observes the running in-process
fak workflow cache and proves cold start, warm admission, explicit invalidation,
and capacity-pressure eviction there. Its provenance sets
`external_backend_claims` to `false`; it does not turn that local observation
into a claim about process-local model KV, shared KV, or a provider prompt cache.
Those layers, plus natural expiry, are typed by the contract but remain
`unsupported` until a backend adapter supplies a safe mechanism and fresh
counters. This is also why the
captured witness is reproducible without a model key, network, or GPU.

The JSONL is the query contract, not a dashboard-specific format. Example with
DuckDB (no import step):

```sql
SELECT model,
       count(*) AS requests,
       quantile_cont(ttft_ms, 0.95) AS ttft_p95_ms,
       quantile_cont(tpot_ms, 0.95) AS tpot_p95_ms,
       quantile_cont(output_tokens_per_second, 0.5) AS output_tok_s_p50
FROM read_json_auto('_scratch/qwen38/model-perf.jsonl')
GROUP BY model;
```

## Reading the signal

- High TTFT with modest TPOT: sweep prompt length at concurrency 1, then sweep
  concurrency at fixed prompt length. The first implicates prefill; the second
  queueing/scheduling.
- High TPOT or low output tok/s: inspect device residency, memory bandwidth,
  quantized kernels, and batch shape.
- Healthy request metrics but poor agent wall time: join observation IDs to
  task outcomes; tool latency, retries, prompt growth, or excess generated tokens
  dominate instead of inference.
- Missing TTFT/TPOT: the workload did not stream. Aggregate duration cannot
  identify the bottleneck, so the report says `missing-stream-timing` rather
  than inventing a diagnosis.

## NVIDIA HBM counter profile

For a deep NVIDIA profile, collect the two cumulative DRAM-byte counters and
kernel duration from a single device. Raw output, base units, and metric names
are part of the import contract:

```bash
ncu --csv --page raw --print-units base --print-metric-name name \
  --devices 0 \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
  --log-file _scratch/modelperfobs/nvidia-hbm-ncu.csv \
  fak <the same fak-native workload and arguments used by the benchmark>
```

Record the profile window at capture time, then import it rather than using the
later file-parse time:

```bash
fak model-observe bandwidth collect \
  --nvidia-ncu-csv _scratch/modelperfobs/nvidia-hbm-ncu.csv \
  --device "NVIDIA H100 80GB HBM3 (0)" \
  --capture-start 2026-08-27T10:00:00Z \
  --capture-end 2026-08-27T10:01:00Z \
  --phase decode --shape large \
  --theoretical-gb-s 3350 \
  --device-roofline-gb-s 3100 \
  --output _scratch/modelperfobs/nvidia-hbm.json
```

The importer groups rows by launch ID, requires one
`dram__bytes_read.sum`, `dram__bytes_write.sum`, and
`gpu__time_duration.sum` base-unit value per launch, then divides cumulative
bytes by cumulative nanoseconds. One byte per nanosecond is one decimal GB/s.
Missing or duplicate metrics, mixed processes, mixed hosts, and mixed devices
fail closed.
`N/A` and `[Not Supported]` make only the affected direction unavailable; total
bandwidth and utilization remain unavailable unless both directions exist.
Unavailable values are omitted from the JSON rather than serialized as zero.
The CSV `Device` column is a profiler
device ID on current Nsight Compute versions; the receipt keeps it separate
from the operator-declared `--device` label. When an older CSV lacks that
column, the label remains explicitly operator-declared provenance. Capture one
device with `--devices` and retain the `.ncu-rep` if an independently
inspectable device identity is required.

`--device-roofline-gb-s` is reserved for a matched, measured roofline from the
same NVIDIA device and operating envelope. The host-memory `--measured-gb-s`
flag is rejected in profile-import mode so a CPU copy result cannot become an
HBM utilization denominator. Host sampling, token, latency, software-byte, and
interval flags are also rejected rather than silently ignored by the importer.

This is profiled-kernel active-time bandwidth, not request-wall-time throughput.
Nsight Compute may replay kernels and has substantial profiling overhead, so
pair the counter capture with an otherwise matched uninstrumented latency and
throughput run. The CSV identifies a process and counters; it does not prove
that the process used the fak-native engine. The receipt therefore records
`engine=fak-native` as operator-asserted, not CSV-proven. In particular,
`nvidia-smi utilization.memory` remains memory-controller active-time and is
never multiplied by a roofline or placed in `LiveBandwidth`.

## Metric semantics

The names follow the request-level practice documented by vLLM's metrics design:
TTFT, inter-token latency, prompt tokens, and generation tokens are the SLO-facing
metrics, while engine counters can provide supporting evidence. This spine starts
at the cross-backend request seam; an optional adapter can join queue, cache,
eviction, and preemption counters through `modelperfobs.AttributeMetrics` without
changing the proxy receipt format.

### Causal limits of server counters

A before/after server counter is shared evidence, not a request fact. Its delta
includes every observed and unobserved request served by that server instance
between the two scrape timestamps. A cache hit, queue event, eviction, or
preemption therefore reaches a request report only when either:

- a backend request label or trace names that request; or
- the server request counter proves the scrape window contains exactly one
  observed request and no background request.

The attribution report preserves the server-instance ID, scrape bounds,
overlapping request IDs and count, background-request count when known, and each
counter's reset or wrap state. Its grades are `request-correlated`,
`isolated-window`, `cohort-only`, `contaminated`, `stale`, and `unavailable`.
Unlabeled deltas from overlapping requests remain visible once in the cohort
report; they are not copied into every request. An unknown background count does
not prove isolation. Scrape failure, server restart, stale data, a counter reset,
or an unrecognized correlation source fails closed instead of manufacturing a
request-level cause. A generic Prometheus adapter may omit request labels and
still retain honest cohort evidence; distributed tracing is optional.

## Capacity-valid serving sweeps (fak.serving-sweep.v1)

Multi-concurrency serving sweeps evaluate throughput saturation curves and p99 SLA boundaries across concurrency points.
The receipt schema `fak.serving-sweep.v1` governs multi-point sweeps, while single-concurrency point evaluations continue to use `fak.serving-parity.v1`.

### Key definitions

- **Workload digest**: Deterministic cryptographic hash of the task dataset, prompt prefix, and generation parameters across every concurrency point.
- **Engine receipt digest**: Digest proving the exact backend engine identity, ensuring no silent engine substitution or mid-sweep drift occurs.
- **Capacity source**: Provenance label (e.g., `declared-manifest`, `probe-bench`) asserting the maximum concurrent batch capacity.
- **Valid point**: A measured concurrency point whose concurrency does not exceed declared batch capacity, whose engine receipt matches the sweep identity, and whose output quality passes verification.
- **Peak**: The maximum aggregate output tokens per second observed strictly among valid points.
- **p99-SLA knee**: The highest concurrency point that simultaneously satisfies optional p99 TTFT (`--ttft-p99-budget-ms`) and p99 ITL (`--itl-p99-budget-ms`) service-level budgets.

### Claim boundaries and invalidation rules

To prevent misleading saturation or capacity claims, the following rules are strictly enforced:
- **Unknown capacity**: A sweep with undeclared or unmeasured capacity (`capacity <= 0` or missing) cannot support a peak or SLA knee claim.
- **Above-capacity load**: Concurrency points exceeding declared batch capacity are flagged as invalid and cannot be selected as peak.
- **Identity drift**: Any change in model, engine, or workload digest between points invalidates the entire sweep.
- **Point sufficiency**: Fewer than two valid points cannot establish a saturation curve or support a peak/knee claim.

Source studied 2026-08-21: [vLLM metrics design](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md).
