# Request timing burndown

Use the observation proxy and the offline reducer to rank time by backend, model,
and declared cache state:

```text
fak model-observe proxy --backend http://127.0.0.1:8080 --ledger observations.jsonl --cache-state warm
fak model-observe time-burndown --input observations.jsonl --format md
fak model-observe time-burndown --input observations.jsonl --format json
fak model-observe time-burndown --input observations.jsonl --format prom
```

The standalone `modelperfobs` binary exposes the same proxy and time-burndown
commands. The existing `report` command remains available.

Cache states are `cold`, `warm`, `steady`, and `unknown`. The proxy's
`--cache-state` is an operator declaration for that capture, not a measured
cache hit or a guarantee that disk pages are cold. Use separate controlled
captures when cache state changes. The default and legacy observations are
unknown. Mark priming captures with `--warmup`; the reducer counts and excludes
them. It never trims an arbitrary number of requests or infers warm from request
order. Error observations are counted separately and excluded from latency
medians and rankings.

Each successful request contributes its full observed duration to exactly one
API partition: `before_first_output` plus `after_first_output` when streaming
TTFT is available, otherwise `unpartitioned_request`. These names describe API
boundaries. Before-first-output can include network, queueing, loading, prefill,
and orchestration; after-first-output can include buffering and response tails.
The report does not identify those partitions as measured prefill or decode.
Cumulative percentages use the sum of request durations in each cohort.
Overlapping requests therefore contribute request time, not elapsed process time
or a measured critical path.

API-only reports expose
`backend-does-not-expose-dispatch-attribution` and leave Active/Wait
unavailable. Unknown attribution never becomes zero device work.
Cohort medians and sample counts are descriptive statistics; this command does
not compute A/B confidence intervals, certify ambient cleanliness, or qualify
hardware. Retain the existing native experiment controls for such claims.

## Join native phase evidence

`--phases phases.jsonl` accepts one JSON envelope per request:

```json
{
  "schema": "fak-request-phases/1",
  "request_id": "exact-observation-request-id",
  "model": "exact-observation-model",
  "backend": "exact-observation-backend",
  "receipt": {
    "engine": "inkernel",
    "backend": "cpu",
    "forward_path": "native",
    "fallback_active": false,
    "wall": 1000000,
    "phases": []
  }
}
```

This envelope illustrates identity and units; the empty phases array is
deliberately incomplete and is rejected. A valid receipt must contain the
existing complete fourteen-phase contract with inclusive and exclusive timing.
Durations inside the native receipt are integer nanoseconds. The reducer invokes
`nativeperf.PhaseReceipt.Validate`, requires an exact request/model/backend
join, rejects duplicates, unmatched receipts, fallback receipts, and native wall
time exceeding observed API duration, and ranks only exclusive phase durations.
Native phase percentages use native wall time. Joined API wall minus native
wall is reported separately as outside-native time; it is not assigned to an
invented host or network phase. Coverage states how many successful requests
have native evidence. Supplying an envelope proves a validated artifact join;
it does not establish that a live runtime emits these receipts.

JSON and Markdown retain cohort identities for local analysis. Treat observation
ledgers and reports as local diagnostic data: backend URLs can contain sensitive
configuration. Prometheus output aggregates across model/backend identities and
uses bounded cache-state, phase, plane, and timing-kind labels only. It is an offline
snapshot, with gauge semantics; this change adds no running exporter.

## Next integration steps and acceptance

1. Emit the native phase envelope on the ordinary serving request path, using
   the proxy's observation ID. Acceptance: a real request produces an exactly
   matched, validated receipt without fallback, with measured emitter overhead
   and a no-emitter compatibility arm.
2. Join tool, provider, and router spans by that same request/run identity.
   Acceptance: measured exclusive spans reconcile within their own clock domain;
   overlapping background work stays separate and unavailable spans stay unknown.
3. Attach existing profile counters to the corresponding measured native phase.
   Acceptance: counter scope, sampling window, backend support, and sustained
   roofline share the same request/phase envelope; unsupported backends retain
   their reason. Select kernel work only from that evidence.
4. Use the existing repeated experiment controls to compare separate warm/cold
   arms, explicit warmups, sampler overhead, ambient state, and uncertainty.
   Acceptance: identical model/workload envelopes and retained per-repetition
   observations support each claimed improvement.

These steps extend current receipt owners. The reducer supplies the joined
reader and missing-evidence view; live emitters, tool spans, counter joins, and
hardware performance gains require their own witnesses.
