---
title: "Dynamo interop: fak governs, Dynamo routes"
description: "Issue #38 decision and bring-up path for running fak in front of a Dynamo-managed prefill/decode worker pool."
---

# Dynamo interop: fak governs, Dynamo routes

Issue #38 is resolved by the **fak-governs / Dynamo-routes** posture.

`internal/engine.DynamoEngine` registers the `dynamo` EngineDriver. It dispatches through
Dynamo's public OpenAI-compatible frontend (`FAK_DYNAMO_BASE_URL`, usually `/v1`) and
observes the pool through Dynamo's public Prometheus scrape (`FAK_DYNAMO_METRICS_URL` or
`<base>/metrics`). Dynamo keeps ownership of P/D routing, worker lifecycle, KVBM/NIXL
movement, and planner decisions. fak stays in front as the governance/adjudication plane and
normalizes Dynamo worker signals into `fak_serving_*`.

## Why this posture

Routing directly to Dynamo-managed workers would duplicate the component Dynamo already
optimizes: KV-aware placement over prefill/decode pools. The existing measured serving
evidence in this repo also says the gateway tax is real and must be kept simple: fak-in-front
of a raw SGLang serve trails raw SGLang at peak concurrency, while the in-process admission
and adjudication floor remains sub-millisecond. Adding a second fak-owned P/D router in front
of Dynamo would add another placement layer without a witnessed locality win.

The locality measurement for Dynamo itself is intentionally not fabricated here. The shipped
stub witness proves the interop contract that is host-independent:

```powershell
go test ./internal/engine -run Dynamo
```

That test drives a faithful OpenAI-compatible Dynamo frontend stub and a P/D Prometheus
fixture with per-worker prefill/decode labels, active decode blocks, queued prefill tokens,
in-flight load, and prefix/KV hit-rate samples. A live Dynamo A/B latency/locality benchmark
belongs to the parity harness once a Dynamo pool is reachable.

## Bring-up

1. Start Dynamo in standalone/disaggregated mode using its public deployment flow.
2. Point fak at Dynamo's frontend and metrics scrape:

```powershell
$env:FAK_DYNAMO_BASE_URL="http://dynamo-frontend:8000/v1"
$env:FAK_DYNAMO_METRICS_URL="http://dynamo-frontend:8000/metrics"
$env:FAK_DYNAMO_MODEL="served-model"
fak run --trace trace.json --engine dynamo
```

3. Scrape the normalized metrics from the adapter or from the host that attaches it. Dynamo
metrics are relabeled into `fak_serving_*` with `engine="dynamo"` and per-worker labels. P/D
role appears on the worker-load gauges:

- `fak_serving_worker_active_decode_blocks{engine="dynamo",worker="decode-0",role="decode"}`
- `fak_serving_worker_active_prefill_tokens{engine="dynamo",worker="prefill-0",role="prefill"}`
- `fak_serving_requests_running{engine="dynamo",worker="decode-0"}`
- `fak_serving_requests_waiting{engine="dynamo",worker="prefill-0"}`

## Trust boundary

No Dynamo source is vendored or forked. The adapter uses only the public frontend and metrics
surface. Because Dynamo owns routing and KV movement, fak can still adjudicate the request
before it reaches the pool and can observe role/load/KV signals after the fact. It does not
move KV bytes and does not claim exact-span eviction inside Dynamo. Exact-span quarantine
remains a fak-owned/native-KV guarantee; on this ridden engine path it degrades to the
coarser control the deployment exposes.
