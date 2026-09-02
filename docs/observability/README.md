---
title: "fak observability route: choose the production signal"
description: "Operator index for choosing fak health, metrics, live-state, or trace-correlated logs by production question."
---

# Observability route

This page is for **operators answering a production question about a running
`fak serve` gateway**. It selects the narrowest shipped signal first, then routes
to the maintained field and query authority.

**Next action:** confirm that the gateway's metrics surface is reachable:

```bash
curl -s http://127.0.0.1:8080/metrics
```

A running gateway returns Prometheus exposition text. Change the host and port to
the deployed address; then choose the question below instead of treating one
signal as a complete diagnosis.

## Start with your own recent context health

```bash
fak session observe             # this workspace, active Codex profile, last 4 calendar days
fak session observe --json      # deterministic aggregate for jq, a cron job, or a dashboard
```

This local, offline command answers the first user question before the fleet telemetry below: did context compaction fire, how much resident context did it shed, and what was the daily input/fire pattern? It reads native Codex rollout counters, includes its inferred scope in the report, and labels resident shedding as occupancy reduction rather than estimated billing savings. Use `fak session compact-audit` for per-fire and regrowth detail.

## Choose by operator question

| Production question | Signal and command | Read the result as | Authority |
|---|---|---|---|
| **Is the process serving?** | `curl -s http://127.0.0.1:8080/healthz` | JSON with `"ok":true` proves the HTTP process is responding and names its engine and model. It is a liveness check, not an SLO or model-quality verdict. | [`fak serve` quickstart](../fak/server-quickstart.md) |
| **Is traffic healthy over time?** | `curl -s http://127.0.0.1:8080/metrics` | Prometheus counters and histograms answer request rate, status, latency, adjudication, cache, and in-kernel runtime questions. Use rates over a window for counters. | [Metrics, logs, and traces](../fak/observability.md#2-prometheus-metrics-get-metrics) |
| **What is this process doing now?** | `curl -s http://127.0.0.1:8080/debug/vars` | The JSON snapshot is break-glass live state for a single process. It complements, rather than replaces, time-series metrics. | [Live snapshot authority](../fak/observability.md#3-the-live-json-snapshot-get-debugvars) |
| **Why did one request get this result?** | Send or capture `X-Trace-Id`, then search the structured log sink for that exact `trace_id`. | The same id joins the response header, gateway operation verdict, and HTTP access record without carrying request bodies or tool arguments. | [Trace-id propagation](../fak/observability.md#trace-id-propagation) |
| **Which tool decisions are being denied?** | Query `fak_gateway_adjudications_total` by verdict and reason, then inspect matching verdict log records. | Metrics show the aggregate trend; structured logs provide per-operation tool, verdict, reason, disposition, duration, and trace id. | [Adjudication metrics](../fak/observability.md#metrics-worth-a-dashboard-panel) |
| **Is context reuse helping?** | Query the `fak_gateway_kv_prefix_*` family for in-kernel traffic. | The family separates eligible turns, real prefix hits, reused tokens, saved prefill work, and reset reasons. A configured cache alone is not a realized hit. | [KV-prefix reuse](../fak/observability.md#metrics-worth-a-dashboard-panel) |
| **Is the service in an incident state?** | Start with `/healthz`, then compare `/metrics`, logs, and `/debug/vars` for the same interval or trace. | Corroborated signals identify the symptom before a restart or configuration change. Continue with the recovery route for remediation. | [Operator recovery route](../operator/README.md#recovery-and-upgrade-order) |

## Shipped production surface

`fak serve` exposes the health endpoint, Prometheus metrics, a live expvar
snapshot, and structured stdout/log-sink records. Metrics, live state, and access
logging are on by default. The gateway does not put request bodies, tool
arguments, or result content in the documented telemetry records; `trace_id`
provides correlation instead.

The default local examples use `127.0.0.1:8080`. Deployment determines the real
address, authentication boundary, scrape path, retention, alert policy, and log
sink. The [server configuration guide](../fak/server-config.md) is the deployment
configuration authority; the running binary and its responses remain the
runtime authority.

## Signal authority and scope

Use [the detailed observability guide](../fak/observability.md) for metric names,
labels, PromQL examples, log fields, cardinality rules, and the correlation
walkthrough. Use [server troubleshooting](../fak/server-troubleshooting.md) only
after a signal identifies a symptom. The documents in this directory about
trajectory control and durable artifacts describe specialized evidence and
control surfaces; they do not replace the `fak serve` production telemetry
contract.

| Context | Operator meaning |
|---|---|
| **Mode** | This route applies to the networked `fak serve` gateway. Offline CLI proofs and contributor test output have different witnesses. |
| **Generation** | This is the current `gen/now` question-to-signal index over shipped endpoints and records. |
| **Lifecycle** | Health is immediate, `/debug/vars` is a live snapshot, metrics become meaningful over collection windows, and logs follow the configured retention lifecycle. |
| **Support** | The linked detailed guide and server configuration define the supported fields and controls. Dashboard, alert, retention, and SIEM policy belong to the operator's deployment. |
| **Runtime authority** | Check the deployed revision's `fak serve --help`, endpoint responses, and emitted records. A historical capture is evidence for its named revision, not a promise for every revision. |

For the broader deploy, observe, recover, and upgrade sequence, return to the
[operator route](../operator/README.md).

## Native-performance evidence

Use these pages for fak-native performance evidence. They keep the metric,
correlation, artifact, and dashboard boundaries separate:

- [Native-performance signal contract](native-performance-contract.md) defines
  the bounded metric inventory, label rules, provenance, freshness, and explicit
  exclusions.
- [Evidence correlation](native-performance-correlation.md) defines the opaque
  lookup key that joins a dashboard point to scrubbed run evidence without
  putting high-cardinality identities in metric labels.
- [Artifact index](native-performance-artifacts.md) defines the bounded artifact
  kinds, locator validation, digest requirements, and unavailable-evidence
  states.
- [Grafana operations runbook](native-performance-grafana-runbook.md) separates
  the currently shipped local stack from planned native panels and names the
  receipt/profile checks required before a performance claim.
- [Grafana panel proof](native-performance-live-proof.md) reproduces the
  deterministic panel-coverage matrix for the four native-performance
  dashboards and separates fixture success from the live fak-native Qwen3.8
  receipt a performance claim additionally requires.

## Trajectory interpretation and views

- [Trajectory interpretation and presentation](trajectory-presentation.md) separates canonical evidence, derived semantics, audience projection, rendering, and live controls.
