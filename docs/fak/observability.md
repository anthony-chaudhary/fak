---
title: "fak observability: metrics, logs, and traces"
description: "How fak serve exposes Prometheus metrics, JSON access logs, and trace_id correlation to audit a security gate without logging request bodies or tool arguments."
---

# Observability guide: metrics, logs, and traces

*Who this is for: operators wiring `fak serve` into a Prometheus/Grafana stack or a SIEM. Prerequisites: a running `fak serve` and comfort with curl and PromQL. You'll learn how to scrape its metrics, read the per-request access log, and follow one request across all three surfaces by `trace_id` — while keeping request bodies and tool arguments out of your telemetry.*

`fak serve` exposes three correlated observability surfaces, all on by default, all
designed so you can run a security gate in production **without ever logging request
bodies, tool arguments, or result content**. Every output block below was captured from a
clean build of `fak` v0.30.0.

| Surface | Route | Format | Use it for |
|---|---|---|---|
| **Metrics** | `GET /metrics` | Prometheus exposition | dashboards, alerts, SLOs |
| **Live snapshot** | `GET /debug/vars` | JSON | break-glass "what is this process doing right now" |
| **Access log** | stdout/log sink | one JSON object per line | per-request audit, incident forensics |

The thread tying them together is the **`trace_id`**: it appears in the response header
(`X-Trace-Id`), in the access-log line, and in the per-operation verdict log — so you can
follow one request across all three surfaces without it carrying any sensitive payload.

---

## 1. The access log (per-request audit trail)

Every HTTP request emits **two** structured JSON lines: one for the kernel operation
(the verdict) and one for the HTTP request (the transport). Real lines from a `read_file`
syscall followed by a denied `refund_payment` adjudication:

```json
{"event":"gateway_operation","operation":"syscall","tool":"read_file","verdict":"ALLOW","duration_ms":5.88,"trace_id":"gw-3"}
{"event":"gateway_http_request","method":"POST","path":"/v1/fak/syscall","route":"/v1/fak/syscall","status":200,"bytes":358,"duration_ms":5.88,"trace_id":"gw-3","user_agent":"curl/8.9.0"}
{"event":"gateway_operation","operation":"adjudicate","tool":"refund_payment","verdict":"DENY","reason":"DEFAULT_DENY","disposition":"TERMINAL","duration_ms":0.511,"trace_id":"gw-4"}
{"event":"gateway_http_request","method":"POST","path":"/v1/fak/adjudicate","route":"/v1/fak/adjudicate","status":200,"bytes":110,"duration_ms":0.861,"trace_id":"gw-4","user_agent":"curl/8.9.0"}
```

**What's in a line — and what's deliberately *not*:**

| Field | Present | Notes |
|---|---|---|
| `event` | ✅ | `gateway_operation` (the verdict) or `gateway_http_request` (the transport) |
| `tool` | ✅ | the tool *name* — never its arguments |
| `verdict` / `reason` / `disposition` | ✅ | the named decision (the audit-relevant part) |
| `duration_ms`, `status`, `bytes`, `route` | ✅ | latency / size / outcome |
| `trace_id` | ✅ | correlates all surfaces |
| request body, tool **arguments**, result **content** | ❌ **never** | a gate can be fully audited without logging the payload it's protecting |

This is the property that lets you ship the log to a central SIEM without it becoming a
secret-leak vector of its own.

### Trace-id propagation

`fak serve` honors an inbound `X-Trace-Id` header; when absent it **mints one**, returns it
in the `X-Trace-Id` response header, and threads it into the kernel operation. So a caller
that already has a request id keeps it end-to-end:

```sh
curl -si -H 'X-Trace-Id: my-req-42' http://127.0.0.1:8137/healthz | grep -i x-trace-id
# X-Trace-Id: my-req-42
```

---

## 2. Prometheus metrics (`GET /metrics`)

Standard Prometheus exposition. The `fak_gateway_*` families cover liveness, a fully
instrumented startup-phase breakdown, in-flight requests, build labels, and per-route HTTP
histograms. A real excerpt:

```
# HELP fak_gateway_up Whether the fak gateway process is scrapeable.
fak_gateway_up 1
# HELP fak_gateway_time_to_ready_seconds Total wall-clock from process start to the gateway becoming ready to serve (0 until ready).
fak_gateway_time_to_ready_seconds 0.0017293
# HELP fak_gateway_startup_phase_duration_seconds Wall-clock cost of each fak gateway boot phase (flag-parse, policy-load, planner-init, vdso-config, kernel-init, listener-bind, and model-load with --gguf).
fak_gateway_startup_phase_duration_seconds{phase="policy-load"} 0
fak_gateway_startup_phase_duration_seconds{phase="kernel-init"} 0
fak_gateway_startup_phase_duration_seconds{phase="listener-bind"} 0.0011925
# HELP fak_gateway_startup_unaccounted_seconds Boot wall-clock not explained by any named startup phase ...
fak_gateway_startup_unaccounted_seconds 0.0005368
# HELP fak_gateway_inflight_requests HTTP requests currently executing in the fak gateway.
fak_gateway_inflight_requests 1
# HELP fak_gateway_build_info Static fak gateway build and runtime labels.
fak_gateway_build_info{version="0.30.0",engine="inkernel",model="smollm2-inkernel",vdso="true"} 1
# HELP fak_gateway_http_requests_total HTTP requests served by route, method, and status.
fak_gateway_http_requests_total{route="/healthz",method="GET",status="200"} 1
fak_gateway_http_requests_total{route="/v1/fak/adjudicate",method="POST",status="200"} 1
fak_gateway_http_requests_total{route="/v1/fak/syscall",method="POST",status="200"} 1
# HELP fak_gateway_http_request_duration_seconds HTTP request latency by route, method, and status.
fak_gateway_http_request_duration_seconds_bucket{route="/healthz",method="GET",status="200",le="0.0001"} 1
fak_gateway_http_request_duration_seconds_bucket{route="/healthz",method="GET",status="200",le="0.001"} 1
...
```

### Metrics worth a dashboard panel

| Metric | Type | Why it matters |
|---|---|---|
| `fak_gateway_up` | gauge | liveness — alert on `== 0` or scrape failure |
| `fak_gateway_build_info{version,engine,model,vdso}` | gauge (labels) | what's actually deployed — pin in a "deployment" panel |
| `fak_gateway_http_requests_total{route,method,status}` | counter | request rate + error rate (`status=~"5.."`) per route |
| `fak_gateway_http_request_duration_seconds_bucket` | histogram | p50/p95/p99 latency per route (`histogram_quantile`) |
| `fak_gateway_inflight_requests` / `_max_age_seconds` | gauge | saturation + stuck-request detection |
| `fak_gateway_time_to_ready_seconds` / `_startup_phase_duration_seconds` | gauge | cold-start budget; which boot phase regressed |

The startup-phase breakdown is unusually granular: `flag-parse`, `policy-load`,
`planner-init`, `vdso-config`, `kernel-init`, `listener-bind`, and (with `--gguf`)
`model-load`, plus a `startup_unaccounted_seconds` that stays near zero when boot is fully
instrumented — so a cold-start regression points at the exact phase.

### Example PromQL

```promql
# error rate per route
sum by (route) (rate(fak_gateway_http_requests_total{status=~"5.."}[5m]))
  / sum by (route) (rate(fak_gateway_http_requests_total[5m]))

# p99 latency per route
histogram_quantile(0.99,
  sum by (route, le) (rate(fak_gateway_http_request_duration_seconds_bucket[5m])))

# alert: gateway down
fak_gateway_up == 0
```

---

## 3. The live JSON snapshot (`GET /debug/vars`)

`/debug/vars` is the same view as `/metrics` but as a single JSON object — the right tool
for a break-glass "what is this one process doing **right now**" probe. Real output (the
kernel-counter and runtime sections are the high-value parts):

```json
{
  "gateway": { "up": true, "version": "0.30.0", "engine": "inkernel", "model": "smollm2-inkernel",
               "vdso": true, "auth_required": false, "uptime_seconds": 4.40, "inflight_requests": 1 },
  "runtime": { "go_version": "go1.26.0", "goos": "windows", "goarch": "amd64",
               "num_cpu": 20, "gomaxprocs": 20, "num_goroutine": 26,
               "memory": { "alloc_bytes": 1483648, "heap_objects": 2739, "num_gc": 1, "...": "..." } },
  "kernel": { "submits": 1, "vdso_hits": 0, "engine_calls": 1, "denies": 0,
              "transforms": 0, "quarantines": 0, "admitted": 1, "vdso_hit_ratio": 0 },
  "metrics": { "http": [ { "route": "/healthz", "method": "GET", "status": "200",
                           "latency": { "count": 1, "sum_seconds": 0, "buckets": [ "..." ] } } ] }
}
```

The `kernel` block is the security-relevant heart: `submits`, `denies`, `transforms`,
`quarantines`, `admitted`, and `vdso_hit_ratio` are the running tally of *what the gate has
been doing* — how many calls it refused, redacted, or walled off, and how much of the load
the local fast-path served for free.

---

## 4. Grafana dashboards, scrape config & alerts

`fak serve` emits standard Prometheus exposition, so any Prometheus + Grafana stack can
scrape it. This repo ships a ready-to-run one under [`tools/grafana/`](https://github.com/anthony-chaudhary/fak/tree/main/tools/grafana):

- **Scrape config** — [`prometheus.yml`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/prometheus.yml) has a
  `fak_gateway` job that scrapes `/metrics` every 30s:

  ```yaml
  scrape_configs:
    - job_name: fak_gateway
      metrics_path: /metrics
      static_configs:
        - targets: ["host.docker.internal:8080"]   # your `fak serve --addr`
          labels: { service: fak-gateway }
  ```

  If the gateway runs with `--require-key-env`, add Prometheus auth for the job (e.g.
  `bearer_token_file`) or front it with a loopback-only scrape proxy.

- **Provisioned dashboards** — under
  [`tools/grafana/dashboards/`](https://github.com/anthony-chaudhary/fak/tree/main/tools/grafana/dashboards), generated from
  [`gen_dashboard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/gen_dashboard.py) (the maintainable source —
  re-run it when a metric name changes so no panel queries a phantom family):
  - `fak-gateway-observability.json` — up / build-info, request & error rate, p50/p95/p99
    latency per route, in-flight saturation.
  - `fak-startup-load.json` — `time_to_ready` and the per-phase boot breakdown.
  - `fak-dogfood-slow-requests.json` — slowest-route drill-down.

- **Alerts** — [`prometheus-alerts.yml`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/prometheus-alerts.yml)
  carries the rule set (e.g. gateway-down on `fak_gateway_up == 0`).

- **One command** — [`up.sh`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/up.sh) brings up Prometheus (`:9091`) +
  Grafana (`:3000`, `admin` / `fleet`) with the scrape config and dashboards
  pre-provisioned; `down.sh` tears it down (`--purge` also drops the data volumes).

### Retention and rotation

fak itself does **not** persist or rotate telemetry — it exposes the live surfaces above
and writes the access log to stdout / its log sink. Retention is owned by the systems you
point at it: set Prometheus TSDB retention with `--storage.tsdb.retention.time`, and
ship/rotate the JSON access log with your log collector (journald, Vector, Fluent Bit, or
a SIEM). Because no line carries request bodies or tool arguments (§1), long-retention
shipping never turns the log into a secret-leak vector of its own.

---

## 5. Securing the surfaces

`/metrics` and `/debug/vars` follow the gateway's auth policy. With `--require-key-env`
set, they require the same bearer token as the `/v1/fak/*` routes; `/healthz` stays
unauthenticated for liveness probes.

```sh
fak serve --addr 0.0.0.0:8080 --base-url … --model … \
  --policy floor.json \
  --require-key-env FAK_TOKEN          # now /metrics, /debug/vars, and /v1/fak/* need the token
```

In the `/debug/vars` snapshot, `gateway.auth_required` reflects whether auth is on — a quick
way to confirm a production process isn't accidentally serving its internals open. Network
hardening (bind address, reverse-proxy scrape isolation) is covered in
[security.md](security.md).

---

## See also

- [tutorial.md §2.5](tutorial.md) — seeing the access log in the guided first session.
- [security.md](security.md) — auth, network exposure, and the threat model for these surfaces.
- [server-config.md](server-config.md) — every `fak serve` flag and env var.
- [`tools/grafana/`](https://github.com/anthony-chaudhary/fak/tree/main/tools/grafana) — the ready-to-run Prometheus + Grafana stack (scrape config, provisioned dashboards, alert rules).
- [`fak/GETTING-STARTED.md` §3](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md) — the full route table.
