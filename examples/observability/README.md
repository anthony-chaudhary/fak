# Operator observability — `/metrics` + `/debug/vars` (zero dependencies)

The SRE walkthrough for the two operator surfaces `fak serve` exposes
([GETTING-STARTED.md §3](../../GETTING-STARTED.md) routes table):

- **`GET /metrics`** — Prometheus exposition: gateway HTTP latency/status, verdict
  counters, kernel counters (vDSO hits / quarantines / denies), inflight requests,
  build labels, and the vDSO hit ratio.
- **`GET /debug/vars`** — an authenticated expvar-style JSON snapshot for *break-glass*:
  config/uptime, runtime memory/goroutines, kernel counters, and the completed
  HTTP/operation metric rows — the live process view without standing up Prometheus.

This is the **runnable example** an operator points at to answer *"what do I monitor and
how"* — distinct from the Grafana dashboard artifacts ([#66](https://github.com/anthony-chaudhary/fak/issues/66)),
the observability *guide* ([#163](https://github.com/anthony-chaudhary/fak/issues/163)),
and the env-knob doc ([#199](https://github.com/anthony-chaudhary/fak/issues/199)).

```bash
./run.sh            # build fak, serve offline (mock engine), drive calls, scrape /metrics + /debug/vars
```

It needs **only a Go toolchain** (to build `fak`) plus `curl`; `jq` is optional (the
script falls back to `python3`, then to raw JSON). No model, API key, or GPU — `fak serve`
with no `--base-url` runs a deterministic offline mock engine. Captured run:
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## A note on metric names

The exposition names are the `fak_gateway_*` / `fak_kernel_*` forms below — the exact
strings asserted by the green test `internal/gateway/metrics_test.go`. Where this doc
phrases a question as a PromQL query, the metric in it is one of those real names (there
is no bare `fak_verdicts_total` / `fak_vdso_hit_ratio` — they are
`fak_gateway_operations_total` and `fak_gateway_vdso_hit_ratio`). Every query here appears
verbatim in the captured `/metrics` scrape, so you can copy them into Prometheus as-is.

---

## The four questions an operator asks

### 1. Is the floor firing? (are we denying / quarantining calls?)

Every adjudicated call is one `fak_gateway_operations_total` sample, labelled by
`operation` (`syscall` / `admit` / `adjudicate`), `verdict`, `reason`, `disposition`, and
`by` (which adjudicator decided — forensics). To watch the *deny* rate broken out by
reason:

```promql
sum by (reason) (rate(fak_gateway_operations_total{verdict="DENY"}[5m]))
```

Quarantines (a poisoned result the result-side floor paged out) are the same family with
`verdict="QUARANTINE"`:

```promql
sum by (reason) (rate(fak_gateway_operations_total{verdict="QUARANTINE"}[5m]))
```

The cheap aggregate kernel counters, if you just want totals without the label set:

```promql
fak_kernel_denies_total        # structural denials (POLICY_BLOCK, …)
fak_kernel_quarantines_total   # results contained by the result-side stack
```

A floor that never fires under real traffic is either perfectly-behaved agents or a
mis-scoped policy — either way, this is the panel that tells you which.

### 2. Is the cache hitting? (is the vDSO saving turns?)

The vDSO pools read-only tool results so an identical call across agents is served from
the pool instead of re-running. The ratio is a single gauge:

```promql
fak_gateway_vdso_hit_ratio     # vdso_hits / submits, 0..1
```

Backing counters, if you want to compute it yourself or alert on a drop:

```promql
fak_kernel_vdso_hits_total / fak_kernel_submits_total
```

A served-from-cache call also shows up in the verdict family with `by="vdso"` — so
`fak_gateway_operations_total{by="vdso"}` is the count of calls the cache answered.

### 3. Is anything stuck inflight?

The aggregate gauge is always present; the oldest-request age is the actual *stuck*
detector (a healthy gateway keeps this near zero):

```promql
fak_gateway_inflight_requests          # in-flight requests right now (all routes)
fak_gateway_inflight_max_age_seconds   # age of the OLDEST in-flight request — the wedge alarm
```

While requests are actually in flight, the gateway also breaks the count out per route, so
you can see *which* endpoint is backed up:

```promql
fak_gateway_inflight_requests_by_route   # gauge, one series per {route="…"} while in flight
```

> `fak_gateway_inflight_requests_by_route` is emitted only for routes with a live request
> at scrape time, so an idle scrape (like the captured one) shows just the aggregate gauge
> and the max-age gauge at 0. Alert on `fak_gateway_inflight_max_age_seconds` crossing your
> request-timeout — that is the "a request is wedged" signal.

### 4. Break-glass: the live process view as JSON

When something is wedged and you want the process state *now* — without standing up a
Prometheus scrape — hit `/debug/vars`. It is the same data Prometheus would compute, as a
single JSON object you can `jq` on the spot. This is the distinct use case: a one-off,
interactive probe, not a time series.

```bash
# the kernel counters block (denies / quarantines / vdso) at this instant
curl -s "$ADDR/debug/vars" | jq '.kernel'

# is anything inflight, and how long has the process been up?
curl -s "$ADDR/debug/vars" | jq '{inflight: .gateway.inflight_requests, uptime_s: .gateway.uptime_seconds}'

# runtime health — goroutine count + heap, the "is it leaking / wedged" probe
curl -s "$ADDR/debug/vars" | jq '{goroutines: .runtime.num_goroutine, alloc_bytes: .runtime.memory.alloc_bytes, num_gc: .runtime.memory.num_gc}'

# the completed operation rows, with which adjudicator decided each
curl -s "$ADDR/debug/vars" | jq '.metrics.operations[] | {operation, verdict, reason, by}'
```

`/debug/vars` carries `auth_required`, so the break-glass responder can confirm at a glance
whether the gateway is authenticated.

---

## Auth: both surfaces honor `--require-key-env`

`/metrics` and `/debug/vars` follow the gateway auth policy — only `/healthz` is ever
unauthenticated. Start the gateway with a required bearer token:

```bash
export FAK_GW_KEY="s3cret-operator-token"
fak serve --addr 127.0.0.1:8080 --require-key-env FAK_GW_KEY
```

Then scrapes must present it:

```bash
curl -s 127.0.0.1:8080/metrics                                     # -> 401 (no token)
curl -s -H "Authorization: Bearer $FAK_GW_KEY" 127.0.0.1:8080/metrics   # -> 200
```

For Prometheus, set the bearer token on the scrape job:

```yaml
scrape_configs:
  - job_name: fak
    authorization:
      type: Bearer
      credentials: s3cret-operator-token   # or credentials_file:
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

`run.sh` demonstrates the `401 -> 200` transition at the end of its run.

---

## Files

| File | What it is |
|---|---|
| [`run.sh`](run.sh) | Builds `fak`, serves it offline, drives allow/deny/quarantine + a vDSO hit, scrapes `/metrics`, fetches `/debug/vars`, demonstrates the auth path, tears down. |
| [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) | A captured run — the scraped metric lines and the `/debug/vars` JSON. |
| [`dashboard.json`](dashboard.json) | A minimal Grafana panel definition over the four questions. Supersede with the full dashboard from [#66](https://github.com/anthony-chaudhary/fak/issues/66) when it lands. |

## See also

- [`GETTING-STARTED.md` §3](../../GETTING-STARTED.md) — the full routes table and the
  access-log / `X-Trace-Id` correlation story.
- [`internal/gateway/metrics_test.go`](../../internal/gateway/metrics_test.go) — the green
  test that pins which labels the exposition emits.
- [`internal/gateway/http.go`](../../internal/gateway/http.go) — route registration and the
  `withAuth` gate.
