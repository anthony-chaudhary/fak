# Fleet Observability — Prometheus + Grafana

The **human + machine** time-series surface for fleet operations and the FAK
gateway. It scrapes:

- [`../fleet_bottleneck.py`](../fleet_bottleneck.py), which scores the
  Top 10 fleet bottlenecks (`FLEET-BOTTLENECK-MASTER-LIST.md` — internal master
  list, not published) from on-disk telemetry and emits `fleet_*` metrics.
- `fak serve`, which emits live gateway/kernel metrics (`fak_gateway_*`,
  `fak_kernel_*`) at `/metrics`.

This folder turns those metrics into provisioned Grafana dashboards + alert rules.

This is the fleet adaptation of the DGX-caching
`metrics-service/grafana` stack (internal companion — not published) — same
Prometheus-scrapes-`:9095/metrics` → Grafana shape, scoped to fleet signals.

## One command (recommended)

```bash
tools/grafana/up.sh        # brings up the whole stack; open http://localhost:3000 (admin / fleet)
tools/grafana/down.sh      # tears it all down  (--purge also drops the data volumes)
```

`up.sh` is idempotent — it starts (or adopts, if already healthy) all four pieces:

1. **the pure kernel** — `fak serve --engine inkernel` on real weights (the
   in-kernel model engine, **not** an upstream proxy). It auto-builds `fak`, and if
   no fak-format export exists it runs `fak/scripts/fetch-model.sh` once to fetch
   SmolLM2-135M. Point `FAK_MODEL_DIR` at a different export to override; pass
   `FAK_NO_GATEWAY=1` to skip the gateway and chart fleet metrics only.
2. **the fleet engine** — `fleet_bottleneck.py serve` (`fleet_*` metrics).
3. **Prometheus** (`:9091`) and **Grafana** (`:3000`) via `docker compose`. On
   macOS it finds the Docker Desktop CLI in the app bundle and starts the daemon if
   it is down.

> **Why the pure kernel, not a proxy.** The default model source is fak's own
> in-kernel engine (`--engine inkernel`) so the dashboards chart REAL adjudicated
> `fak_kernel_*` traffic through the kernel-owned KV cache. It is a **byte-level
> reference forward pass**, not an English chat surface (GETTING-STARTED.md §4b,
> issue #69). To proxy a chat-quality upstream instead, run `fak serve` with
> `--base-url`/`--api-key-env` yourself — that is a separate, non-kernel tier.

The manual steps below are still documented for piecemeal / CI / Linux-host use.

## Two ways to feed it

**A. Live (long-running engines):** the recommended path.

```bash
python tools/fleet_bottleneck.py serve --port 9095   # serves /metrics (+ HTML, JSON API)
./fak serve --addr 127.0.0.1:8080           # serves /metrics for gateway/kernel telemetry
cd tools/grafana && docker compose up -d
```

**B. Cron / textfile collector (no long-running server):** every
`report` / `json` / `prometheus` run also writes
`tools/_registry/fleet_bottleneck.prom`. Point a node_exporter textfile collector
at `_registry/`, or scrape a tiny static file server, and schedule:

```bash
python tools/fleet_bottleneck.py prometheus --out /var/lib/node_exporter/textfile/fleet_bottleneck.prom
```

## Quick start (manual / piecemeal)

Equivalent to `up.sh`, one piece per shell — useful for CI or a Linux host:

```bash
python tools/fleet_bottleneck.py serve --port 9095            # fleet_* metrics
export FAK_MODEL_DIR="$PWD/fak/internal/model/.cache/smollm2-135m"   # real weights (fetch-model.sh exports these)
go build -o fak ./cmd/fak && \
  ./fak serve --addr 0.0.0.0:8080 --engine inkernel --model smollm2-135m   # pure kernel + gateway/kernel metrics
cd tools/grafana && docker compose up -d                      # Prometheus :9091 + Grafana :3000
```

Bind the gateway to `0.0.0.0` (not `127.0.0.1`) so the Prometheus container can
reach it via `host.docker.internal`. The gateway is optional — omit it to chart
fleet metrics only; the **FAK Gateway Observability** dashboard simply shows no
data until `fak serve` is scrapeable.

Open [http://localhost:3000](http://localhost:3000), log in `admin` / `fleet`.
The **Fleet Bottleneck & Visibility** dashboard is the home dashboard; the
**FAK Gateway Observability**, **FAK Dogfood Slow Requests**, and **FAK Startup &
Model Load** dashboards appear alongside it and populate once Prometheus scrapes
`fak serve`.

The startup dashboard is the one-time boot timeline of a fak gateway: time-to-ready,
a bottleneck-first breakdown of every boot phase (flag-parse, policy-load,
planner-init, vdso-config, kernel-init, listener-bind, and model-load with `--gguf`),
the GGUF weight-load phases, and a **Boot unaccounted** tile that flags any boot
wall-clock the named phases do not explain — near-zero means startup is fully
instrumented. Populate the model-load panels with `fak serve --gguf MODEL.gguf`.

The dogfood dashboard is the focused debugging view for Claude Code sessions through
`fak serve` `/v1/messages`: p50/p95/p99 latency, status mix, in-flight requests,
slow-threshold rates, and kernel activity. It does not show raw request bodies; use
the dogfood launcher's default Claude API debug log (`/tmp/fak-claude.log` for
`--probe`) and the gateway log (`/tmp/fak-serve.log`) for request-level detail.

## How it works

```
fleet_sessions.py + session_audit.py            (telemetry on disk)
        │
        ▼
fleet_bottleneck.py  :9095/metrics   ──→  Prometheus :9091  ──→  Grafana :3000
   (render_prometheus, namespace fleet_)      (scrape + alerts)     (dashboard)

fak serve           :8080/metrics    ──→  Prometheus :9091  ──→  Grafana :3000
   (gateway/kernel counters)              (scrape + alerts)     (dashboard)
```

`render_prometheus` emits one HELP/TYPE per family with bounded cardinality
(per-bottleneck ≤10, per-account ≤ n_accounts, leaderboards ≤6). `fak serve`
emits bounded route/status/verdict labels plus kernel counters. Dashboard JSON is
generated by [`gen_dashboard.py`](gen_dashboard.py) so panel queries stay in
lock-step with emitted metric names (`fleet_bottleneck_test.py` asserts this).

## Ports

| Port | Service | Purpose |
|------|---------|---------|
| 8080 | `fak serve` | Gateway/API + Prometheus scrape target (host) |
| 9095 | fleet_bottleneck.py | Prometheus scrape target + HTML dashboard (host) |
| 9091 | Prometheus | Prometheus UI + API |
| 3000 | Grafana | Dashboard UI (localhost only) |

> These match the metrics-service stack's ports. If you run **both** stacks on one
> box, offset one set (e.g. Grafana `3001:3000`, Prometheus `9092:9091`) to avoid
> a collision. The two engines never share the same `:9095` source — fleet's is
> `fleet_bottleneck.py`, the GPU server project's is `metricsd`.

## Regenerate the dashboard

```bash
python tools/grafana/gen_dashboard.py     # rewrites dashboards/*.json
```

Edit `gen_dashboard.py` (not the JSON) when the metric set changes, then re-run.
It currently writes:

- `fleet-bottleneck-overview.json`
- `fak-gateway-observability.json`
- `fak-dogfood-slow-requests.json`
- `fak-startup-load.json`

## Alert rules

`prometheus-alerts.yml` covers the Top 10 by layer:

- **Availability** — engine unscrapeable, health CRITICAL, every account throttled
- **Account ceiling** — auth-blocked, throttle saturation
- **Recovery** — resume backlog, dead-crash surfacing backlog, recovery plumbing stale
- **Provider** — API-error stalls
- **Cost** — low cache reuse (token waste)
- **Observability** — stale telemetry, snapshot not refreshing
- **Generic Top-10** — any class crossing CRITICAL (≥80) or HIGH (≥55), with the
  class title/layer in the alert labels (new classes covered automatically)

It also covers the FAK gateway:

- **Availability** — gateway target down or `fak_gateway_up` not 1
- **Request path** — 5xx responses, high HTTP p95, high inflight requests
- **Kernel path** — high deny ratio, quarantine observed, high operation p95

If `fak serve` uses `--require-key-env`, add Prometheus bearer-token auth in
`prometheus.yml` or run a loopback scrape proxy; otherwise the gateway target will
correctly show as down.

## Linux / DGX (host networking)

Uncomment `network_mode: host` in `docker-compose.yml`, comment the
`ports`/`extra_hosts` blocks, and update:

- `prometheus.yml`: targets → `localhost:9095` and `localhost:8080`
- `provisioning/datasources/datasource.yml`: Prometheus URL → `http://localhost:9091`
