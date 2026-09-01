# Fleet Observability — Prometheus + Grafana

The **human + machine** time-series surface for fleet operations and the FAK
gateway. It scrapes:

- [`../fleet_bottleneck.py`](../fleet_bottleneck.py), which scores the
  Top 10 fleet bottlenecks (`FLEET-BOTTLENECK-MASTER-LIST.md` — internal master
  list, not published) from on-disk telemetry and emits `fleet_*` metrics.
- `fak serve`, which emits live gateway/kernel metrics (`fak_gateway_*`,
  `fak_kernel_*`) at `/metrics`.
- `fak cachevalue metrics --serve`, which re-folds the nightrun cache-value ledgers
  and the offline `fak ablate` arms on each scrape and emits the cache-value roll-up
  (`fak_cachevalue_*`, `fak_ablation_*`) at `/metrics`.
- `fak fleet metrics --serve`, which re-folds the live session registry, durable
  root-run registrations, and the gateway-usage ledger on each scrape and emits the **fleet-first** families
  (`fak_fleet_*`) at `/metrics` — the only source carrying a per-`session` label, and
  therefore the one that makes a fleet roll-up drill down to a named session.

This folder turns those metrics into provisioned Grafana dashboards + alert rules.

This is the fleet adaptation of the GPU server-caching
`metrics-service/grafana` stack (internal companion — not published) — same
Prometheus-scrapes-`:9095/metrics` → Grafana shape, scoped to fleet signals.

## One command (recommended)

```bash
tools/grafana/up.sh        # brings up the whole stack; open http://localhost:3000 (admin / fleet)
tools/grafana/down.sh      # stops only processes this stack owns (--purge drops Docker volumes)
```

`up.sh` is idempotent — it starts (or adopts, if already healthy) all the pieces:

1. **the pure kernel** — `fak serve --engine inkernel` on real weights (the
   in-kernel model engine, **not** an upstream proxy). It auto-builds `fak`, and if
   no fak-format export exists it runs `fak/scripts/fetch-model.sh` once to fetch
   SmolLM2-135M. Point `FAK_MODEL_DIR` at a different export to override; pass
   `FAK_NO_GATEWAY=1` to skip the gateway and chart fleet metrics only.
2. **the fleet engine** — `fleet_bottleneck.py serve` (`fleet_*` metrics).
3. **the cache-value exporter** — `fak cachevalue metrics --serve` on `:9097`
   (`fak_cachevalue_*`, `fak_ablation_*`). It reuses the `fak` binary built for the
   gateway (no weights needed) and is skipped under `FAK_NO_GATEWAY=1`. Its panels
   read *No data* until the `docs/nightrun/*.jsonl` ledgers exist — run
   `fak cachevalue feed --once` (or the nightrun) to populate them.
4. **the fleet-session exporter** — `fak fleet metrics --serve` on `:9098`
   (`fak_fleet_*`). It needs no gateway, no weights and no network — it folds the
   live session registry, durable run registrations, and gateway-usage ledger straight
   off disk — so it
   starts even under `FAK_NO_GATEWAY=1`, which is the "chart fleet metrics only" mode
   this exporter most belongs to.
5. **Prometheus** (`127.0.0.1:9091`) and **Grafana** (`127.0.0.1:3000`). Docker
   Compose remains the default. On macOS, `up.sh` first finds the Docker Desktop
   CLI in the app bundle and starts its daemon if needed; only when Docker remains
   unavailable does it use locally installed Homebrew Prometheus and Grafana.

### macOS without Docker

Install the native services once:

```bash
brew install prometheus grafana
tools/grafana/up.sh
```

The fallback runs the formula binaries directly rather than using `brew services`,
so ownership stays attributable to this invocation. It generates all runtime state
under the ignored `tools/grafana/.run/native/` directory:

- Prometheus scrape targets are rewritten from `host.docker.internal` to
  `127.0.0.1`, and the container-only Alertmanager dependency is omitted.
- Grafana receives a generated Prometheus datasource at
  `http://127.0.0.1:9091` and provisions the tracked `dashboards/` directory,
  including the bundled native-performance dashboards.
- Prometheus binds only `127.0.0.1:9091`; Grafana binds only
  `127.0.0.1:3000`.

Healthy services already on those endpoints are adopted, not restarted. `up.sh`
records only the supervisor processes it actually launches, with an invocation
token checked against the live command line. `down.sh` therefore leaves adopted,
stale-PID, and ownership-mismatched processes running. Native data remains under
`.run/native/`; `--purge` retains its existing Docker-volume meaning.

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

> **Remote gateway (e.g. the always-on Mac).** The `fak_gateway` scrape job target
> need not be localhost — point it at any reachable `fak serve` over the tailnet,
> for example `node-macos-a.local:8080` in `prometheus.yml`. This is how the
> `claude-mac-fak --overlay` / preflight-panel workflow
> ([`docs/fak/mac-agent-ui.md`](../../docs/fak/mac-agent-ui.md)) charts the Mac
> gateway. If that gateway runs with `--require-key-env`, add Prometheus bearer
> auth for the job (the `prometheus.yml` comment shows where).

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

Open `http://localhost:3000`, log in `admin` / `fleet`.
The **FAK Run Operations** dashboard (stable uid `fak-fleet-overview`) is the default
home; the
**FAK Gateway Observability**, **FAK Dogfood Slow Requests**, **FAK Startup &
Model Load**, and **FAK Guard — Kernel Adjudication** dashboards appear alongside it
and populate once Prometheus scrapes `fak serve` (or, for the guard view, a
`fak guard --addr 127.0.0.1:8080` session — see below).

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

The **FAK Guard — Kernel Adjudication** dashboard is the live, in-Grafana twin of the
summary `fak guard` prints when the wrapped agent exits. It foregrounds the
adjudication-referee story the gateway dashboard does not: the verdict mix
(allowed/denied/repaired/quarantined), the trust floor firing by the closed refusal
vocabulary (`internal/abi/reasons.go` — `TRUST_VIOLATION`, `SECRET_EXFIL`,
`SELF_MODIFY`, `POLICY_BLOCK`, `DEFAULT_DENY`, …), the NET provider-cache economics the
kernel hop preserved (`fak_vcache_*`), and the history-compaction / tool-floor token
shedding. `fak guard` binds the **same** gateway `fak serve` does and serves the same
`/metrics`, so the dashboard introduces no new metric — it is a re-framing of the
gateway families around a single guarded session.

**Scrape a guard session.** Guard binds a private OS-picked loopback port by default
and tears it down when the agent exits, so to chart a session, pin the port the
Prometheus `fak_gateway` job already targets:

```bash
fak guard --addr 127.0.0.1:8080 -- claude
```

Now the existing `fak_gateway` scrape job (`host.docker.internal:8080`) reads guard's
live `/metrics` and the dashboard populates. (`fak serve` and a pinned `fak guard`
share the one port, so run one at a time.) Grafana shows the verdicts and savings; it
never shows raw request bodies — the durable, hash-chained **decision journal**
(`guard-audit.jsonl`, on by default) is the tamper-evident record of the same verdicts,
replayable with `fak audit verify <path>`, and `--log FILE` emits the structured
per-request / per-verdict stream.

The **FAK Cache Value — Roll-up & Ablation** dashboard is the live Grafana twin of the
`fak cachevalue feed` Slack card, answering one question: *is fak's cache work paying
off, and which feature earned it?* It keeps the report's honesty fence intact across two
tracks that are shown side by side and **never blended**: *Track 1* is the **WITNESSED**
realized KV-prefix reuse (`fak_cachevalue_latest_reuse_ratio`, session counts); *Track 2*
is an **OBSERVED / projected-$** P&L (every `_usd` / `saved_token_equiv` family), split by
an `owner` label (`provider` prompt-cache vs `fak`-authored vs `total`) so a provider-only
corpus can never read as a fak win. Alongside them, the offline `fak ablate` arms
(`fak_ablation_*`) attribute the speedup to a single toggled feature by replaying one
frozen trace per arm — a deterministic, $0 measurement, not a live provider number.
Unlike the other dashboards it does **not** read `fak serve`; scrape the dedicated
`fak_cachevalue` job instead (`fak cachevalue metrics --serve --addr 127.0.0.1:9097`,
folded from `docs/nightrun/*.jsonl` + `experiments/ablate/*.json`).

## Run Operations: home → run drill-down

The **FAK Run Operations** home (uid `fak-fleet-overview`) and **FAK Run Operations —
Run Drill-down** (uid `fak-fleet-session`) are one surface at two zoom levels. The UIDs
stay stable so saved links continue to work after the operator-facing rename.

Run Operations starts with source readiness, then keeps two inventories visibly separate:
**Open / live runs** come only from the current descriptor registry, while **Completed runs**
come only from durable root registrations. The exporter reads the default registration ledger
automatically; `--registration-ledger PATH` selects a different durable store. Launch kind,
runtime, lifecycle state, outcome, and `source=durable_registration` stay on historical rows.

> *Which sessions are alive right now, and what is each one costing?*

Every other dashboard here is either **aggregate** (the `fleet_*` scores are fleet-wide
scalars plus ≤6-entry leaderboards) or **single-process** (the `fak_gateway_*` families
describe whichever process Prometheus happens to scrape). Neither carries a per-session
dimension, so an operator could see the fleet was slow and never see *which* session
was the reason. `fak fleet metrics` adds that axis:

```bash
./fak fleet metrics --serve --addr 127.0.0.1:9098     # live; re-folds per scrape
./fak fleet metrics                                   # one-shot exposition to stdout
./fak fleet metrics --textfile fleet.prom             # for a textfile collector
./fak fleet metrics --fleet                           # also fold PEER nodes' C2 refs
```

Start on Run Operations: the readiness row distinguishes a down exporter from an unreadable
live or registration store. The live table includes every non-terminal descriptor state
(running, stalled, throttled, paused, and draining). The completed table retains terminal
root registrations after their processes exit. Clicking either table opens the run drill-down
with the time range preserved; its selector unions live session IDs with the session IDs retained
on durable run registrations.

**Three projections, never blended** — the same honesty fence `fak info` renders:

| Family | Tier | Source | Read it as |
|---|---|---|---|
| `fak_fleet_sessions*`, `fak_fleet_session_*` | **LIVE** | the durable session registry, via an oracle (PCB state, heartbeat freshness, resume's idle-vs-TTL posture) | *who is alive* — the same fold `fak session ls --durable` prints, so the dashboard and the CLI can never disagree |
| `fak_fleet_run_*`, `fak_fleet_registered_runs*` | **DURABLE RUN HISTORY** | root records in the session-registration ledger | *what ran and how it ended* — launch/runtime/source are recorded, never inferred from process liveness |
| `fak_fleet_usage_*` | **HISTORICAL USAGE** | the append-only gateway-usage ledger | *what it cost* — OBSERVED (provider-relayed) tokens, except `kv_prefix_*` which fak authored itself |

Three panels exist purely so the dashboard cannot lie to you:

- **Live inventory readable / Run history readable** — 0 means the exporter could not
  read that source. Every corresponding panel then shows zero. Without this tile,
  "no sessions" and "could not look" are the
  same picture.
- **Per-session rows dropped** — the `--max-sessions` cardinality bound confessing. A
  truncated table that looked complete would be worse than no table.
- **Drill-down coverage** — the ledger's `session_id` is optional, so a per-session cost
  panel can cover a sliver of the fleet while looking whole. `fak guard` rows carry an
  id (one process, one agent session); a `fak serve` row carries one **only when that
  process hosted exactly one session**, because a multiplexed gateway's exit counters are
  a process total and stamping one of its traces would report the whole process's spend
  as that session's; and a non-durable guard launch's shared `guard` sentinel is dropped
  rather than collapsing thousands of unrelated sessions into one fake series. Each case
  that declines to stamp lands in the unidentified census instead, which is what this
  tile divides by — an honestly absent id, never a wrong one that reads as right.

Historical usage reaches back only as far as the usage ledger: a session that has not exited
yet has no usage row. Durable run history is independent and retains terminal registrations
even without a usage row. The live tier's time series is only what Prometheus sampled; it is
not a substitute for completed-run registration history.

## How it works

```
fleet_sessions.py + session_audit.py            (telemetry on disk)
        │
        ▼
fleet_bottleneck.py  :9095/metrics   ──→  Prometheus :9091  ──→  Grafana :3000
   (render_prometheus, namespace fleet_)      (scrape + alerts)     (dashboard)

fak serve           :8080/metrics    ──→  Prometheus :9091  ──→  Grafana :3000
   (gateway/kernel counters)              (scrape + alerts)     (dashboard)

live session registry ─────┐
run registration ledger ───┼→ fak fleet metrics :9098/metrics ──→ Prometheus ──→ Grafana
gateway-usage.jsonl ────────┘     (namespace fak_fleet_)        (Run Operations → drill-down)
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
| 9097 | `fak cachevalue metrics --serve` | Cache-value roll-up + ablation exposition (host) |
| 9098 | `fak fleet metrics --serve` | Live fleet-session inventory + roll-up exposition (host) |
| 9091 | Prometheus | Prometheus UI + API |
| 9093 | Alertmanager | Docker-only alert routing UI + API (POSTs the webhook) |
| 9096 | `fak slack alert --serve` | Webhook receiver → durable Slack outbox (host) |
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
- `fak-guard-adjudication.json`
- `fak-cache-value-rollup.json`
- `fak-fleet-overview.json`
- `fak-fleet-session.json`

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

And the fleet sessions (`fak_fleet_sessions` group):

- **Stalled sessions** — one or more sessions claiming work without a heartbeat for 15m
  (`FakFleetSessionsStalled`), and the fleet-wide variant where *every* session is
  stalled (`FakFleetAllSessionsStalled`, critical — suspect an account ceiling or host
  wedge, not one wedged agent)
- **Blind spots** — the exporter unscrapeable, the registry unreadable while the exporter
  is up, the per-session tier truncated, and a drill-down with zero coverage. These fire
  on the *observability* failing, which otherwise looks exactly like a calm fleet.

If `fak serve` uses `--require-key-env`, add Prometheus bearer-token auth in
`prometheus.yml` or run a loopback scrape proxy; otherwise the gateway target will
correctly show as down.

## Alerts → Slack (Alertmanager → fak receiver → durable outbox)

Firing rules are only useful if they reach someone. The Docker stack routes them
to Slack:

```
Prometheus :9091 ──(alerting: block)──▶ Alertmanager :9093 ──(webhook)──▶
    fak slack alert --serve :9096 ──▶ durable Slack outbox ──▶ Slack #grafana
```

Why a fak receiver instead of Alertmanager's built-in `slack_config`: the built-in
Slack integration posts through a Slack **incoming-webhook URL**, but every fak surface
authenticates with the shared **bot token** (`.env.slack.local`, see `fak slack check`).
The `fak slack alert` receiver reuses that one token *and* the **durable outbox**
(`internal/slackoutbox`), so an alert survives a crash / 429 / token blip and its
delivery is witnessable with `fak slack outbox status | calls` — not fire-and-forgotten.

The macOS/Homebrew fallback still evaluates and displays the Prometheus rules, but
intentionally has no native Alertmanager dependency. Use Docker Compose for the
bundled Alertmanager → Slack route.

**Bring it up:**

```bash
# 1. the receiver runs on the HOST (it needs the bot token). Channel resolves from
#    FAK_ALERTS_CHANNEL, else the public #grafana default.
./fak slack alert --serve --addr 127.0.0.1:9096

# 2. Alertmanager (added to docker-compose.yml) POSTs to it via host.docker.internal.
cd tools/grafana && docker compose up -d      # now includes fleet-alertmanager :9093
```

**Test the path without waiting for a real alert** — feed a webhook body straight
through the receiver (`--dry-run` renders without posting; drop it to deliver live):

```bash
./fak slack alert --dry-run <<'JSON'
{"version":"4","status":"firing","groupLabels":{"alertname":"FakGatewayDown"},
 "commonLabels":{"alertname":"FakGatewayDown","severity":"critical","job":"fak_gateway"},
 "alerts":[{"status":"firing","labels":{"alertname":"FakGatewayDown","severity":"critical","instance":"h:8080"},
   "annotations":{"summary":"fak gateway metrics are not scrapeable"},
   "startsAt":"2026-07-09T12:00:00Z","generatorURL":"http://prometheus:9091/graph"}]}
JSON
```

Or fire a real one end-to-end from the Alertmanager UI/API (`amtool alert add`, or
POST to `:9093/api/v2/alerts`) and watch it land in Slack, then confirm with
`fak slack outbox status`. Route to a dedicated channel by setting `FAK_ALERTS_CHANNEL`
(or `--channel`); the webhook target and cadence live in
[`alertmanager.yml`](alertmanager.yml).

## Linux / GPU server (host networking)

Uncomment `network_mode: host` in `docker-compose.yml`, comment the
`ports`/`extra_hosts` blocks, and update:

- `prometheus.yml`: targets → `localhost:9095` and `localhost:8080`
- `provisioning/datasources/datasource.yml`: Prometheus URL → `http://localhost:9091`
