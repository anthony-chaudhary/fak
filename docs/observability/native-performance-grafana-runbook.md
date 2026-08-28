---
title: "Native-performance Grafana operations runbook"
description: "Clean-checkout startup, fak-native evidence checks, Grafana diagnosis, and cleanup without claiming unfinished native-observability dependencies."
---

# Native-performance Grafana operations runbook

Use this runbook to bring up the **currently shipped** local Prometheus/Grafana stack,
drive a fak-native Qwen3.8 request, and decide whether the available telemetry is enough
to support a performance claim. It is deliberately provenance-honest: the generic
fleet/gateway dashboards exist today, while the native-performance metric contract,
dedicated dashboard, cross-artifact correlation, device drilldowns, regression views,
artifact links, and live panel proof are still tracked by issues #9809-#9817.

## 1. Know the shipped boundary

| Surface | Status in this checkout | Operator consequence |
|---|---|---|
| `tools/grafana/up.sh` / `down.sh` | **Shipped** | Starts/stops host exporters plus pinned Prometheus, Alertmanager, and Grafana containers. The scripts are tracked without an executable bit, so invoke them with `bash`. |
| Prometheus datasource and file dashboard provider | **Shipped** | `tools/grafana/provisioning/` is mounted read-only and provisions datasource UID `fleet-prometheus` plus every JSON dashboard under `tools/grafana/dashboards/`. |
| Gateway/kernel metrics | **Shipped** | `fak serve` exposes `/metrics`; the current dashboards cover gateway requests, cache, startup/model load, fleet state, guard decisions, and cache value. |
| Native-performance overview and phase/device panels | **Planned**: #9810, #9811, #9813, #9814 | Do not describe an empty or absent native panel as proof that a phase or device is idle. Use receipts and profiles instead. |
| Request/run/commit/receipt/trace/profile correlation and panel links | **Planned**: #9812, #9816 | Navigate artifacts from their recorded paths; Grafana cannot yet guarantee an exact receipt/profile/trace link. |
| Regression annotations/SLOs and live panel contract tests | **Planned**: #9815, #9817 | A rendered panel is not, by itself, promotion evidence. Validate the underlying receipt/profile and preserve the command/envelope. |

The existing `tools/grafana/up.sh` default is a small SmolLM2 pure-kernel metrics
demo. It does **not** constitute Qwen3.8 evidence. For Qwen3.8, start the stack without
its demo gateway and launch the accepted native recipe separately.

## 2. Clean-checkout prerequisites

Run from the repository root. Required local tools are Go 1.26+ (`GOTOOLCHAIN=auto`),
Python 3, `curl`, and a running Docker daemon with Compose v2. The first model pull and
container start require network access and sufficient disk space.

Validate the checked-in surfaces before starting anything:

```bash
test -f tools/grafana/docker-compose.yml
test -f tools/grafana/prometheus.yml
test -f tools/grafana/provisioning/datasources/datasource.yml
test -f tools/grafana/provisioning/dashboards/dashboards.yml
test -f docs/supported/qwen38-27b.md
bash -n tools/grafana/up.sh tools/grafana/down.sh
docker compose -f tools/grafana/docker-compose.yml config --quiet
```

`docker compose ... config --quiet` is the compose/schema check. If `docker` is missing,
install or start Docker before continuing; do not treat that as a model or Grafana bug.

## 3. Start the provisioned stack

Start the shipped stack with its demo gateway disabled so port `8080` remains available
for the Qwen3.8 process:

```bash
FAK_NO_GATEWAY=1 bash tools/grafana/up.sh
```

The script builds a temporary binary under `tools/grafana/.run/`, starts the fleet
exporters on `9095` and `9098`, and starts these pinned containers:

- Prometheus `prom/prometheus:v3.2.1` on `http://localhost:9091`;
- Alertmanager `prom/alertmanager:v0.27.0` on `http://localhost:9093`;
- Grafana `grafana/grafana:11.5.2` on `http://localhost:3000` (`admin` / `fleet`).

Confirm health and provisioning:

```bash
curl -fsS http://localhost:3000/api/health
curl -fsS http://localhost:9091/-/ready
curl -fsS -u admin:fleet http://localhost:3000/api/datasources/uid/fleet-prometheus
curl -fsS -u admin:fleet 'http://localhost:3000/api/search?query=FAK'
```

The datasource should report UID `fleet-prometheus` and URL
`http://prometheus:9091`. The dashboard provider scans
`/var/lib/grafana/dashboards` every 30 seconds, backed by the read-only host directory
`tools/grafana/dashboards/`. Re-run the API search after 30 seconds if Grafana is healthy
but the first search is empty.

Prometheus' tracked bridge-network target is `host.docker.internal:8080`. On Linux GPU
hosts using host networking, follow the exact edits documented under **Linux / GPU server
(host networking)** in `tools/grafana/README.md`; do not silently mix bridge and host
networking.

## 4. Start Qwen3.8 through fak-native

Use the accepted recipes in `docs/supported/qwen38-27b.md`. They resolve the alias first
and keep execution inside fak's `inkernel` engine; neither recipe names a llama.cpp
server or a remote `--base-url`.

### Metal: measured 36 GiB streamed-Q4_K route

```bash
fak model pull qwen38:27b
FAK_METAL_STREAM_Q4K=1 FAK_Q4K=1 fak serve \
  --addr 0.0.0.0:8080 \
  --gguf qwen38:27b --model qwen38:27b --metal \
  --context-budget-tokens 4096
```

### CUDA: measured A100 GGUF route

```bash
fak model pull qwen38:27b
FAK_Q4K=1 fak serve \
  --addr 0.0.0.0:8080 \
  --gguf qwen38:27b --model qwen38:27b --backend cuda \
  --context-budget-tokens 4096
```

Run one recipe only, in a separate terminal. Do not add `--base-url`, `--local`,
`--llama-server`, or `--qwen38-runtime llama-mtp`; those select a different runtime and
cannot prove fak-native execution. Do not substitute Qwen3.6 unless the task records an
explicit compatibility, regression, historical, or hardware/artifact exception.

Wait for actual readiness, not merely a bound socket:

```bash
until curl -fsS http://127.0.0.1:8080/healthz | grep -q '"ok":true'; do sleep 2; done
curl -fsS http://127.0.0.1:8080/metrics | grep -E '^fak_(gateway|kernel)_'
```

Drive a bounded request through the shipped Anthropic-compatible route:

```bash
curl -fsS http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen38:27b","max_tokens":32,"messages":[{"role":"user","content":"Reply with exactly: native-ok"}]}'
```

Then verify that Prometheus sees the gateway target:

```bash
curl -fsS -G http://localhost:9091/api/v1/query \
  --data-urlencode 'query=up{job="fak_gateway"}'
```

A successful scrape proves reachability and metric ingestion. It does **not** prove the
model, backend, quality, or absence of fallback.

## 5. Evidence required before calling a run fak-native Qwen3.8

A promotion-quality packet needs all of the following, even if the generic dashboards
look healthy:

1. **Exact model identity:** `qwen38:27b`, immutable artifact identity/hash, quantization,
   and model revision where applicable.
2. **Native engine identity:** a receipt with `execution.engine` equal to `fak-native`, an
   explicit native `forward_path`, and `fallback_count: 0`. The canonical receipt shape is
   `fak-native-performance-receipt/v1`; see
   `docs/_witnesses/issue-8819-qwen38-a100-roofline/receipt.json`.
3. **Backend and hardware:** scrubbed machine ID, platform, Metal or CUDA backend, device
   model/count, memory/capacity envelope, and the exact launch command.
4. **Matched controls:** prompt/decode/context tokens, batch/concurrency, sampling,
   cache state, warmups, repetitions, and unchanged controls.
5. **Quality:** exact-output or task-quality result. A receipt marked HOLD or below the
   quality floor may support attribution, but not a performance promotion.
6. **End-to-end accounting:** startup/recovery/verification overhead when relevant,
   end-to-end latency, throughput, prefill/decode split when observed, and peak/resident
   memory. Never promote a gain from a kernel-only timer as if it were end to end.
7. **Artifacts:** scrubbed receipt plus any profile/trace/raw rows, each with a stable path
   and digest. Keep private prompts, credentials, hostnames, and raw internal logs out of
   public artifacts.

Validate a native profile bundle with the shipped CLI:

```bash
fak native-performance --profile path/to/profile.json
fak native-performance --profile-next path/to/profile.json
```

The second command can legitimately refuse when the profile's recommendation is not an
unwitnessed dependency-ready lever. Preserve that result; do not rewrite it as success.

## 6. Grafana navigation and diagnosis

Open `http://localhost:3000` and select **FAK Gateway Observability** for request rate,
latency, errors, in-flight work, cache, and kernel counters. Use **FAK Startup & Model
Load** for startup/load observations. The other provisioned dashboards are fleet, guard,
cache-value, and dogfood views; none is currently a complete native Qwen3.8 phase/device
view.

Work from symptom to the narrowest supported evidence:

| Symptom | Check now | Honest interpretation |
|---|---|---|
| Requests queue or in-flight work rises | Gateway request/in-flight panels; compare request timestamps and end-to-end receipt rows | Queue attribution is provisional until #9813 lands explicit native queue metrics. |
| First request is slow | `/healthz`, startup/load dashboard, receipt warmup and cache-state controls | Separate cold load/warmup from steady-state prefill/decode. |
| Prompt-heavy requests regress | Receipt `prefill_milliseconds`, prompt tokens, context, cache state | Dedicated prefill/tokenize panels are planned in #9813. |
| Decode throughput regresses | Receipt `decode_milliseconds` and `tokens_per_second`; validate the profile | Use `fak native-performance --profile`; do not infer the kernel from GPU utilization alone. |
| KV/cache behavior changes | Cache dashboard plus receipt cache state and unchanged controls | Native KV allocation/restore/eviction phase coverage is incomplete until #9810/#9813. |
| Transfer or synchronization suspected | Profile events and backend-specific artifacts | Host/device transfer and synchronization panels are planned in #9814. |
| Memory pressure or OOM | Receipt peak/resident bytes, startup logs, device tooling captured in the witness | A missing Grafana device-memory series is “unavailable,” not zero. |
| Kernel appears slow | Profile classification, exact forward path, matched repetitions | Correlate with a receipt; a generic process graph cannot name the native kernel. |

### No-data decision tree

Run these in order:

```bash
curl -fsS http://127.0.0.1:8080/metrics >/dev/null
curl -fsS -G http://localhost:9091/api/v1/query \
  --data-urlencode 'query=up{job="fak_gateway"}'
curl -fsS http://localhost:9091/api/v1/targets
curl -fsS -u admin:fleet http://localhost:3000/api/datasources/uid/fleet-prometheus
cd tools/grafana && docker compose ps
```

- `/metrics` fails: diagnose the `fak serve` process, readiness, address, and model load.
- `/metrics` works but `up{job="fak_gateway"}` is not `1`: diagnose the Prometheus target
  and Docker-to-host networking.
- Prometheus is up but Grafana is empty: verify datasource UID/URL, dashboard provisioning,
  dashboard time range, and that a request occurred inside that range.
- Only a native phase/device/artifact panel is absent or empty: check #9810-#9817 status.
  On this checkout those surfaces are planned dependencies, not evidence of zero work.
- Cache-value panels alone are empty: `FAK_NO_GATEWAY=1` intentionally skips the
  cache-value exporter, and the cache-value rollup also needs its documented nightrun
  ledgers/ablation inputs. This is independent of the Qwen3.8 request path.

Inspect producer/container logs without starting another copy:

```bash
sed -n '1,240p' tools/grafana/.run/fak_fleet.log
test ! -f tools/grafana/.run/fak_gateway.log || sed -n '1,240p' tools/grafana/.run/fak_gateway.log
cd tools/grafana && docker compose logs --tail=200 prometheus grafana
```

With `FAK_NO_GATEWAY=1`, the shipped launcher does not create `fak_gateway.log`; use the
terminal or supervisor log for the separately launched Qwen3.8 process.

## 7. Bounded-label and cardinality checks

Prometheus stores 30 days in the shipped compose configuration. The current gateway
exporter is designed around bounded route/status/verdict labels; the native-performance
schema and budgets are not frozen until #9809/#9810 land.

Capture a before/after series count around a bounded test:

```bash
curl -fsS -G http://localhost:9091/api/v1/query \
  --data-urlencode 'query=count({job="fak_gateway"})'
# Run the fixed request corpus/repetitions here.
curl -fsS -G http://localhost:9091/api/v1/query \
  --data-urlencode 'query=count({job="fak_gateway"})'
```

Inspect label names and current values through the Prometheus HTTP API:

```bash
curl -fsS http://localhost:9091/api/v1/labels
curl -fsS 'http://localhost:9091/api/v1/label/job/values'
```

Reject a proposed metric or dashboard query if a label can contain prompt text, response
text, request/trace IDs, artifact paths, raw commit SHAs, unbounded model strings,
user/account/session IDs, or error text. Put high-cardinality correlation identifiers in a
scrubbed receipt/trace/profile artifact, not a Prometheus label. Until #9809 publishes the
native budget, record the before/after counts and explain every new series rather than
inventing a numeric limit.

## 8. Receipt, trace, and profile navigation

Today, navigate from the run ledger or witness directory, not from Grafana:

1. Open the scrubbed receipt and confirm `schema`, envelope, revision, model artifact,
   machine/backend, controls, `execution.engine`, `forward_path`, and `fallback_count`.
2. Follow only paths listed under `profiler_artifacts` (or the equivalent versioned
   artifact field) and verify their digests.
3. Run `fak native-performance --profile <profile.json>` to classify the captured profile.
4. If an OTLP collector was explicitly configured with
   `fak serve --otlp-traces-endpoint <endpoint>`, use its trace store and the correlation
   data recorded in the witness. The shipped Grafana compose file does not provision a
   trace datasource.

Grafana deep links to an exact receipt/profile/trace are a dependency of #9812/#9816.
Do not paste a private filesystem path, signed object URL, prompt, token, hostname, or raw
log into a public dashboard annotation as a workaround.

## 9. Metal and CUDA unavailable states

Treat “unavailable” as a typed operating state, not as `0` utilization.

- **Metal unavailable:** `--metal` is only a supported native path on a compatible
  Apple-Silicon build/device. A requested unavailable Metal path must fail loud; never
  remove `--metal` and report the CPU run as Metal.
- **CUDA unavailable:** `--backend cuda` requires a CUDA-capable build, driver, device,
  and adequate capacity. Never remove the flag, switch to a provider, or attach a
  llama.cpp server to make the request pass.
- **Scrape available, device detail unavailable:** gateway reachability is still valid,
  but utilization, residency, transfer, synchronization, and device-memory conclusions
  remain unsupported. Preserve the receipt/log failure and use a sanctioned compute node
  from `docs/fleet-compute-nodes.md` when the requested hardware is not local.
- **Dashboard behavior:** until #9814 lands explicit Metal/CUDA unavailable-state panels,
  annotate the witness as unavailable and use receipt/profile evidence. Never convert a
  missing series to zero in a performance claim.

## 10. Shutdown, cleanup, and rollback

Stop the foreground Qwen3.8 `fak serve` with `Ctrl-C`, then tear down only what the shipped
launcher started:

```bash
bash tools/grafana/down.sh
```

Verify no launcher-owned producers or containers remain:

```bash
for pidfile in tools/grafana/.run/*.pid; do
  test ! -e "$pidfile" || { pid=$(cat "$pidfile"); ! kill -0 "$pid" 2>/dev/null; }
done
cd tools/grafana && docker compose ps
```

Normal teardown preserves Prometheus, Alertmanager, and Grafana volumes. To delete the
local 30-day metrics history, Grafana state, and Alertmanager data as well:

```bash
bash tools/grafana/down.sh --purge
```

`--purge` is irreversible for those Docker volumes. Copy any required scrubbed witness
artifacts outside `tools/grafana/.run/` before purging; `.run/` is operational scratch, not
a durable evidence location.

If a new dashboard/config revision is bad, stop the stack, restore the last known-good
tracked files/commit, restart with `FAK_NO_GATEWAY=1 bash tools/grafana/up.sh`, and re-run
the health, datasource, target, bounded-corpus, receipt, and cleanup checks above. Never
roll back by silently changing the native engine or Qwen generation.

## 11. Public-safe completion witness

A complete clean-room walkthrough records:

- repository commit and `fak version`;
- exact Qwen3.8 artifact/revision and launch argv;
- scrubbed machine/backend/capacity identity;
- Grafana/Prometheus health and provisioned datasource/dashboard API responses;
- `up{job="fak_gateway"} == 1` after the bounded request;
- before/after series counts and any new label values;
- the versioned native receipt, quality result, no-fallback proof, and validated profile;
- screenshots with secrets, prompts, user/session IDs, hostnames, and private paths removed;
- shutdown evidence showing no launcher-owned producer or compose service remains.

Until #9810-#9817 ship, the valid conclusion is: **the existing Grafana stack ingested the
fak gateway/kernel metrics, while native Qwen3.8 execution and bottleneck attribution were
proved separately by the receipt/profile packet**. Do not claim the planned native Grafana
experience is already present.
