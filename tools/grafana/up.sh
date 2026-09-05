#!/usr/bin/env bash
# up.sh — one command to bring up the whole fleet observability stack:
#
#   pure-kernel model engine  →  fak serve --engine inkernel (real weights)   :8080/metrics
#   fleet bottleneck engine   →  fleet_bottleneck.py serve                    :9095/metrics
#   scrape + alerts           →  Prometheus (Docker; Homebrew fallback)       :9091
#   dashboards                →  Grafana (Docker; Homebrew fallback)          :3000
#
# It is idempotent: re-running adopts anything already healthy instead of
# colliding on a port. Tear it all down with ./down.sh.
#
# Usage:
#   tools/grafana/up.sh                 # bring everything up (pure kernel, SmolLM2-135M)
#   FAK_MODEL_DIR=/path tools/grafana/up.sh   # use a different fak-format export
#   FAK_NO_GATEWAY=1 tools/grafana/up.sh      # fleet metrics only (skip fak serve)
#
# Honest scope: the inkernel engine is a byte-level reference forward pass on real
# weights, NOT an English chat surface (see ../../fak/GETTING-STARTED.md §4b and
# issue #69). It exists here to drive REAL fak_kernel_* / fak_gateway_* metrics
# into the dashboards through the adjudicated dispatch path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRAFANA_DIR="$ROOT/tools/grafana"
RUN_DIR="$GRAFANA_DIR/.run"
mkdir -p "$RUN_DIR"
chmod 700 "$RUN_DIR"
RUN_ID="${FAK_GRAFANA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM:-0}}"

GATEWAY_HOSTPORT="127.0.0.1:8080"                  # where WE health-check it
BOTTLENECK_PORT="${FLEET_BOTTLENECK_PORT:-9095}"
CACHEVALUE_PORT="${FAK_CACHEVALUE_PORT:-9097}"
FLEET_METRICS_PORT="${FAK_FLEET_METRICS_PORT:-9098}"
MODEL_LABEL="${FAK_DOGFOOD_MODEL:-smollm2-135m}"

log()  { printf '\033[1;36m[up]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[up]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[up] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- Docker CLI discovery (Docker Desktop ships the CLI inside the app bundle) ----
find_docker() {
  if command -v docker >/dev/null 2>&1; then echo docker; return; fi
  for c in /Applications/Docker.app/Contents/Resources/bin/docker "$HOME/.docker/bin/docker"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  return 1
}
DOCKER=""

ensure_docker_daemon() {
  DOCKER="$(find_docker)" || return 1
  if "$DOCKER" info >/dev/null 2>&1; then return; fi
  if [ "$(uname)" = "Darwin" ] && [ -d /Applications/Docker.app ]; then
    log "Docker daemon down — starting Docker Desktop…"
    open -a Docker
    for i in $(seq 1 30); do
      "$DOCKER" info >/dev/null 2>&1 && { log "Docker daemon up after ~$((i*3))s"; return; }
      sleep 3
    done
    warn "Docker daemon did not come up within 90s."
  fi
  return 1
}

select_stack_mode() {
  STACK_MODE=native
  if ! command -v docker >/dev/null 2>&1; then
    [ "$(uname)" = "Darwin" ] \
      || die "Docker CLI is unavailable. Install Docker engine, or use the macOS/Homebrew fallback on a Mac."
    return
  fi
  if ensure_docker_daemon; then
    STACK_MODE=docker
    return
  fi
  [ "$(uname)" = "Darwin" ] \
    || die "Docker daemon is unavailable. Install Docker engine, or use the macOS/Homebrew fallback on a Mac."
}

configure_gateway_addr() {
  if [ -n "${FAK_GATEWAY_ADDR:-}" ]; then
    GATEWAY_ADDR="$FAK_GATEWAY_ADDR"
  elif [ "$STACK_MODE" = native ]; then
    GATEWAY_ADDR="127.0.0.1:8080"
  else
    # Docker Prometheus reaches host services through host.docker.internal.
    GATEWAY_ADDR="0.0.0.0:8080"
  fi
}

# ---- a host process is already serving this port? ----
port_live() { curl -sf -m 3 "http://127.0.0.1:$1$2" >/dev/null 2>&1; }

start_bg() {  # name port-check-path command… → records only processes this invocation owns
  local name="$1" check="$2"; shift 2
  local pid_file="$RUN_DIR/$name.pid"
  if port_live "${check%% *}" "${check#* }"; then
    if [ "$(uname)" = "Darwin" ] && [ -s "$pid_file" ]; then
      local recorded_owner recorded_label recorded_runtime expected_label
      recorded_owner="$(sed -n 's/^owner=//p' "$pid_file" 2>/dev/null | head -1)"
      recorded_label="$(sed -n 's/^label=//p' "$pid_file" 2>/dev/null | head -1)"
      recorded_runtime="$(sed -n 's/^runtime=//p' "$pid_file" 2>/dev/null | head -1)"
      expected_label="com.fak.grafana.$(id -u).${name//_/-}.$recorded_owner"
      if [[ "$recorded_owner" =~ ^[A-Za-z0-9.-]+$ ]] \
        && [ "$recorded_label" = "$expected_label" ] \
        && [[ "$recorded_runtime" == "${TMPDIR:-/tmp}"/fak-grafana-* ]] \
        && [ -d "$recorded_runtime" ] \
        && launchctl print "gui/$(id -u)/$recorded_label" >/dev/null 2>&1; then
        log "$name already healthy — retaining owned launchd job."
        return
      fi
    fi
    rm -f "$pid_file"
    log "$name already healthy — adopting it."
    return
  fi
  rm -f "$pid_file"
  log "starting ${name}…"
  if [ "$(uname)" = "Darwin" ]; then
    [[ "$RUN_ID" =~ ^[A-Za-z0-9.-]+$ ]] \
      || die "FAK_GRAFANA_RUN_ID must contain only letters, digits, dots, and hyphens."
    local label="com.fak.grafana.$(id -u).${name//_/-}.$RUN_ID"
    local runtime runtime_fak runtime_log command_path
    runtime="$(mktemp -d "${TMPDIR:-/tmp}/fak-grafana-${name}.${RUN_ID}.XXXXXX")" \
      || die "could not allocate a private launchd runtime for $name."
    chmod 700 "$runtime"
    runtime_fak="$runtime/fak"
    runtime_log="$runtime/$name.log"
    command_path="$1"
    if [ "$command_path" = "$FAK_BIN" ]; then
      cp "$FAK_BIN" "$runtime_fak"
      chmod 700 "$runtime_fak"
      shift
      set -- "$runtime_fak" "$@"
    fi
    # launchd owns the wrapper after this launcher exits. Its executable and logs
    # live under a private TMPDIR runtime, so protected-checkout cwd/log access is
    # never required. The wrapper atomically publishes exact teardown metadata.
    launchctl submit \
      -p /bin/bash \
      -l "$label" \
      -o "$runtime_log" \
      -e "$runtime_log" \
      -- /bin/bash -c '
        pid_file="$1"
        owner="$2"
        label="$3"
        runtime="$4"
        workspace="$5"
        shift 5
        child=""
        pid_file_tmp="$pid_file.tmp.$$"
        stop_child() {
          [ -z "$child" ] || kill "$child" 2>/dev/null || true
          [ -z "$child" ] || wait "$child" 2>/dev/null || true
          exit 0
        }
        trap stop_child TERM INT
        umask 077
        {
          printf "pid=%s\n" "$$"
          printf "owner=%s\n" "$owner"
          printf "supervisor=launchd\n"
          printf "label=%s\n" "$label"
          printf "runtime=%s\n" "$runtime"
        } >"$pid_file_tmp"
        /bin/mv "$pid_file_tmp" "$pid_file"
        cd "$runtime"
        FAK_WORKSPACE_ROOT="$workspace" FAK_LEDGER_ROOT="$workspace" "$@" &
        child=$!
        wait "$child"
      ' "fak-grafana-owner=$RUN_ID" "$pid_file" "$RUN_ID" "$label" "$runtime" "$ROOT" "$@"
    for _ in $(seq 1 50); do
      if [ -s "$pid_file" ] && grep -qx "runtime=$runtime" "$pid_file"; then
        return
      fi
      sleep 0.02
    done
    launchctl remove "$label" 2>/dev/null || true
    rm -rf "$runtime"
    die "$name launchd supervisor did not publish ownership metadata."
  fi
  # nohup makes the supervisor independent of the launcher's shell lifetime. Keep
  # TERM/INT trapped so non-macOS down.sh can still stop the owned child by PID.
  nohup bash -c '
    child=""
    stop_child() {
      [ -z "$child" ] || kill "$child" 2>/dev/null || true
      [ -z "$child" ] || wait "$child" 2>/dev/null || true
      exit 0
    }
    trap stop_child TERM INT
    "$@" &
    child=$!
    wait "$child"
  ' "fak-grafana-owner=$RUN_ID" "$@" </dev/null >"$RUN_DIR/$name.log" 2>&1 &
  local pid=$!
  {
    printf 'pid=%s\n' "$pid"
    printf 'owner=%s\n' "$RUN_ID"
  } >"$RUN_DIR/$name.pid"
  chmod 600 "$RUN_DIR/$name.pid"
}

# ---- native macOS configuration (generated; tracked Docker config is untouched) ----
NATIVE_DIR="$RUN_DIR/native"
NATIVE_PROVISIONING="$NATIVE_DIR/provisioning"

write_native_configs() {
  mkdir -p \
    "$NATIVE_PROVISIONING/datasources" \
    "$NATIVE_PROVISIONING/dashboards" \
    "$NATIVE_DIR/prometheus-data" \
    "$NATIVE_DIR/grafana-data/plugins" \
    "$NATIVE_DIR/grafana-logs"

  awk -v rules="$GRAFANA_DIR/prometheus-alerts.yml" '
    /^# Where firing alerts go\./ { skip_alerting = 1 }
    skip_alerting && /^scrape_configs:/ { skip_alerting = 0 }
    skip_alerting { next }
    {
      gsub(/host[.]docker[.]internal/, "127.0.0.1")
      if ($0 == "  - \"prometheus-alerts.yml\"") {
        print "  - \"" rules "\""
      } else {
        print
      }
    }
  ' "$GRAFANA_DIR/prometheus.yml" >"$NATIVE_DIR/prometheus.yml"

  sed 's#http://prometheus:9091#http://127.0.0.1:9091#g' \
    "$GRAFANA_DIR/provisioning/datasources/datasource.yml" \
    >"$NATIVE_PROVISIONING/datasources/datasource.yml"

  cat >"$NATIVE_PROVISIONING/dashboards/dashboards.yml" <<EOF
apiVersion: 1

providers:
  - name: FleetBottleneck
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 30
    options:
      path: $GRAFANA_DIR/dashboards
      foldersFromFilesStructure: false
EOF
}

find_brew() {
  if command -v brew >/dev/null 2>&1; then echo brew; return; fi
  for c in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  return 1
}

start_native_stack() {
  local brew prometheus_bin prometheus_prefix grafana_bin grafana_prefix grafana_home
  local -a grafana_command
  brew="$(find_brew)" || die "Docker is unavailable and Homebrew was not found. Install Homebrew, then run: brew install prometheus grafana"

  prometheus_bin="${FAK_PROMETHEUS_BIN:-}"
  [ -n "$prometheus_bin" ] || prometheus_bin="$(command -v prometheus 2>/dev/null || true)"
  if [ -z "$prometheus_bin" ]; then
    prometheus_prefix="$("$brew" --prefix prometheus 2>/dev/null || true)"
    [ -z "$prometheus_prefix" ] || prometheus_bin="$prometheus_prefix/bin/prometheus"
  fi

  grafana_bin="${FAK_GRAFANA_BIN:-}"
  [ -n "$grafana_bin" ] || grafana_bin="$(command -v grafana 2>/dev/null || true)"
  [ -n "$grafana_bin" ] || grafana_bin="$(command -v grafana-server 2>/dev/null || true)"
  grafana_prefix="$("$brew" --prefix grafana 2>/dev/null || true)"
  if [ -z "$grafana_bin" ] && [ -n "$grafana_prefix" ]; then
    grafana_bin="$grafana_prefix/bin/grafana"
  fi

  [ -x "$prometheus_bin" ] && [ -x "$grafana_bin" ] \
    || die "Docker is unavailable and the Homebrew services are missing. Run: brew install prometheus grafana"

  [ -n "$grafana_prefix" ] || grafana_prefix="$(cd "$(dirname "$grafana_bin")/.." && pwd)"
  grafana_home="${FAK_GRAFANA_HOME:-$grafana_prefix/share/grafana}"
  [ -d "$grafana_home" ] || die "Grafana home not found at $grafana_home (override with FAK_GRAFANA_HOME)."
  if [ "$(basename "$grafana_bin")" = grafana-server ]; then
    grafana_command=("$grafana_bin")
  else
    grafana_command=("$grafana_bin" server)
  fi

  write_native_configs
  log "Docker unavailable — using Homebrew Prometheus :9091 and Grafana :3000."
  start_bg prometheus "9091 /-/ready" \
    "$prometheus_bin" \
      --config.file="$NATIVE_DIR/prometheus.yml" \
      --storage.tsdb.path="$NATIVE_DIR/prometheus-data" \
      --storage.tsdb.retention.time=30d \
      --web.enable-lifecycle \
      --web.listen-address=127.0.0.1:9091

  start_bg grafana "3000 /api/health" \
    env \
      GF_PATHS_PROVISIONING="$NATIVE_PROVISIONING" \
      GF_PATHS_DATA="$NATIVE_DIR/grafana-data" \
      GF_PATHS_LOGS="$NATIVE_DIR/grafana-logs" \
      GF_PATHS_PLUGINS="$NATIVE_DIR/grafana-data/plugins" \
      GF_SECURITY_ADMIN_USER=admin \
      GF_SECURITY_ADMIN_PASSWORD=fleet \
      GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH="$GRAFANA_DIR/dashboards/fleet-bottleneck-overview.json" \
      GF_SERVER_HTTP_ADDR=127.0.0.1 \
      GF_SERVER_HTTP_PORT=3000 \
      "${grafana_command[@]}" --homepath "$grafana_home" --packaging=brew
}

if [ "${FAK_GRAFANA_NATIVE_CONFIG_ONLY:-0}" = "1" ]; then
  write_native_configs
  exit 0
fi

# ===== 1. real weights for the pure kernel =====
# $ROOT is the repository root and the Go module root (AGENTS.md: "the Go module is the
# repository root"), so the paths below are $ROOT/… — NOT $ROOT/fak/…, which pointed at
# a nested checkout this repo does not have and made every `cd` here fail.
export FAK_MODEL_DIR="${FAK_MODEL_DIR:-$ROOT/internal/model/.cache/smollm2-135m}"
if [ "${FAK_NO_GATEWAY:-0}" != "1" ]; then
  if [ ! -f "$FAK_MODEL_DIR/weights.f32" ]; then
    warn "no fak-format export at $FAK_MODEL_DIR — running fetch-model.sh (downloads SmolLM2-135M, ~one-time)…"
    ( cd "$ROOT" && FAK_MODEL_DIR="$FAK_MODEL_DIR" bash ./scripts/fetch-model.sh ) \
      || die "fetch-model.sh failed; set FAK_MODEL_DIR to an existing export or run with FAK_NO_GATEWAY=1."
  fi
  log "pure-kernel weights: $FAK_MODEL_DIR"
fi

# ===== 2. build fak (native; go build is fine on this host) =====
# NOT gated on FAK_NO_GATEWAY any more: that flag means "chart fleet metrics only", and
# the fleet-session exporter below is exactly what such an operator wants. It needs the
# binary but neither the gateway nor any weights — it folds files on disk.
FAK_BIN="$RUN_DIR/fak"
if [ ! -x "$FAK_BIN" ] || [ "${FAK_REBUILD:-0}" = "1" ]; then
  log "building fak → $FAK_BIN"
  ( cd "$ROOT" && go build -o "$FAK_BIN" ./cmd/fak ) || die "go build failed."
fi

# Decide before starting the gateway: the native stack keeps it loopback-only,
# while the Docker stack retains its host.docker.internal scrape path.
select_stack_mode
configure_gateway_addr

# ===== 3. metrics sources on the host =====
start_bg fleet_bottleneck "$BOTTLENECK_PORT /metrics" \
  python3 "$ROOT/tools/fleet_bottleneck.py" serve --port "$BOTTLENECK_PORT"

# The fleet-session exporter: the LIVE per-session inventory (which sessions are alive
# right now) plus the HISTORICAL cost/cache roll-ups, re-folded from disk on every
# scrape. It is the only source carrying a per-session label, so it is what makes the
# fleet dashboards drill down. Deliberately OUTSIDE the FAK_NO_GATEWAY guard: it needs
# no gateway, no weights and no network.
start_bg fak_fleet "$FLEET_METRICS_PORT /metrics" \
  "$FAK_BIN" fleet metrics --serve --addr "0.0.0.0:$FLEET_METRICS_PORT" --usage-ledger "$ROOT/.fak/nightrun/gateway-usage.jsonl"

if [ "${FAK_NO_GATEWAY:-0}" != "1" ]; then
  start_bg fak_gateway "${GATEWAY_ADDR##*:} /metrics" \
    "$FAK_BIN" serve --addr "$GATEWAY_ADDR" --engine inkernel --model "$MODEL_LABEL"
  # The cache-value roll-up exporter re-folds the nightrun ledgers + ablate arms on
  # each scrape (no model/weights needed). Its panels read "No data" until the
  # docs/nightrun/*.jsonl ledgers exist — run `fak cachevalue feed --once` to populate.
  start_bg fak_cachevalue "$CACHEVALUE_PORT /metrics" \
    "$FAK_BIN" cachevalue metrics --serve --addr "0.0.0.0:$CACHEVALUE_PORT" \
      --ledger "$ROOT/docs/nightrun/cache-value.jsonl" \
      --savings-ledger "$ROOT/docs/nightrun/cache-savings.jsonl" \
      --usage-ledger "$ROOT/.fak/nightrun/gateway-usage.jsonl" \
      --ablation-dir "$ROOT/experiments/ablate"
else
  warn "FAK_NO_GATEWAY=1 — skipping fak serve; the FAK Gateway + Cache Value dashboards will show no data."
fi

# ===== 4. Prometheus + Grafana =====
if [ "$STACK_MODE" = docker ]; then
  log "docker compose up -d (Prometheus :9091, Grafana :3000)…"
  ( cd "$GRAFANA_DIR" && "$DOCKER" compose up -d ) || die "docker compose up failed."
else
  start_native_stack
fi
printf '%s\n' "$STACK_MODE" >"$RUN_DIR/stack.mode"

# ===== 5. wait for health =====
log "waiting for the stack to report healthy…"
ok=1
for i in $(seq 1 20); do
  g=$(curl -sf -m 3 http://localhost:3000/api/health >/dev/null 2>&1 && echo y || echo n)
  p=$(curl -sf -m 3 http://localhost:9091/-/ready  >/dev/null 2>&1 && echo y || echo n)
  b=$(port_live "$BOTTLENECK_PORT" /metrics && echo y || echo n)
  [ "$g" = y ] && [ "$p" = y ] && [ "$b" = y ] && { ok=0; break; }
  sleep 3
done

echo
if [ "$ok" = 0 ]; then
  log "✅ observability stack is up."
else
  if [ "$STACK_MODE" = docker ]; then
    warn "stack started but not all health checks passed in time — see $RUN_DIR/*.log and 'docker compose ps'."
  else
    warn "stack started but not all health checks passed in time — see $RUN_DIR/*.log."
  fi
fi
cat >&2 <<EOF

  Grafana     http://localhost:3000      (admin / fleet)
  Prometheus  http://localhost:9091
  fleet src   http://localhost:$BOTTLENECK_PORT/metrics
  gateway     http://$GATEWAY_HOSTPORT/metrics   (engine=inkernel model=$MODEL_LABEL)
  cache value http://localhost:$CACHEVALUE_PORT/metrics   (nightrun ledgers + ablate arms)
  fleet sess  http://localhost:$FLEET_METRICS_PORT/metrics   (live session inventory + roll-ups)

  Drive a real kernel decode (populates fak_kernel_* / fak_gateway_*):
    curl -s http://$GATEWAY_HOSTPORT/v1/fak/syscall -H 'Content-Type: application/json' \\
      -d '{"tool":"search_flights","arguments":{"origin":"SFO","destination":"JFK"}}'

  Tear down:  tools/grafana/down.sh
EOF
