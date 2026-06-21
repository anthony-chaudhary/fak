#!/usr/bin/env bash
# up.sh — one command to bring up the whole fleet observability stack:
#
#   pure-kernel model engine  →  fak serve --engine inkernel (real weights)   :8080/metrics
#   fleet bottleneck engine   →  fleet_bottleneck.py serve                    :9095/metrics
#   scrape + alerts           →  Prometheus (docker)                          :9091
#   dashboards                →  Grafana (docker, admin/fleet)                :3000
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

GATEWAY_ADDR="${FAK_GATEWAY_ADDR:-0.0.0.0:8080}"   # 0.0.0.0 so Docker's host.docker.internal can scrape it
GATEWAY_HOSTPORT="127.0.0.1:8080"                  # where WE health-check it
BOTTLENECK_PORT="${FLEET_BOTTLENECK_PORT:-9095}"
MODEL_LABEL="${FAK_DOGFOOD_MODEL:-smollm2-135m}"

log()  { printf '\033[1;36m[up]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[up]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[up] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- docker CLI discovery (Docker Desktop ships the CLI inside the app bundle) ----
find_docker() {
  if command -v docker >/dev/null 2>&1; then echo docker; return; fi
  for c in /Applications/Docker.app/Contents/Resources/bin/docker "$HOME/.docker/bin/docker"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  die "docker not found. Install Docker Desktop (macOS/Windows) or docker engine (Linux)."
}
DOCKER="$(find_docker)"

ensure_docker_daemon() {
  if "$DOCKER" info >/dev/null 2>&1; then return; fi
  if [ "$(uname)" = "Darwin" ] && [ -d /Applications/Docker.app ]; then
    log "Docker daemon down — starting Docker Desktop…"
    open -a Docker
    for i in $(seq 1 30); do
      "$DOCKER" info >/dev/null 2>&1 && { log "Docker daemon up after ~$((i*3))s"; return; }
      sleep 3
    done
    die "Docker daemon did not come up within 90s."
  fi
  die "Docker daemon is not running (and not Docker Desktop on macOS to auto-start)."
}

# ---- a host process is already serving this port? ----
port_live() { curl -sf -m 3 "http://127.0.0.1:$1$2" >/dev/null 2>&1; }

start_bg() {  # name port-check-path "command…"  → records a pidfile
  local name="$1" check="$2"; shift 2
  if port_live "${check%% *}" "${check#* }"; then
    log "$name already healthy — adopting it."
    return
  fi
  log "starting ${name}…"
  ( "$@" >"$RUN_DIR/$name.log" 2>&1 & echo $! >"$RUN_DIR/$name.pid" )
}

# ===== 1. real weights for the pure kernel =====
export FAK_MODEL_DIR="${FAK_MODEL_DIR:-$ROOT/fak/internal/model/.cache/smollm2-135m}"
if [ "${FAK_NO_GATEWAY:-0}" != "1" ]; then
  if [ ! -f "$FAK_MODEL_DIR/weights.f32" ]; then
    warn "no fak-format export at $FAK_MODEL_DIR — running fetch-model.sh (downloads SmolLM2-135M, ~one-time)…"
    ( cd "$ROOT/fak" && FAK_MODEL_DIR="$FAK_MODEL_DIR" bash ./scripts/fetch-model.sh ) \
      || die "fetch-model.sh failed; set FAK_MODEL_DIR to an existing export or run with FAK_NO_GATEWAY=1."
  fi
  log "pure-kernel weights: $FAK_MODEL_DIR"
fi

# ===== 2. build fak (native; go build is fine on this host) =====
if [ "${FAK_NO_GATEWAY:-0}" != "1" ]; then
  FAK_BIN="$RUN_DIR/fak"
  if [ ! -x "$FAK_BIN" ] || [ "${FAK_REBUILD:-0}" = "1" ]; then
    log "building fak → $FAK_BIN"
    ( cd "$ROOT/fak" && go build -o "$FAK_BIN" ./cmd/fak ) || die "go build failed."
  fi
fi

# ===== 3. metrics sources on the host =====
start_bg fleet_bottleneck "$BOTTLENECK_PORT /metrics" \
  python3 "$ROOT/tools/fleet_bottleneck.py" serve --port "$BOTTLENECK_PORT"

if [ "${FAK_NO_GATEWAY:-0}" != "1" ]; then
  start_bg fak_gateway "8080 /metrics" \
    "$FAK_BIN" serve --addr "$GATEWAY_ADDR" --engine inkernel --model "$MODEL_LABEL"
else
  warn "FAK_NO_GATEWAY=1 — skipping fak serve; the FAK Gateway dashboard will show no data."
fi

# ===== 4. Prometheus + Grafana =====
ensure_docker_daemon
log "docker compose up -d (Prometheus :9091, Grafana :3000)…"
( cd "$GRAFANA_DIR" && "$DOCKER" compose up -d ) || die "docker compose up failed."

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
  warn "stack started but not all health checks passed in time — see $RUN_DIR/*.log and 'docker compose ps'."
fi
cat >&2 <<EOF

  Grafana     http://localhost:3000      (admin / fleet)
  Prometheus  http://localhost:9091
  fleet src   http://localhost:$BOTTLENECK_PORT/metrics
  gateway     http://$GATEWAY_HOSTPORT/metrics   (engine=inkernel model=$MODEL_LABEL)

  Drive a real kernel decode (populates fak_kernel_* / fak_gateway_*):
    curl -s http://$GATEWAY_HOSTPORT/v1/fak/syscall -H 'Content-Type: application/json' \\
      -d '{"tool":"search_flights","arguments":{"origin":"SFO","destination":"JFK"}}'

  Tear down:  tools/grafana/down.sh
EOF
