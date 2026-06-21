#!/usr/bin/env bash
# down.sh — tear down everything up.sh started: the Docker stack + the host-side
# metrics sources (fak serve, fleet_bottleneck.py).
#
# Usage:
#   tools/grafana/down.sh          # stop containers + host processes (keep data volumes)
#   tools/grafana/down.sh --purge  # also remove Prometheus/Grafana data volumes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRAFANA_DIR="$ROOT/tools/grafana"
RUN_DIR="$GRAFANA_DIR/.run"

log() { printf '\033[1;36m[down]\033[0m %s\n' "$*" >&2; }

find_docker() {
  command -v docker >/dev/null 2>&1 && { echo docker; return; }
  for c in /Applications/Docker.app/Contents/Resources/bin/docker "$HOME/.docker/bin/docker"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  echo docker
}
DOCKER="$(find_docker)"

# Docker stack
if "$DOCKER" info >/dev/null 2>&1; then
  if [ "${1:-}" = "--purge" ]; then
    log "docker compose down -v (removing data volumes)…"
    ( cd "$GRAFANA_DIR" && "$DOCKER" compose down -v )
  else
    log "docker compose down…"
    ( cd "$GRAFANA_DIR" && "$DOCKER" compose down )
  fi
else
  log "docker daemon not running — skipping container teardown."
fi

# Host-side processes started by up.sh
if [ -d "$RUN_DIR" ]; then
  for pf in "$RUN_DIR"/*.pid; do
    [ -e "$pf" ] || continue
    pid="$(cat "$pf" 2>/dev/null || true)"
    name="$(basename "$pf" .pid)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      log "stopping $name (pid $pid)…"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
  done
fi

log "✅ stack torn down."
