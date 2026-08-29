#!/usr/bin/env bash
# down.sh — tear down the Docker stack or owned native processes, plus host-side
# metrics sources started by up.sh. Adopted processes are never stopped.
#
# Usage:
#   tools/grafana/down.sh          # stop containers + host processes (keep data volumes)
#   tools/grafana/down.sh --purge  # also remove Docker Prometheus/Grafana data volumes
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
  return 1
}
DOCKER="$(find_docker || true)"
STACK_MODE="$(cat "$RUN_DIR/stack.mode" 2>/dev/null || true)"

# Docker stack
if [ "$STACK_MODE" = native ]; then
  log "native Homebrew stack recorded — skipping container teardown."
elif [ -n "$DOCKER" ] && "$DOCKER" info >/dev/null 2>&1; then
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
    pid="$(sed -n 's/^pid=//p' "$pf" 2>/dev/null | head -1)"
    owner="$(sed -n 's/^owner=//p' "$pf" 2>/dev/null | head -1)"
    supervisor="$(sed -n 's/^supervisor=//p' "$pf" 2>/dev/null | head -1)"
    label="$(sed -n 's/^label=//p' "$pf" 2>/dev/null | head -1)"
    runtime="$(sed -n 's/^runtime=//p' "$pf" 2>/dev/null | head -1)"
    name="$(basename "$pf" .pid)"
    expected_label="com.fak.grafana.$(id -u).${name//_/-}.$owner"
    remove_pid_file=true
    if [ "$supervisor" = launchd ] && [ "$(uname)" = "Darwin" ]; then
      job="$(launchctl print "gui/$(id -u)/$label" 2>/dev/null || true)"
      command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
      runtime_safe=false
      runtime_parent="$(dirname -- "$runtime")"
      runtime_base="$(basename -- "$runtime")"
      tmp_root="$(cd "${TMPDIR:-/tmp}" 2>/dev/null && pwd -P || true)"
      canonical_parent="$(cd "$runtime_parent" 2>/dev/null && pwd -P || true)"
      if [ -n "$tmp_root" ] && [ "$canonical_parent" = "$tmp_root" ] \
        && [[ "$runtime_base" == fak-grafana-* ]] \
        && [ -d "$runtime" ] && [ ! -L "$runtime" ]; then
        runtime_safe=true
      fi
      if [[ "$owner" =~ ^[A-Za-z0-9.-]+$ ]] && [ "$label" = "$expected_label" ] \
        && [ -n "$job" ] && [[ "$pid" =~ ^[0-9]+$ ]] \
        && [[ "$command_line" == *"fak-grafana-owner=$owner"* ]] \
        && [ "$runtime_safe" = true ]; then
        log "stopping $name (owned launchd job $label)…"
        launchctl remove "$label" 2>/dev/null || true
        rm -rf -- "$runtime"
      else
        remove_pid_file=false
        log "leaving $name launchd job/runtime running — ownership was not verified."
      fi
    elif [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
      if [ -n "$owner" ] && [[ "$command_line" == *"fak-grafana-owner=$owner"* ]]; then
        log "stopping $name (owned pid $pid)…"
        kill "$pid" 2>/dev/null || true
      else
        log "leaving $name (pid $pid) running — ownership was not verified."
      fi
    fi
    [ "$remove_pid_file" != true ] || rm -f "$pf"
  done
fi

log "✅ stack torn down."
