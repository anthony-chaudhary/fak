#!/usr/bin/env bash
# gcp-glm-demo.sh — the ONE-COMMAND GCP-H100 GLM-5.2 kernel demo, end to end.
#
# This is the orchestrator the goal "demoable working fak kernel GLM-5.2 on GCP
# H100(s)" names: it chains the four steps that a live demo needs into a single
# reviewable command, defaulting to the 8x H100 Mega tier (a3-mega-h100, 640 GB —
# GLM-5.2 is 343-487 GB so it needs the 8-GPU shape, never 1x80 GB):
#
#   1. PROVISION + SERVE — stand GLM-5.2 up through the PURE FAK KERNEL on a GCP
#      H100 node. Delegated verbatim to scripts/gcp-glm-serve.sh (the canonical
#      bring-up); this script never re-implements the gcloud/serve rendering.
#   2. PROBE — two `claude-glm-gcp --probe` turns that SHARE a system+tools prefix,
#      so turn 2 hits the prefix fak already holds resident.
#   3. PERFORMANCE — fold the probe JSON into a duration/throughput witness.
#   4. CACHE-VALUE — scrape fak's OWN RadixAttention KV-prefix reuse off the serve
#      /metrics surface (fak_gateway_kv_prefix_reused_tokens_total). reuse > 0 on
#      turn 2..N is the WITNESSED demo datum (#1010): the prefill fak did NOT redo.
#   5. TEARDOWN — delete the VM (always, on a trap) so the demo leaves zero cost.
#
#       ┌────────────────┐  /v1   ┌────────────────────────┐  /v1   ┌──────────────────┐
#       │ claude-glm-gcp │ ─────▶ │ fak serve (the kernel) │ ─────▶ │ GLM-5.2 on 8x     │
#       │  (Claude Code) │ ◀───── │ glm_moe_dsa, adjud.    │ ◀───── │ H100 (a3-high)    │
#       └────────────────┘        └────────────────────────┘        └──────────────────┘
#                 turn 2 reuses turn 1's system+tools prefix  ── fak_gateway_kv_prefix_reused_tokens_total > 0
#
# PLAN BY DEFAULT. With no creds this prints every step — the gcloud create (via the
# serve script), the two probes, the cache-value scrape, and the teardown — and exits
# 0, so the whole demo is reviewable before a dollar is spent. `--apply` runs it and
# needs `gcloud` + GCP_PROJECT (and HF_TOKEN for the gated checkpoint).
#
# Usage:
#   ./scripts/gcp-glm-demo.sh                                 # PLAN: the full demo, no creds
#   GCP_PROJECT=my-proj ./scripts/gcp-glm-demo.sh --apply     # run it on 8x H100, then tear down
#   GCP_TIER=a3-high-h100 ./scripts/gcp-glm-demo.sh           # plan it on standard H100 instead
#   GCP_TIER=a3-ultra-h200 ./scripts/gcp-glm-demo.sh          # plan it on H200 instead
#   KEEP=1 GCP_PROJECT=my-proj ./scripts/gcp-glm-demo.sh --apply  # skip teardown (debug the node)
#
# Knobs (env) — the serve knobs (SERVE, GLM_GGUF_*, HF_TOKEN, ...) pass straight through
# to scripts/gcp-glm-serve.sh; this script adds only the demo-specific ones:
#   GCP_TIER          gcp_accel.py tier slug          (default a3-mega-h100 — the 8x H100 Mega demo tier)
#   EP_RANKS          pure-fak resident EP ranks       (default 8 — one rank per H100)
#   VM_NAME           instance name                   (default fak-glm-demo)
#   PROBE_PROMPT      the headless probe turn          (default "say pong")
#   PROBE_TURNS       how many cache-warming turns     (default 2 — turn 2 is where reuse must bite)
#   LOCAL_TUNNEL_PORT local port the tunnel binds      (default 8200 — the preset default)
#   READY_TIMEOUT_MINUTES  max wait for the VM-side fak serve to become ready (default 180)
#   PROBE_TIMEOUT_S   timeout floor for each Claude Code probe turn (default 900)
#   WITNESS_DIR       where apply-mode probe/metrics evidence is copied
#   PERF_MAX_DURATION_MS  max per-turn Claude probe duration (default 60000; 0 disables)
#   FAK_STARTUP_PATCH_FILE  optional local patch applied after the VM clones FAK_REPO_URL
#   FAK_REPO_REF    optional commit/ref checked out before applying FAK_STARTUP_PATCH_FILE
#   CACHE_VALUE_MIN_REUSED  minimum reused-token counter required after probes (default 1)
#   KEEP              1 = skip teardown (debug)         (default empty = always tear down)
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The demo tier is the 8x H100 Mega shape by default — the current GCP fallback that has
# succeeded for GLM-5.2 and carries the separate NVIDIA_H100_MEGA quota family. GLM-5.2 at
# UD-Q4_K_S still needs the 640 GB of an 8x80GB H100/H100-Mega node, and keeps the demo on
# the pure-Q4_K expert path instead of UD-Q4_K_M's slow mixed Q5_K/Q6_K host seam.
GCP_TIER="${GCP_TIER:-a3-mega-h100}"
VM_NAME="${VM_NAME:-fak-glm-demo}"
# The demo is the PURE FAK KERNEL demo: force SERVE=fak even on sm_90+ H100, where the
# serve script would otherwise default to stock SGLang. This is load-bearing twice over —
# it's the whole point of "demoable working fak kernel", AND the cache-value witness
# (fak_gateway_kv_prefix_reused_tokens_total) only exists when fak itself serves the model.
SERVE="${SERVE:-fak}"
EP_RANKS="${EP_RANKS:-8}"
GLM_PORT="${GLM_PORT:-8000}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8200}"
PROBE_PROMPT="${PROBE_PROMPT:-say pong}"
PROBE_TURNS="${PROBE_TURNS:-2}"
READY_TIMEOUT_MINUTES="${READY_TIMEOUT_MINUTES:-180}"
REMOTE_WAIT_POLL_S="${REMOTE_WAIT_POLL_S:-30}"
TUNNEL_READY_TIMEOUT_S="${TUNNEL_READY_TIMEOUT_S:-90}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-900}"
PERF_MAX_DURATION_MS="${PERF_MAX_DURATION_MS:-60000}"
CACHE_VALUE_MIN_REUSED="${CACHE_VALUE_MIN_REUSED:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WITNESS_DIR="${WITNESS_DIR:-$ROOT/experiments/agent-live/gcp-glm-demo-$RUN_ID}"
KEEP="${KEEP:-}"
TUNNEL_PID=""

MODE="plan"
case "${1:-}" in
  --apply)    MODE="apply" ;;
  --plan|"")  MODE="plan" ;;
  --help|-h)  sed -n '2,52p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

log()  { printf '\033[36m[gcp-glm-demo]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[gcp-glm-demo] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[gcp-glm-demo] %s\033[0m\n' "$*" >&2; exit 1; }

# --- resolve the tier from the single registry (tools/gcp_accel.py) so the demo's own
# summary + teardown command name the same zone/shape the serve script provisions. ---
command -v python >/dev/null 2>&1 && PY=python || PY=python3
TIER_SHELL="$("$PY" "$ROOT/tools/gcp_accel.py" --emit-shell "$GCP_TIER")" \
  || die "unknown GCP_TIER='$GCP_TIER' — run: $PY tools/gcp_accel.py  (to list the ladder)"
eval "$TIER_SHELL"   # defines GLM_MACHINE_TYPE, GLM_DEFAULT_ZONE, GLM_GPU_COUNT, ...
GCP_ZONE="${GCP_ZONE:-${GLM_DEFAULT_ZONE}}"

case "$PROBE_TURNS" in ''|*[!0-9]*) die "PROBE_TURNS must be a positive integer (got '$PROBE_TURNS')" ;; esac
[ "$PROBE_TURNS" -ge 1 ] || die "PROBE_TURNS must be >= 1 (got '$PROBE_TURNS')"
case "$READY_TIMEOUT_MINUTES" in ''|*[!0-9]*) die "READY_TIMEOUT_MINUTES must be a positive integer (got '$READY_TIMEOUT_MINUTES')" ;; esac
[ "$READY_TIMEOUT_MINUTES" -ge 1 ] || die "READY_TIMEOUT_MINUTES must be >= 1 (got '$READY_TIMEOUT_MINUTES')"
case "$REMOTE_WAIT_POLL_S" in ''|*[!0-9]*) die "REMOTE_WAIT_POLL_S must be a positive integer (got '$REMOTE_WAIT_POLL_S')" ;; esac
[ "$REMOTE_WAIT_POLL_S" -ge 1 ] || die "REMOTE_WAIT_POLL_S must be >= 1 (got '$REMOTE_WAIT_POLL_S')"
case "$TUNNEL_READY_TIMEOUT_S" in ''|*[!0-9]*) die "TUNNEL_READY_TIMEOUT_S must be a positive integer (got '$TUNNEL_READY_TIMEOUT_S')" ;; esac
[ "$TUNNEL_READY_TIMEOUT_S" -ge 1 ] || die "TUNNEL_READY_TIMEOUT_S must be >= 1 (got '$TUNNEL_READY_TIMEOUT_S')"
case "$PROBE_TIMEOUT_S" in ''|*[!0-9]*) die "PROBE_TIMEOUT_S must be a positive integer (got '$PROBE_TIMEOUT_S')" ;; esac
[ "$PROBE_TIMEOUT_S" -ge 1 ] || die "PROBE_TIMEOUT_S must be >= 1 (got '$PROBE_TIMEOUT_S')"
case "$PERF_MAX_DURATION_MS" in ''|*[!0-9]*) die "PERF_MAX_DURATION_MS must be a non-negative integer (got '$PERF_MAX_DURATION_MS')" ;; esac

# --- step 1 render: delegate the provision+serve plan to the canonical bring-up. ------
# We never re-implement the gcloud/serve rendering; we COMPOSE it, then add the demo's
# probe + cache-value + teardown steps around it.
render_serve_plan() {
  GCP_TIER="$GCP_TIER" VM_NAME="$VM_NAME" GLM_PORT="$GLM_PORT" SERVE="$SERVE" EP_RANKS="$EP_RANKS" \
  LOCAL_TUNNEL_PORT="$LOCAL_TUNNEL_PORT" GCP_ZONE="$GCP_ZONE" FAK_STARTUP_PATCH_FILE="${FAK_STARTUP_PATCH_FILE:-}" \
  FAK_REPO_REF="${FAK_REPO_REF:-}" \
    bash "$ROOT/scripts/gcp-glm-serve.sh" --plan
}

gcloud_common_args() {
  printf '%s\0' --zone "$GCP_ZONE" --tunnel-through-iap
  if [ -n "${GCP_PROJECT:-}" ]; then
    printf '%s\0' --project "$GCP_PROJECT"
  fi
}

wait_for_remote_ready() {
  log "waiting for VM-side fak serve to become ready (timeout ${READY_TIMEOUT_MINUTES}m)"
  local remote_cmd
  remote_cmd="$(cat <<REMOTE
set -euo pipefail
deadline=\$(( \$(date +%s) + ${READY_TIMEOUT_MINUTES} * 60 ))
poll=${REMOTE_WAIT_POLL_S}
port=${GLM_PORT}
phase_file='${GLM_STAGE_DIR:-/opt/glm52-q4}/PHASE'
last_fail=""
fail_count=0
while [ "\$(date +%s)" -lt "\$deadline" ]; do
  phase="\$(cat "\$phase_file" 2>/dev/null || true)"
  if printf '%s' "\$phase" | grep -q 'GLM52_FAK_NATIVE_SERVE_READY' && curl -sf -m 5 "http://127.0.0.1:\$port/v1/models" >/tmp/glm52-models.json 2>/dev/null; then
    echo "READY phase=\$phase"
    cat /tmp/glm52-models.json
    exit 0
  fi
  if [ -z "\$phase" ] && journalctl -u google-startup-scripts.service -n 40 --no-pager 2>/dev/null | grep -Eq 'startup-script.*failed|error while communicating with "startup-script"|Script "startup-script" failed'; then
    echo "REMOTE_FAIL phase=STARTUP_SCRIPT_FAIL confirmation=1" >&2
    journalctl -u google-startup-scripts.service -n 160 --no-pager >&2 || true
    exit 2
  fi
  if printf '%s' "\$phase" | grep -Eq '(^| )(BAD_|.*_FAIL|.*_TIMEOUT|.*_EXITED_EARLY|SMOKE_FAIL|NO_HF_CLI|HEALTH_TIMEOUT)( |$)'; then
    if [ "\$phase" = "\$last_fail" ]; then
      fail_count=\$((fail_count + 1))
    else
      last_fail="\$phase"
      fail_count=1
    fi
    echo "REMOTE_FAIL phase=\$phase confirmation=\$fail_count" >&2
    if [ "\$fail_count" -ge 2 ]; then
      journalctl -u glm52serve -n 120 --no-pager >&2 || true
      exit 2
    fi
  else
    last_fail=""
    fail_count=0
  fi
  echo "WAIT phase=\${phase:-none}" >&2
  sleep "\$poll"
done
echo "TIMEOUT waiting for GLM52_FAK_NATIVE_SERVE_READY on port \$port" >&2
journalctl -u glm52serve -n 80 --no-pager >&2 || true
exit 1
REMOTE
)"
  local args=()
  while IFS= read -r -d '' a; do args+=("$a"); done < <(gcloud_common_args)
  gcloud compute ssh "$VM_NAME" "${args[@]}" --command "$remote_cmd"
}

collect_remote_witness() {
  mkdir -p "$WITNESS_DIR"
  local label="${1:-remote}"
  local out="$WITNESS_DIR/${label}-logs.tgz"
  log "collecting remote serve logs -> $out"
  local remote_cmd
  remote_cmd="$(cat <<REMOTE
set -euo pipefail
dir='${GLM_STAGE_DIR:-/opt/glm52-q4}'
tmp="\$(mktemp -d)"
cleanup(){ rm -rf "\$tmp"; }
trap cleanup EXIT
if [ -d "\$dir" ]; then
  for name in PHASE fak_native_serve.log stage_serve.log server.log; do
    [ -f "\$dir/\$name" ] && cp "\$dir/\$name" "\$tmp/\$name" || true
  done
  for f in "\$dir"/server-rank*.log; do
    [ -f "\$f" ] && cp "\$f" "\$tmp/\$(basename "\$f")" || true
  done
fi
journalctl -u glm52serve -n 240 --no-pager >"\$tmp/journal-glm52serve.txt" 2>&1 || true
tar -C "\$tmp" -czf - .
REMOTE
)"
  local args=()
  while IFS= read -r -d '' a; do args+=("$a"); done < <(gcloud_common_args)
  if ! gcloud compute ssh "$VM_NAME" "${args[@]}" --command "$remote_cmd" >"$out"; then
    warn "remote log collection failed"
    rm -f "$out"
    return 1
  fi
}

start_tunnel() {
  mkdir -p "$WITNESS_DIR"
  log "opening IAP tunnel localhost:${LOCAL_TUNNEL_PORT} -> ${VM_NAME}:${GLM_PORT}"
  local args=()
  while IFS= read -r -d '' a; do args+=("$a"); done < <(gcloud_common_args)
  gcloud compute ssh "$VM_NAME" "${args[@]}" -- -N -L "${LOCAL_TUNNEL_PORT}:localhost:${GLM_PORT}" \
    >"$WITNESS_DIR/tunnel.log" 2>&1 &
  TUNNEL_PID=$!
  for _ in $(seq 1 "$TUNNEL_READY_TIMEOUT_S"); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      tail -40 "$WITNESS_DIR/tunnel.log" >&2 || true
      die "IAP tunnel exited before becoming ready"
    fi
    if curl -sf -m 3 "http://127.0.0.1:${LOCAL_TUNNEL_PORT}/v1/models" >"$WITNESS_DIR/models.json" 2>/dev/null; then
      log "tunnel ready; copied /v1/models -> $WITNESS_DIR/models.json"
      return 0
    fi
    sleep 1
  done
  tail -40 "$WITNESS_DIR/tunnel.log" >&2 || true
  die "IAP tunnel did not expose /v1/models within ${TUNNEL_READY_TIMEOUT_S}s"
}

stop_tunnel() {
  if [ -n "${TUNNEL_PID:-}" ]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""
  fi
}

run_probe_turns() {
  mkdir -p "$WITNESS_DIR"
  log "running ${PROBE_TURNS} Claude Code probe turn(s) through the local kernel-fronted tunnel"
  for turn in $(seq 1 "$PROBE_TURNS"); do
    local log_file="$WITNESS_DIR/probe-turn-${turn}.log"
    local json_src="$ROOT/experiments/agent-live/dogfood-claude-probe.json"
    local json_win_src="$ROOT/experiments/agent-live/dogfood-claude-probe-win.json"
    local json_dst="$WITNESS_DIR/probe-turn-${turn}.json"
    log "probe turn ${turn}/${PROBE_TURNS}: ${PROBE_PROMPT}"
    if command -v claude >/dev/null 2>&1; then
      FAK_DOGFOOD_PRESET=glm-gcp \
      FAK_GLM_GCP_BASE_URL="http://127.0.0.1:${LOCAL_TUNNEL_PORT}/v1" \
      FAK_DOGFOOD_TIMEOUT_S="$PROBE_TIMEOUT_S" \
      FAK_DOGFOOD_NO_ATTACH=1 \
        bash "$ROOT/scripts/dogfood-claude.sh" --probe "$PROBE_PROMPT" 2>&1 | tee "$log_file"
      [ -f "$json_src" ] || die "probe turn $turn completed but $json_src was not written"
      cp "$json_src" "$json_dst"
    elif command -v powershell.exe >/dev/null 2>&1 && [ -f "$ROOT/scripts/dogfood-claude.ps1" ]; then
      local ps_script
      ps_script="$(wslpath -w "$ROOT/scripts/dogfood-claude.ps1")"
      # We are invoking a Win32 process (powershell.exe) FROM wsl, so the glm-gcp preset
      # env vars must cross with the /w flag (WSL->Win32). /u is the OPPOSITE direction
      # (Win32->WSL) and would silently drop them, leaving the .ps1 with no preset -> it
      # falls back to the local SmolLM2-135M shim instead of fronting the GLM tunnel. These
      # are plain string/URL values (no path), so /w alone, never /wp (no path translation).
      local wsl_env="FAK_DOGFOOD_PRESET/w:FAK_GLM_GCP_BASE_URL/w:FAK_DOGFOOD_TIMEOUT_S/w:FAK_DOGFOOD_NO_ATTACH/w"
      [ -z "${WSLENV:-}" ] || wsl_env="$wsl_env:$WSLENV"
      WSLENV="$wsl_env" \
      FAK_DOGFOOD_PRESET=glm-gcp \
      FAK_GLM_GCP_BASE_URL="http://127.0.0.1:${LOCAL_TUNNEL_PORT}/v1" \
      FAK_DOGFOOD_TIMEOUT_S="$PROBE_TIMEOUT_S" \
      FAK_DOGFOOD_NO_ATTACH=1 \
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ps_script" --probe "$PROBE_PROMPT" 2>&1 | tee "$log_file"
      [ -f "$json_win_src" ] || die "probe turn $turn completed but $json_win_src was not written"
      cp "$json_win_src" "$json_dst"
    else
      die "no Claude Code runner found: install a native 'claude' in PATH or expose powershell.exe + scripts/dogfood-claude.ps1"
    fi
  done
}

summarize_probe_perf() {
  mkdir -p "$WITNESS_DIR"
  local summary="$WITNESS_DIR/performance-summary.json"
  log "folding probe performance witness -> $summary"
  "$PY" - "$WITNESS_DIR" "$summary" "$PERF_MAX_DURATION_MS" <<'PY'
import glob
import json
import math
import sys
from pathlib import Path

witness_dir = Path(sys.argv[1])
summary = Path(sys.argv[2])
max_duration_ms = int(sys.argv[3])

turns = []
for path in sorted(glob.glob(str(witness_dir / "probe-turn-*.json"))):
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
    model_usage = d.get("modelUsage") if isinstance(d.get("modelUsage"), dict) else {}
    if model_usage:
        first = next(iter(model_usage.values()))
        if isinstance(first, dict):
            usage = {**usage, **{
                "input_tokens": first.get("inputTokens", usage.get("input_tokens", 0)),
                "output_tokens": first.get("outputTokens", usage.get("output_tokens", 0)),
            }}
    duration_ms = float(d.get("duration_ms") or 0)
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = input_tokens + output_tokens
    seconds = duration_ms / 1000 if duration_ms > 0 else math.inf
    turns.append({
        "file": str(p),
        "terminal_reason": d.get("terminal_reason"),
        "duration_ms": duration_ms,
        "ttft_ms": d.get("ttft_ms"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "e2e_tokens_per_s": (total_tokens / seconds) if seconds and math.isfinite(seconds) else 0,
        "result_prefix": str(d.get("result") or "")[:120],
    })

if not turns:
    raise SystemExit("no probe-turn JSON artifacts found")

max_seen = max(t["duration_ms"] for t in turns)
min_rate = min(t["e2e_tokens_per_s"] for t in turns)
all_completed = all(t.get("terminal_reason") == "completed" for t in turns)
duration_ok = (max_duration_ms == 0) or all(t["duration_ms"] <= max_duration_ms for t in turns)
doc = {
    "schema": "fak.gcp-glm-demo-performance.v1",
    "thresholds": {
        "max_duration_ms": max_duration_ms,
    },
    "summary": {
        "turns": len(turns),
        "all_completed": all_completed,
        "max_duration_ms": max_seen,
        "min_e2e_tokens_per_s": min_rate,
        "pass": all_completed and duration_ok,
    },
    "turns": turns,
}
summary.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(doc["summary"], sort_keys=True))
if not doc["summary"]["pass"]:
    raise SystemExit(3)
PY
}

scrape_cache_value() {
  mkdir -p "$WITNESS_DIR"
  local metrics="$WITNESS_DIR/metrics.prom"
  local summary="$WITNESS_DIR/cache-value-summary.json"
  log "scraping remote pure-kernel /metrics for fak KV-prefix reuse"
  curl -sf "http://127.0.0.1:${LOCAL_TUNNEL_PORT}/metrics" >"$metrics"
  "$PY" - "$metrics" "$summary" "$CACHE_VALUE_MIN_REUSED" <<'PY'
import json
import re
import sys
from pathlib import Path

metrics = Path(sys.argv[1])
summary = Path(sys.argv[2])
minimum = float(sys.argv[3])
values = {}
for line in metrics.read_text(encoding="utf-8", errors="replace").splitlines():
    if line.startswith("#"):
        continue
    m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$", line)
    if m:
        values[m.group(1)] = float(m.group(2))
reused = values.get("fak_gateway_kv_prefix_reused_tokens_total", 0.0)
prompt = values.get("fak_gateway_kv_prefix_prompt_tokens_total", 0.0)
doc = {
    "schema": "fak.gcp-glm-demo-cache-value.v1",
    "metric_source": str(metrics),
    "kv_prefix": {
        "reused_tokens": reused,
        "prompt_tokens": prompt,
        "cache_bit": reused >= minimum,
        "min_reused_tokens": minimum,
    },
}
summary.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(doc["kv_prefix"], sort_keys=True))
if reused < minimum:
    raise SystemExit(3)
PY
  log "cache-value witness -> $summary"
}

# --- step 2..4 render: shared-prefix probes, a performance fold, then fak's OWN realized
# KV-prefix reuse off the serve /metrics surface. ------------------------------------
print_cache_value_steps() {
  cat <<DEMO
# === DEMO step 2 — drive ${PROBE_TURNS} headless turns through the kernel ===
# Each turn carries the SAME system+tools prefix, so turn 2..N reuse the KV prefix fak
# already holds resident. (Run scripts/dogfood-claude.sh --install once for the preset.)
export FAK_GLM_GCP_BASE_URL=http://127.0.0.1:${LOCAL_TUNNEL_PORT}/v1
for turn in \$(seq 1 ${PROBE_TURNS}); do
  claude-glm-gcp --probe "${PROBE_PROMPT}"   # one witnessable headless turn
done

# === DEMO step 3 — WITNESS performance ===
# apply mode writes performance-summary.json and fails if any probe exceeds
# PERF_MAX_DURATION_MS=${PERF_MAX_DURATION_MS} (0 disables the duration gate).

# === DEMO step 4 — WITNESS the cache value (the #1010 lever) ===
# fak's RadixAttention prefix cache eliminates the repeated system+tools+repo prefill on
# turns 2..N. Scrape the counter fak itself authored (WITNESSED, not relayed):
curl -s http://127.0.0.1:${LOCAL_TUNNEL_PORT}/metrics \\
  | grep '^fak_gateway_kv_prefix_reused_tokens_total'
# DEMO PASSES when fak_gateway_kv_prefix_reused_tokens_total > 0 — the prefill fak did NOT redo.
DEMO
}

# --- step 5 render: teardown so the demo is self-contained and leaves zero cost. ------
print_teardown_steps() {
  if [ -n "$KEEP" ]; then
    echo "# === DEMO step 5 — teardown SKIPPED (KEEP=1); delete it yourself when done: ==="
  else
    echo "# === DEMO step 5 — teardown (always, so the demo leaves zero residual cost) ==="
  fi
  printf 'gcloud compute instances delete %q --zone %q' "$VM_NAME" "$GCP_ZONE"
  echo "${GCP_PROJECT:+ --project ${GCP_PROJECT}} --quiet"
}

if [ "$MODE" = "plan" ]; then
  log "PLAN (no apply). One-command GLM-5.2 demo on tier '$GCP_TIER' ($GLM_MACHINE_TYPE, ${GLM_GPU_COUNT}x ${GLM_GPU_LABEL}, sm_${GLM_COMPUTE_CAP}) as '$VM_NAME' in $GCP_ZONE."
  echo "# ============================================================================"
  echo "# DEMO step 1 — PROVISION + SERVE (delegated to scripts/gcp-glm-serve.sh):"
  echo "# ============================================================================"
  render_serve_plan
  echo
  print_cache_value_steps
  echo
  print_teardown_steps
  echo
  log "to run it for real: GCP_PROJECT=<id> HF_TOKEN=<hf> $0 --apply"
  exit 0
fi

# --- apply --------------------------------------------------------------------
command -v gcloud >/dev/null || die "gcloud not found — install the Cloud SDK"
[ -n "${GCP_PROJECT:-}" ] || die "GCP_PROJECT is required for --apply"

# Teardown ALWAYS runs (unless KEEP=1): a crashed demo must not leave an 8x H100 node
# burning. The trap fires on any exit after the VM is created. The local tunnel is always
# stopped, including KEEP=1 debug runs.
teardown() {
  [ -n "$KEEP" ] && { warn "KEEP=1 — leaving $VM_NAME up; delete it with: gcloud compute instances delete $VM_NAME --zone $GCP_ZONE --quiet"; return 0; }
  log "tearing down $VM_NAME (zero residual cost)"
  gcloud compute instances delete "$VM_NAME" --zone "$GCP_ZONE" ${GCP_PROJECT:+--project "$GCP_PROJECT"} --quiet || true
}
cleanup() {
  stop_tunnel
  teardown
}
trap cleanup EXIT

log "DEMO step 1 — provision + serve GLM-5.2 on $VM_NAME via the pure fak kernel"
GCP_TIER="$GCP_TIER" VM_NAME="$VM_NAME" GLM_PORT="$GLM_PORT" SERVE="$SERVE" EP_RANKS="$EP_RANKS" \
LOCAL_TUNNEL_PORT="$LOCAL_TUNNEL_PORT" GCP_ZONE="$GCP_ZONE" GCP_PROJECT="$GCP_PROJECT" \
FAK_STARTUP_PATCH_FILE="${FAK_STARTUP_PATCH_FILE:-}" \
FAK_REPO_REF="${FAK_REPO_REF:-}" \
  bash "$ROOT/scripts/gcp-glm-serve.sh" --apply

if ! wait_for_remote_ready; then
  collect_remote_witness "failed-ready" || true
  exit 1
fi
collect_remote_witness "ready" || true
start_tunnel
run_probe_turns
summarize_probe_perf
scrape_cache_value
collect_remote_witness "post-probe" || true

log "DEMO complete; witnesses copied under $WITNESS_DIR"
