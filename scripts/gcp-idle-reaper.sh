#!/usr/bin/env bash
# gcp-idle-reaper.sh — the on-VM idle gardener for a fak GPU serve node. It runs ON
# the serving VM (fired by a systemd timer every few minutes), watches the kernel's
# OWN inference counter, and STOPS or DELETES its own instance after a configurable
# idle window — so a crashed control-session on a laptop can never leave an A100
# burning unattended.
#
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │ this GPU VM                                                            │
#   │   fak serve  ──/metrics──▶  fak_gateway_inference_requests_total       │
#   │       ▲                              │                                 │
#   │       │ no new turns for IDLE_MINUTES│                                 │
#   │   gcp-idle-reaper.sh (this) ─────────┘                                 │
#   │       │ reap = stop|delete SELF via the metadata-server identity        │
#   │       ▼                                                                 │
#   │   gcloud compute instances {stop,delete} <self> --zone <self> --quiet   │
#   └──────────────────────────────────────────────────────────────────────┘
#
# WHY ON THE VM (not a laptop cron): the whole point is to survive the death of the
# session that launched the box. A gardener on the dev box dies WITH the session it
# is meant to outlive. So the reaper lives on the VM and needs no external control
# plane — it reads its own identity from the GCP metadata server and acts on itself
# with the VM's attached service account. This mirrors the self-clean teardown the
# bench VMs already use (tools/gcp_bench.py).
#
# IDLE SIGNAL — fak_gateway_inference_requests_total (model turns), NOT
# fak_gateway_http_requests_total. The inference counter advances ONLY on a real
# completion turn, so /healthz and /metrics scrapes do not keep the box alive, and
# an in-flight workflow does. A flat counter for >= IDLE_MINUTES means genuinely idle.
#
# FALSE-REAP GUARDS (a multi-hundred-GB GLM load takes ~40 min):
#   * GRACE_MINUTES boot floor — never reap within this many minutes of boot.
#   * /metrics unreachable => the serve is still coming up, NOT idle: reset the clock.
#   * counter line absent but /metrics reachable => zero turns served yet (sum 0),
#     which is a normal just-loaded state; the grace floor covers it.
#
# House style mirrors tools/runaway_process_reaper.ps1: DRY-RUN is the safe default
# (set ON_IDLE_LIVE=1 or pass --live to actually reap), every tick writes a JSONL
# provenance record, and a single threshold trips the action.
#
# Usage (normally invoked by the systemd timer, but runnable by hand):
#   ./gcp-idle-reaper.sh                 # one tick, DRY-RUN: log what it WOULD do
#   ./gcp-idle-reaper.sh --live          # one tick, may actually stop/delete
#   IDLE_MINUTES=30 ON_IDLE=stop ./gcp-idle-reaper.sh --live
#
# Knobs (env):
#   PORT           served gateway port to scrape /metrics on   (default 8000)
#   IDLE_MINUTES   reap after this many minutes of NO new turns (default 60)
#   GRACE_MINUTES  never reap within this many minutes of boot  (default 90)
#   ON_IDLE        stop | delete — the action when idle         (default delete)
#   ON_IDLE_LIVE   1 => actually act (same as --live)           (default 0 = dry-run)
#   STATE_DIR      where to persist the idle clock + log        (default /var/lib/fak-idle-reaper)
#   METRICS_HOST   host to scrape                               (default 127.0.0.1)
set -euo pipefail

PORT="${PORT:-8000}"
IDLE_MINUTES="${IDLE_MINUTES:-60}"
GRACE_MINUTES="${GRACE_MINUTES:-90}"
ON_IDLE="${ON_IDLE:-delete}"
STATE_DIR="${STATE_DIR:-/var/lib/fak-idle-reaper}"
METRICS_HOST="${METRICS_HOST:-127.0.0.1}"
LIVE="${ON_IDLE_LIVE:-0}"

case "${1:-}" in
  --live)        LIVE=1 ;;
  --dry-run|"")  ;;
  --help|-h)     sed -n '2,60p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

case "$ON_IDLE" in
  stop|delete) ;;
  *) echo "ON_IDLE must be 'stop' or 'delete' (got '$ON_IDLE')" >&2; exit 2 ;;
esac

MODE=$([ "$LIVE" = "1" ] && echo LIVE || echo DRY-RUN)
NOW="$(date -u +%s)"
NOW_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
METRICS_URL="http://${METRICS_HOST}:${PORT}/metrics"
META="http://metadata.google.internal/computeMetadata/v1/instance"
MDH=(-H "Metadata-Flavor: Google")

mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/state.json"
JSONL="$STATE_DIR/reaper.jsonl"

log()  { printf '%s  %s\n' "$NOW_ISO" "$*" >&2; }
# Append one structured provenance record per tick (best-effort; never fails the tick).
emit() { # emit <action> <reason> <sum> <idle_secs>
  printf '{"ts":"%s","mode":"%s","action":"%s","reason":"%s","on_idle":"%s","inference_sum":%s,"idle_secs":%s,"idle_minutes_cfg":%s,"grace_minutes_cfg":%s,"vm":"%s","zone":"%s"}\n' \
    "$NOW_ISO" "$MODE" "$1" "$2" "$ON_IDLE" "${3:-null}" "${4:-null}" "$IDLE_MINUTES" "$GRACE_MINUTES" "${VM_NAME:-?}" "${ZONE_SHORT:-?}" \
    >> "$JSONL" 2>/dev/null || true
}

# --- boot grace floor: never reap a box that may still be staging the model -------
# /proc/uptime field 1 is seconds since boot. A fresh GLM load (~40 min) must never
# read as "60 min idle"; the grace floor blocks any reap inside the staging window.
UPTIME_SEC="$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null || echo 0)"
GRACE_SEC=$(( GRACE_MINUTES * 60 ))
IDLE_SEC=$(( IDLE_MINUTES * 60 ))

# --- read the inference counter (sum across finish reasons) -----------------------
# A reachable /metrics with NO inference line => 0 turns served yet (just-loaded),
# which is fine — the grace floor handles that case. An UNREACHABLE /metrics means
# the serve is not up (still staging or crashed-and-restarting): treat as "starting,
# not idle" and RESET the clock so we never reap a box that is mid-load.
if ! RAW="$(curl -fsS --max-time 10 "$METRICS_URL" 2>/dev/null)"; then
  log "TICK $MODE metrics UNREACHABLE ($METRICS_URL) — serve still coming up or restarting; treating as STARTING, clock reset."
  emit "starting" "metrics_unreachable"
  # Reset the idle clock: record NOW with a sentinel sum so the next reachable tick
  # starts a fresh window rather than inheriting a stale flat-counter timestamp.
  printf '{"sum":-1,"since":%s,"updated":%s}\n' "$NOW" "$NOW" > "$STATE"
  exit 0
fi

SUM="$(printf '%s\n' "$RAW" \
  | awk '/^fak_gateway_inference_requests_total\{/ { s += $NF } END { printf "%d", s }')"
SUM="${SUM:-0}"

# --- self identity from the metadata server (only needed when we might act) -------
VM_NAME="$(curl -fsS --max-time 5 "${MDH[@]}" "$META/name" 2>/dev/null || echo '')"
ZONE_PATH="$(curl -fsS --max-time 5 "${MDH[@]}" "$META/zone" 2>/dev/null || echo '')"
ZONE_SHORT="${ZONE_PATH##*/}"   # ".../zones/asia-southeast1-b" -> "asia-southeast1-b"

# --- compare against the persisted idle clock -------------------------------------
PREV_SUM=""; SINCE=""
if [ -f "$STATE" ]; then
  PREV_SUM="$(awk -F'[:,]' '/"sum"/ {gsub(/[^0-9-]/,"",$2); print $2; exit}' "$STATE" 2>/dev/null || echo '')"
  SINCE="$(awk -F'[:,]' '{for(i=1;i<=NF;i++) if($i ~ /"since"/){gsub(/[^0-9]/,"",$(i+1)); print $(i+1); exit}}' "$STATE" 2>/dev/null || echo '')"
fi

if [ -z "$PREV_SUM" ] || [ "$PREV_SUM" = "-1" ] || [ "$SUM" != "$PREV_SUM" ]; then
  # First reachable observation, recovering from a STARTING sentinel, or the counter
  # advanced => the box is (or just was) busy. Stamp the new sum and reset the clock.
  printf '{"sum":%s,"since":%s,"updated":%s}\n' "$SUM" "$NOW" "$NOW" > "$STATE"
  log "TICK $MODE busy/observed: inference_sum=$SUM (was '${PREV_SUM:-none}') — idle clock reset."
  emit "observe" "counter_advanced_or_first" "$SUM" 0
  exit 0
fi

# Counter is FLAT vs the persisted value. How long has it been flat?
SINCE="${SINCE:-$NOW}"
FLAT_SEC=$(( NOW - SINCE ))
[ "$FLAT_SEC" -lt 0 ] && FLAT_SEC=0

if [ "$UPTIME_SEC" -lt "$GRACE_SEC" ]; then
  log "TICK $MODE flat ${FLAT_SEC}s but within boot grace (uptime ${UPTIME_SEC}s < ${GRACE_SEC}s) — no reap."
  emit "grace" "within_boot_grace" "$SUM" "$FLAT_SEC"
  exit 0
fi

if [ "$FLAT_SEC" -lt "$IDLE_SEC" ]; then
  log "TICK $MODE idle ${FLAT_SEC}s / threshold ${IDLE_SEC}s (sum=$SUM) — not yet."
  emit "waiting" "below_idle_threshold" "$SUM" "$FLAT_SEC"
  exit 0
fi

# --- threshold tripped: reap (or, in dry-run, log what we WOULD do) ----------------
if [ -z "$VM_NAME" ] || [ -z "$ZONE_SHORT" ]; then
  log "TICK $MODE IDLE ${FLAT_SEC}s but could not resolve self identity from metadata (name='$VM_NAME' zone='$ZONE_SHORT') — refusing to act."
  emit "abort" "no_self_identity" "$SUM" "$FLAT_SEC"
  exit 0
fi

if [ "$LIVE" != "1" ]; then
  log "WOULD-$ON_IDLE $VM_NAME (zone $ZONE_SHORT): idle ${FLAT_SEC}s >= ${IDLE_SEC}s, sum flat at $SUM. [DRY-RUN]"
  emit "would_${ON_IDLE}" "idle_threshold_met" "$SUM" "$FLAT_SEC"
  exit 0
fi

log "REAP: ${ON_IDLE} $VM_NAME (zone $ZONE_SHORT) — idle ${FLAT_SEC}s >= ${IDLE_SEC}s, inference_sum flat at $SUM."
emit "${ON_IDLE}" "idle_threshold_met" "$SUM" "$FLAT_SEC"

if ! command -v gcloud >/dev/null 2>&1; then
  log "REAP_DENIED: gcloud not found on the VM — cannot self-${ON_IDLE}."
  emit "reap_denied" "no_gcloud" "$SUM" "$FLAT_SEC"
  exit 1
fi

# Self-act with the VM's attached service account. --quiet skips the confirm prompt.
# delete tears down the VM + boot disk (zero residual cost, full reload on relaunch);
# stop halts compute (GPU billing ends) but keeps the disk billing.
if gcloud compute instances "$ON_IDLE" "$VM_NAME" --zone "$ZONE_SHORT" --quiet; then
  log "REAPED: ${ON_IDLE} $VM_NAME issued OK."
  emit "reaped_ok" "idle_threshold_met" "$SUM" "$FLAT_SEC"
else
  rc=$?
  log "REAP_DENIED: gcloud compute instances ${ON_IDLE} $VM_NAME failed (rc=$rc) — likely the attached service account lacks compute.instances.${ON_IDLE}. The box stays UP; fix the SA role or stop it manually."
  emit "reap_denied" "gcloud_failed_rc_${rc}" "$SUM" "$FLAT_SEC"
  exit "$rc"
fi
