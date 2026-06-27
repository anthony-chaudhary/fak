#!/usr/bin/env bash
# gcp-fleet-janitor.sh — the DURABLE, operator-free idle-GPU janitor. It runs from a
# control plane under an owner/compute-admin identity, sweeps EVERY GPU instance in a
# GCP project on a timer, and reaps the ones that are genuinely idle — with no human
# in the loop and no dependency on each VM being able to reap itself.
#
# WHY THIS EXISTS (the lesson that built it): on 2026-06-27 an 8xA100 box (~$29/hr)
# served ONE smoke turn, its `fak serve` process then died, and the box sat at 0% GPU
# for 8+ hours burning ~$250 because nothing was watching. The on-VM self-reaper
# (scripts/gcp-idle-reaper.sh) is the right primitive for FUTURE boxes, but it cannot
# help a box whose serve has already crashed (its /metrics is gone) or a box whose
# attached SA lacks the compute scope/role to delete itself. This janitor closes both
# gaps from OUTSIDE: it judges idle by GPU UTILIZATION (ground truth — catches a dead
# serve, not just zero traffic) read over IAP, and it acts with the control plane's
# own owner identity, so it works on any box regardless of that box's SA.
#
# IDLE SIGNAL — max GPU utilization across all GPUs, via `nvidia-smi` over IAP SSH.
# A box is "busy" if ANY GPU is above UTIL_PCT (default 5%). A box is "idle" only when
# every GPU has been at/below that for the whole IDLE_MINUTES window. This is strictly
# better than polling a serve endpoint: a crashed serve reads as idle (correctly),
# whereas an endpoint poll reads as "unreachable" and is ambiguous.
#
# POLICY (declarative, travels with the instance — no operator memory required):
#   * an instance labelled  fak-reaper=keep   is NEVER reaped (e.g. fak-realmodel).
#   * an instance labelled  fak-reaper=stop   is STOPPED when idle (keeps its disk).
#   * everything else with a GPU defaults to ON_IDLE (delete) when idle.
#   * GRACE_MINUTES from the instance's lastStartTimestamp blocks reaping a box that
#     may still be staging a multi-hundred-GB model.
#
# SAFE BY DEFAULT: DRY-RUN unless --live. Every decision is logged as JSONL. Idle is
# tracked per-instance in a local state dir, so a box must be idle across CONSECUTIVE
# sweeps spanning IDLE_MINUTES — one transient 0% reading never reaps anything.
#
# Usage:
#   ./gcp-fleet-janitor.sh                         # one DRY-RUN sweep of the project
#   ./gcp-fleet-janitor.sh --live                  # one sweep that may actually reap
#   PROJECT=my-proj IDLE_MINUTES=60 ./gcp-fleet-janitor.sh --live
#   ./gcp-fleet-janitor.sh --install-task          # register the unattended Windows task
#
# Knobs (env):
#   PROJECT        gcloud project to sweep        (default: active gcloud project)
#   IDLE_MINUTES   reap after this long at <=UTIL (default 60)
#   GRACE_MINUTES  skip a box younger than this   (default 90)
#   UTIL_PCT       GPU% at/below = idle           (default 5)
#   ON_IDLE        delete | stop default action   (default delete)
#   STATE_DIR      local idle-clock + log dir      (default $LOCALAPPDATA/Fleet/janitor or /tmp)
#   SSH_TIMEOUT    per-box IAP SSH timeout secs    (default 60)
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
IDLE_MINUTES="${IDLE_MINUTES:-60}"
GRACE_MINUTES="${GRACE_MINUTES:-90}"
UTIL_PCT="${UTIL_PCT:-5}"
ON_IDLE="${ON_IDLE:-delete}"
SSH_TIMEOUT="${SSH_TIMEOUT:-60}"
LIVE=0

case "${1:-}" in
  --live)          LIVE=1 ;;
  --dry-run|"")    ;;
  --install-task)  INSTALL_TASK=1 ;;
  --help|-h)       sed -n '2,60p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

case "$ON_IDLE" in delete|stop) ;; *) echo "ON_IDLE must be delete|stop (got '$ON_IDLE')" >&2; exit 2 ;; esac
[ -n "$PROJECT" ] || { echo "no project (set PROJECT or 'gcloud config set project')" >&2; exit 2; }

# State dir: prefer the fleet state root this box already uses for watchdog tooling.
if [ -z "${STATE_DIR:-}" ]; then
  if [ -n "${FLEET_STATE_DIR:-}" ]; then STATE_DIR="$FLEET_STATE_DIR/janitor"
  elif [ -n "${LOCALAPPDATA:-}" ]; then STATE_DIR="$LOCALAPPDATA/Fleet/janitor"
  else STATE_DIR="${TMPDIR:-/tmp}/fak-fleet-janitor"; fi
fi
mkdir -p "$STATE_DIR"
JSONL="$STATE_DIR/janitor.jsonl"
NOW="$(date -u +%s)"
NOW_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODE=$([ "$LIVE" = "1" ] && echo LIVE || echo DRY-RUN)

log()  { printf '%s  %s\n' "$NOW_ISO" "$*" >&2; }
emit() { printf '%s\n' "$1" >> "$JSONL" 2>/dev/null || true; }

# --- optional: register the unattended Windows scheduled task and exit ------------
# Durability: a control-plane sweep is only operator-free if it runs as a managed job.
# This box already drives the fleet via Windows scheduled tasks, so we register there.
if [ "${INSTALL_TASK:-0}" = "1" ]; then
  SELF_WIN="$(cygpath -w "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" 2>/dev/null || echo "$0")"
  log "registering unattended task 'FakFleetJanitor' (every 15 min, --live) for $SELF_WIN"
  cat <<EOF >&2
Run this once (elevated PowerShell) to install the operator-free sweep:

  \$bash = (Get-Command bash).Source
  schtasks /Create /TN FakFleetJanitor /SC MINUTE /MO 15 /RL HIGHEST /F ^
    /TR "\$bash -lc 'PROJECT=$PROJECT IDLE_MINUTES=$IDLE_MINUTES ON_IDLE=$ON_IDLE $SELF_WIN --live'"

It sweeps every 15 min under your logged-in gcloud identity and reaps idle GPU boxes
per the fak-reaper=keep|stop labels. Remove with:  schtasks /Delete /TN FakFleetJanitor /F
EOF
  exit 0
fi

# --- enumerate every GPU instance in the project ----------------------------------
# RUNNING instances with at least one guestAccelerator. TSV: name<TAB>zone<TAB>start<TAB>label.
mapfile -t ROWS < <(gcloud compute instances list --project "$PROJECT" \
  --filter="status=RUNNING AND guestAccelerators:*" \
  --format="value(name, zone.basename(), lastStartTimestamp, labels.fak-reaper)" 2>/dev/null || true)

log "SWEEP $MODE project=$PROJECT gpu-instances=${#ROWS[@]} idle>=${IDLE_MINUTES}m util<=${UTIL_PCT}% grace=${GRACE_MINUTES}m default=$ON_IDLE"

reaped=0
for row in "${ROWS[@]}"; do
  [ -z "$row" ] && continue
  name="$(printf '%s' "$row" | cut -f1)"
  zone="$(printf '%s' "$row" | cut -f2)"
  start="$(printf '%s' "$row" | cut -f3)"
  label="$(printf '%s' "$row" | cut -f4)"
  st="$STATE_DIR/${name}.${zone}.since"

  # KEEP label => never touch (e.g. fak-realmodel, the long-lived demo host).
  if [ "$label" = "keep" ]; then
    log "  KEEP $name ($zone): labelled fak-reaper=keep — skipped."
    rm -f "$st" 2>/dev/null || true
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"keep\",\"reason\":\"label_keep\"}"
    continue
  fi

  action="$([ "$label" = "stop" ] && echo stop || echo "$ON_IDLE")"

  # Boot grace: skip a box that may still be staging the model.
  if [ -n "$start" ]; then
    start_s="$(date -u -d "$start" +%s 2>/dev/null || echo 0)"
    age_min=$(( (NOW - start_s) / 60 ))
    if [ "$start_s" -gt 0 ] && [ "$age_min" -lt "$GRACE_MINUTES" ]; then
      log "  GRACE $name ($zone): age ${age_min}m < ${GRACE_MINUTES}m — skipped."
      rm -f "$st" 2>/dev/null || true
      emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"grace\",\"age_min\":$age_min}"
      continue
    fi
  fi

  # GPU utilization over IAP — the ground-truth idle signal (catches a dead serve).
  util="$(timeout "$SSH_TIMEOUT" gcloud compute ssh "$name" --zone "$zone" --project "$PROJECT" \
    --tunnel-through-iap --command='nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1' \
    2>/dev/null | tr -dc '0-9' || true)"

  if [ -z "$util" ]; then
    # Could not read GPU (SSH/IAP hiccup or driver not ready). Be conservative: do NOT
    # treat unreadable as idle — reset the clock so a transient failure never reaps.
    log "  UNREADABLE $name ($zone): nvidia-smi over IAP returned nothing — not counting as idle (clock reset)."
    rm -f "$st" 2>/dev/null || true
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"unreadable\",\"reason\":\"no_gpu_read\"}"
    continue
  fi

  if [ "$util" -gt "$UTIL_PCT" ]; then
    log "  BUSY $name ($zone): max GPU ${util}% > ${UTIL_PCT}% — idle clock reset."
    rm -f "$st" 2>/dev/null || true
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"busy\",\"gpu_pct\":$util}"
    continue
  fi

  # Idle reading. Stamp/age the per-instance idle clock.
  if [ ! -f "$st" ]; then echo "$NOW" > "$st"; fi
  since="$(cat "$st" 2>/dev/null || echo "$NOW")"
  idle_min=$(( (NOW - since) / 60 ))
  if [ "$idle_min" -lt "$IDLE_MINUTES" ]; then
    log "  IDLE $name ($zone): GPU ${util}% for ${idle_min}m / ${IDLE_MINUTES}m — waiting."
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"waiting\",\"gpu_pct\":$util,\"idle_min\":$idle_min}"
    continue
  fi

  # Threshold met — reap (or, dry-run, log what we WOULD do).
  if [ "$LIVE" != "1" ]; then
    log "  WOULD-$action $name ($zone): idle ${idle_min}m >= ${IDLE_MINUTES}m, GPU ${util}%. [DRY-RUN]"
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"would_$action\",\"gpu_pct\":$util,\"idle_min\":$idle_min}"
    continue
  fi

  log "  REAP: $action $name ($zone) — idle ${idle_min}m, GPU ${util}%."
  if gcloud compute instances "$action" "$name" --zone "$zone" --project "$PROJECT" --quiet 2>&1 | tail -1 >&2; then
    reaped=$((reaped+1))
    rm -f "$st" 2>/dev/null || true
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"reaped_$action\",\"gpu_pct\":$util,\"idle_min\":$idle_min}"
    log "  REAPED: $action $name OK."
  else
    rc=$?
    log "  REAP_FAILED: $action $name rc=$rc (kept up)."
    emit "{\"ts\":\"$NOW_ISO\",\"mode\":\"$MODE\",\"vm\":\"$name\",\"zone\":\"$zone\",\"action\":\"reap_failed\",\"rc\":$rc}"
  fi
done

log "SWEEP done: reaped=$reaped (mode=$MODE)."
