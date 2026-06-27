#!/usr/bin/env bash
# gcp-idle-reaper-install.sh — retrofit the on-VM idle gardener (gcp-idle-reaper.sh)
# onto an ALREADY-RUNNING fak GPU serve node, over IAP, without a redeploy. This is
# the "cover today's burn now" half of the gardener: the provision scripts bake the
# reaper into every NEW box, but the boxes already up need it pushed onto them.
#
# It copies the reaper + a systemd service/timer pair onto the VM and enables the
# timer (fires every 5 min). After this, the box watches its OWN inference counter
# and self-stops or self-deletes after IDLE_MINUTES of no model turns — so a crashed
# laptop control-session can't leave it burning. See scripts/gcp-idle-reaper.sh.
#
# PLAN BY DEFAULT. With no --apply it prints the cost label, the SA-permission
# preflight it WOULD run, and the exact remote steps, then exits 0 — reviewable
# without touching the VM. --apply runs them (needs authenticated gcloud + IAP).
#
# Usage:
#   ./gcp-idle-reaper-install.sh fak-qwen-serve us-central1-f                 # PLAN
#   ./gcp-idle-reaper-install.sh fak-qwen-serve us-central1-f --apply         # install (delete on idle)
#   ./gcp-idle-reaper-install.sh fak-glm-serve asia-southeast1-b --on-idle delete --apply
#   ./gcp-idle-reaper-install.sh fak-qwen-serve us-central1-f --on-idle stop --idle-minutes 60 --apply
#
# Args:  <vm-name> <zone>   (required, positional)
# Flags:
#   --on-idle stop|delete   action when idle           (default delete)
#   --idle-minutes N        idle window in minutes      (default 60)
#   --grace-minutes N       boot grace floor in minutes (default 90)
#   --port N                served /metrics port         (default 8000; qwen uses 8080)
#   --apply                 actually install (default: PLAN only)
#   --project ID            gcloud project               (default: active gcloud project)
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REAPER="$SCRIPT_DIR/gcp-idle-reaper.sh"

VM_NAME=""; ZONE=""; ON_IDLE="delete"; IDLE_MINUTES=60; GRACE_MINUTES=90; PORT=8000
MODE="plan"; PROJECT="${GCP_PROJECT:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --on-idle)      ON_IDLE="$2"; shift 2 ;;
    --idle-minutes) IDLE_MINUTES="$2"; shift 2 ;;
    --grace-minutes)GRACE_MINUTES="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --project)      PROJECT="$2"; shift 2 ;;
    --apply)        MODE="apply"; shift ;;
    --help|-h)      sed -n '2,40p' "$0"; exit 0 ;;
    -*)             echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
    *)              if [ -z "$VM_NAME" ]; then VM_NAME="$1"; elif [ -z "$ZONE" ]; then ZONE="$1"; else echo "unexpected arg: $1" >&2; exit 2; fi; shift ;;
  esac
done

log() { printf '\033[36m[reaper-install]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[reaper-install] %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "$VM_NAME" ] && [ -n "$ZONE" ] || die "usage: $0 <vm-name> <zone> [flags] (see --help)"
[ -f "$REAPER" ] || die "reaper not found at $REAPER (run from the fak checkout)"
case "$ON_IDLE" in stop|delete) ;; *) die "--on-idle must be stop|delete (got '$ON_IDLE')";; esac
PROJ_FLAG=""; [ -n "$PROJECT" ] && PROJ_FLAG="--project $PROJECT"

# Cost ladder for the box's machine type, from the single tier registry, so the
# operator can confirm this box's hourly burn before arming a delete.
cost_hint() {
  command -v python >/dev/null 2>&1 && PY=python || PY=python3
  "$PY" "$ROOT/tools/gcp_accel.py" 2>/dev/null | awk 'NR>1{print "    "$0}' | head -12
}

# --- the systemd unit + timer (rendered remotely) ---------------------------------
render_service() {
  cat <<UNIT
[Unit]
Description=fak idle-GPU reaper (self-${ON_IDLE} after idle)
After=network-online.target

[Service]
Type=oneshot
Environment=PORT=${PORT}
Environment=IDLE_MINUTES=${IDLE_MINUTES}
Environment=GRACE_MINUTES=${GRACE_MINUTES}
Environment=ON_IDLE=${ON_IDLE}
Environment=ON_IDLE_LIVE=1
ExecStart=/usr/bin/env bash /opt/fak/scripts/gcp-idle-reaper.sh --live
UNIT
}
render_timer() {
  cat <<'UNIT'
[Unit]
Description=run the fak idle-GPU reaper every 5 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
UNIT
}

# The remote install: stage the reaper into /opt/fak/scripts (the repo clone), drop
# the unit + timer, and enable. Idempotent — re-running just refreshes the files.
remote_install_cmd() {
  cat <<'REMOTE'
set -euxo pipefail
sudo install -d -m 0755 /opt/fak/scripts /var/lib/fak-idle-reaper
sudo install -m 0755 /tmp/gcp-idle-reaper.sh /opt/fak/scripts/gcp-idle-reaper.sh
sudo install -m 0644 /tmp/fak-idle-reaper.service /etc/systemd/system/fak-idle-reaper.service
sudo install -m 0644 /tmp/fak-idle-reaper.timer   /etc/systemd/system/fak-idle-reaper.timer
sudo systemctl daemon-reload
sudo systemctl enable --now fak-idle-reaper.timer
echo "--- timer status ---"; systemctl status --no-pager fak-idle-reaper.timer || true
echo "--- one immediate (live) tick ---"; sudo systemctl start fak-idle-reaper.service || true
echo "--- last reaper log lines ---"; sudo journalctl -u fak-idle-reaper --no-pager -n 20 || true
REMOTE
}

echo "# idle-reaper retrofit plan for: $VM_NAME (zone $ZONE)"
echo "#   on-idle=$ON_IDLE  idle-minutes=$IDLE_MINUTES  grace-minutes=$GRACE_MINUTES  port=$PORT"
echo "# cost ladder (tools/gcp_accel.py) — confirm this box's hourly burn:"
cost_hint
echo
echo "# 1) PREFLIGHT — the attached service account must be able to self-${ON_IDLE}:"
echo "gcloud compute instances describe $VM_NAME --zone $ZONE $PROJ_FLAG --format='value(serviceAccounts[].email,serviceAccounts[].scopes)'"
echo "#    the SA needs compute.instances.${ON_IDLE} on this instance (roles/compute.instanceAdmin.v1"
echo "#    or a tight custom role). If it lacks it, the reaper logs REAP_DENIED and the box stays UP."
echo
echo "# 2) STAGE the reaper + units onto the VM and ENABLE the timer (over IAP):"
echo "gcloud compute scp $REAPER $VM_NAME:/tmp/gcp-idle-reaper.sh --zone $ZONE --tunnel-through-iap $PROJ_FLAG"
echo "gcloud compute scp <rendered service> <rendered timer> $VM_NAME:/tmp/ --zone $ZONE --tunnel-through-iap $PROJ_FLAG"
echo "gcloud compute ssh $VM_NAME --zone $ZONE --tunnel-through-iap $PROJ_FLAG -- '<remote install>'"

if [ "$MODE" = "plan" ]; then
  echo
  log "PLAN only. Re-run with --apply to perform the install."
  exit 0
fi

# --- apply --------------------------------------------------------------------
command -v gcloud >/dev/null || die "gcloud not found — install the Cloud SDK"

log "PREFLIGHT: reading $VM_NAME's attached service account…"
SA_LINE="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" $PROJ_FLAG \
  --format='value(serviceAccounts[].email)' 2>/dev/null || true)"
[ -n "$SA_LINE" ] && log "attached SA: $SA_LINE" || log "WARN: could not read attached SA (continuing; the remote tick will report REAP_DENIED if it lacks perms)."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
render_service > "$TMP/fak-idle-reaper.service"
render_timer   > "$TMP/fak-idle-reaper.timer"

log "staging reaper + units onto $VM_NAME over IAP…"
gcloud compute scp "$REAPER" "$VM_NAME:/tmp/gcp-idle-reaper.sh" \
  --zone "$ZONE" --tunnel-through-iap $PROJ_FLAG
gcloud compute scp "$TMP/fak-idle-reaper.service" "$TMP/fak-idle-reaper.timer" \
  "$VM_NAME:/tmp/" --zone "$ZONE" --tunnel-through-iap $PROJ_FLAG

log "installing + enabling the timer on $VM_NAME…"
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --tunnel-through-iap $PROJ_FLAG \
  -- "$(remote_install_cmd)"

log "done. The reaper now ticks every 5 min on $VM_NAME (on-idle=$ON_IDLE, idle=${IDLE_MINUTES}m)."
log "watch it:  gcloud compute ssh $VM_NAME --zone $ZONE --tunnel-through-iap $PROJ_FLAG -- 'journalctl -u fak-idle-reaper -f'"
