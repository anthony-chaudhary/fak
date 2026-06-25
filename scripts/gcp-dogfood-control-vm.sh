#!/usr/bin/env bash
# gcp-dogfood-control-vm.sh — stand up the GCP Tier-2 always-on dogfood control VM
# (issue #732). docs/fak/always-on-dogfood-server.md describes Tier 2 (a cheap e2-small
# running the guarded fleet + a shared `fak serve` gateway 24/7); this is the script that
# actually provisions it. A VM never sleeps, so it is the steady-state overflow lane next
# to the Mac (Tier 1) — and unlike launchd, the units are systemd (the Linux analogue of
# tools/com.fak.serve-gateway.plist + tools/com.fak.dogfood-fleet.plist).
#
# PLAN BY DEFAULT. With no creds this prints the exact `gcloud` command + the VM startup
# script and exits 0 — so the whole deploy is reviewable without a GCP project. `--apply`
# runs it, and requires `gcloud` + an authenticated project (GCP_PROJECT).
#
# Usage:
#   ./scripts/gcp-dogfood-control-vm.sh                 # PLAN: print gcloud + startup script
#   GCP_PROJECT=my-proj ./scripts/gcp-dogfood-control-vm.sh --apply   # create the VM
#
# Knobs (env):
#   GCP_PROJECT     the GCP project id            (REQUIRED for --apply)
#   GCP_ZONE        compute zone                  (default us-central1-a)
#   GCP_MACHINE     machine type                  (default e2-small — the Tier-2 steady state)
#   VM_NAME         instance name                 (default fak-dogfood-control)
#   FAK_REPO_URL    repo to clone on the VM       (default the public fak remote)
#   TAILSCALE_AUTHKEY  optional tailscale authkey to join the private overlay on boot
set -euo pipefail

GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_MACHINE="${GCP_MACHINE:-e2-small}"
VM_NAME="${VM_NAME:-fak-dogfood-control}"
FAK_REPO_URL="${FAK_REPO_URL:-https://github.com/anthony-chaudhary/fak.git}"

MODE="plan"
case "${1:-}" in
  --apply)    MODE="apply" ;;
  --plan|"")  MODE="plan" ;;
  --help|-h)  sed -n '2,24p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

log() { printf '\033[36m[gcp-tier2]\033[0m %s\n' "$*" >&2; }

# --- the VM startup script (cloud-init) ---------------------------------------
# Runs once on first boot: installs Go + python + git, clones the repo, builds fak, and
# installs two systemd units — a 24/7 `fak serve` gateway and a guarded-dispatch tick
# timer (every 30 min). Self-contained so the VM converges with no further SSH.
render_startup_script() {
  cat <<STARTUP
#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git golang-go python3 curl

# Optional: join the Tailscale overlay so other nodes can dial the gateway.
if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey "${TAILSCALE_AUTHKEY:-}" --hostname "${VM_NAME}" || true
fi

# Clone + build into a stable location owned by a service account user.
install -d -o root -g root /opt
git clone "${FAK_REPO_URL}" /opt/fak || (cd /opt/fak && git pull --ff-only)
cd /opt/fak
go build -o /opt/fak/tools/.bin/fak ./cmd/fak

# --- systemd: the 24/7 fak serve gateway (Linux analogue of the serve-gateway plist) ---
# Dogfood the 100k-session lever by DEFAULT: --compact-history-budget shrinks a long
# Anthropic-passthrough session's OLD turns to a resident-token budget while keeping the
# cache_control prefix byte-identical, so we eat our own dog food on the real dev gateway.
# Override the budget (or disable it: 0) with FAK_COMPACT_BUDGET before running this script;
# watch it on /metrics (fak_gateway_compaction_*) and in the `fak guard` exit summary.
: "${FAK_COMPACT_BUDGET:=8000}"
cat >/etc/systemd/system/fak-serve-gateway.service <<UNIT
[Unit]
Description=fak serve — shared dogfood gateway (anthropic passthrough, history compaction on)
After=network-online.target
Wants=network-online.target
[Service]
WorkingDirectory=/opt/fak
Environment=FAK_AUDIT_JOURNAL=/opt/fak/tools/_watchdog/serve_audit.jsonl
ExecStart=/opt/fak/tools/.bin/fak serve --provider anthropic --base-url https://api.anthropic.com --addr 127.0.0.1:8080 --compact-history-budget ${FAK_COMPACT_BUDGET}
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
UNIT

# --- systemd: the guarded dispatch tick (Linux analogue of the dogfood-fleet plist) ----
cat >/etc/systemd/system/fak-dogfood-fleet.service <<'UNIT'
[Unit]
Description=fak guarded dogfood dispatch tick (one bounded, preflight-gated worker)
After=fak-serve-gateway.service
[Service]
Type=oneshot
WorkingDirectory=/opt/fak
Environment=FLEET_DOGFOOD_GUARD=1
ExecStart=/usr/bin/python3 /opt/fak/tools/issue_dispatch.py --live --max-workers 1
UNIT

cat >/etc/systemd/system/fak-dogfood-fleet.timer <<'UNIT'
[Unit]
Description=fire the guarded dogfood dispatch tick every 30 minutes
[Timer]
OnBootSec=2min
OnUnitActiveSec=30min
[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now fak-serve-gateway.service
systemctl enable --now fak-dogfood-fleet.timer
STARTUP
}

# --- the gcloud create command ------------------------------------------------
build_gcloud_args() {
  printf 'gcloud compute instances create %q' "$VM_NAME"
  printf ' \\\n  --zone=%q' "$GCP_ZONE"
  printf ' \\\n  --machine-type=%q' "$GCP_MACHINE"
  printf ' \\\n  --image-family=debian-12 --image-project=debian-cloud'
  printf ' \\\n  --metadata-from-file=startup-script=<(rendered startup script)'
  if [ -n "${GCP_PROJECT:-}" ]; then printf ' \\\n  --project=%q' "$GCP_PROJECT"; fi
  return 0
}

if [ "$MODE" = "plan" ]; then
  log "PLAN (no apply). Tier-2 control VM: $GCP_MACHINE in $GCP_ZONE as '$VM_NAME'."
  echo "# gcloud command this would run:"
  build_gcloud_args
  echo
  echo
  echo "# --- VM startup script (cloud-init) ---"
  render_startup_script
  echo
  log "to apply: GCP_PROJECT=<id> $0 --apply   (requires authenticated gcloud)"
  exit 0
fi

# --- apply --------------------------------------------------------------------
command -v gcloud >/dev/null || { echo "gcloud not found — install the Cloud SDK" >&2; exit 1; }
[ -n "${GCP_PROJECT:-}" ] || { echo "GCP_PROJECT is required for --apply" >&2; exit 1; }

TMP_STARTUP="$(mktemp)"
trap 'rm -f "$TMP_STARTUP"' EXIT
render_startup_script > "$TMP_STARTUP"

log "creating $VM_NAME ($GCP_MACHINE) in $GCP_ZONE under project $GCP_PROJECT"
exec gcloud compute instances create "$VM_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$GCP_ZONE" \
  --machine-type="$GCP_MACHINE" \
  --image-family=debian-12 --image-project=debian-cloud \
  --metadata-from-file=startup-script="$TMP_STARTUP"
