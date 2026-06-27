#!/usr/bin/env bash
# gcp-qwen-serve.sh — stand up Qwen3.6-27B on a single GCP A100 via the PURE FAK KERNEL and
# make it usable DIRECTLY from Claude Code on this machine — the coding fallback for when a
# subscription account is unavailable.
#
#   ┌────────────────┐  /v1/messages  ┌──────────────────────────────┐  in-kernel  ┌──────────────────┐
#   │ this machine   │ ─────────────▶ │ fak serve (the PURE KERNEL)   │ ─────────▶ │ Qwen3.6-27B q4_k │
#   │ claude (Code)  │ ◀───────────── │ --backend cuda, OWN forward,  │ ◀───────── │ resident on ONE  │
#   │                │                │ adjudicates every tool call   │            │ A100-40GB (sm_80)│
#   └────────────────┘                └──────────────────────────────┘            └──────────────────┘
#       ANTHROPIC_BASE_URL = http://<tailscale-or-tunnel>:8080   (connect-fak-node.ps1/.sh)
#
# Unlike GLM-5.2 (a 466 GB MoE that needs --cpu-offload-experts across 8 GPUs), Qwen3.6-27B
# q4_k_m is ~16-17 GB resident: it fits ONE A100-40GB whole. So this is the simple, cheap,
# "actually available" single-GPU coding-fallback tier. See scripts/gcp-glm-serve.sh for the
# frontier-MoE sibling.
#
# PLAN BY DEFAULT. With no creds this prints the exact `gcloud` create command, the VM startup
# script, and the reach-from-this-machine steps, then exits 0 — the whole deploy is reviewable
# without a GCP project. `--apply` runs it and needs `gcloud` + GCP_PROJECT.
#
# Usage:
#   ./scripts/gcp-qwen-serve.sh                                  # PLAN: gcloud + startup + reach steps
#   GCP_PROJECT=my-proj ./scripts/gcp-qwen-serve.sh --apply      # create the A100 VM and serve
#   GCP_TIER=a2-ultra-a100-80gb-1g ./scripts/gcp-qwen-serve.sh   # plan on a single 80GB A100 instead
#   CUDA_GRAPH=1 ./scripts/gcp-qwen-serve.sh                     # serve with the #483 graph-replay decode lever on
#
# Knobs (env):
#   GCP_PROJECT     the GCP project id                 (REQUIRED for --apply)
#   GCP_TIER        gcp_accel.py tier slug             (default a2-high-a100-40gb-1g = ONE A100-40GB;
#                                                        also a2-ultra-a100-80gb-1g — see tools/gcp_accel.py)
#   GCP_ZONE        compute zone                       (default: the tier's first common zone)
#   VM_NAME         instance name                      (default fak-qwen-serve)
#   QWEN_REPO       HF repo for the GGUF               (default lmstudio-community/Qwen3.6-27B-GGUF)
#   QWEN_PORT       served gateway port on the VM      (default 8080 — the connect-fak-node default)
#   CUDA_GRAPH      1 => serve with --cuda-graph (#483) (default 0; witness tok/s before trusting)
#   FAK_GATEWAY_KEY inbound bearer key clients present  (default: generated with openssl on --apply)
#   HF_TOKEN        Hugging Face token (only if the repo is gated; this one is public)
#   FAK_REPO_URL    repo to clone on the VM            (default the public fak remote)
#   TAILSCALE_AUTHKEY  optional tailscale authkey to join the private overlay on boot (recommended:
#                      it gives the client a stable private IP with no SSH tunnel to babysit)
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GCP_TIER="${GCP_TIER:-a2-high-a100-40gb-1g}"
VM_NAME="${VM_NAME:-fak-qwen-serve}"
QWEN_REPO="${QWEN_REPO:-lmstudio-community/Qwen3.6-27B-GGUF}"
QWEN_PORT="${QWEN_PORT:-8080}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"
FAK_REPO_URL="${FAK_REPO_URL:-https://github.com/anthony-chaudhary/fak.git}"

MODE="plan"
case "${1:-}" in
  --apply)    MODE="apply" ;;
  --plan|"")  MODE="plan" ;;
  --help|-h)  sed -n '2,33p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

log() { printf '\033[36m[gcp-qwen]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[gcp-qwen] %s\033[0m\n' "$*" >&2; exit 1; }

# --- resolve the tier from the single registry (tools/gcp_accel.py) -----------
command -v python >/dev/null 2>&1 && PY=python || PY=python3
TIER_SHELL="$("$PY" "$ROOT/tools/gcp_accel.py" --emit-shell "$GCP_TIER")" \
  || die "unknown GCP_TIER='$GCP_TIER' — run: $PY tools/gcp_accel.py  (to list the ladder)"
eval "$TIER_SHELL"   # defines GLM_MACHINE_TYPE, GLM_ACCEL_FLAG, GLM_IMAGE_FAMILY, GLM_COMPUTE_CAP, ...

GCP_ZONE="${GCP_ZONE:-${GLM_DEFAULT_ZONE}}"
cap="${GLM_COMPUTE_CAP:-0}"
# A100 is sm_80, below the sm_90 DSA floor — exactly why the pure fak kernel (which runs on
# Ampere) is the serve path here. A Hopper/Blackwell tier works too, but is overkill for 27B.
if [ "$cap" -ge 90 ] 2>/dev/null; then
  log "note: tier '$GCP_TIER' is sm_${cap} (Hopper/Blackwell) — fine, but a single A100 (a2-high-a100-40gb) is cheaper and plenty for Qwen3.6-27B q4_k_m."
fi

# --- inbound bearer key (secret hygiene mirrors gcp-glm-serve.sh) --------------
# PLAN prints the startup script to stdout (meant to be read/shared), so it must NEVER contain
# a live key. APPLY writes the script to a mode-0600 temp file and bakes the real key in.
if [ "$MODE" = "apply" ]; then
  RENDER_KEY="${FAK_GATEWAY_KEY:-sk-fak-$(openssl rand -hex 24 2>/dev/null || head -c24 /dev/urandom | xxd -p)}"
  RENDER_HF_TOKEN="${HF_TOKEN:-}"
  RENDER_TS_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
else
  RENDER_KEY='***GENERATED-ON-APPLY (or set FAK_GATEWAY_KEY) — paste into connect-fak-node***'
  RENDER_HF_TOKEN="$([ -n "${HF_TOKEN:-}" ] && printf '%s' '***REDACTED — set HF_TOKEN in the --apply env***' || printf '')"
  RENDER_TS_AUTHKEY="$([ -n "${TAILSCALE_AUTHKEY:-}" ] && printf '%s' '***REDACTED — set TAILSCALE_AUTHKEY in the --apply env***' || printf '')"
fi

# --- the VM startup script (cloud-init) ---------------------------------------
# Runs once on first boot of the CUDA Deep-Learning image (driver + toolkit preinstalled):
# installs base tools, clones the repo, and launches tools/qwen36_a100_fak_serve.sh as a
# durable systemd unit (Restart=on-failure). Binds 0.0.0.0:${QWEN_PORT}; reach it ONLY over
# Tailscale or an SSH/IAP tunnel (the create command adds no public ingress), and the serve
# REQUIRES the bearer key on every request.
render_startup_script() {
  cat <<PRE
#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y git python3 python3-pip curl build-essential cmake || true

# Optional: join the Tailscale overlay so this machine can dial the gateway privately.
if [ -n "${RENDER_TS_AUTHKEY}" ]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey "${RENDER_TS_AUTHKEY}" --hostname "${VM_NAME}" || true
fi

install -d -o root -g root /opt
git clone "${FAK_REPO_URL}" /opt/fak || (cd /opt/fak && git pull --ff-only)
cd /opt/fak

python3 -m pip install -q --upgrade "huggingface_hub[hf_transfer,hf_xet]" >/dev/null 2>&1 || pip3 install -q --upgrade huggingface_hub || true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${RENDER_HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}"

# Durable unit; restarts on a transient crash. Logs: journalctl -u qwen36serve and
# /opt/qwen36-q4k/fak_native_serve.log. Poll readiness: cat /opt/qwen36-q4k/PHASE
systemd-run --unit=qwen36serve --collect \\
  --setenv=QWEN_REPO="${QWEN_REPO}" \\
  --setenv=PORT="${QWEN_PORT}" \\
  --setenv=MODEL_ID="qwen3.6-27b" \\
  --setenv=CUDA_GRAPH="${CUDA_GRAPH}" \\
  --setenv=FAK_CUDA_ARCH="sm_${cap}" \\
  --setenv=FAK_GATEWAY_KEY="${RENDER_KEY}" \\
  --setenv=HF_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HF_HUB_ENABLE_HF_TRANSFER=1 \\
  --property=Restart=on-failure --property=RestartSec=15 \\
  /usr/bin/env bash /opt/fak/tools/qwen36_a100_fak_serve.sh
PRE
}

# --- the gcloud create command (printed in plan, run in apply) -----------------
print_gcloud() {
  printf 'gcloud compute instances create %q' "$VM_NAME"
  printf ' \\\n  --zone=%q' "$GCP_ZONE"
  printf ' \\\n  --machine-type=%q' "$GLM_MACHINE_TYPE"
  printf ' \\\n  --accelerator=%s --maintenance-policy=TERMINATE' "$GLM_ACCEL_FLAG"
  printf ' \\\n  --image-family=%q --image-project=%q' "$GLM_IMAGE_FAMILY" "$GLM_IMAGE_PROJECT"
  printf ' \\\n  --boot-disk-size=200GB --boot-disk-type=pd-ssd'
  printf ' \\\n  --metadata-from-file=startup-script=<(rendered startup script)'
  if [ -n "${GCP_PROJECT:-}" ]; then printf ' \\\n  --project=%q' "$GCP_PROJECT"; fi
  echo
}

# --- the reach-from-this-machine steps (the connect-fak-node hand-off) ----------
print_reach_steps() {
  cat <<REACH
# 1) Reach the VM's gateway (it has no public ingress). Either:
#    a. Tailscale (recommended) — get the VM's private IP:
#       gcloud compute ssh ${VM_NAME} --zone ${GCP_ZONE}${GCP_PROJECT:+ --project ${GCP_PROJECT}} -- tailscale ip -4
#    b. or an SSH/IAP tunnel — local :${QWEN_PORT} -> VM :${QWEN_PORT}:
gcloud compute ssh ${VM_NAME} --zone ${GCP_ZONE}${GCP_PROJECT:+ --project ${GCP_PROJECT}} \\
  --tunnel-through-iap -- -N -L ${QWEN_PORT}:localhost:${QWEN_PORT}
#       (then GatewayHost is 127.0.0.1 below)

# 2) Point Claude Code on THIS machine at the kernel — one connect command:
#    Windows (PowerShell):
. scripts/connect-fak-node.ps1 -GatewayHost <tailscale-ip-or-127.0.0.1> -GatewayKey ${RENDER_KEY} -GatewayPort ${QWEN_PORT} -Probe
#    macOS / Linux:
source scripts/connect-fak-node.sh <tailscale-ip-or-127.0.0.1> ${RENDER_KEY} ${QWEN_PORT}

# 3) Use Qwen3.6-27B from here, through the kernel, as your coding fallback:
claude     # every turn now decodes on fak's own forward on the A100
REACH
}

if [ "$MODE" = "plan" ]; then
  log "PLAN (no apply). Qwen3.6-27B serve node: tier '$GCP_TIER' ($GLM_MACHINE_TYPE, ${GLM_GPU_COUNT}x ${GLM_GPU_LABEL}, sm_${cap}) in $GCP_ZONE as '$VM_NAME'."
  log "serve=PURE FAK KERNEL (FAK_Q4K=1 fak serve --backend cuda --gguf <q4_k_m>${CUDA_GRAPH:+ --cuda-graph}) gguf=${QWEN_REPO} port=${QWEN_PORT}  (~\$${GLM_APPROX_USD_HR}/hr while up)"
  echo "# gcloud command this would run:"
  print_gcloud
  echo "# --- VM startup script (cloud-init) ---"
  render_startup_script
  echo
  echo "# --- then, from this machine ---"
  print_reach_steps
  echo
  log "to apply: GCP_PROJECT=<id> $0 --apply   (requires authenticated gcloud + GPU quota for $GLM_ACCEL_FLAG)"
  exit 0
fi

# --- apply --------------------------------------------------------------------
command -v gcloud >/dev/null || die "gcloud not found — install the Cloud SDK"
[ -n "${GCP_PROJECT:-}" ] || die "GCP_PROJECT is required for --apply"

TMP_STARTUP="$(mktemp)"
chmod 600 "$TMP_STARTUP"
trap 'rm -f "$TMP_STARTUP"' EXIT
render_startup_script > "$TMP_STARTUP"

log "creating $VM_NAME ($GLM_MACHINE_TYPE, $GLM_ACCEL_FLAG) in $GCP_ZONE under project $GCP_PROJECT"
gcloud compute instances create "$VM_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$GCP_ZONE" \
  --machine-type="$GLM_MACHINE_TYPE" \
  --accelerator="$GLM_ACCEL_FLAG" --maintenance-policy=TERMINATE \
  --image-family="$GLM_IMAGE_FAMILY" --image-project="$GLM_IMAGE_PROJECT" \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --metadata-from-file=startup-script="$TMP_STARTUP"

log "VM created. The 27B q4_k_m load is ~16-17 GB and takes a few minutes; watch it with:"
log "  gcloud compute ssh $VM_NAME --zone $GCP_ZONE -- 'journalctl -u qwen36serve -f'"
log "Inbound bearer key (paste into connect-fak-node): $RENDER_KEY"
echo
echo "# --- then, from this machine ---"
print_reach_steps
