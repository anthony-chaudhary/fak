#!/usr/bin/env bash
# gcp-glm-serve.sh — stand up GLM-5.2 on a GCP GPU node and make it usable from this
# laptop through the kernel via the `claude-glm-gcp` preset. This is the "kernel gcp
# setup" half of that one-command experience: it provisions an sm_90+ serving VM, runs
# the preflight-gated SGLang/vLLM serve (tools/glm52_sglang_vllm_serve.sh), and prints
# the two follow-up steps — tunnel to its /v1, then `claude-glm-gcp`.
#
#   ┌────────────────┐         ┌──────────────────────────────┐         ┌──────────────────┐
#   │ this laptop    │  /v1    │ fak serve (the kernel)        │  /v1    │ GLM-5.2 on a GCP  │
#   │ claude-glm-gcp │ ──────▶ │ openai backend, adjudicates   │ ──────▶ │ sm_90+ GPU node   │
#   │ (Claude Code)  │ ◀────── │ every tool call               │ ◀────── │ (SGLang / vLLM)   │
#   └────────────────┘         └──────────────────────────────┘         └──────────────────┘
#       FAK_GLM_GCP_BASE_URL = http://<tunnel-or-tailscale>:PORT/v1
#
# PLAN BY DEFAULT. With no creds this prints the exact `gcloud` create command, the VM
# startup script, and the reach-from-laptop steps, then exits 0 — so the whole deploy is
# reviewable without a GCP project. `--apply` runs it and needs `gcloud` + GCP_PROJECT.
#
# WHY sm_90+: GLM-5.2 uses DeepSeek-Sparse-Attention; stock SGLang/vLLM DSA kernels are
# gated to Hopper (sm_90) / Blackwell (sm_100). The default tier (a3-ultra-h200) clears
# that floor, and the on-node preflight (tools/glm52_serve_preflight.py) fails CLOSED if
# the tier somehow doesn't. On Ampere (A100, sm_80) use the llama.cpp MLA path on a GPU
# server instead: tools/glm52_serve.sh.
#
# Usage:
#   ./scripts/gcp-glm-serve.sh                          # PLAN: print gcloud + startup + reach steps
#   GCP_PROJECT=my-proj ./scripts/gcp-glm-serve.sh --apply       # create the GPU VM
#   GCP_TIER=a4-b200 ./scripts/gcp-glm-serve.sh         # plan a Blackwell node instead
#
# Knobs (env):
#   GCP_PROJECT     the GCP project id                 (REQUIRED for --apply)
#   GCP_TIER        gcp_accel.py tier slug             (default a3-ultra-h200; the registry
#                                                        is tools/gcp_accel.py — see its ladder)
#   GCP_ZONE        compute zone                       (default: the tier's first common zone)
#   VM_NAME         instance name                      (default fak-glm-serve)
#   ENGINE          sglang | vllm                      (default sglang)
#   QUANT           fp8 | w4afp8 | nvfp4 | bf16         (default fp8 — the 8x H200 checkpoint)
#   GLM_PORT        served /v1 port on the VM           (default 8000)
#   HF_TOKEN        Hugging Face token for the checkpoint (passed to the VM if set)
#   FAK_REPO_URL    repo to clone on the VM            (default the public fak remote)
#   TAILSCALE_AUTHKEY  optional tailscale authkey to join the private overlay on boot
#   LOCAL_TUNNEL_PORT  local port the printed tunnel binds (default 8200 — the preset default)
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GCP_TIER="${GCP_TIER:-a3-ultra-h200}"
VM_NAME="${VM_NAME:-fak-glm-serve}"
ENGINE="${ENGINE:-sglang}"
QUANT="${QUANT:-fp8}"
GLM_PORT="${GLM_PORT:-8000}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8200}"
FAK_REPO_URL="${FAK_REPO_URL:-https://github.com/anthony-chaudhary/fak.git}"

MODE="plan"
case "${1:-}" in
  --apply)    MODE="apply" ;;
  --plan|"")  MODE="plan" ;;
  --help|-h)  sed -n '2,40p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

log() { printf '\033[36m[gcp-glm]\033[0m %s\n' "$*" >&2; }
warn(){ printf '\033[33m[gcp-glm] %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m[gcp-glm] %s\033[0m\n' "$*" >&2; exit 1; }

if [ "$ENGINE" != "sglang" ] && [ "$ENGINE" != "vllm" ]; then
  die "ENGINE must be 'sglang' or 'vllm' (got '$ENGINE')"
fi

# --- resolve the tier from the single registry (tools/gcp_accel.py) -----------
# emit-shell prints eval-able GLM_* assignments (machine type, accelerator flag, GPU
# image, default zone) so the exact gcloud strings live in exactly one file.
command -v python >/dev/null 2>&1 && PY=python || PY=python3
TIER_SHELL="$("$PY" "$ROOT/tools/gcp_accel.py" --emit-shell "$GCP_TIER")" \
  || die "unknown GCP_TIER='$GCP_TIER' — run: $PY tools/gcp_accel.py  (to list the ladder)"
eval "$TIER_SHELL"   # defines GLM_MACHINE_TYPE, GLM_ACCEL_FLAG, GLM_IMAGE_FAMILY, ...

GCP_ZONE="${GCP_ZONE:-${GLM_DEFAULT_ZONE}}"

# sm_90 is the stock-DSA floor. Warn loudly in the plan if the chosen tier is below it —
# the on-node preflight will BLOCK, and the operator should pick a Hopper/Blackwell tier
# (or the llama.cpp path on a GPU server). Compute capability is sm_XX without the "sm_".
if [ "${GLM_COMPUTE_CAP:-0}" -lt 90 ] 2>/dev/null; then
  warn "tier '$GCP_TIER' is sm_${GLM_COMPUTE_CAP} (< sm_90): stock ${ENGINE} DSA kernels need Hopper/Blackwell."
  warn "the on-node preflight WILL block it. Pick GCP_TIER=a3-ultra-h200 (or a4-b200), or use"
  warn "tools/glm52_serve.sh (llama.cpp MLA) on an Ampere GPU server instead."
fi

# --- the VM startup script (cloud-init) ---------------------------------------
# Runs once on first boot of the CUDA Deep-Learning image (driver + toolkit preinstalled):
# installs git/python, clones the repo, and launches the preflight-gated GLM-5.2 serve as
# a durable systemd unit (Restart=on-failure) so a transient crash self-heals instead of
# orphaning a multi-hundred-GB load. The serve binds 0.0.0.0:${GLM_PORT}; reach it ONLY
# over Tailscale or an SSH/IAP tunnel (the create command below adds no public ingress).
render_startup_script() {
  cat <<STARTUP
#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y git python3 python3-pip curl || true

# Optional: join the Tailscale overlay so this laptop can dial the /v1 privately.
if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey "${TAILSCALE_AUTHKEY:-}" --hostname "${VM_NAME}" || true
fi

install -d -o root -g root /opt
git clone "${FAK_REPO_URL}" /opt/fak || (cd /opt/fak && git pull --ff-only)
cd /opt/fak

# The GLM-5.2 checkpoint is gated on Hugging Face; export the token for the engine fetch.
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# Launch the preflight-gated serve as a durable unit. It fails CLOSED on the wrong GPU
# arch and health-checks itself; the unit restarts it on a transient crash. Logs land in
# the journal (journalctl -u glm52serve) and /opt/fak/glm52-${ENGINE}-server.log.
systemd-run --unit=glm52serve --collect \
  --setenv=ENGINE="${ENGINE}" \
  --setenv=QUANT="${QUANT}" \
  --setenv=PORT="${GLM_PORT}" \
  --setenv=HF_TOKEN="${HF_TOKEN:-}" \
  --setenv=HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  --property=Restart=on-failure --property=RestartSec=10 \
  /usr/bin/env bash /opt/fak/tools/glm52_sglang_vllm_serve.sh
STARTUP
}

# --- the gcloud create command (printed in plan, run in apply) -----------------
print_gcloud() {
  printf 'gcloud compute instances create %q' "$VM_NAME"
  printf ' \\\n  --zone=%q' "$GCP_ZONE"
  printf ' \\\n  --machine-type=%q' "$GLM_MACHINE_TYPE"
  printf ' \\\n  --accelerator=%s --maintenance-policy=TERMINATE' "$GLM_ACCEL_FLAG"
  printf ' \\\n  --image-family=%q --image-project=%q' "$GLM_IMAGE_FAMILY" "$GLM_IMAGE_PROJECT"
  printf ' \\\n  --boot-disk-size=1000GB --boot-disk-type=pd-ssd'
  printf ' \\\n  --metadata-from-file=startup-script=<(rendered startup script)'
  if [ -n "${GCP_PROJECT:-}" ]; then printf ' \\\n  --project=%q' "$GCP_PROJECT"; fi
  echo
}

# --- the reach-from-laptop steps (the claude-glm-gcp hand-off) ------------------
print_reach_steps() {
  cat <<REACH
# 1) Open a private path to the VM's /v1 (it has no public ingress). Either:
#    a. SSH/IAP tunnel — local :${LOCAL_TUNNEL_PORT} -> VM :${GLM_PORT}:
gcloud compute ssh ${VM_NAME} --zone ${GCP_ZONE}${GCP_PROJECT:+ --project ${GCP_PROJECT}} \\
  --tunnel-through-iap -- -N -L ${LOCAL_TUNNEL_PORT}:localhost:${GLM_PORT}
#       then point the preset at the tunnel (this is the preset's default port):
export FAK_GLM_GCP_BASE_URL=http://127.0.0.1:${LOCAL_TUNNEL_PORT}/v1
#    b. or, if the VM joined Tailscale, dial it directly:
# export FAK_GLM_GCP_BASE_URL=http://${VM_NAME}:${GLM_PORT}/v1

# 2) Use GLM-5.2 from here through the kernel — one preset command:
claude-glm-gcp --probe "say pong"     # one witnessable headless turn
claude-glm-gcp                         # interactive Claude Code on GLM-5.2
# (run scripts/dogfood-claude.sh --install once so claude-glm-gcp is on PATH;
#  on Windows: .\\scripts\\dogfood-claude.ps1 --install)
REACH
}

if [ "$MODE" = "plan" ]; then
  log "PLAN (no apply). GLM-5.2 serve node: tier '$GCP_TIER' ($GLM_MACHINE_TYPE, ${GLM_GPU_COUNT}x ${GLM_GPU_LABEL}, sm_${GLM_COMPUTE_CAP}) in $GCP_ZONE as '$VM_NAME'."
  log "engine=$ENGINE quant=$QUANT served-port=$GLM_PORT  (~\$${GLM_APPROX_USD_HR}/hr while up)"
  echo "# gcloud command this would run:"
  print_gcloud
  echo "# --- VM startup script (cloud-init) ---"
  render_startup_script
  echo
  echo "# --- then, from this laptop ---"
  print_reach_steps
  echo
  log "to apply: GCP_PROJECT=<id> $0 --apply   (requires authenticated gcloud + GPU quota for $GLM_ACCEL_FLAG)"
  exit 0
fi

# --- apply --------------------------------------------------------------------
command -v gcloud >/dev/null || die "gcloud not found — install the Cloud SDK"
[ -n "${GCP_PROJECT:-}" ] || die "GCP_PROJECT is required for --apply"

TMP_STARTUP="$(mktemp)"
trap 'rm -f "$TMP_STARTUP"' EXIT
render_startup_script > "$TMP_STARTUP"

log "creating $VM_NAME ($GLM_MACHINE_TYPE, $GLM_ACCEL_FLAG) in $GCP_ZONE under project $GCP_PROJECT"
gcloud compute instances create "$VM_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$GCP_ZONE" \
  --machine-type="$GLM_MACHINE_TYPE" \
  --accelerator="$GLM_ACCEL_FLAG" --maintenance-policy=TERMINATE \
  --image-family="$GLM_IMAGE_FAMILY" --image-project="$GLM_IMAGE_PROJECT" \
  --boot-disk-size=1000GB --boot-disk-type=pd-ssd \
  --metadata-from-file=startup-script="$TMP_STARTUP"

log "VM created. The GLM-5.2 load is multi-hundred-GB and takes minutes; watch it with:"
log "  gcloud compute ssh $VM_NAME --zone $GCP_ZONE -- 'journalctl -u glm52serve -f'"
echo
echo "# --- then, from this laptop ---"
print_reach_steps
