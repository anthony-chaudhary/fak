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
# THREE SERVE PATHS, picked from the tier's GPU arch (override with SERVE=):
#   * fak       — the PURE FAK KERNEL: fak serves GLM-5.2 (glm_moe_dsa) through its OWN CUDA
#                 kernels (tools/glm52_fak_native_serve.sh). DEFAULT on Ampere (A100, sm_80);
#                 the forward is bit-exact vs the CPU reference (cosine 1.0, witnessed on
#                 sm_80). PREFERRED — fak runs the weights, not an external engine.
#   * llamacpp  — the BENCHMARK baseline: the SAME checkpoint under llama.cpp MLA + CPU
#                 expert-offload (tools/glm52_stage_serve_dgx3.sh, the DGX A100 example).
#                 Stand it up to compare fak apples-to-apples. SERVE=llamacpp.
#   * sglang|vllm — stock DSA engines, sm_90+ only (tools/glm52_sglang_vllm_serve.sh), gated
#                 by tools/glm52_serve_preflight.py (fails CLOSED below sm_90). DEFAULT on
#                 Hopper/Blackwell (the witnessed sm_90 real-serving path).
#
# So "whatever is available" includes A100: GCP_TIER=a2-ultra-a100-80gb serves GLM-5.2 via
# the pure fak kernel by default, with SERVE=llamacpp for the benchmark comparison.
#
# Usage:
#   ./scripts/gcp-glm-serve.sh                                # PLAN: gcloud + startup + reach steps
#   GCP_PROJECT=my-proj ./scripts/gcp-glm-serve.sh --apply    # create the GPU VM
#   GCP_TIER=a2-ultra-a100-80gb ./scripts/gcp-glm-serve.sh    # plan an A100 (pure fak kernel) node
#   GCP_TIER=a2-ultra-a100-80gb SERVE=llamacpp ./scripts/gcp-glm-serve.sh  # the llama.cpp benchmark node
#   GCP_TIER=a4-b200 ./scripts/gcp-glm-serve.sh               # plan a Blackwell (SGLang/vLLM) node
#
# Knobs (env):
#   GCP_PROJECT     the GCP project id                 (REQUIRED for --apply)
#   GCP_TIER        gcp_accel.py tier slug             (default a3-ultra-h200; the registry is
#                                                        tools/gcp_accel.py — A100 tiers:
#                                                        a2-ultra-a100-80gb / a2-high-a100-40gb)
#   SERVE           fak | llamacpp | sglang | vllm     (default: by arch — sm_80→fak, sm_90+→sglang)
#   GCP_ZONE        compute zone                       (default: the tier's first common zone)
#   VM_NAME         instance name                      (default fak-glm-serve)
#   ENGINE          sglang | vllm                      (the sm_90 stock engine; default sglang)
#   QUANT           fp8 | w4afp8 | nvfp4 | bf16         (sm_90 stock quant; default fp8)
#   GLM_PORT        served /v1 port on the VM           (default 8000)
#   GLM_GGUF_REPO   HF repo for the fak/llama.cpp GGUF  (default unsloth/GLM-5.2-GGUF)
#   GLM_GGUF_SUBDIR GGUF quant subdir                   (default UD-Q4_K_M, ~466 GB)
#   NCPU_MOE        llama.cpp experts-on-host count     (default 999 = all; fak offload is flag-driven)
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
SERVE="${SERVE:-}"   # empty => resolved from the tier's arch below (sm_80→fak, sm_90+→sglang/vllm)
GLM_GGUF_REPO="${GLM_GGUF_REPO:-unsloth/GLM-5.2-GGUF}"
GLM_GGUF_SUBDIR="${GLM_GGUF_SUBDIR:-UD-Q4_K_M}"
NCPU_MOE="${NCPU_MOE:-999}"
GLM_STAGE_DIR="${GLM_STAGE_DIR:-/opt/glm52-q4}"
LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"

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

# --- resolve the serve path: explicit SERVE wins; else by the tier's GPU arch -----------
# sm_90+ (Hopper/Blackwell) defaults to the stock SGLang/vLLM DSA path; sm_80 (Ampere/A100)
# is BELOW the DSA kernel floor, so it defaults to the PURE FAK KERNEL native serve. SERVE=
# overrides on any tier (e.g. SERVE=llamacpp for the benchmark baseline; SERVE=fak to prefer
# the pure kernel on Hopper too). Compute capability is sm_XX without the "sm_".
cap="${GLM_COMPUTE_CAP:-0}"
if [ -z "$SERVE" ]; then
  if [ "$cap" -ge 90 ] 2>/dev/null; then SERVE="$ENGINE"; else SERVE="fak"; fi
fi
case "$SERVE" in
  fak|llamacpp|sglang|vllm) ;;
  *) die "SERVE must be one of: fak | llamacpp | sglang | vllm (got '$SERVE')" ;;
esac
# The stock DSA engines cannot run below sm_90; fak + llamacpp run on Ampere by design
# (that is the whole point of the A100 path), so only the stock engines are arch-gated here.
if { [ "$SERVE" = "sglang" ] || [ "$SERVE" = "vllm" ]; } && [ "$cap" -lt 90 ] 2>/dev/null; then
  die "SERVE=$SERVE needs sm_90+ (tier '$GCP_TIER' is sm_${cap}). Use SERVE=fak (pure kernel) or SERVE=llamacpp (benchmark) on A100, or pick a Hopper/Blackwell tier."
fi
if [ "$cap" -lt 90 ] 2>/dev/null; then
  if [ "$SERVE" = "fak" ]; then
    log "tier '$GCP_TIER' is sm_${cap} (Ampere, below the sm_90 DSA floor): serving via the PURE FAK KERNEL (native glm_moe_dsa; forward witnessed at q8 on sm_80, a live serve turn is hardware+load-gated)."
  else
    log "tier '$GCP_TIER' is sm_${cap} (Ampere): serving via llama.cpp MLA (the DGX A100 benchmark baseline)."
  fi
fi

# Secret hygiene: PLAN prints the startup script to stdout (meant to be read/shared), so it
# must NEVER contain live credentials. Render secrets as a redacted placeholder in plan; only
# APPLY (which writes the script to a mode-0600 temp file) bakes the real values in.
if [ "$MODE" = "apply" ]; then
  RENDER_HF_TOKEN="${HF_TOKEN:-}"
  RENDER_TS_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
else
  RENDER_HF_TOKEN="$([ -n "${HF_TOKEN:-}" ] && printf '%s' '***REDACTED — set HF_TOKEN in the --apply env***' || printf '')"
  RENDER_TS_AUTHKEY="$([ -n "${TAILSCALE_AUTHKEY:-}" ] && printf '%s' '***REDACTED — set TAILSCALE_AUTHKEY in the --apply env***' || printf '')"
fi

# --- the VM startup script (cloud-init) ---------------------------------------
# Runs once on first boot of the CUDA Deep-Learning image (driver + toolkit preinstalled):
# installs base tools, clones the repo, and launches the chosen GLM-5.2 serve as a durable
# systemd unit (Restart=on-failure) so a transient crash self-heals instead of orphaning a
# multi-hundred-GB load. The serve binds 0.0.0.0:${GLM_PORT}; reach it ONLY over Tailscale or
# an SSH/IAP tunnel (the create command below adds no public ingress). The startup script is
# assembled from a shared preamble + a serve-path-specific tail (fak | llamacpp | sglang/vllm).
render_startup_preamble() {
  # $1 = extra apt packages for the chosen serve path (e.g. build-essential cmake).
  cat <<PRE
#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y git python3 python3-pip curl $1 || true

# Optional: join the Tailscale overlay so this laptop can dial the /v1 privately.
if [ -n "${RENDER_TS_AUTHKEY}" ]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey "${RENDER_TS_AUTHKEY}" --hostname "${VM_NAME}" || true
fi

install -d -o root -g root /opt
git clone "${FAK_REPO_URL}" /opt/fak || (cd /opt/fak && git pull --ff-only)
cd /opt/fak

# The GLM-5.2 checkpoint is gated on Hugging Face; export the token for the fetch.
export HF_TOKEN="${RENDER_HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}"
PRE
}

# PURE FAK KERNEL tail (Ampere/A100 default; also usable on sm_90 via SERVE=fak): build the
# -tags cuda fak binary and serve GLM-5.2 through fak's OWN glm_moe_dsa kernels.
render_startup_tail_fak() {
  cat <<TAIL
python3 -m pip install -q --upgrade "huggingface_hub[hf_transfer,hf_xet]" >/dev/null 2>&1 || pip3 install -q --upgrade huggingface_hub || true
export HF_HUB_ENABLE_HF_TRANSFER=1
# Durable unit; restarts on a transient crash. Logs: journalctl -u glm52serve and
# ${GLM_STAGE_DIR}/fak_native_serve.log. FAK_CUDA_ARCH tracks the tier's compute capability.
systemd-run --unit=glm52serve --collect \\
  --setenv=GLM_DIR="${GLM_STAGE_DIR}" \\
  --setenv=GLM_REPO="${GLM_GGUF_REPO}" \\
  --setenv=GLM_SUBDIR="${GLM_GGUF_SUBDIR}" \\
  --setenv=PORT="${GLM_PORT}" \\
  --setenv=MODEL_ID="glm-5.2" \\
  --setenv=FAK_CUDA_ARCH="sm_${cap}" \\
  --setenv=HF_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HF_HUB_ENABLE_HF_TRANSFER=1 \\
  --property=Restart=on-failure --property=RestartSec=15 \\
  /usr/bin/env bash /opt/fak/tools/glm52_fak_native_serve.sh
TAIL
}

# llama.cpp BENCHMARK tail (SERVE=llamacpp): the DGX A100 example, brought to GCP. Self-stages
# the SAME checkpoint and serves it via MLA + CPU expert-offload for the apples-to-apples
# comparison against the pure fak kernel.
render_startup_tail_llamacpp() {
  cat <<TAIL
python3 -m pip install -q --upgrade "huggingface_hub[hf_transfer,hf_xet]" >/dev/null 2>&1 || pip3 install -q --upgrade huggingface_hub || true
export HF_HUB_ENABLE_HF_TRANSFER=1
# Durable unit. Logs: journalctl -u glm52serve and ${GLM_STAGE_DIR}/stage_serve.log.
systemd-run --unit=glm52serve --collect \\
  --setenv=GLM_DIR="${GLM_STAGE_DIR}" \\
  --setenv=LLAMA="${LLAMA_DIR}" \\
  --setenv=GLM_REPO="${GLM_GGUF_REPO}" \\
  --setenv=GLM_SUBDIR="${GLM_GGUF_SUBDIR}" \\
  --setenv=PORT="${GLM_PORT}" \\
  --setenv=NCPU_MOE="${NCPU_MOE}" \\
  --setenv=HF_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HF_HUB_ENABLE_HF_TRANSFER=1 \\
  --property=Restart=on-failure --property=RestartSec=15 \\
  /usr/bin/env bash /opt/fak/tools/glm52_stage_serve_dgx3.sh
TAIL
}

# stock SGLang/vLLM DSA tail (sm_90+ default): the preflight-gated serve. It fails CLOSED on
# the wrong GPU arch and health-checks itself.
render_startup_tail_dsa() {
  cat <<TAIL
# Logs: journalctl -u glm52serve and /opt/fak/glm52-${ENGINE}-server.log.
systemd-run --unit=glm52serve --collect \\
  --setenv=ENGINE="${ENGINE}" \\
  --setenv=QUANT="${QUANT}" \\
  --setenv=PORT="${GLM_PORT}" \\
  --setenv=HF_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}" \\
  --property=Restart=on-failure --property=RestartSec=10 \\
  /usr/bin/env bash /opt/fak/tools/glm52_sglang_vllm_serve.sh
TAIL
}

render_startup_script() {
  case "$SERVE" in
    fak)      render_startup_preamble "build-essential cmake"; render_startup_tail_fak ;;
    llamacpp) render_startup_preamble "build-essential cmake"; render_startup_tail_llamacpp ;;
    *)        render_startup_preamble ""; render_startup_tail_dsa ;;
  esac
}

# one-line human description of the resolved serve path, for the plan summary.
describe_serve() {
  case "$SERVE" in
    fak)      echo "serve=PURE FAK KERNEL (native glm_moe_dsa: --backend cuda --cpu-offload-experts) gguf=${GLM_GGUF_REPO}/${GLM_GGUF_SUBDIR}" ;;
    llamacpp) echo "serve=llama.cpp MLA BENCHMARK baseline (n-cpu-moe=${NCPU_MOE}) gguf=${GLM_GGUF_REPO}/${GLM_GGUF_SUBDIR}" ;;
    *)        echo "serve=${SERVE} (stock DSA, sm_90+) quant=${QUANT}" ;;
  esac
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
  log "$(describe_serve) served-port=$GLM_PORT  (~\$${GLM_APPROX_USD_HR}/hr while up)"
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
