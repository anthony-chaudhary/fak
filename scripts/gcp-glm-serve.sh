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
#                 expert-offload (tools/glm52_stage_serve_gpu_server.sh, the DGX A100 example).
#                 Stand it up to compare fak apples-to-apples. SERVE=llamacpp.
#   * sglang|vllm — stock DSA engines, sm_90+ only (tools/glm52_sglang_vllm_serve.sh), gated
#                 by tools/glm52_serve_preflight.py (fails CLOSED below sm_90). DEFAULT on
#                 Hopper/Blackwell (the witnessed sm_90 real-serving path). This is a
#                 temporary stock compatibility path; the intended kernel target remains
#                 SERVE=fak on every tier where the pure FAK kernel can serve the model.
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
#   QUANT           fp8 | w4afp8 | nvfp4 | bf16         (sm_90 stock quant; default fp8,
#                                                        except H100 tiers default w4afp8)
#   GLM_PORT        served /v1 port on the VM           (default 8000)
#   CTX             stock SGLang/vLLM served context    (default 65536; fits Claude's 32k+32k probe)
#   EP_RANKS        pure-fak resident expert-parallel ranks (default 1; set 8 on full-HBM H100/H200/A100-80GB to avoid cpu-offload)
#   FAK_EP_REQUIRE_DEVICE_PG  for EP_RANKS>1, require the multi-process device-NCCL reduce
#                   instead of falling back to host DistComm (default: 1 when EP_RANKS>1)
#   GLM_GGUF_REPO   HF repo for the fak/llama.cpp GGUF  (default unsloth/GLM-5.2-GGUF)
#   GLM_GGUF_SUBDIR GGUF quant subdir                   (default UD-Q4_K_S, pure-Q4_K experts for CUDA)
#   FAK_STARTUP_PATCH_FILE  optional local patch to apply after the VM clones FAK_REPO_URL
#   NCCL_APT_VERSION optional apt version for libnccl2/libnccl-dev (default 2.30.7-1+cuda12.9)
#   NCPU_MOE        llama.cpp experts-on-host count     (default 999 = all; fak offload is flag-driven)
#   HF_TOKEN        Hugging Face token for the checkpoint (passed to the VM if set)
#   FAK_REPO_URL    repo to clone on the VM            (default the public fak remote)
#   FAK_REPO_REF    optional commit/ref to checkout before applying a startup patch
#   TAILSCALE_AUTHKEY  optional tailscale authkey to join the private overlay on boot
#   LOCAL_TUNNEL_PORT  local port the printed tunnel binds (default 8200 — the preset default)
#   ON_IDLE         stop | delete — idle-gardener action when this box serves no model turns
#                   for IDLE_MINUTES (default delete: zero residual cost; a crashed control
#                   session can't leave this multi-A100 box burning). scripts/gcp-idle-reaper.sh.
#                   ON_IDLE=none disables the reaper.
#   IDLE_MINUTES    idle window before the reaper acts        (default 60)
#   GRACE_MINUTES   boot grace floor (never reap mid-load)    (default 90; GLM loads ~40min)
#   MAX_DAILY_USD   hard operating envelope for this VM       (default 500; 0 disables)
#   MAX_RUNTIME_MINUTES  override the computed max lifetime   (default: floor(MAX_DAILY_USD / tier_hourly * 60))
#   MAX_RUNTIME_ACTION   stop | delete when lifetime expires  (default delete)
#   PROVISIONING_MODEL   STANDARD | SPOT | FLEX_START         (default STANDARD/fail-fast)
#   REQUEST_VALID_FOR_DURATION  Flex-start queue window       (default 2h when FLEX_START)
#   VM_MAX_RUN_DURATION  gcloud max-run-duration              (default MAX_RUNTIME_MINUTES minutes for SPOT/FLEX_START)
#   VM_TERMINATION_ACTION  STOP | DELETE for gcloud timeout   (default mirrors MAX_RUNTIME_ACTION)
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GCP_TIER="${GCP_TIER:-a3-ultra-h200}"
VM_NAME="${VM_NAME:-fak-glm-serve}"
ENGINE="${ENGINE:-sglang}"
QUANT="${QUANT:-}"
GLM_PORT="${GLM_PORT:-8000}"
CTX="${CTX:-65536}"
EP_RANKS="${EP_RANKS:-1}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8200}"
FAK_REPO_URL="${FAK_REPO_URL:-https://github.com/anthony-chaudhary/fak.git}"
FAK_REPO_REF="${FAK_REPO_REF:-}"
FAK_STARTUP_PATCH_FILE="${FAK_STARTUP_PATCH_FILE:-}"
SERVE="${SERVE:-}"   # empty => resolved from the tier's arch below (sm_80→fak, sm_90+→sglang/vllm)
GLM_GGUF_REPO="${GLM_GGUF_REPO:-unsloth/GLM-5.2-GGUF}"
GLM_GGUF_SUBDIR="${GLM_GGUF_SUBDIR:-UD-Q4_K_S}"
NCCL_APT_VERSION="${NCCL_APT_VERSION:-2.30.7-1+cuda12.9}"
NCPU_MOE="${NCPU_MOE:-999}"
GLM_STAGE_DIR="${GLM_STAGE_DIR:-/opt/glm52-q4}"
LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
ON_IDLE="${ON_IDLE:-delete}"        # idle-gardener action; ON_IDLE=none disables the reaper
IDLE_MINUTES="${IDLE_MINUTES:-60}"  # reap after this many minutes of no model turns
GRACE_MINUTES="${GRACE_MINUTES:-90}" # never reap within this many minutes of boot (GLM load ~40min)
MAX_DAILY_USD="${MAX_DAILY_USD:-500}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-}"
MAX_RUNTIME_ACTION="${MAX_RUNTIME_ACTION:-delete}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-}"
REQUEST_VALID_FOR_DURATION="${REQUEST_VALID_FOR_DURATION:-2h}"
VM_MAX_RUN_DURATION="${VM_MAX_RUN_DURATION:-}"
VM_TERMINATION_ACTION="${VM_TERMINATION_ACTION:-}"

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
case "$EP_RANKS" in
  ''|*[!0-9]*) die "EP_RANKS must be a positive integer (got '$EP_RANKS')" ;;
esac
if [ "$EP_RANKS" -lt 1 ]; then
  die "EP_RANKS must be >= 1 (got '$EP_RANKS')"
fi
FAK_EP_REQUIRE_DEVICE_PG="${FAK_EP_REQUIRE_DEVICE_PG:-}"
if [ -z "$FAK_EP_REQUIRE_DEVICE_PG" ]; then
  if [ "$EP_RANKS" -gt 1 ]; then FAK_EP_REQUIRE_DEVICE_PG=1; else FAK_EP_REQUIRE_DEVICE_PG=0; fi
fi
case "$FAK_EP_REQUIRE_DEVICE_PG" in
  0|1|true|false|yes|no|on|off) ;;
  *) die "FAK_EP_REQUIRE_DEVICE_PG must be boolean-ish (0/1/true/false/yes/no/on/off; got '$FAK_EP_REQUIRE_DEVICE_PG')" ;;
esac
case "$MAX_RUNTIME_ACTION" in
  stop|delete) ;;
  *) die "MAX_RUNTIME_ACTION must be 'stop' or 'delete' (got '$MAX_RUNTIME_ACTION')" ;;
esac
case "$PROVISIONING_MODEL" in
  ""|STANDARD|SPOT|FLEX_START) ;;
  *) die "PROVISIONING_MODEL must be STANDARD, SPOT, or FLEX_START (got '$PROVISIONING_MODEL')" ;;
esac
if [ "$PROVISIONING_MODEL" = "STANDARD" ]; then PROVISIONING_MODEL=""; fi

# --- resolve the tier from the single registry (tools/gcp_accel.py) -----------
# emit-shell prints eval-able GLM_* assignments (machine type, accelerator flag, GPU
# image, default zone) so the exact gcloud strings live in exactly one file.
command -v python >/dev/null 2>&1 && PY=python || PY=python3
TIER_SHELL="$("$PY" "$ROOT/tools/gcp_accel.py" --emit-shell "$GCP_TIER")" \
  || die "unknown GCP_TIER='$GCP_TIER' — run: $PY tools/gcp_accel.py  (to list the ladder)"
eval "$TIER_SHELL"   # defines GLM_MACHINE_TYPE, GLM_ACCEL_FLAG, GLM_IMAGE_FAMILY, ...

GCP_ZONE="${GCP_ZONE:-${GLM_DEFAULT_ZONE}}"
if [ -z "$QUANT" ]; then
  case "${GLM_ACCEL_TYPE:-}" in
    nvidia-h100*|nvidia-h100-mega*) QUANT="w4afp8" ;;
    *) QUANT="fp8" ;;
  esac
fi

compute_budget_minutes() {
  "$PY" - "$GLM_APPROX_USD_HR" "$MAX_DAILY_USD" <<'PY'
import math
import sys

try:
    hourly = float(sys.argv[1])
    budget = float(sys.argv[2])
except ValueError:
    raise SystemExit(2)

if hourly <= 0 or budget <= 0:
    print(0)
else:
    print(max(1, math.floor((budget / hourly) * 60)))
PY
}
if [ -z "$MAX_RUNTIME_MINUTES" ]; then
  if ! MAX_RUNTIME_MINUTES="$(compute_budget_minutes)"; then
    die "MAX_DAILY_USD must be numeric (got '$MAX_DAILY_USD')"
  fi
fi
case "$MAX_RUNTIME_MINUTES" in
  ''|*[!0-9]*) die "MAX_RUNTIME_MINUTES must be a non-negative integer (got '$MAX_RUNTIME_MINUTES')" ;;
esac
if [ -n "$PROVISIONING_MODEL" ] && [ -z "$VM_MAX_RUN_DURATION" ] && [ "$MAX_RUNTIME_MINUTES" != "0" ]; then
  VM_MAX_RUN_DURATION="${MAX_RUNTIME_MINUTES}m"
fi
if [ -n "$VM_MAX_RUN_DURATION" ] && [ -z "$VM_TERMINATION_ACTION" ]; then
  case "$MAX_RUNTIME_ACTION" in
    stop) VM_TERMINATION_ACTION="STOP" ;;
    delete) VM_TERMINATION_ACTION="DELETE" ;;
  esac
fi
case "$VM_TERMINATION_ACTION" in
  ""|STOP|DELETE) ;;
  *) die "VM_TERMINATION_ACTION must be STOP or DELETE (got '$VM_TERMINATION_ACTION')" ;;
esac
if [ "$PROVISIONING_MODEL" = "FLEX_START" ] && [ -z "$REQUEST_VALID_FOR_DURATION" ]; then
  die "REQUEST_VALID_FOR_DURATION is required for PROVISIONING_MODEL=FLEX_START"
fi

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
if [ "$EP_RANKS" -gt 1 ] && [ "$SERVE" != "fak" ]; then
  die "EP_RANKS=$EP_RANKS only applies to SERVE=fak (pure fak resident expert-parallel); got SERVE=$SERVE"
fi
if [ "$EP_RANKS" -gt "$GLM_GPU_COUNT" ]; then
  die "EP_RANKS=$EP_RANKS exceeds tier '$GCP_TIER' GPU count ($GLM_GPU_COUNT); choose a full-HBM tier or lower EP_RANKS"
fi
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
  if [ -n "$FAK_STARTUP_PATCH_FILE" ]; then
    [ -f "$FAK_STARTUP_PATCH_FILE" ] || die "FAK_STARTUP_PATCH_FILE does not exist: $FAK_STARTUP_PATCH_FILE"
    RENDER_HAS_STARTUP_PATCH=1
  else
    RENDER_HAS_STARTUP_PATCH=0
  fi
else
  RENDER_HF_TOKEN="$([ -n "${HF_TOKEN:-}" ] && printf '%s' '***REDACTED — set HF_TOKEN in the --apply env***' || printf '')"
  RENDER_TS_AUTHKEY="$([ -n "${TAILSCALE_AUTHKEY:-}" ] && printf '%s' '***REDACTED — set TAILSCALE_AUTHKEY in the --apply env***' || printf '')"
  RENDER_HAS_STARTUP_PATCH="$([ -n "$FAK_STARTUP_PATCH_FILE" ] && printf 1 || printf 0)"
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
# Bounded like every other curl in this script (#3479): an unattended GCE startup
# script runs SEQUENTIALLY, so a stalled download here wedges provisioning of the
# GPU VM *and* the idle-GPU + budget reapers installed further down.
if [ -n "${RENDER_TS_AUTHKEY}" ]; then
  curl -fsSL --connect-timeout 15 --max-time 120 https://tailscale.com/install.sh | sh
  tailscale up --authkey "${RENDER_TS_AUTHKEY}" --hostname "${VM_NAME}" || true
fi

install -d -o root -g root /opt
git clone "${FAK_REPO_URL}" /opt/fak || (cd /opt/fak && git pull --ff-only)
cd /opt/fak
if [ -n "${FAK_REPO_REF}" ]; then
  git checkout --detach "${FAK_REPO_REF}"
fi
if [ "${RENDER_HAS_STARTUP_PATCH}" = "1" ]; then
  { set +x; } 2>/dev/null || true
  PATCH_ATTR_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes/fak-startup-patch-b64"
  if ! curl -fsS --max-time 30 -H 'Metadata-Flavor: Google' "\${PATCH_ATTR_URL}" -o /tmp/fak-startup.patch.b64; then
    echo "STARTUP_PATCH_METADATA_MISSING: fak-startup-patch-b64 metadata item was required but unavailable" >&2
    exit 18
  fi
  if [ ! -s /tmp/fak-startup.patch.b64 ]; then
    echo "STARTUP_PATCH_METADATA_EMPTY: fak-startup-patch-b64 was empty" >&2
    exit 18
  fi
  base64 -d /tmp/fak-startup.patch.b64 | gzip -dc >/tmp/fak-startup.patch
  git apply --whitespace=nowarn /tmp/fak-startup.patch
  git status --short
  { set -x; } 2>/dev/null || true
fi

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
export HF_HUB_DISABLE_XET="\${HF_HUB_DISABLE_XET:-1}"
install_nccl_dev() {
  apt-get update -y || true
  if ! apt-cache show libnccl-dev >/dev/null 2>&1; then
    . /etc/os-release
    repo="ubuntu\${VERSION_ID//./}/x86_64"
    curl -fsSL --connect-timeout 15 --max-time 120 "https://developer.download.nvidia.com/compute/cuda/repos/\${repo}/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -y
  fi
  apt-get install -y "libnccl2=${NCCL_APT_VERSION}" "libnccl-dev=${NCCL_APT_VERSION}" || apt-get install -y libnccl2 libnccl-dev
}
if [ "${EP_RANKS}" -gt 1 ]; then
  install_nccl_dev
  if ! { [ -f /usr/include/nccl.h ] || [ -f /usr/local/cuda/include/nccl.h ] || [ -f /usr/local/cuda/targets/x86_64-linux/include/nccl.h ]; }; then
    echo "NCCL_HEADERS_MISSING: install libnccl-dev or set NCCL_HOME before launching EP_RANKS=${EP_RANKS}" >&2
    exit 17
  fi
fi
# Durable unit; restarts on a transient crash. Logs: journalctl -u glm52serve and
# ${GLM_STAGE_DIR}/fak_native_serve.log. FAK_CUDA_ARCH tracks the tier's compute capability.
systemd-run --unit=glm52serve --collect \\
  --setenv=GLM_DIR="${GLM_STAGE_DIR}" \\
  --setenv=GLM_REPO="${GLM_GGUF_REPO}" \\
  --setenv=GLM_SUBDIR="${GLM_GGUF_SUBDIR}" \\
  --setenv=HF_HUB_DISABLE_XET="\${HF_HUB_DISABLE_XET}" \\
  --setenv=PORT="${GLM_PORT}" \\
  --setenv=MODEL_ID="glm-5.2" \\
  --setenv=EP_RANKS="${EP_RANKS}" \\
  --setenv=EP_JOIN_TIMEOUT_S=1800 \\
  --setenv=FAK_EP_REQUIRE_DEVICE_PG="${FAK_EP_REQUIRE_DEVICE_PG}" \\
  --setenv=FAK_CUDA_ARCH="sm_${cap}" \\
  --setenv=HF_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HF_HUB_ENABLE_HF_TRANSFER=1 \\
  --setenv=HF_XET_HIGH_PERFORMANCE=1 \\
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
  --setenv=HF_XET_HIGH_PERFORMANCE=1 \\
  --property=Restart=on-failure --property=RestartSec=15 \\
  /usr/bin/env bash /opt/fak/tools/glm52_stage_serve_gpu_server.sh
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
  --setenv=CTX="${CTX}" \\
  --setenv=HF_HUB_DISABLE_XET="\${HF_HUB_DISABLE_XET:-1}" \\
  --setenv=HF_TOKEN="${RENDER_HF_TOKEN}" \\
  --setenv=HUGGING_FACE_HUB_TOKEN="${RENDER_HF_TOKEN}" \\
  --property=Restart=on-failure --property=RestartSec=10 \\
  /usr/bin/env bash /opt/fak/tools/glm52_sglang_vllm_serve.sh
TAIL
}

# render_idle_reaper <serve-port> — emit the cloud-init lines that install the idle
# gardener (scripts/gcp-idle-reaper.sh, already in the /opt/fak clone) as a systemd
# timer. The box watches its OWN inference counter and self-${ON_IDLE}s after
# ${IDLE_MINUTES}min of no model turns, so a crashed control-session can't leave this
# multi-A100 node burning. ON_IDLE=none disables it. The systemd unit bodies use a
# QUOTED ('UNIT') heredoc so the inner cat never expands $-vars; the Environment=
# values are baked from the OUTER shell here. Shared shape with gcp-qwen-serve.sh.
render_idle_reaper() {
  local port="$1"
  if [ "$ON_IDLE" = "none" ]; then
    echo "# idle reaper disabled (ON_IDLE=none)"
    return 0
  fi
  echo "# --- idle-GPU gardener (scripts/gcp-idle-reaper.sh) ---"
  echo "install -d -m 0755 /var/lib/fak-idle-reaper"
  echo "cat >/etc/systemd/system/fak-idle-reaper.service <<'UNIT'"
  echo "[Unit]"
  echo "Description=fak idle-GPU reaper (self-${ON_IDLE} after idle)"
  echo "After=network-online.target"
  echo "[Service]"
  echo "Type=oneshot"
  printf 'Environment=PORT=%s\n' "$port"
  printf 'Environment=IDLE_MINUTES=%s\n' "$IDLE_MINUTES"
  printf 'Environment=GRACE_MINUTES=%s\n' "$GRACE_MINUTES"
  printf 'Environment=ON_IDLE=%s\n' "$ON_IDLE"
  echo "Environment=ON_IDLE_LIVE=1"
  echo "ExecStart=/usr/bin/env bash /opt/fak/scripts/gcp-idle-reaper.sh --live"
  echo "UNIT"
  echo "cat >/etc/systemd/system/fak-idle-reaper.timer <<'UNIT'"
  echo "[Unit]"
  echo "Description=run the fak idle-GPU reaper every 5 minutes"
  echo "[Timer]"
  echo "OnBootSec=5min"
  echo "OnUnitActiveSec=5min"
  echo "AccuracySec=30s"
  echo "[Install]"
  echo "WantedBy=timers.target"
  echo "UNIT"
  echo "systemctl daemon-reload"
  echo "systemctl enable --now fak-idle-reaper.timer"
}

# render_budget_reaper — emit a one-shot max-lifetime guard that self-stops/deletes the
# VM after the budget-derived runtime. This is separate from the idle reaper: idle protects
# against abandoned sessions, while this protects the operator's daily spend envelope even
# if the node stays busy. The price is the registry's indicative hourly estimate; the
# Cloud Billing budget remains the authoritative spend monitor.
render_budget_reaper() {
  if [ "$MAX_RUNTIME_MINUTES" = "0" ]; then
    echo "# max-lifetime budget reaper disabled (MAX_DAILY_USD=0 or MAX_RUNTIME_MINUTES=0)"
    return 0
  fi
  echo "# --- max-lifetime budget guard (MAX_DAILY_USD=${MAX_DAILY_USD}, ~\$${GLM_APPROX_USD_HR}/hr => ${MAX_RUNTIME_MINUTES}min) ---"
  cat <<'BUDGET'
cat >/usr/local/sbin/fak-budget-reaper.sh <<'UNIT'
#!/usr/bin/env bash
set -euo pipefail
ACTION="${MAX_RUNTIME_ACTION:-delete}"
META="http://metadata.google.internal/computeMetadata/v1/instance"
VM_NAME="$(curl -fsS --max-time 5 -H 'Metadata-Flavor: Google' "$META/name" 2>/dev/null || true)"
ZONE_PATH="$(curl -fsS --max-time 5 -H 'Metadata-Flavor: Google' "$META/zone" 2>/dev/null || true)"
ZONE_SHORT="${ZONE_PATH##*/}"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$ts budget-reaper action=$ACTION max_daily_usd=${MAX_DAILY_USD:-?} runtime_min=${MAX_RUNTIME_MINUTES:-?} hourly_usd=${GLM_APPROX_USD_HR:-?} vm=$VM_NAME zone=$ZONE_SHORT" >&2
case "$ACTION" in stop|delete) ;; *) echo "invalid MAX_RUNTIME_ACTION=$ACTION" >&2; exit 2 ;; esac
[ -n "$VM_NAME" ] && [ -n "$ZONE_SHORT" ] || { echo "metadata identity unavailable; refusing to act" >&2; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo "gcloud unavailable; refusing to act" >&2; exit 1; }
exec gcloud compute instances "$ACTION" "$VM_NAME" --zone "$ZONE_SHORT" --quiet
UNIT
chmod 0755 /usr/local/sbin/fak-budget-reaper.sh
BUDGET
  printf 'systemd-run --unit=fak-budget-reaper --collect --on-active=%smin \\\n' "$MAX_RUNTIME_MINUTES"
  printf '  --setenv=MAX_RUNTIME_ACTION=%q \\\n' "$MAX_RUNTIME_ACTION"
  printf '  --setenv=MAX_DAILY_USD=%q \\\n' "$MAX_DAILY_USD"
  printf '  --setenv=MAX_RUNTIME_MINUTES=%q \\\n' "$MAX_RUNTIME_MINUTES"
  printf '  --setenv=GLM_APPROX_USD_HR=%q \\\n' "$GLM_APPROX_USD_HR"
  echo "  /usr/bin/env bash /usr/local/sbin/fak-budget-reaper.sh"
}

render_startup_script() {
  case "$SERVE" in
    fak)      render_startup_preamble "build-essential cmake"; render_startup_tail_fak ;;
    llamacpp) render_startup_preamble "build-essential cmake"; render_startup_tail_llamacpp ;;
    *)        render_startup_preamble ""; render_startup_tail_dsa ;;
  esac
  render_idle_reaper "$GLM_PORT"
  render_budget_reaper
}

# one-line human description of the resolved serve path, for the plan summary.
describe_serve() {
  case "$SERVE" in
    fak)      echo "serve=PURE FAK KERNEL (native glm_moe_dsa: EP_RANKS=${EP_RANKS}; 1=--cpu-offload-experts, >1=resident expert-parallel) gguf=${GLM_GGUF_REPO}/${GLM_GGUF_SUBDIR}" ;;
    llamacpp) echo "serve=llama.cpp MLA BENCHMARK baseline (n-cpu-moe=${NCPU_MOE}) gguf=${GLM_GGUF_REPO}/${GLM_GGUF_SUBDIR}" ;;
    *)        echo "serve=${SERVE} (stock DSA, sm_90+) quant=${QUANT} ctx=${CTX}" ;;
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
  # cloud-platform scope so the baked-in idle reaper can self-delete via the attached
  # SA (which also needs roles/compute.instanceAdmin.v1). Without it the reaper would
  # only ever log REAP_DENIED. ON_IDLE=none if you don't want the box self-managing.
  printf ' \\\n  --scopes=cloud-platform'
  if [ -n "$PROVISIONING_MODEL" ]; then
    printf ' \\\n  --provisioning-model=%q' "$PROVISIONING_MODEL"
  fi
  if [ -n "$VM_MAX_RUN_DURATION" ]; then
    printf ' \\\n  --max-run-duration=%q' "$VM_MAX_RUN_DURATION"
  fi
  if [ -n "$VM_TERMINATION_ACTION" ]; then
    printf ' \\\n  --instance-termination-action=%q' "$VM_TERMINATION_ACTION"
  fi
  if [ "$PROVISIONING_MODEL" = "FLEX_START" ]; then
    printf ' \\\n  --request-valid-for-duration=%q' "$REQUEST_VALID_FOR_DURATION"
  fi
  printf ' \\\n  --metadata-from-file=startup-script=<(rendered startup script)'
  if [ "$RENDER_HAS_STARTUP_PATCH" = "1" ]; then
    printf ',fak-startup-patch-b64=<(gzip -9c FAK_STARTUP_PATCH_FILE | base64)'
  fi
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
  if [ "$MAX_RUNTIME_MINUTES" = "0" ]; then
    log "budget guard disabled; Cloud Billing budget alerts are still recommended."
  else
    log "budget guard: MAX_DAILY_USD=$MAX_DAILY_USD => self-${MAX_RUNTIME_ACTION} after ${MAX_RUNTIME_MINUTES}min on this tier (indicative price; verify billing before apply)."
  fi
  if [ -n "$PROVISIONING_MODEL" ]; then
    log "capacity request: PROVISIONING_MODEL=$PROVISIONING_MODEL max-run=${VM_MAX_RUN_DURATION:-none} termination=${VM_TERMINATION_ACTION:-default}${PROVISIONING_MODEL:+ request-valid=${REQUEST_VALID_FOR_DURATION}}."
  fi
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
TMP_STARTUP_PATCH_B64=""
trap 'rm -f "$TMP_STARTUP" ${TMP_STARTUP_PATCH_B64:+"$TMP_STARTUP_PATCH_B64"}' EXIT
render_startup_script > "$TMP_STARTUP"
if [ "$RENDER_HAS_STARTUP_PATCH" = "1" ]; then
  TMP_STARTUP_PATCH_B64="$(mktemp)"
  gzip -9c "$FAK_STARTUP_PATCH_FILE" | base64 | fold -w 76 > "$TMP_STARTUP_PATCH_B64"
fi
METADATA_FROM_FILE="startup-script=$TMP_STARTUP"
if [ "$RENDER_HAS_STARTUP_PATCH" = "1" ]; then
  METADATA_FROM_FILE+=",fak-startup-patch-b64=$TMP_STARTUP_PATCH_B64"
fi

log "creating $VM_NAME ($GLM_MACHINE_TYPE, $GLM_ACCEL_FLAG) in $GCP_ZONE under project $GCP_PROJECT"
CREATE_ARGS=(
  compute instances create "$VM_NAME"
  --project="$GCP_PROJECT"
  --zone="$GCP_ZONE"
  --machine-type="$GLM_MACHINE_TYPE"
  --accelerator="$GLM_ACCEL_FLAG" --maintenance-policy=TERMINATE
  --image-family="$GLM_IMAGE_FAMILY" --image-project="$GLM_IMAGE_PROJECT"
  --boot-disk-size=1000GB --boot-disk-type=pd-ssd
  --scopes=cloud-platform
  --metadata-from-file="$METADATA_FROM_FILE"
)
if [ -n "$PROVISIONING_MODEL" ]; then
  CREATE_ARGS+=(--provisioning-model="$PROVISIONING_MODEL")
fi
if [ -n "$VM_MAX_RUN_DURATION" ]; then
  CREATE_ARGS+=(--max-run-duration="$VM_MAX_RUN_DURATION")
fi
if [ -n "$VM_TERMINATION_ACTION" ]; then
  CREATE_ARGS+=(--instance-termination-action="$VM_TERMINATION_ACTION")
fi
if [ "$PROVISIONING_MODEL" = "FLEX_START" ]; then
  CREATE_ARGS+=(--request-valid-for-duration="$REQUEST_VALID_FOR_DURATION")
fi
gcloud "${CREATE_ARGS[@]}"

log "VM created. The GLM-5.2 load is multi-hundred-GB and takes minutes; watch it with:"
log "  gcloud compute ssh $VM_NAME --zone $GCP_ZONE -- 'journalctl -u glm52serve -f'"
echo
echo "# --- then, from this laptop ---"
print_reach_steps
