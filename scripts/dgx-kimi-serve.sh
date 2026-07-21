#!/usr/bin/env bash
# Turnkey Moonshot Kimi (open-weights) bring-up on a GPU node/pod, fronted by `fak serve`.
#
#   [this script on the GPU node] → vLLM|SGLang|NIM OpenAI server :8000
#        → then, anywhere: fak serve --provider openai-compatible → that server
#
# Sibling of scripts/dgx-deepseek-serve.sh. It does the boring, error-prone
# parts of the Kimi self-host baseline runbook
# (docs/benchmarks/KIMI-SELFHOST-BASELINE-RUNBOOK.md) turnkey:
#   1. detect the GPUs, GATE on the Kimi architecture floor, and CHECK that the
#      visible VRAM plausibly holds the chosen checkpoint (Kimi K2 is a ~1T-param
#      MoE — a MULTI-NODE deployment, not a single-DGX resident model),
#   2. launch an OpenAI-compatible server for the chosen model+engine on :8000,
#   3. health-check GET /models until the roster is up,
#   4. print the exact `fak serve` line AND the keyless go smoke command that
#      witnesses the wire (readiness + non-streaming + streaming).
#
# It makes NO performance claim. It is the wire/bring-up floor, not a benchmark.
#
# ── Which Kimi route belongs on the DGX? ──────────────────────────────────────
# There are TWO Kimi routes and only ONE is a self-host GPU workload:
#   * Kimi K3   — Moonshot's CLOUD OpenAI-compatible API (api.moonshot.ai). It is
#                 NOT self-hostable; test it with scripts/claude-kimi-k3.sh, no GPU.
#   * Kimi K2   — OPEN WEIGHTS (moonshotai/Kimi-K2-Instruct), block-FP8, ~1T total
#                 / ~32B active MoE. THIS is the DGX self-host target — and it is a
#                 multi-node expert-parallel deployment, not a single-node fit.
# See the runbook's route-decision table before reserving GPU time.
#
# ── Kimi K2 architecture floor (why the GPU gate exists) ──────────────────────
# K2 ships native block-FP8 (fp8_e4m3) weights with MLA attention. Stock FP8
# kernels need modern GPUs:
#   * sm_100 (Blackwell, B200/GB200) — native FP8/FP4.
#   * sm_90  (Hopper, H100/H200)     — native FP8: the intended K2 path.
#   * sm_89  (Ada, L40S)             — FP8 yes, but a single card is far too small.
#   * sm_80  (Ampere, A100)          — no native FP8: a BF16 upcast ~doubles the
#                                      already-multi-node footprint. Generally not viable.
#   * sm_70  (Volta, V100-class)     — none of the above: not viable.
# The gate WARNS (does not silently proceed) below sm_90. Confirm the exact floor
# against your engine's Kimi K2 kernel requirements before reserving GPU time.
set -euo pipefail

# ── knobs (env-overridable) ───────────────────────────────────────────────────
ENGINE="${FAK_KIMI_ENGINE:-vllm}"                  # vllm | sglang | nim
# K2-Instruct is the open-weights default. It is a ~1T-total MoE and is a
# MULTI-NODE expert-parallel deployment — do not point a single node at it and
# expect it to fit. There is no small self-hostable Kimi variant; for a
# single-box smoke, front the Moonshot cloud K3 route instead (see above).
MODEL="${FAK_KIMI_MODEL:-moonshotai/Kimi-K2-Instruct}"
PORT="${FAK_KIMI_PORT:-8000}"
HOST="${FAK_KIMI_HOST:-0.0.0.0}"
ADVERTISE_HOST="${FAK_KIMI_ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
NIM_IMAGE="${FAK_KIMI_NIM_IMAGE:-}"                # required only when ENGINE=nim
TP="${FAK_KIMI_TP:-auto}"                           # tensor-parallel size; auto = #GPUs
# Approximate resident weight footprint in GiB used only for the VRAM-fit CHECK.
# Default ~1000 GiB ≈ Kimi K2 in native FP8 (~1T params @ ~1 byte). Override for a
# different checkpoint/precision; set 0 to skip the fit check entirely.
FOOTPRINT_GIB="${FAK_KIMI_FOOTPRINT_GIB:-1000}"
HEALTH_TIMEOUT="${FAK_KIMI_HEALTH_TIMEOUT:-1800}"   # seconds to wait for /models
DRY_RUN="${FAK_KIMI_DRY_RUN:-0}"                    # 1 = print plan, do not launch
FORCE="${FAK_KIMI_FORCE:-0}"                        # 1 = proceed past the gates

log()  { printf '\033[36m[dgx-kimi]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[dgx-kimi] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[dgx-kimi] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. detect GPUs + gate on the architecture floor + VRAM-fit check ──────────
GPU_COUNT=0
MIN_CC=""
TOTAL_VRAM_GIB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  # compute_cap is emitted as e.g. "9.0"; take the minimum across visible GPUs.
  mapfile -t _caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
  GPU_COUNT="${#_caps[@]}"
  for c in "${_caps[@]}"; do
    [[ -z "$c" ]] && continue
    if [[ -z "$MIN_CC" ]] || awk -v a="$c" -v b="$MIN_CC" 'BEGIN{exit !(a<b)}'; then MIN_CC="$c"; fi
  done
  # Sum visible VRAM (MiB → GiB) for the fit check.
  while IFS= read -r m; do
    m="$(printf '%s' "$m" | tr -dc '0-9')"
    [[ -z "$m" ]] && continue
    TOTAL_VRAM_GIB=$(awk -v t="$TOTAL_VRAM_GIB" -v m="$m" 'BEGIN{printf "%.1f", t + m/1024}')
  done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)
  log "detected ${GPU_COUNT} GPU(s); min compute capability: ${MIN_CC:-unknown}; total VRAM: ${TOTAL_VRAM_GIB} GiB"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[dgx-kimi]   gpu /' >&2 || true
else
  warn "nvidia-smi not found — cannot verify the Kimi architecture floor on this host."
fi

if [[ -n "$MIN_CC" ]]; then
  # Gate: below sm_90 (Hopper) the stock native-FP8 Kimi K2 path is not expected to run.
  if awk -v cc="$MIN_CC" 'BEGIN{exit !(cc+0 < 9.0)}'; then
    warn "compute capability ${MIN_CC} is below sm_90 (Hopper)."
    warn "Kimi K2's native block-FP8 weights generally require sm_90+ (Hopper) or sm_100 (Blackwell)."
    warn "An Ampere node (A100, sm_80) has no native FP8; a BF16 upcast roughly doubles the footprint."
    warn "Options: use a Hopper/Blackwell pod, or front the Moonshot cloud K3 route (scripts/claude-kimi-k3.sh) instead."
    if [[ "$FORCE" != "1" ]]; then
      die "refusing to launch below the Kimi K2 arch floor. Set FAK_KIMI_FORCE=1 to override (expect kernel/precision failures)."
    fi
    warn "FAK_KIMI_FORCE=1 set — proceeding anyway; kernel or precision failures are likely."
  fi
fi

# VRAM-fit check: a ~1T-param FP8 MoE does not fit a single DGX (8×80GB = 640GB).
# This is a plausibility gate, not an exact planner — leave headroom for KV cache.
if awk -v f="$FOOTPRINT_GIB" 'BEGIN{exit !(f+0 > 0)}' && awk -v v="$TOTAL_VRAM_GIB" 'BEGIN{exit !(v+0 > 0)}'; then
  if awk -v v="$TOTAL_VRAM_GIB" -v f="$FOOTPRINT_GIB" 'BEGIN{exit !(v < f)}'; then
    warn "visible VRAM ${TOTAL_VRAM_GIB} GiB is below the ~${FOOTPRINT_GIB} GiB resident-weight estimate for ${MODEL}."
    warn "Kimi K2 (~1T params, FP8) is a MULTI-NODE expert-parallel deployment; a single DGX (640 GiB) does not hold it."
    warn "Add nodes (pipeline/expert parallel across a pod), pick a smaller checkpoint, or override FAK_KIMI_FOOTPRINT_GIB=0 to skip this check."
    if [[ "$FORCE" != "1" ]]; then
      die "refusing to launch: the chosen model will not fit the visible VRAM. Set FAK_KIMI_FORCE=1 to override (expect OOM)."
    fi
    warn "FAK_KIMI_FORCE=1 set — proceeding anyway; an out-of-memory failure at load is likely."
  fi
fi

if [[ "$TP" == "auto" ]]; then
  TP="${GPU_COUNT:-1}"; [[ "$TP" -lt 1 ]] && TP=1
fi

BASE_URL="http://${ADVERTISE_HOST}:${PORT}/v1"

# ── 2. build the launch command for the chosen engine ─────────────────────────
declare -a CMD
case "$ENGINE" in
  vllm)
    command -v vllm >/dev/null 2>&1 || die "vllm not on PATH. Install it in the serving venv (see scripts/requirements-sglang-serving.txt for the SGLang path)."
    CMD=(vllm serve "$MODEL" --served-model-name "$MODEL" --host "$HOST" --port "$PORT" --tensor-parallel-size "$TP" --trust-remote-code)
    ;;
  sglang)
    python -c 'import sglang' >/dev/null 2>&1 || die "sglang not importable. Install the SGLang serving venv first."
    CMD=(python -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port "$PORT" --tp "$TP" --trust-remote-code)
    ;;
  nim)
    command -v docker >/dev/null 2>&1 || die "docker not on PATH (required for ENGINE=nim)."
    [[ -n "$NIM_IMAGE" ]] || die "ENGINE=nim requires FAK_KIMI_NIM_IMAGE=<nvcr.io/nim/... Kimi image tag>. Confirm the tag exists in NGC before use."
    [[ -n "${NGC_API_KEY:-}" ]] || warn "NGC_API_KEY unset — 'docker login nvcr.io' / NIM pull may fail."
    CMD=(docker run --rm --gpus all --shm-size=16g -p "${PORT}:8000" -e "NGC_API_KEY=${NGC_API_KEY:-}" "$NIM_IMAGE")
    ;;
  *) die "unknown FAK_KIMI_ENGINE='$ENGINE' (want: vllm | sglang | nim)" ;;
esac

log "engine=$ENGINE model=$MODEL tp=$TP port=$PORT"
log "launch: ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  log "FAK_KIMI_DRY_RUN=1 — plan only, not launching."
else
  # ── 3. launch in the background, health-check GET /models until the roster is up
  "${CMD[@]}" &
  SERVER_PID=$!
  trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
  log "server pid=$SERVER_PID; waiting up to ${HEALTH_TIMEOUT}s for GET ${BASE_URL}/models …"
  deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then die "server process exited before /models became ready (check its logs above)."; fi
    if [[ $(date +%s) -ge $deadline ]]; then die "timed out after ${HEALTH_TIMEOUT}s waiting for /models."; fi
    sleep 5
  done
  log "READY: ${BASE_URL}/models is serving."
  trap - EXIT   # leave the server running; the operator owns its lifecycle
fi

# ── 4. print the exact fak + witness commands ─────────────────────────────────
cat >&2 <<EOF

$(printf '\033[32m[dgx-kimi] Kimi K2 OpenAI endpoint is up.\033[0m')

  Front it with fak (from this or any host that can reach ${ADVERTISE_HOST}:${PORT}):

    fak serve --provider openai-compatible \\
      --base-url "${BASE_URL}" \\
      --model "${MODEL}"

  Witness the wire (readiness + non-streaming + streaming), keyless-from-fak:

    KIMI_SELFHOST_BASE_URL="${BASE_URL}" \\
    KIMI_SELFHOST_MODEL="${MODEL}" \\
      go test ./internal/gateway -run TestKimiSelfHost -v

  (If your engine needs an auth token, also export KIMI_SELFHOST_API_KEY.)
  Perf headline is deferred to a tuned EP/EPLB baseline — this is wire readiness only.
EOF
