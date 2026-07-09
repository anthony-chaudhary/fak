#!/usr/bin/env bash
# Turnkey DeepSeek V4 bring-up on a GPU node (any CUDA node), fronted by `fak serve`.
#
#   [this script on the GPU node] → vLLM|SGLang|NIM OpenAI server :8000
#        → then, anywhere: fak serve --provider openai-compatible → that server
#
# It does the boring, error-prone parts of the self-host baseline runbook
# (docs/benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md) turnkey:
#   1. detect the GPUs and GATE on the DeepSeek-V4 architecture floor (below),
#   2. launch an OpenAI-compatible server for the chosen model+engine on :8000,
#   3. health-check GET /models until the roster is up,
#   4. print the exact `fak serve` line AND the keyless go smoke command that
#      witnesses the wire (readiness + non-streaming + streaming).
#
# It makes NO performance claim. It is the wire/bring-up floor, not a benchmark.
#
# ── DeepSeek V4 architecture floor (why the GPU gate exists) ──────────────────
# V4 checkpoints are FP4 (MoE experts) + FP8 (most other params) with an FP4
# attention indexer, and the attention stack is DSA (DeepSeek Sparse Attention).
# Stock kernels for these paths need modern GPUs:
#   * sm_100 (Blackwell, B200/GB200) — native NVFP4: the intended V4-Pro path.
#   * sm_90  (Hopper, H100/H200)     — native FP8, DSA kernels, FP4 via emulation.
#   * sm_89  (Ada, L40S)             — FP8 yes, DSA stock path usually NOT present.
#   * sm_80  (Ampere)                — no FP8/FP4, no DSA: BF16 dense fallback only,
#                                      generally cannot run a stock V4 resident path.
#   * sm_70  (Volta, V100-class)     — none of the above: not viable.
# The gate WARNS (does not silently proceed) below sm_90. Confirm the exact floor
# against your engine's V4 kernel requirements before reserving GPU time.
set -euo pipefail

# ── knobs (env-overridable) ───────────────────────────────────────────────────
ENGINE="${FAK_DGX_ENGINE:-vllm}"                 # vllm | sglang | nim
# V4-Flash is the single-node default (284B total / 13B active). V4-Pro is a
# 1.6T-total MoE and is a MULTI-NODE expert-parallel deployment — do not point a
# single node at it and expect it to fit.
MODEL="${FAK_DGX_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
PORT="${FAK_DGX_PORT:-8000}"
HOST="${FAK_DGX_HOST:-0.0.0.0}"
ADVERTISE_HOST="${FAK_DGX_ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
NIM_IMAGE="${FAK_DGX_NIM_IMAGE:-}"               # required only when ENGINE=nim
TP="${FAK_DGX_TP:-auto}"                          # tensor-parallel size; auto = #GPUs
HEALTH_TIMEOUT="${FAK_DGX_HEALTH_TIMEOUT:-1800}"  # seconds to wait for /models
DRY_RUN="${FAK_DGX_DRY_RUN:-0}"                   # 1 = print plan, do not launch
FORCE="${FAK_DGX_FORCE:-0}"                       # 1 = proceed past the arch-floor gate

log()  { printf '\033[36m[dgx-deepseek]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[dgx-deepseek] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[dgx-deepseek] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. detect GPUs + gate on the architecture floor ───────────────────────────
GPU_COUNT=0
MIN_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  # compute_cap is emitted as e.g. "9.0"; take the minimum across visible GPUs.
  mapfile -t _caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
  GPU_COUNT="${#_caps[@]}"
  for c in "${_caps[@]}"; do
    [[ -z "$c" ]] && continue
    if [[ -z "$MIN_CC" ]] || awk -v a="$c" -v b="$MIN_CC" 'BEGIN{exit !(a<b)}'; then MIN_CC="$c"; fi
  done
  log "detected ${GPU_COUNT} GPU(s); min compute capability: ${MIN_CC:-unknown}"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[dgx-deepseek]   gpu /' >&2 || true
else
  warn "nvidia-smi not found — cannot verify the V4 architecture floor on this host."
fi

if [[ -n "$MIN_CC" ]]; then
  # Gate: below sm_90 (Hopper) the stock FP4/FP8/DSA V4 paths are not expected to run.
  if awk -v cc="$MIN_CC" 'BEGIN{exit !(cc+0 < 9.0)}'; then
    warn "compute capability ${MIN_CC} is below sm_90 (Hopper)."
    warn "DeepSeek V4's stock FP4/FP8 + DSA kernels generally require sm_90+ (Hopper) or sm_100 (Blackwell for native FP4)."
    warn "A Volta-class node (V100, sm_70) cannot serve a stock V4 resident path."
    warn "Options: use a Hopper/Blackwell node, a V4-Flash BF16 dense-attention fallback if your engine provides one, or route to a hosted V4 endpoint instead."
    if [[ "$FORCE" != "1" ]]; then
      die "refusing to launch below the V4 arch floor. Set FAK_DGX_FORCE=1 to override (expect kernel/precision failures)."
    fi
    warn "FAK_DGX_FORCE=1 set — proceeding anyway; kernel or precision failures are likely."
  fi
fi

if [[ "$TP" == "auto" ]]; then
  TP="${GPU_COUNT:-1}"; [[ "$TP" -lt 1 ]] && TP=1
fi
case "$MODEL" in
  *V4-Pro*|*v4-pro*) warn "MODEL names V4-Pro (1.6T total) — this is a MULTI-NODE expert-parallel deployment; a single node will not fit it. V4-Flash is the single-node default." ;;
esac

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
    [[ -n "$NIM_IMAGE" ]] || die "ENGINE=nim requires FAK_DGX_NIM_IMAGE=<nvcr.io/nim/... V4 image tag>. Confirm the tag exists in NGC before use."
    [[ -n "${NGC_API_KEY:-}" ]] || warn "NGC_API_KEY unset — 'docker login nvcr.io' / NIM pull may fail."
    CMD=(docker run --rm --gpus all --shm-size=16g -p "${PORT}:8000" -e "NGC_API_KEY=${NGC_API_KEY:-}" "$NIM_IMAGE")
    ;;
  *) die "unknown FAK_DGX_ENGINE='$ENGINE' (want: vllm | sglang | nim)" ;;
esac

log "engine=$ENGINE model=$MODEL tp=$TP port=$PORT"
log "launch: ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  log "FAK_DGX_DRY_RUN=1 — plan only, not launching."
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

$(printf '\033[32m[dgx-deepseek] DeepSeek V4 OpenAI endpoint is up.\033[0m')

  Front it with fak (from this or any host that can reach ${ADVERTISE_HOST}:${PORT}):

    fak serve --provider openai-compatible \\
      --base-url "${BASE_URL}" \\
      --model "${MODEL}"

  Witness the wire (readiness + non-streaming + streaming), keyless-from-fak:

    DEEPSEEK_SELFHOST_BASE_URL="${BASE_URL}" \\
    DEEPSEEK_SELFHOST_MODEL="${MODEL}" \\
      go test ./internal/gateway -run TestDeepSeekV4SelfHost -v

  (If your engine needs an auth token, also export DEEPSEEK_SELFHOST_API_KEY.)
  Perf headline is deferred to the tuned EP/EPLB baseline — this is wire readiness only.
EOF
