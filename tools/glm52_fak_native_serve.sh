#!/usr/bin/env bash
# glm52_fak_native_serve.sh - durable stage + build + serve of GLM-5.2 on an A100 (sm_80)
# node via fak's OWN in-kernel engine (the PURE FAK KERNEL), not llama.cpp. It is the
# pure-fak-kernel sibling of tools/glm52_stage_serve_dgx3.sh (which stands up the SAME
# checkpoint under llama.cpp as the BENCHMARK baseline). Prefer THIS; keep llama.cpp for
# the apples-to-apples comparison.
#
# WHY THIS IS THE PREFERRED PATH (vs the llama.cpp baseline):
#   * fak serves GLM-5.2 (glm_moe_dsa) through its OWN CUDA kernels — the forward is bit-exact
#     vs the CPU reference AT Q8 (cosine 1.000000, argmax-exact) on sm_80, witnessed at
#     experiments/glm-gpu-witness/a100-glm52-*.json (incl. the cpu-offload hybrid). That
#     correctness guarantee is fak's differentiator; the stock engines do not make it. NOTE:
#     the --cpu-offload-experts serve below runs the resident-Q4_K path, which is not yet
#     covered by a full-forward cosine witness (the q8 forward is).
#   * It is the SAME wire llama.cpp serves (OpenAI /v1) so `fak guard` / the #413 witness
#     front it identically — but the weights run in fak's kernel, not an external engine.
#   * llama.cpp stays the honest THROUGHPUT baseline for the comparison ladder
#     (docs/notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md). Never put the
#     two numbers side by side without holding {weights, hardware, precision, ctx} equal.
#
# HOW IT SERVES (the load-speed doc, GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25):
#   fak serve --gguf <shard1> --backend cuda --cpu-offload-experts --context-budget-tokens 8192
#   * --backend cuda  : prefill+decode run on the GPU HAL (needs a -tags cuda build + a GPU).
#   * --cpu-offload-experts : the ~424 GB MoE experts stay on host RAM (the A100s hold
#     attention/shared/dense); the device load uses the direct-resident-Q4_K path (no
#     Q4_K->f32->Q8 round-trip).
#   * --context-budget-tokens 8192 : the default 1M context plans a 533 GiB KV -> FitTooBig.
#
# HONEST SCOPE: this asserts NO throughput/quality number, and makes NO live-serve claim. It
# builds the cuda fak binary, stages the checkpoint, stands the endpoint up, and health-checks
# a REAL chat completion — a live GLM-5.2 serve turn is hardware+load-gated until that gate
# passes on a real A100. The witnessed claim is the Q8 forward correctness (cosine 1.0, sm_80);
# the resident-Q4_K serve path is not yet cosine-witnessed, and load time on the dynamic-mixed
# UD-Q4_K_M is the open perf item (the resident-Q4_K path only fully fires on pure-Q4_K
# tensors). Run tools/glm52_e2e_after_serve_dgx3.sh against this endpoint for the #413 evidence.
#
# Usage (RUN ON THE GPU HOST, detached so a disconnect does not orphan a large load):
#   systemd-run --unit=glm52serve --collect bash tools/glm52_fak_native_serve.sh
# then poll:  cat "$GLM_DIR/PHASE"   and on GLM52_FAK_NATIVE_SERVE_READY run the witness.
set -uo pipefail

GLM_DIR="${GLM_DIR:-/opt/glm52-q4}"
REPO="${GLM_REPO:-unsloth/GLM-5.2-GGUF}"
SUBDIR="${GLM_SUBDIR:-UD-Q4_K_M}"
PORT="${PORT:-8000}"
ADDR="${ADDR:-0.0.0.0:${PORT}}"
MODEL_ID="${MODEL_ID:-glm-5.2}"
CTX="${CTX:-8192}"
FAK_BIN="${FAK_BIN:-/usr/local/bin/fak}"
GO_VERSION="${GO_VERSION:-1.26.4}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"

# locate the fak checkout root from this script's location (tools/<this>).
SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PHASE="$GLM_DIR/PHASE"
LOG="$GLM_DIR/fak_native_serve.log"
mkdir -p "$GLM_DIR" "$GOCACHE" "$GOPATH"
ph(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; echo "$*" > "$PHASE"; }

export PATH="/usr/local/go/bin:${CUDA_HOME}/bin:$PATH"

# 1. Ensure Go. The GCP Deep-Learning CUDA image ships nvcc but not always the Go toolchain
#    build_cuda.sh expects at /usr/local/go/bin; install it once if missing.
if ! command -v go >/dev/null 2>&1; then
  ph "INSTALL_GO ${GO_VERSION}"
  if curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz >>"$LOG" 2>&1 \
     && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz >>"$LOG" 2>&1; then :; else
    ph "GO_INSTALL_FAIL"; exit 11
  fi
fi
ph "GO $(go version 2>/dev/null || echo missing)"

# 2. Download the GGUF shards (resumable; the HF CLI skips already-complete files).
ph "DOWNLOAD_START repo=$REPO subdir=$SUBDIR dir=$GLM_DIR"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --include "$SUBDIR/*" --local-dir "$GLM_DIR" >>"$LOG" 2>&1; DL_RC=$?
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --include "$SUBDIR/*" --local-dir "$GLM_DIR" >>"$LOG" 2>&1; DL_RC=$?
else
  ph "NO_HF_CLI install huggingface_hub first"; exit 10
fi
SHARDS=$(ls "$GLM_DIR/$SUBDIR"/*.gguf 2>/dev/null | wc -l)
ph "DOWNLOAD_DONE rc=$DL_RC shards=$SHARDS"
[ "${DL_RC:-1}" -eq 0 ] && [ "${SHARDS:-0}" -ge 1 ] || { ph "DOWNLOAD_FAIL"; exit 20; }
SHARD1=$(ls "$GLM_DIR/$SUBDIR"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -n "$SHARD1" ] || SHARD1=$(ls "$GLM_DIR/$SUBDIR"/*.gguf 2>/dev/null | sort | head -1)
ph "SHARD1=$SHARD1"

# 3. Build the -tags cuda fak binary (libfakcuda sm_80 + cmd/fak) via the canonical recipe
#    (internal/compute/build_cuda.sh resolves the CGO -I/-L/-rpath set for this host).
if [ ! -x "$FAK_BIN" ] || [ "${REBUILD_FAK:-0}" = "1" ]; then
  ph "BUILD_FAK_CUDA arch=$FAK_CUDA_ARCH out=$FAK_BIN"
  if ( cd "$ROOT" && bash internal/compute/build_cuda.sh binary ./cmd/fak "$FAK_BIN" ) >>"$LOG" 2>&1; then :; else
    ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
  fi
fi
[ -x "$FAK_BIN" ] || { ph "BUILD_FAK_FAIL"; exit 30; }
ph "FAK_BIN_READY $FAK_BIN"

# Runtime: put the CUDA shared libs (cudart/cublas) on LD_LIBRARY_PATH. The binary also
# bakes an rpath to them at link time; this is belt-and-braces for systemd's clean env.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# 4. Serve via the PURE FAK KERNEL. The embedded GGUF tokenizer makes /v1/chat/completions
#    serve real in-kernel chat; the eager load binds the listener only AFTER the weights are
#    resident, so /v1/models answering means the model is loaded.
ph "LAUNCH fak serve --gguf $SHARD1 --backend cuda --cpu-offload-experts --context-budget-tokens $CTX --model $MODEL_ID (large load; resident-Q4_K path)"
"$FAK_BIN" serve \
  --addr "$ADDR" \
  --gguf "$SHARD1" \
  --backend cuda \
  --cpu-offload-experts \
  --context-budget-tokens "$CTX" \
  --model "$MODEL_ID" \
  > "$GLM_DIR/server.log" 2>&1 &
SRV=$!
ph "SERVER_PID=$SRV"

# 5. Health-check: detect a crashed load immediately, and assert a REAL chat answer before
#    declaring ready (a server that bound but cannot decode must NOT greenlight a witness).
#    360 x 20s ~= 2 h, covering the large UD-Q4_K_M load.
for _ in $(seq 1 360); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -40 "$GLM_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    smoke=$(curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":8}")
    echo "SMOKE: $smoke" >>"$LOG"
    if printf '%s' "$smoke" | grep -q '"content"' && ! printf '%s' "$smoke" | grep -q '"error"'; then
      ph "GLM52_FAK_NATIVE_SERVE_READY port=$PORT model=$MODEL_ID"; exit 0
    fi
    ph "SMOKE_FAIL"; exit 41
  fi
  sleep 20
done
ph "HEALTH_TIMEOUT"; tail -20 "$GLM_DIR/server.log" >>"$LOG" 2>&1; exit 42
