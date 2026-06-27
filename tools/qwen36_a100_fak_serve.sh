#!/usr/bin/env bash
# qwen36_a100_fak_serve.sh — durable stage + build + serve of Qwen3.6-27B on a single
# A100 (sm_80) via fak's OWN in-kernel CUDA engine (the PURE FAK KERNEL), so it can be
# used DIRECTLY from Claude Code here when a subscription account is unavailable.
#
# This is the Qwen sibling of tools/glm52_fak_native_serve.sh, but SIMPLER on purpose:
# Qwen3.6-27B q4_k_m is ~16-17 GB resident, so it fits a single A100-40GB whole — NO
# --cpu-offload-experts, NO multi-shard staging, NO 466 GB MoE. One GPU, one file, one serve.
#
# WHY THE PURE FAK KERNEL (vs an external engine):
#   * fak serves Qwen3.6-27B through its OWN forward. That forward is cosine-witnessed
#     >= 0.9999 vs HF and argmax-exact on the CPU reference (the qwen35 oracle gates, #442
#     — see docs/qwen36-claude-dogfood-playbook.md "Native parity witnesses"). That
#     correctness guarantee is fak's differentiator; the stock engines do not make it.
#   * It exposes the SAME OpenAI /v1 + Anthropic /v1/messages wire Claude Code already
#     speaks, so `fak guard` / connect-fak-node front it identically.
#
# HOW IT SERVES (the Qwen3.6-27B decode lever, QWEN36-NATIVE-PERF-PLAN P1/P2):
#   FAK_Q4K=1 fak serve --gguf <Qwen3.6-27B.q4_k_m.gguf> --backend cuda --addr 0.0.0.0:PORT
#   * --backend cuda  : prefill+decode run on the GPU HAL (needs a -tags cuda build + a GPU).
#   * FAK_Q4K=1       : the direct-resident-Q4_K path — holds the q4_k matmul tensors raw on
#                       device (no Q4_K->f32->Q8 round-trip), the 27B decode lever.
#   * CUDA_GRAPH=1    : pass --cuda-graph — capture each decode token's op stream into a CUDA
#                       graph and replay it as ONE launch (#483). The per-token launch-overhead
#                       lever for large single-stream decode; OFF by default (a measured no-win
#                       on a tiny 0.5B/L4). WITNESS tok/s before/after on THIS node before
#                       relying on it — the 27B/A100 calculus differs from the 0.5B/L4 one.
#
# HONEST SCOPE: this asserts NO throughput number and makes NO live-serve claim by itself.
# It builds the cuda fak binary, fetches the checkpoint, stands the endpoint up, and
# health-checks a REAL chat completion. A live 27B serve turn — and its tok/s — is
# hardware+load-gated until this runs on a real A100. The witnessed claim is the CPU forward
# correctness (cosine, #442); the CUDA-resident-Q4_K-27B decode RATE is the open perf item to
# measure here.
#
# Usage (RUN ON THE A100 HOST, detached so a disconnect does not orphan the load):
#   FAK_GATEWAY_KEY=sk-fak-... systemd-run --unit=qwen36serve --collect \
#     bash tools/qwen36_a100_fak_serve.sh
# then poll:  cat "$QWEN_DIR/PHASE"   and on QWEN36_A100_FAK_SERVE_READY connect from the client.
set -uo pipefail

QWEN_DIR="${QWEN_DIR:-/opt/qwen36-q4k}"
REPO="${QWEN_REPO:-lmstudio-community/Qwen3.6-27B-GGUF}"
# Glob that matches the q4_k_m shard in the repo (the file name carries the quant tag).
QWEN_FILE_GLOB="${QWEN_FILE_GLOB:-*[Qq]4_[Kk]_[Mm]*.gguf}"
PORT="${PORT:-8080}"
ADDR="${ADDR:-0.0.0.0:${PORT}}"
MODEL_ID="${MODEL_ID:-qwen3.6-27b}"
FAK_BIN="${FAK_BIN:-/usr/local/bin/fak}"
GO_VERSION="${GO_VERSION:-1.26.4}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"        # 1 => pass --cuda-graph (#483); witness tok/s before trusting
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"

# locate the fak checkout root from this script's location (tools/<this>).
SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PHASE="$QWEN_DIR/PHASE"
LOG="$QWEN_DIR/fak_native_serve.log"
mkdir -p "$QWEN_DIR" "$GOCACHE" "$GOPATH"
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

# 2. Download the single q4_k_m GGUF (resumable; the HF CLI skips an already-complete file).
ph "DOWNLOAD_START repo=$REPO glob=$QWEN_FILE_GLOB dir=$QWEN_DIR"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --include "$QWEN_FILE_GLOB" --local-dir "$QWEN_DIR" >>"$LOG" 2>&1; DL_RC=$?
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --include "$QWEN_FILE_GLOB" --local-dir "$QWEN_DIR" >>"$LOG" 2>&1; DL_RC=$?
else
  ph "NO_HF_CLI install huggingface_hub first"; exit 10
fi
# shellcheck disable=SC2086
GGUF=$(ls "$QWEN_DIR"/$QWEN_FILE_GLOB 2>/dev/null | sort | head -1)
ph "DOWNLOAD_DONE rc=$DL_RC gguf=$GGUF"
[ "${DL_RC:-1}" -eq 0 ] && [ -n "$GGUF" ] && [ -f "$GGUF" ] || { ph "DOWNLOAD_FAIL"; exit 20; }

# 3. Build the -tags cuda fak binary (libfakcuda sm_80 + cmd/fak) via the canonical recipe.
if [ ! -x "$FAK_BIN" ] || [ "${REBUILD_FAK:-0}" = "1" ]; then
  ph "BUILD_FAK_CUDA arch=$FAK_CUDA_ARCH out=$FAK_BIN"
  if ( cd "$ROOT" && bash internal/compute/build_cuda.sh binary ./cmd/fak "$FAK_BIN" ) >>"$LOG" 2>&1; then :; else
    ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
  fi
fi
[ -x "$FAK_BIN" ] || { ph "BUILD_FAK_FAIL"; exit 30; }
ph "FAK_BIN_READY $FAK_BIN"

# Runtime: put the CUDA shared libs (cudart/cublas) on LD_LIBRARY_PATH (belt-and-braces; the
# binary also bakes an rpath at link time).
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# 4. Serve via the PURE FAK KERNEL, resident-Q4_K decode. The embedded GGUF tokenizer makes
#    /v1/chat/completions AND /v1/messages serve real in-kernel chat; the eager load binds the
#    listener only AFTER the weights are resident, so /v1/models answering means it is loaded.
SERVE_ARGS=(serve --addr "$ADDR" --gguf "$GGUF" --backend cuda --model "$MODEL_ID")
[ "$CUDA_GRAPH" = "1" ] && SERVE_ARGS+=(--cuda-graph)
# Inbound bearer auth: a network-facing gateway must require a key. connect-fak-node uses it.
if [ -n "${FAK_GATEWAY_KEY:-}" ]; then
  export FAK_GATEWAY_KEY
  SERVE_ARGS+=(--require-key-env FAK_GATEWAY_KEY)
else
  ph "WARN no FAK_GATEWAY_KEY set — serving with NO inbound auth (loopback/tunnel only!)"
fi
ph "LAUNCH FAK_Q4K=1 fak ${SERVE_ARGS[*]} (resident-Q4_K 27B decode; cuda-graph=$CUDA_GRAPH)"
FAK_Q4K=1 "$FAK_BIN" "${SERVE_ARGS[@]}" > "$QWEN_DIR/server.log" 2>&1 &
SRV=$!
ph "SERVER_PID=$SRV"

# 5. Health-check: detect a crashed load immediately, and assert a REAL chat answer before
#    declaring ready. 90 x 20s = 30 min, ample for a ~16 GB single-file load.
AUTH_HDR=()
[ -n "${FAK_GATEWAY_KEY:-}" ] && AUTH_HDR=(-H "Authorization: Bearer ${FAK_GATEWAY_KEY}")
for _ in $(seq 1 90); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -40 "$QWEN_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
  if curl -sf -m 5 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 \
     || curl -sf -m 5 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    smoke=$(curl -s -m 120 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":8}")
    echo "SMOKE: $smoke" >>"$LOG"
    if printf '%s' "$smoke" | grep -q '"content"' && ! printf '%s' "$smoke" | grep -q '"error"'; then
      ph "QWEN36_A100_FAK_SERVE_READY port=$PORT model=$MODEL_ID"; exit 0
    fi
    ph "SMOKE_FAIL"; exit 41
  fi
  sleep 20
done
ph "HEALTH_TIMEOUT"; tail -20 "$QWEN_DIR/server.log" >>"$LOG" 2>&1; exit 42
