#!/usr/bin/env bash
# deepseek2_fak_native_serve.sh - durable stage + build + serve of a DeepSeek-V2/V3 (arch
# "deepseek2") GGUF on an A100 (sm_80) node via fak's OWN in-kernel engine (the PURE FAK
# KERNEL), not llama.cpp. It is the DeepSeek sibling of tools/glm52_fak_native_serve.sh.
#
# WHY THIS RUNS ON sm_80 (the A100 "not supported" overcome):
#   DeepSeek (deepseek2) is exactly the glm_moe_dsa architecture fak already serves MINUS the
#   DSA "lightning indexer": same Multi-head Latent Attention (q_a/q_b, kv_a/kv_b, decoupled
#   RoPE), same MoE router + shared experts + batched routed experts. With no indexer
#   (IndexNHeads==0) every query attends its FULL causal prefix — dense MLA, which needs NO
#   sm_90 kernel and runs on Ampere, exactly as GLM-5.2's MLA does under llama.cpp. So fak's
#   native forward serves DeepSeek on sm_80 where the stock V4 FP4/DSA stack cannot
#   (docs/benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md: sm_80 is below the stock V4 floor).
#   fak routes deepseek2 through the same glm MLA+MoE forward via Config.usesMLAMoELayout
#   (internal/model/config.go) + archUsesMLAMoELayout (internal/ggufload); ModelType stays
#   "deepseek2" (first-class, honestly labeled — NOT relabeled glm).
#
# HONEST SCOPE (read before quoting anything): this stands the endpoint UP and health-checks a
# REAL chat completion; it asserts NO throughput/quality number. The dense-MLA seam is WITNESSED
# on a synthetic deepseek2 fixture (internal/model/deepseek2_dense_mla_test.go: full-causal
# selection, decode self-consistency, context-dependence) — a live serve of a REAL DeepSeek
# checkpoint through fak's kernel is what THIS script stands up and is the next gate. If a real
# GGUF surfaces a loader/forward gap (tensor-name mapping, MoE expert split, rope/YaRN scaling,
# tokenizer), that is a concrete fak-owned finding to file — not a reason to fake a pass. The
# engine-honest THROUGHPUT baseline is llama.cpp serving the SAME deepseek2 GGUF (llama.cpp has
# mature deepseek2 support); keep that as the comparison arm and never put the two numbers side
# by side without holding {weights, hardware, precision, ctx} equal.
#
# DEFAULT TARGET: DeepSeek-V2-Lite (15.7B total / 2.4B active, deepseek2 arch). At Q4_K_M it is
# ~10 GB and fits RESIDENT on a single 80 GB card — the right first live-serve bring-up (no
# cpu-offload needed). For the big V3 (671B) set DS_REPO/DS_SUBDIR to a V3 quant and
# CPU_OFFLOAD=1 (the ~400 GB experts stay on host RAM; attention/shared/dense on the GPUs).
#
# Usage (RUN ON THE GPU HOST, detached so a disconnect does not orphan a large load):
#   systemd-run --unit=deepseek2serve --collect bash tools/deepseek2_fak_native_serve.sh
# then poll:  cat "$DS_DIR/PHASE"   and on DEEPSEEK2_FAK_NATIVE_SERVE_READY run the witness/e2e.
set -uo pipefail

DS_DIR="${DS_DIR:-/opt/deepseek2}"
# A real deepseek2 GGUF. V2-Lite is the small resident default; override for V3.
# Deliberately NOT bartowski/DeepSeek-V2-Lite-Chat-GGUF: that repo is not publicly
# fetchable. The HF model API answers 401 for it while openai-community/gpt2 and this
# repo both answer 200, so it is that repo specifically, not an unauthenticated host.
# The failure is silent in the worst way -- `hf download` prints "Invalid username or
# password", downloads nothing, and STILL exits 0 -- so rc is not evidence here; only
# the resolve_gguf re-check below catches it (DOWNLOAD_FAIL rc=0 no gguf matching...).
# mradermacher's is public, ungated, and carries DeepSeek-V2-Lite-Chat.Q4_K_M.gguf,
# which the DS_GLOB default already matches.
DS_REPO="${DS_REPO:-mradermacher/DeepSeek-V2-Lite-Chat-GGUF}"
DS_SUBDIR="${DS_SUBDIR:-}"                       # some repos put shards flat (no quant subdir)
DS_GLOB="${DS_GLOB:-*Q4_K_M*.gguf}"             # which quant file(s) to fetch/serve
PORT="${PORT:-8000}"
ADDR="${ADDR:-0.0.0.0:${PORT}}"
MODEL_ID="${MODEL_ID:-deepseek-v2-lite}"
CTX="${CTX:-4096}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"                  # 1 for a checkpoint whose experts dwarf VRAM (V3)
FAK_BIN="${FAK_BIN:-/usr/local/bin/fak}"
GO_VERSION="${GO_VERSION:-1.26.4}"
SMOKE_TIMEOUT_S="${SMOKE_TIMEOUT_S:-900}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-8}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export FAK_HTTP_WRITE_TIMEOUT_S="${FAK_HTTP_WRITE_TIMEOUT_S:-0}"
export FAK_KQ_INT8="${FAK_KQ_INT8:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PHASE="$DS_DIR/PHASE"
LOG="$DS_DIR/fak_native_serve.log"
mkdir -p "$DS_DIR" "$GOCACHE" "$GOPATH"
ph(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; echo "$*" > "$PHASE"; }

# resolve the first (or only) GGUF file matching DS_GLOB under a dir. Handles both a single-file
# quant (V2-Lite) and a *-00001-of-NNNNN sharded quant (V3). Echoes the path or returns 1.
resolve_gguf() {
  _d="$1"
  _s1=$(ls "$_d"/*-00001-of-*.gguf 2>/dev/null | sort | head -1) || true
  [ -n "$_s1" ] && { echo "$_s1"; return 0; }
  _one=$(ls "$_d"/$DS_GLOB 2>/dev/null | sort | head -1) || true
  [ -n "$_one" ] && { echo "$_one"; return 0; }
  return 1
}

export PATH="/usr/local/go/bin:${CUDA_HOME}/bin:$PATH"

# 1. Ensure Go for the CUDA build.
if ! command -v go >/dev/null 2>&1; then
  ph "INSTALL_GO ${GO_VERSION}"
  if curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz >>"$LOG" 2>&1 \
     && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz >>"$LOG" 2>&1; then :; else
    ph "GO_INSTALL_FAIL"; exit 11
  fi
fi
ph "GO $(go version 2>/dev/null || echo missing)"

# 2. Stage the GGUF (prefer a complete local copy; else resumable HF download).
FETCH_DIR="$DS_DIR${DS_SUBDIR:+/$DS_SUBDIR}"
if SHARD1=$(resolve_gguf "$FETCH_DIR"); then
  ph "USING_PRESTAGED shard1=$SHARD1 (HF download skipped)"
else
  ph "DOWNLOAD_START repo=$DS_REPO subdir=${DS_SUBDIR:-<flat>} glob=$DS_GLOB dir=$DS_DIR"
  _inc="${DS_SUBDIR:+$DS_SUBDIR/}$DS_GLOB"
  if command -v hf >/dev/null 2>&1; then
    hf download "$DS_REPO" --include "$_inc" --local-dir "$DS_DIR" >>"$LOG" 2>&1; DL_RC=$?
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$DS_REPO" --include "$_inc" --local-dir "$DS_DIR" >>"$LOG" 2>&1; DL_RC=$?
  else
    ph "NO_HF_CLI install huggingface_hub first"; exit 10
  fi
  if SHARD1=$(resolve_gguf "$FETCH_DIR"); then
    ph "DOWNLOAD_DONE rc=${DL_RC:-?} shard1=$SHARD1"
  else
    ph "DOWNLOAD_FAIL rc=${DL_RC:-?} no gguf matching '$DS_GLOB' under $FETCH_DIR"; exit 20
  fi
fi
[ -n "$SHARD1" ] && [ -s "$SHARD1" ] || { ph "NO_SHARD"; exit 20; }

# 3. Build the -tags cuda fak binary (libfakcuda sm_80 + cmd/fak) via the canonical recipe.
if [ ! -x "$FAK_BIN" ] || [ "${REBUILD_FAK:-0}" = "1" ]; then
  ph "BUILD_FAK_CUDA arch=$FAK_CUDA_ARCH out=$FAK_BIN"
  if ( cd "$ROOT" && bash internal/compute/build_cuda.sh binary ./cmd/fak "$FAK_BIN" ) >>"$LOG" 2>&1; then :; else
    ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
  fi
fi
[ -x "$FAK_BIN" ] || { ph "BUILD_FAK_FAIL"; exit 30; }
ph "FAK_BIN_READY $FAK_BIN"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# 4. Serve via the PURE FAK KERNEL. The embedded GGUF tokenizer makes /v1/chat/completions serve
#    real in-kernel chat; the eager load binds the listener only AFTER the weights are resident,
#    so /v1/models answering means the model is loaded.
OFFLOAD_ARG=()
[ "$CPU_OFFLOAD" = "1" ] && OFFLOAD_ARG=(--cpu-offload-experts)
ph "LAUNCH fak serve --gguf $SHARD1 --backend cuda ${OFFLOAD_ARG[*]:-} --context-budget-tokens $CTX --model $MODEL_ID"
"$FAK_BIN" serve \
  --addr "$ADDR" \
  --gguf "$SHARD1" \
  --backend cuda \
  "${OFFLOAD_ARG[@]}" \
  --context-budget-tokens "$CTX" \
  --model "$MODEL_ID" \
  > "$DS_DIR/server.log" 2>&1 &
SRV=$!
ph "SERVER_PID=$SRV"

smoke_body() {
  printf '{"model":"%s","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":%s,"temperature":0}' "$MODEL_ID" "$SMOKE_MAX_TOKENS"
}

# 5. Health-check: crash-detect immediately, and assert a REAL chat answer before declaring ready
#    (a server that bound but cannot decode must NOT greenlight a witness). 180 x 20s ~= 1h.
for _ in $(seq 1 180); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -60 "$DS_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    smoke=$(curl -sS -m "$SMOKE_TIMEOUT_S" "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$(smoke_body)" 2>&1)
    echo "SMOKE: $smoke" >>"$LOG"
    if printf '%s' "$smoke" | grep -q '"choices"' && ! printf '%s' "$smoke" | grep -q '"error"'; then
      ph "DEEPSEEK2_FAK_NATIVE_SERVE_READY port=$PORT model=$MODEL_ID gguf=$SHARD1"
      wait "$SRV"; rc=$?; ph "SERVER_EXITED rc=$rc"; exit "$rc"
    fi
    ph "SMOKE_FAIL"; tail -40 "$DS_DIR/server.log" >>"$LOG" 2>&1; exit 41
  fi
  sleep 20
done
ph "HEALTH_TIMEOUT"; tail -30 "$DS_DIR/server.log" >>"$LOG" 2>&1; exit 42
