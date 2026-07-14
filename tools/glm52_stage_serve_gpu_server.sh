#!/usr/bin/env bash
# glm52_stage_serve_dgx3.sh - durable stage+build+serve of GLM-5.2 on an A100 (sm_80) node
# via llama.cpp MLA + CPU expert-offload. The self-staging, unsloth-GGUF sibling of
# tools/glm52_serve.sh.
#
# WHY THIS EXISTS (the A100 "not supported" overcome):
#   Stock SGLang/vLLM CANNOT serve GLM-5.2 (glm_moe_dsa) on A100 (sm_80): its
#   DeepSeek-Sparse-Attention kernels hard-depend on Hopper/Blackwell DeepGemm
#   (vLLM #35021). tools/glm52_serve_preflight.py returns BLOCKED_ARCH below sm_90.
#   llama.cpp CAN: it loads glm_moe_dsa GGUF and runs it as full MLA attention
#   (DSA indexer WIP upstream) - MLA needs no sm_90 kernel, so it runs on Ampere.
#
# DIFFERENCE FROM tools/glm52_serve.sh:
#   * glm52_serve.sh expects 8 PRE-STAGED shards (GLM-5.2-Q4_K_M-*-of-00008.gguf) and a
#     manually fetched checkpoint; it does not download.
#   * THIS script self-stages unsloth/GLM-5.2-GGUF UD-Q4_K_M (the published 11-shard,
#     ~466 GB dynamic-Q4 quant) via the huggingface CLI, resumable, then builds and serves.
#   * It writes a PHASE file each step so a control-bridge with an unreliable live tail can
#     poll progress out-of-band (DOWNLOAD_START -> DOWNLOAD_DONE -> BUILD_LLAMA ->
#     SERVER_READY -> GLM52_SERVE_READY).
#
# HONEST SCOPE: this script asserts NO throughput/quality number. It stands the endpoint
# up and health-checks a real chat completion; the #413 serving witness
# (tools/glm52_serving_witness.py) is what records the measured evidence against it.
# UD-Q4_K_M is a dynamic 4-bit quant (mostly-lossless per unsloth), not the bf16/fp8 master;
# the witness records the served quant in-band so no number is over-claimed.
#
# Memory note: UD-Q4_K_M is ~466 GB. On 8xA100-80GB (640 GB VRAM) it does NOT fit weights-
# resident; NCPU_MOE keeps the MoE experts on host RAM (the node needs ~512 GB+ free) while
# attention/shared tensors live on the GPUs. There is no per-node VRAM math here - run
# tools/glm52_serve_preflight.py first for the footprint gate.
#
# Usage (RUN ON THE GPU HOST, detached so a disconnect does not orphan a ~466 GB load):
#   systemd-run --user --unit=glm52stage --collect bash tools/glm52_stage_serve_dgx3.sh
# then poll:  cat "$GLM_DIR/PHASE"   and on GLM52_SERVE_READY run the #413 witness.
set -uo pipefail

GLM_DIR="${GLM_DIR:-/projects/glm52-q4}"
LLAMA="${LLAMA:-/projects/llama.cpp}"
REPO="${GLM_REPO:-unsloth/GLM-5.2-GGUF}"
SUBDIR="${GLM_SUBDIR:-UD-Q4_K_M}"
PORT="${PORT:-8000}"
NGL="${NGL:-99}"
# Keep ALL expert layers on host RAM (errs safe against VRAM OOM); lower if VRAM is spare.
NCPU_MOE="${NCPU_MOE:-999}"
CTX="${CTX:-8192}"
PHASE="$GLM_DIR/PHASE"
LOG="$GLM_DIR/stage_serve.log"
# CUDA toolkit bin on PATH for the sm_80 llama.cpp build under systemd's clean env. The DGX
# example ships CUDA 12.8; the GCP Deep-Learning image ships 12.9 at /usr/local/cuda. Prefer
# CUDA_BIN, then the generic symlink, then known versions — so the SAME script builds on the
# DGX A100 and a GCP a2 A100 node (the "bring the DGX example to GCP" path) with no edit.
export PATH="/usr/local/go/bin:$PATH"
for _cuda in "${CUDA_BIN:-}" /usr/local/cuda/bin /usr/local/cuda-12.9/bin /usr/local/cuda-12.8/bin; do
  if [ -n "$_cuda" ] && [ -d "$_cuda" ]; then export PATH="$_cuda:$PATH"; break; fi
done

mkdir -p "$GLM_DIR"
ph(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; echo "$*" > "$PHASE"; }

# 0. Prefer a COMPLETE local-NVMe copy over the slow /projects NFS. A verified local copy
#    reads ~2.9 GB/s (NVMe) vs ~0.055 GB/s (NFSv3) — ~53x — dropping the cold load from
#    ~1h41m to ~44m. The verified copy lives FLAT in /mnt/sglang_dv3/glm52-q4/ (no UD-Q4_K_M
#    subdir). Resolve NVMe-first and LOG the winner LOUDLY so a silent fall-through to the slow
#    NFS path can never go unnoticed (that precedence bug is what caused a 62-min cold load).
PRESTAGED_SHARD1=""
for _d in /mnt/sglang_dv3/glm52-q4 "$GLM_DIR/$SUBDIR" "$GLM_DIR"; do
  _s1=$(ls "$_d"/*-00001-of-*.gguf 2>/dev/null | head -1) || true
  [ -n "$_s1" ] && [ "$(ls "$_d"/GLM-5.2-UD-Q4_K_M-*-of-*.gguf 2>/dev/null | wc -l)" -ge 11 ] || continue
  PRESTAGED_SHARD1="$_s1"; break
done
if [ -n "$PRESTAGED_SHARD1" ]; then
  case "$PRESTAGED_SHARD1" in
    /mnt/*|/nvme*|/local*|/raid*|/scratch*) ph "USING_LOCAL_NVME shard1=$PRESTAGED_SHARD1 (fast local read; HF download skipped)";;
    *) ph "USING_PRESTAGED shard1=$PRESTAGED_SHARD1 (WARN: NOT local NVMe — if this is /projects NFS the load is ~53x slower; stage to /mnt/sglang_dv3/glm52-q4 for the fast path)";;
  esac
fi

# 1. download the GGUF shards (resumable; the HF CLI skips already-complete files).
if [ -z "$PRESTAGED_SHARD1" ]; then
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
[ "$DL_RC" -eq 0 ] && [ "${SHARDS:-0}" -ge 1 ] || { ph "DOWNLOAD_FAIL"; exit 20; }
SHARD1=$(ls "$GLM_DIR/$SUBDIR"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -n "$SHARD1" ] || SHARD1=$(ls "$GLM_DIR/$SUBDIR"/*.gguf 2>/dev/null | sort | head -1)
ph "SHARD1=$SHARD1"
else
  SHARD1="$PRESTAGED_SHARD1"
  ph "SHARD1=$SHARD1 (pre-staged; HF download skipped)"
fi

# 2. build llama.cpp (CUDA sm_80) if the server binary is missing.
SERVER="$LLAMA/build/bin/llama-server"
if [ ! -x "$SERVER" ]; then
  [ -d "$LLAMA" ] || { ph "CLONE_LLAMA"; git clone https://github.com/ggml-org/llama.cpp "$LLAMA" >>"$LOG" 2>&1; }
  ph "BUILD_LLAMA sm_80"
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF >>"$LOG" 2>&1
  cmake --build "$LLAMA/build" --config Release -j "$(nproc)" --target llama-server >>"$LOG" 2>&1
fi
[ -x "$SERVER" ] || { ph "BUILD_FAIL"; exit 30; }
ph "SERVER_READY $SERVER"

# 3. launch llama-server (--jinja applies the GGUF chat template; --alias pins the model id).
ph "LAUNCH ngl=$NGL n-cpu-moe=$NCPU_MOE port=$PORT (model load of ~466 GB takes minutes)"
"$SERVER" --model "$SHARD1" --alias glm-5.2 --jinja \
  --n-gpu-layers "$NGL" --n-cpu-moe "$NCPU_MOE" \
  --host 0.0.0.0 --port "$PORT" --ctx-size "$CTX" \
  > "$GLM_DIR/server.log" 2>&1 &
SRV=$!
ph "SERVER_PID=$SRV"

# 4. health-check: detect a crashed load immediately, and assert a real chat answer before
#    declaring ready (a server that loaded but cannot complete must NOT greenlight a witness).
for _ in $(seq 1 180); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -30 "$GLM_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    smoke=$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":8}')
    echo "SMOKE: $smoke" >>"$LOG"
    if printf '%s' "$smoke" | grep -q '"content"' && ! printf '%s' "$smoke" | grep -q '"error"'; then
      ph "GLM52_SERVE_READY port=$PORT"; exit 0
    fi
    ph "SMOKE_FAIL"; exit 41
  fi
  sleep 20
done
ph "HEALTH_TIMEOUT"; tail -20 "$GLM_DIR/server.log" >>"$LOG" 2>&1; exit 42
