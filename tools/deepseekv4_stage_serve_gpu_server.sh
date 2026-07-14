#!/usr/bin/env bash
# deepseekv4_stage_serve_gpu_server.sh - durable stage+build+serve of DeepSeek-V4-Flash on an
# A100 (sm_80) node via a V4-aware llama.cpp fork. The DeepSeek-V4 sibling of
# tools/glm52_stage_serve_gpu_server.sh, and the sm_80 counterpart to scripts/dgx-deepseek-serve.sh
# (which fronts a *stock* vLLM/SGLang engine and correctly REFUSES below sm_90).
#
# WHY THIS EXISTS (the A100 "V4 not supported" overcome):
#   Stock SGLang/vLLM CANNOT serve DeepSeek-V4 on A100 (sm_80): V4's checkpoint is FP4
#   (MoE experts) + FP8 + an FP4 attention indexer, and the attention stack is DSA
#   (CSA/HCA + lightning indexer). Those stock kernels need sm_90 (Hopper) / sm_100
#   (Blackwell). scripts/dgx-deepseek-serve.sh encodes that arch-floor gate and refuses
#   below sm_90. This is the SAME wall that blocks GLM-5.2 on A100 (vLLM #35021).
#
#   A V4-aware llama.cpp fork CAN serve it on sm_80: it implements V4's five custom ops
#   (compressor decode, hyperconnection, lightning indexer, FP8-KV simulation, NextN
#   heads) with CUDA kernels, and on sm_80 falls into the *software-emulated* FP8 path
#   (correct, just not HW-accelerated) - the same "llama.cpp overcomes the sm_90 kernel
#   floor" move already proven for GLM-5.2 on GPU server.
#
#   MATCHED PAIRS (fork <-> GGUF must agree; they are different code paths):
#     * teamblobfish GGUF  <-> cchuter fork  feat/v4-port-cuda  (5 V4 ops, CUDA-validated)
#     * unsloth GGUF       <-> upstream llama.cpp with DeepSeek-V4 support (PR #24162)
#   Default is the teamblobfish+cchuter pair (self-contained V4 ops). Override with the
#   env knobs below to use the unsloth+upstream pair once #24162 has merged.
#
# CAPACITY - the key difference from GLM-5.2 on this box:
#   DeepSeek-V4-Flash is 284B total / 13B active. Its GGUF quants (UD-IQ3_XXS ~103 GB,
#   UD-Q4_K_XL ~142 GB, UD-Q8_K_XL ~162 GB) all FIT RESIDENT across 8x A100-80GB
#   (640 GB VRAM). So unlike GLM-5.2's 466 GB (which forces MoE experts onto host RAM and
#   a <0.1 tok/s decode floor), V4-Flash runs experts GPU-RESIDENT: -n-gpu-layers 999 and
#   NO -n-cpu-moe by default. Set DSV4_NCPU_MOE>0 only for the 162 GB Q8 if you want VRAM
#   headroom for a very long context.
#
# HONEST SCOPE: this script asserts NO throughput/quality number. It stands the endpoint
# up, health-checks it, and records a three-rung WIRE witness (models roster / non-stream
# completion with usage / streamed completion to [DONE]) - the same rungs as the self-host
# runbook (docs/benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md), captured on-box via
# curl because GPU server has no Go toolchain for the Go smoke. The served quant is recorded
# in-band so nothing is over-claimed. Any perf headline is deferred to a tuned baseline.
#
# Usage (RUN ON THE GPU HOST, detached so a disconnect does not orphan a ~100+ GB load):
#   tmux new-session -d -s dsv4 'bash tools/deepseekv4_stage_serve_gpu_server.sh > /tmp/fakgpu/dsv4.log 2>&1'
# then poll:  cat "$DSV4_DIR/PHASE"   and on DSV4_SERVE_READY read "$DSV4_DIR/WITNESS".
set -uo pipefail

DSV4_DIR="${DSV4_DIR:-/projects/deepseek-v4-flash}"
LLAMA="${LLAMA:-/projects/llama.cpp-v4}"
# Matched pair (default: teamblobfish GGUF + cchuter V4 fork). Override for unsloth+upstream.
REPO="${DSV4_REPO:-teamblobfish/DeepSeek-V4-Flash-GGUF}"
SUBDIR="${DSV4_SUBDIR:-Q4_K_M-XL}"
LLAMA_GIT="${DSV4_LLAMA_GIT:-https://github.com/cchuter/llama.cpp}"
LLAMA_BRANCH="${DSV4_LLAMA_BRANCH:-feat/v4-port-cuda}"
PORT="${PORT:-8100}"                 # 8100, NOT 8000/8001 - those belong to other lab users
NGL="${NGL:-999}"
# 0 = experts GPU-resident (default; V4-Flash fits on 640 GB). >0 offloads that many MoE
# layers to host RAM (only needed for the 162 GB Q8 with a very long context).
NCPU_MOE="${DSV4_NCPU_MOE:-0}"
CTX="${CTX:-16384}"
CUDA_ARCH="${FAK_CUDA_ARCH:-80}"     # A100 = Ampere = sm_80
ALIAS="${DSV4_ALIAS:-deepseek-v4-flash}"
PHASE="$DSV4_DIR/PHASE"
LOG="$DSV4_DIR/stage_serve.log"
WITNESS="$DSV4_DIR/WITNESS"

# CUDA toolkit bin on PATH for the sm_80 build. GPU server ships CUDA 12.8; prefer CUDA_BIN, then
# the generic symlink, then known versions, so the same script builds on the DGX A100 and a
# GCP A100 node with no edit.
for _cuda in "${CUDA_BIN:-}" /usr/local/cuda/bin /usr/local/cuda-12.9/bin /usr/local/cuda-12.8/bin; do
  if [ -n "$_cuda" ] && [ -d "$_cuda" ]; then export PATH="$_cuda:$PATH"; break; fi
done

mkdir -p "$DSV4_DIR"
ph(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; echo "$*" > "$PHASE"; }

# PREFLIGHT (good-neighbor gate on a SHARED box) - run ON the box so a caller who cannot
# read an interactive recon back over a flaky control channel still gets a safe abort with a
# machine-pollable PHASE verdict. Aborts BEFORE any download/build if headroom is short.
#   * free VRAM must fit the resident model + KV (default need 200 GB across the fleet)
#   * free disk at DSV4_DIR must fit the GGUF + build (default 180 GB)
#   * HF must be reachable IF nothing is pre-staged (else the download can't run)
# Override the thresholds with DSV4_MIN_FREE_VRAM_MB / DSV4_MIN_FREE_DISK_G / DSV4_MIN_FREE_RAM_G.
MIN_FREE_VRAM_MB="${DSV4_MIN_FREE_VRAM_MB:-200000}"
MIN_FREE_DISK_G="${DSV4_MIN_FREE_DISK_G:-180}"
MIN_FREE_RAM_G="${DSV4_MIN_FREE_RAM_G:-16}"
preflight(){
  local fv fr fd hf prestaged=0
  ls "$DSV4_DIR/$SUBDIR"/*-00001-of-*.gguf /mnt/*/deepseek-v4-flash/*-00001-of-*.gguf >/dev/null 2>&1 && prestaged=1
  fv=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  fr=$(free -g 2>/dev/null | awk '/Mem:/{print $7}')
  fd=$(df -PBG "$DSV4_DIR" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4);print $4}')
  ph "PREFLIGHT free_vram_mb=${fv:-0} free_ram_g=${fr:-0} free_disk_g=${fd:-0} prestaged=$prestaged"
  if [ "${fv:-0}" -lt "$MIN_FREE_VRAM_MB" ]; then ph "PREFLIGHT_FAIL_VRAM have=${fv}MB need=${MIN_FREE_VRAM_MB}MB (a neighbor likely holds the GPUs - do not evict; retry later or lower DSV4_MIN_FREE_VRAM_MB for a smaller quant)"; return 1; fi
  if [ -n "$fd" ] && [ "$fd" -lt "$MIN_FREE_DISK_G" ]; then ph "PREFLIGHT_FAIL_DISK have=${fd}G need=${MIN_FREE_DISK_G}G at $DSV4_DIR"; return 1; fi
  if [ -n "$fr" ] && [ "$fr" -lt "$MIN_FREE_RAM_G" ]; then ph "PREFLIGHT_FAIL_RAM have=${fr}G need=${MIN_FREE_RAM_G}G"; return 1; fi
  if [ "$prestaged" -eq 0 ]; then
    hf=$(timeout 20 curl -s -o /dev/null -w '%{http_code}' https://huggingface.co 2>/dev/null || echo 000)
    [ "$hf" = 200 ] || [ "$hf" = 301 ] || [ "$hf" = 302 ] || { ph "PREFLIGHT_FAIL_HF status=$hf (HF unreachable and nothing pre-staged - operator must pre-stage the GGUF to $DSV4_DIR/$SUBDIR)"; return 1; }
  fi
  ph "PREFLIGHT_OK"; return 0
}
if [ "${DSV4_SKIP_PREFLIGHT:-0}" != 1 ]; then preflight || { ph "ABORTED_PREFLIGHT"; exit 3; }; fi

# 0. prefer a COMPLETE local-NVMe copy over slow /projects NFS (same ~53x lesson as GLM-5.2).
PRESTAGED_SHARD1=""
for _d in /mnt/*/deepseek-v4-flash "$DSV4_DIR/$SUBDIR" "$DSV4_DIR"; do
  _s1=$(ls "$_d"/*-00001-of-*.gguf 2>/dev/null | head -1) || true
  [ -n "$_s1" ] || continue
  PRESTAGED_SHARD1="$_s1"; break
done
if [ -n "$PRESTAGED_SHARD1" ]; then
  case "$PRESTAGED_SHARD1" in
    /mnt/*|/nvme*|/local*|/raid*|/scratch*) ph "USING_LOCAL_NVME shard1=$PRESTAGED_SHARD1 (fast local read; HF download skipped)";;
    *) ph "USING_PRESTAGED shard1=$PRESTAGED_SHARD1 (WARN: if this is /projects NFS the load is far slower; stage to local NVMe for the fast path)";;
  esac
fi

# 1. download the GGUF shards (resumable; the HF CLI skips already-complete files).
if [ -z "$PRESTAGED_SHARD1" ]; then
  ph "DOWNLOAD_START repo=$REPO subdir=$SUBDIR dir=$DSV4_DIR"
  if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" --include "$SUBDIR/*" --local-dir "$DSV4_DIR" >>"$LOG" 2>&1; DL_RC=$?
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" --include "$SUBDIR/*" --local-dir "$DSV4_DIR" >>"$LOG" 2>&1; DL_RC=$?
  else
    ph "NO_HF_CLI install huggingface_hub first"; exit 10
  fi
  SHARDS=$(ls "$DSV4_DIR/$SUBDIR"/*.gguf 2>/dev/null | wc -l)
  ph "DOWNLOAD_DONE rc=$DL_RC shards=$SHARDS"
  [ "$DL_RC" -eq 0 ] && [ "${SHARDS:-0}" -ge 1 ] || { ph "DOWNLOAD_FAIL"; exit 20; }
  SHARD1=$(ls "$DSV4_DIR/$SUBDIR"/*-00001-of-*.gguf 2>/dev/null | head -1)
  [ -n "$SHARD1" ] || SHARD1=$(ls "$DSV4_DIR/$SUBDIR"/*.gguf 2>/dev/null | sort | head -1)
  ph "SHARD1=$SHARD1"
else
  SHARD1="$PRESTAGED_SHARD1"
  ph "SHARD1=$SHARD1 (pre-staged; HF download skipped)"
fi

# 2. build the V4-aware llama.cpp (CUDA sm_80) if the server binary is missing.
#    V4's dense per-layer inputs exceed llama.cpp's default scheduler split cap (30) at
#    multi-GPU split boundaries, so GGML_SCHED_MAX_SPLIT_INPUTS=128 is REQUIRED for an
#    8-way split (cost ~200 MB extra scheduler RAM).
SERVER="$LLAMA/build/bin/llama-server"
if [ ! -x "$SERVER" ]; then
  if [ ! -d "$LLAMA" ]; then
    ph "CLONE_LLAMA $LLAMA_GIT@$LLAMA_BRANCH"
    git clone -b "$LLAMA_BRANCH" "$LLAMA_GIT" "$LLAMA" >>"$LOG" 2>&1 || { ph "CLONE_FAIL"; exit 25; }
  fi
  ph "BUILD_LLAMA sm_$CUDA_ARCH (GGML_SCHED_MAX_SPLIT_INPUTS=128)"
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLLAMA_CURL=OFF \
    -DCMAKE_CXX_FLAGS=-DGGML_SCHED_MAX_SPLIT_INPUTS=128 \
    -DCMAKE_CUDA_FLAGS=-DGGML_SCHED_MAX_SPLIT_INPUTS=128 >>"$LOG" 2>&1
  cmake --build "$LLAMA/build" --config Release -j "$(nproc)" --target llama-server >>"$LOG" 2>&1
fi
[ -x "$SERVER" ] || { ph "BUILD_FAIL"; exit 30; }
ph "SERVER_READY $SERVER"

# 3. launch llama-server. V4 flags: --jinja applies the GGUF chat template; --no-repack is
#    required for V4 quants; --flash-attn on; deterministic sampling for the smoke. NOTE the
#    V4 KV quirk: --cache-type-k|v q8_0 is silently overridden to f16 (V4 writes K as FP8),
#    so we do NOT request a quantized KV cache here.
MOE_FLAG=(); [ "${NCPU_MOE:-0}" -gt 0 ] && MOE_FLAG=(--n-cpu-moe "$NCPU_MOE")
ph "LAUNCH ngl=$NGL n-cpu-moe=${NCPU_MOE} port=$PORT ctx=$CTX (resident load of ~100-160 GB takes minutes)"
"$SERVER" --model "$SHARD1" --alias "$ALIAS" --jinja --no-repack --flash-attn on \
  --n-gpu-layers "$NGL" "${MOE_FLAG[@]}" \
  --host 0.0.0.0 --port "$PORT" --ctx-size "$CTX" \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  > "$DSV4_DIR/server.log" 2>&1 &
SRV=$!
ph "SERVER_PID=$SRV"

# 4. health-check + three-rung wire witness (models / non-streaming+usage / streaming [DONE]).
BASE="http://127.0.0.1:$PORT"
for _ in $(seq 1 240); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -40 "$DSV4_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
  if curl -sf -m 5 "$BASE/health" >/dev/null 2>&1; then
    : > "$WITNESS"
    echo "# DeepSeek-V4-Flash on-box wire witness ($(date -u))" >>"$WITNESS"
    echo "served_quant_shard1=$SHARD1" >>"$WITNESS"
    echo "repo=$REPO subdir=$SUBDIR llama=$LLAMA_GIT@$LLAMA_BRANCH sm=$CUDA_ARCH" >>"$WITNESS"
    # rung 1: models roster
    models=$(curl -s -m 15 "$BASE/v1/models")
    echo "RUNG_models: $models" >>"$WITNESS"
    # rung 2: non-streaming completion (expect content + a usage block)
    nonstream=$(curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":16,\"stream\":false}")
    echo "RUNG_nonstream: $nonstream" >>"$WITNESS"
    # rung 3: streaming completion (expect SSE deltas terminated by [DONE])
    stream=$(curl -s -m 120 -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Count: one two three\"}],\"max_tokens\":24,\"stream\":true}")
    echo "RUNG_stream_tail: $(printf '%s' "$stream" | tail -5)" >>"$WITNESS"
    ok_ns=0; printf '%s' "$nonstream" | grep -q '"content"' && ! printf '%s' "$nonstream" | grep -q '"error"' && ok_ns=1
    ok_st=0; printf '%s' "$stream" | grep -q '\[DONE\]' && ok_st=1
    echo "VERDICT ok_nonstream=$ok_ns ok_stream=$ok_st" >>"$WITNESS"
    if [ "$ok_ns" = 1 ] && [ "$ok_st" = 1 ]; then
      ph "DSV4_SERVE_READY port=$PORT (wire witness PASS - see $WITNESS)"; exit 0
    fi
    ph "SMOKE_FAIL (see $WITNESS)"; exit 41
  fi
  sleep 20
done
ph "HEALTH_TIMEOUT"; tail -20 "$DSV4_DIR/server.log" >>"$LOG" 2>&1; exit 42
