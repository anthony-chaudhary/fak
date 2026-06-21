#!/usr/bin/env bash
# glm52_serve.sh — stand up GLM-5.2 (Q4_K_M GGUF) on an 8x A100-40GB host
# via llama.cpp with CPU expert-offload, then health-check it for the #130 serving
# witness. RUN THIS ON THE GPU HOST, not a laptop.
#
# Why this exists / what it encodes (verified 2026-06-21):
#   * Stock SGLang/vLLM CANNOT serve GLM-5.2 on A100 (sm_80): its DeepSeek-Sparse-
#     Attention kernels hard-depend on Hopper/Blackwell DeepGemm (vLLM #35021).
#   * llama.cpp CAN: it supports glm_moe_dsa GGUF and runs it as full MLA attention
#     (DSA indexer runtime still WIP upstream) — MLA needs no sm_90 kernel, so it
#     runs on Ampere/CPU. Quality is slightly suboptimal vs true sparse DSA.
#   * Memory: Q4_K_M ~454GB. The 320GB VRAM holds ~70%; the rest offloads to the
#     ~728GB host RAM via --n-cpu-moe (experts on CPU, attention/shared on the GPUs).
#   * DURABILITY: launch long jobs with `systemd-run --scope` — many session/shell
#     managers cgroup-kill their child processes on disconnect, and `setsid`/`nohup`
#     do NOT escape a cgroup (this killed the build 3x). Run THIS script under
#     systemd-run too.
#
# Usage (on the GPU host):
#   systemd-run --scope --unit=glm52serve bash tools/glm52_serve.sh
# then from the laptop, once it reports healthy:
#   fak serve --base-url http://<dgx>:8000/v1 ...   # front it
#   python tools/glm52_serving_witness.py --base-url http://127.0.0.1:8000/v1   # #130 evidence
set -euo pipefail

GLM_DIR="${GLM_DIR:-/mnt/glm/glm52-q4}"
LLAMA="${LLAMA:-/mnt/glm/llama.cpp}"
SHARD1="${GLM_DIR}/GLM-5.2-Q4_K_M-00001-of-00008.gguf"
PORT="${PORT:-8000}"
NGL="${NGL:-99}"            # offload all layers' non-expert tensors to GPU
NCPU_MOE="${NCPU_MOE:-40}"  # keep this many layers' MoE experts on CPU RAM; lower if VRAM spare, raise on OOM
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}"

log(){ echo "[$(date +%T)] $*"; }

# 1. Wait for all 8 shards to finish downloading (aria2 pre-allocates, so check the
#    .aria2 control files are gone — that is aria2's 'complete' signal).
log "waiting for 8 GGUF shards in ${GLM_DIR} ..."
while :; do
  have=$(ls "${GLM_DIR}"/GLM-5.2-Q4_K_M-*-of-00008.gguf 2>/dev/null | wc -l)
  pending=$(ls "${GLM_DIR}"/*.aria2 2>/dev/null | wc -l)
  log "  shards=${have}/8 in-progress(.aria2)=${pending}"
  [ "${have}" -eq 8 ] && [ "${pending}" -eq 0 ] && break
  sleep 30
done
log "all shards present."

# 2. Build llama.cpp (CUDA sm_80) if the server binary is missing.
SERVER="${LLAMA}/build/bin/llama-server"
if [ ! -x "${SERVER}" ]; then
  log "building llama.cpp (CUDA sm_80) ..."
  cmake -S "${LLAMA}" -B "${LLAMA}/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF
  cmake --build "${LLAMA}/build" --config Release -j "$(nproc)" --target llama-server
fi
log "server binary: ${SERVER}"

# 3. Launch the server. llama.cpp loads a sharded GGUF from shard 1.
log "launching llama-server on :${PORT} (ngl=${NGL} n-cpu-moe=${NCPU_MOE}) ..."
"${SERVER}" \
  --model "${SHARD1}" \
  --n-gpu-layers "${NGL}" \
  --n-cpu-moe "${NCPU_MOE}" \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 8192 \
  > "${GLM_DIR}/server.log" 2>&1 &
SRV_PID=$!
log "server pid=${SRV_PID}; tailing until /health is ok (model load of ~454GB takes minutes) ..."

# 4. Health-check.
for _ in $(seq 1 120); do
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    log "HEALTHY. models:"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
    log "smoke completion:"
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":8}' \
      | head -c 500; echo
    log "GLM52_SERVE_READY on :${PORT} — now run the #130 witness against it."
    exit 0
  fi
  sleep 10
done
log "server did not become healthy in time; see ${GLM_DIR}/server.log"
tail -20 "${GLM_DIR}/server.log" || true
exit 1
