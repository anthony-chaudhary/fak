#!/usr/bin/env bash
# glm52_serve.sh — stand up GLM-5.2 (Q4_K_M GGUF) on an 8-GPU server-40GB host
# via llama.cpp with CPU expert-offload, then health-check it for the #130 serving
# witness. RUN THIS ON THE GPU HOST, not a laptop.
#
# HARDWARE FORK: this is the A100/Ampere (sm_80) llama.cpp MLA path. On Hopper or
# Blackwell (H100/H200/B200/B300/GB300, sm_90+) do NOT use this — use
# tools/glm52_sglang_vllm_serve.sh (stock SGLang/vLLM DSA), which gates on
# tools/glm52_serve_preflight.py (it returns BLOCKED_ARCH below sm_90). The
# preflight's arch verdict is the authority for which script to run.
#
# Why this exists / what it encodes (verified 2026-06-21):
#   * Stock SGLang/vLLM CANNOT serve GLM-5.2 on A100 (sm_80): its DeepSeek-Sparse-
#     Attention kernels hard-depend on Hopper/Blackwell DeepGemm (vLLM #35021).
#   * llama.cpp CAN: it supports glm_moe_dsa GGUF and runs it as full MLA attention
#     (DSA indexer runtime still WIP upstream) — MLA needs no sm_90 kernel, so it
#     runs on Ampere/CPU. Quality is slightly suboptimal vs true sparse DSA.
#   * Memory: Q4_K_M ~454GB. The 320GB VRAM holds ~70%; the rest offloads to the
#     ~728GB host RAM via --n-cpu-moe (experts on CPU, attention/shared on the GPUs).
#   * DURABILITY: a bare `systemd-run --scope` is a transient cgroup with NO restart
#     policy — once this script exits, the backgrounded server is an orphan and a
#     crash/OOM/reboot is permanent until a full ~454GB reload. For a server you
#     want to survive, prefer `systemd-run --unit=glm52serve --collect` (the sibling
#     sglang/vllm script's pattern) or a real .service unit with Restart=on-failure.
#     Either way wrap THIS script: many session/shell managers cgroup-kill their
#     children on disconnect, and setsid/nohup do NOT escape a cgroup (killed the
#     build 3x).
#
# Prerequisites (this script does NOT auto-fetch — stage these first, or it fails
# fast with the exact command):
#   * the 8 Q4_K_M GGUF shards in $GLM_DIR (set $GLM_REPO to your HF source repo)
#   * a glm_moe_dsa/MLA-capable llama.cpp checkout at $LLAMA (auto-cloned if absent)
#
# Usage (on the GPU host):
#   GLM_REPO=<hf-org/glm-5.2-q4-gguf> systemd-run --unit=glm52serve --collect \
#       bash tools/glm52_serve.sh
# then from the laptop, once it reports healthy:
#   fak serve --base-url http://<dgx>:8000/v1 ...   # front it
#   python tools/glm52_serving_witness.py --base-url http://127.0.0.1:8000/v1 --model glm-5.2   # #130 evidence
# Scope of this witness: it proves the fak-fronts-an-external-engine (llama.cpp)
# form of #130 only -- fak governs/fronts the weights the outside engine serves. It
# does NOT prove native in-kernel GLM-5.2 serving, which is the separate native track
# (docs/notes/native-753b-track-staged-plan.md; the external-vs-native evidence
# boundary is drawn in docs/serving/glm52-full-size-serving-witness.md section 6).
set -euo pipefail

GLM_DIR="${GLM_DIR:-/mnt/glm/glm52-q4}"
LLAMA="${LLAMA:-/mnt/glm/llama.cpp}"
LLAMA_REF="${LLAMA_REF:-}"   # optional: pin a glm_moe_dsa-capable llama.cpp commit/tag
GLM_REPO="${GLM_REPO:-}"     # HF repo holding the Q4_K_M shards (for the fetch hint)
SHARD_GLOB="GLM-5.2-Q4_K_M-*-of-00008.gguf"
SHARD1="${GLM_DIR}/GLM-5.2-Q4_K_M-00001-of-00008.gguf"
PORT="${PORT:-8000}"
NGL="${NGL:-99}"            # offload all layers' non-expert tensors to GPU
# Keep this many layers' MoE experts on CPU RAM. Q4_K_M (~454GB) cannot fit 320GB
# VRAM, so a large fraction of experts MUST stay on the ~728GB host. This default
# errs HIGH (more on CPU = safer against VRAM OOM); LOWER it if VRAM is spare. There
# is no per-node VRAM math here — see glm52_serve_preflight.py for the footprint gate.
NCPU_MOE="${NCPU_MOE:-40}"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda-12.8/bin}"
[ -d "${CUDA_BIN}" ] && export PATH="${CUDA_BIN}:${PATH}" || echo "WARN: ${CUDA_BIN} missing; set CUDA_BIN before building"
export HOME="${HOME:-/root}"

log(){ echo "[$(date +%T)] $*"; }

# 1. Ensure the 8 shards are present. This script does NOT download them; it waits
#    (bounded) for an out-of-band fetch, then fails fast with the exact command so a
#    fresh node never hangs forever at shards=0/8. The .aria2 control files gone is
#    aria2's 'complete' signal.
mkdir -p "${GLM_DIR}"
SHARD_WAIT_TRIES="${SHARD_WAIT_TRIES:-40}"   # x30s ≈ 20 min; set 0 to require shards already staged
fetch_hint(){
  if [ -n "${GLM_REPO}" ]; then
    echo "  huggingface-cli download \"${GLM_REPO}\" --include '${SHARD_GLOB}' --local-dir \"${GLM_DIR}\""
  else
    echo "  set GLM_REPO=<hf-org/glm-5.2-q4-gguf> and run:"
    echo "  huggingface-cli download \"\$GLM_REPO\" --include '${SHARD_GLOB}' --local-dir \"${GLM_DIR}\""
  fi
}
log "checking for 8 GGUF shards in ${GLM_DIR} ..."
tries=0
while :; do
  # `|| true`: a non-matching glob makes ls exit 2, which pipefail+set -e would
  # otherwise treat as fatal — defeating the very wait this loop performs.
  have=$(ls "${GLM_DIR}"/${SHARD_GLOB} 2>/dev/null | wc -l || true)
  pending=$(ls "${GLM_DIR}"/*.aria2 2>/dev/null | wc -l || true)
  [ "${have}" -eq 8 ] && [ "${pending}" -eq 0 ] && break
  if [ "${tries}" -ge "${SHARD_WAIT_TRIES}" ]; then
    log "shards not ready (have=${have}/8 in-progress=${pending}) after ${SHARD_WAIT_TRIES} tries."
    log "download them first, e.g.:"
    fetch_hint
    exit 1
  fi
  log "  shards=${have}/8 in-progress(.aria2)=${pending} — waiting (try $((tries+1))/${SHARD_WAIT_TRIES})"
  tries=$((tries+1))
  sleep 30
done
log "all 8 shards present."

# 2. Build llama.cpp (CUDA sm_80) if the server binary is missing. The source tree
#    is NOT auto-staged, so clone it first (a missing dir would otherwise crash
#    cmake -S with an opaque error). NOTE: glm_moe_dsa MLA support is recent and the
#    DSA indexer is WIP upstream — pin a known-good commit via LLAMA_REF if HEAD
#    regresses.
SERVER="${LLAMA}/build/bin/llama-server"
if [ ! -x "${SERVER}" ]; then
  if [ ! -d "${LLAMA}" ]; then
    log "cloning llama.cpp into ${LLAMA} ..."
    git clone https://github.com/ggml-org/llama.cpp "${LLAMA}"
  fi
  if [ -n "${LLAMA_REF}" ]; then
    log "checking out llama.cpp ${LLAMA_REF} ..."
    git -C "${LLAMA}" checkout "${LLAMA_REF}"
  fi
  log "building llama.cpp (CUDA sm_80) ..."
  cmake -S "${LLAMA}" -B "${LLAMA}/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF
  cmake --build "${LLAMA}/build" --config Release -j "$(nproc)" --target llama-server
fi
log "server binary: ${SERVER}"

# 3. Launch the server. llama.cpp loads a sharded GGUF from shard 1.
#    --jinja applies the GGUF's embedded chat template on /v1/chat/completions
#    (without it llama-server mis-formats GLM's turn/reasoning tokens or errors).
#    --alias fixes the served model id so /v1/models and the witness agree on it.
log "launching llama-server on :${PORT} (ngl=${NGL} n-cpu-moe=${NCPU_MOE}) ..."
"${SERVER}" \
  --model "${SHARD1}" \
  --alias glm-5.2 \
  --jinja \
  --n-gpu-layers "${NGL}" \
  --n-cpu-moe "${NCPU_MOE}" \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 8192 \
  > "${GLM_DIR}/server.log" 2>&1 &
SRV_PID=$!
log "server pid=${SRV_PID}; tailing until /health is ok (model load of ~454GB takes minutes) ..."

# 4. Health-check. Detect a CRASHED server immediately (an OOM mid-load otherwise
#    looks identical to 'still loading' for the whole timeout), and ASSERT the smoke
#    completion actually answers — a server that loaded but can't complete a chat
#    turn must NOT report READY and greenlight a false-positive #130 witness.
for _ in $(seq 1 120); do
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    log "server process ${SRV_PID} exited before becoming healthy; tail of log:"
    tail -40 "${GLM_DIR}/server.log" || true
    exit 1
  fi
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    log "HEALTHY. models:"
    # `|| true`: head closing the pipe early can SIGPIPE curl (141); pipefail+set -e
    # would then abort right before we report READY on an otherwise-good server.
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400 || true; echo
    log "smoke completion:"
    smoke=$(curl -s -m 30 "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":8}')
    printf '%s\n' "${smoke}" | head -c 500; echo
    if printf '%s' "${smoke}" | grep -q '"error"' || ! printf '%s' "${smoke}" | grep -q '"content"'; then
      log "smoke completion did not return a usable chat answer — NOT ready (check --jinja / template)."
      exit 1
    fi
    log "GLM52_SERVE_READY on :${PORT} — now run the #130 witness against it."
    exit 0
  fi
  sleep 10
done
log "server did not become healthy in time; see ${GLM_DIR}/server.log"
tail -20 "${GLM_DIR}/server.log" || true
exit 1
