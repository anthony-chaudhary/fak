#!/usr/bin/env bash
# glm52_sglang_vllm_serve.sh — stand up GLM-5.2 (753B, glm_moe_dsa) on a serving
# node with SGLang or vLLM, GATED by the portable readiness preflight so it fails
# CLOSED on the wrong GPU generation instead of crashing deep in a DSA kernel.
#
# RUN THIS ON THE SERVING NODE (a Hopper/Blackwell box: H200/B200/B300/GB300),
# not the laptop and not the GPU server. The preflight will refuse Ampere (sm_80).
#
# Why this exists / what it encodes (verified 2026-06-21):
#   * GLM-5.2 uses DeepSeek-Sparse-Attention (DSA). Stock SGLang/vLLM DSA kernels
#     are gated to Hopper (sm_90) / Blackwell (sm_100). A100 (sm_80) is BELOW the
#     floor — vLLM #35021 reports >=3 layers of sm_80 incompatibility. The Ada
#     (sm_89, RTX 4090) community port (renning22/glm-5.2-4090, ada_dsa.py) is the
#     only lower-arch path and is NOT Ampere and NOT stock.
#   * On a GPU server, use the llama.cpp MLA path instead: tools/glm52_serve.sh.
#   * SGLang GLM-5.2 cookbook: `sglang serve --model-path zai-org/GLM-5.2-FP8 --tp 8`
#     plus EAGLE speculative decode (MTP). FP8 fits 8x H200 (~1128 GB > ~866 GB).
#   * DURABILITY: launch under `systemd-run --unit=NAME --collect` on a host where
#     a private control bridge cgroup-kills its sessions (see glm52_serve.sh).
#
# Usage (on the serving node):
#   ENGINE=sglang bash tools/glm52_sglang_vllm_serve.sh
#   ENGINE=vllm   bash tools/glm52_sglang_vllm_serve.sh
#   export GLM52_TOOL_CALL_PARSER=replace-with-parser-name
#   ENGINE=vllm ENGINE_ARGS="--enable-auto-tool-choice --tool-call-parser ${GLM52_TOOL_CALL_PARSER}" \
#       bash tools/glm52_sglang_vllm_serve.sh
# then, once healthy, capture the #130 evidence from any box:
#   python tools/glm52_serving_witness.py --base-url http://<node>:8000/v1 \
#       --model zai-org/GLM-5.2 --engine-cache-engine "${ENGINE}"
# Scope of this witness: it proves the fak-fronts-an-external-engine (SGLang/vLLM)
# form of #130 only -- fak governs/fronts the weights the outside engine serves. It
# does NOT prove native in-kernel GLM-5.2 serving, which is the separate native track
# (docs/notes/native-753b-track-staged-plan.md; the external-vs-native evidence
# boundary is drawn in docs/serving/glm52-full-size-serving-witness.md section 6).
set -euo pipefail

ENGINE="${ENGINE:-sglang}"          # sglang | vllm
SERVED_NAME="${SERVED_NAME:-glm-5.2}"
QUANT="${QUANT:-fp8}"               # fp8 | w4afp8 | nvfp4 | int4 | bf16 (matches the checkpoint)
# Default checkpoint per quant (override with MODEL=...). w4afp8 = the Phala
# Hopper-validated checkpoint (368 GB, runs on H100/H200); fp8 = the 8x H200 default.
case "${QUANT}" in
  fp8)    DEFAULT_MODEL="zai-org/GLM-5.2-FP8" ;;
  w4afp8) DEFAULT_MODEL="PhalaCloud/GLM-5.2-W4AFP8" ;;     # AWQ INT4 experts + FP8 acts, Hopper
  nvfp4)  DEFAULT_MODEL="Mapika/GLM-5.2-NVFP4" ;;          # community NVFP4 (Blackwell only)
  bf16)   DEFAULT_MODEL="zai-org/GLM-5.2" ;;
  *)      DEFAULT_MODEL="" ;;                              # int4: no official repo (self-quant)
esac
MODEL="${MODEL:-${DEFAULT_MODEL}}"
if [ -z "${MODEL}" ]; then
  echo "no MODEL for QUANT=${QUANT} (no default checkpoint) — set MODEL=<hf-repo>"; exit 2
fi
TP="${TP:-8}"                       # tensor-parallel size = GPU count
PORT="${PORT:-8000}"
CTX="${CTX:-131072}"               # served context; GLM-5.2 supports up to 1M
PREFLIGHT="${PREFLIGHT:-1}"         # set 0 to bypass the gate (NOT recommended)
ENGINE_ARGS="${ENGINE_ARGS:-}"      # extra engine-specific flags, recorded in the shell command
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log(){ echo "[$(date +%T)] $*"; }

if [ "${ENGINE}" != "sglang" ] && [ "${ENGINE}" != "vllm" ]; then
  log "ENGINE must be 'sglang' or 'vllm' (got '${ENGINE}')"; exit 2
fi

# 1. Readiness gate. Fails closed unless this node clears the DSA arch + VRAM floor.
if [ "${PREFLIGHT}" = "1" ]; then
  log "preflight: GLM-5.2 ${ENGINE} readiness on this node (quant=${QUANT}) ..."
  if ! python "${HERE}/tools/glm52_serve_preflight.py" \
        --engine "${ENGINE}" --quant "${QUANT}" --require-ready \
        --markdown "${HERE}/glm52-${ENGINE}-preflight.md"; then
    log "PREFLIGHT BLOCKED — this node cannot serve GLM-5.2 in ${ENGINE}."
    log "See glm52-${ENGINE}-preflight.md. On a GPU server use tools/glm52_serve.sh (llama.cpp)."
    exit 1
  fi
  log "preflight OK."
fi

# 2. Quant-specific engine flags. NVFP4 is Blackwell-only (FP4 tensor cores);
#    on Hopper use INT4 (AWQ/GPTQ) or W4AFP8. Override with QUANT_FLAGS=...
#    The exact flag depends on the checkpoint — confirm against its model card.
case "${QUANT}" in
  fp8)    SG_QFLAGS="" ;;                                                   # native FP8, auto-detected
  nvfp4)  SG_QFLAGS="--quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass" ;;  # Blackwell
  int4)   SG_QFLAGS="--quantization awq_marlin" ;;                          # AWQ INT4 checkpoint (Hopper)
  w4afp8) SG_QFLAGS="--quantization w4afp8 --disable-shared-experts-fusion --kv-cache-dtype fp8_e4m3" ;;  # Phala Hopper recipe
  *)      SG_QFLAGS="" ;;
esac
SG_QFLAGS="${QUANT_FLAGS:-${SG_QFLAGS}}"

# 3. Launch the engine. SGLang >= v0.5.13.post1 registers GlmMoeDsaForCausalLM.
log "launching ${ENGINE} for ${MODEL} (tp=${TP} ctx=${CTX} quant=${QUANT}) on :${PORT} ..."
if [ "${ENGINE}" = "sglang" ]; then
  # SGLang GLM-5.2 cookbook (H200 low-latency profile): EAGLE MTP speculative decode.
  # SGLang auto-selects the DSA KV-cache dtype per arch (bf16 on Hopper, fp8 on Blackwell).
  sglang serve \
    --model-path "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --tp "${TP}" \
    --context-length "${CTX}" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --mem-fraction-static 0.85 \
    ${SG_QFLAGS} \
    ${ENGINE_ARGS} \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --host 0.0.0.0 --port "${PORT}" \
    > "${HERE}/glm52-sglang-server.log" 2>&1 &
else
  # vLLM (GlmMoeDsaForCausalLM); requires a build whose DSA kernels target this arch.
  # vLLM loads modelopt NVFP4 with --quantization modelopt (Blackwell); INT4 via the
  # checkpoint's own AWQ/GPTQ metadata. FP8 is the validated 8x H200 default.
  VL_QFLAGS=""
  [ "${QUANT}" = "nvfp4" ] && VL_QFLAGS="--quantization modelopt"
  VL_QFLAGS="${QUANT_FLAGS:-${VL_QFLAGS}}"
  vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${CTX}" \
    --trust-remote-code \
    ${VL_QFLAGS} \
    ${ENGINE_ARGS} \
    --host 0.0.0.0 --port "${PORT}" \
    > "${HERE}/glm52-vllm-server.log" 2>&1 &
fi
SRV_PID=$!
log "server pid=${SRV_PID}; waiting for /health (multi-hundred-GB load takes minutes) ..."

# 4. Health-check.
for _ in $(seq 1 180); do
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
     || curl -sf -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    log "HEALTHY. models:"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
    log "smoke completion:"
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: GLM52_OK\"}],\"max_tokens\":8,\"temperature\":0}" \
      | head -c 500; echo
    log "GLM52_${ENGINE}_READY on :${PORT} — now run tools/glm52_serving_witness.py against it."
    exit 0
  fi
  sleep 10
done
log "server did not become healthy in time; see ${HERE}/glm52-${ENGINE}-server.log"
tail -20 "${HERE}/glm52-${ENGINE}-server.log" || true
exit 1
