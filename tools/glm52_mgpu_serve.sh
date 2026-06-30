#!/usr/bin/env bash
# glm52_mgpu_serve.sh - stand GLM-5.2 UD-Q4_K_M up FULLY GPU-RESIDENT across a
# multi-GPU sm_80 box (e.g. an 8x 80GB datacenter GPU server). The PERFORMANT
# counterpart to the cpu-offload staging sibling in tools/ (the --n-cpu-moe path).
#
# WHY THIS EXISTS (the "use the full GPUs and RAM" overcome):
#   The cpu-offload staging sibling pins ALL MoE experts to host RAM
#   (--n-cpu-moe 999) so the 753B/~466GB checkpoint serves at all on a single
#   GPU's worth of VRAM. That path is host-expert-GEMM-bound: only 1 of 8 GPUs
#   does work and decode crawls (~0.23 tok/s steady-state — slower than pure-CPU
#   llama.cpp; the #971 wall). But UD-Q4_K_M is ~434 GiB and 7x80GB = 560 GiB of
#   VRAM, so the WHOLE model fits resident across the GPUs. This script does
#   exactly that: it OMITS --n-cpu-moe, so -ngl 999 keeps every expert on GPU HBM,
#   layer-split across the visible devices.
#
# WITNESSED (8-GPU server, 2026-06-29, GPUs 1-7, llama.cpp build 9801, NVMe-staged):
#   decode 23.4 tok/s steady-state (~100x the cpu-offload 0.2324 tok/s; ~26x the
#   0.89 tok/s pure-CPU llama.cpp baseline), prefill ~54 tok/s, ~433 GiB resident
#   across the 7 GPUs. See experiments/glm-gpu-witness/mgpu-glm52-fullgpu-serve-witness-2026-06-29.json.
#
# By DEFAULT it serves on GPUs 1-7 and leaves GPU0 for a peer serve on a shared box;
# set DEVICES=0,1,2,3,4,5,6,7 to use all 8 on a dedicated box.
#
# Usage (RUN ON THE GPU HOST, detached so a disconnect does not orphan the load):
#   systemd-run --user --unit=glm52mgpu --collect bash tools/glm52_mgpu_serve.sh
#   # or: setsid bash tools/glm52_mgpu_serve.sh >/tmp/glm52_mgpu/wrap.log 2>&1 &
# then poll:  cat /tmp/glm52_mgpu/PHASE   (LAUNCH -> ... -> GLM52_MGPU_READY)
set -uo pipefail

SERVER="${LLAMA_SERVER:-/projects/llama.cpp/build/bin/llama-server}"
DEVICES="${DEVICES:-1,2,3,4,5,6,7}"        # GPU0 left for a peer serve by default
PORT="${PORT:-8002}"
CTX="${CTX:-8192}"
ALIAS="${ALIAS:-glm-5.2}"
RUN="${RUN:-/tmp/glm52_mgpu}"
PHASE="$RUN/PHASE"
SRVLOG="$RUN/server.log"
mkdir -p "$RUN"
ph(){ echo "$(date -u +%H:%M:%S) $*"; echo "$*" > "$PHASE"; }

# Resolve shard 1 NVMe-first (a local NVMe read is ~50x the /projects NFS; a silent
# fall-through to NFS is the #1 cause of a multi-hour cold load).
SHARD1="${SHARD1:-}"
if [ -z "$SHARD1" ]; then
  for d in /mnt/sglang_dv3/glm52-q4 /mnt/nvme-glm/glm52-q4/UD-Q4_K_M /projects/glm52-q4/UD-Q4_K_M; do
    s1=$(ls "$d"/GLM-5.2-UD-Q4_K_M-*-00001-of-*.gguf 2>/dev/null | head -1) || true
    [ -n "$s1" ] && [ "$(ls "$d"/GLM-5.2-UD-Q4_K_M-*-of-*.gguf 2>/dev/null | wc -l)" -ge 11 ] || continue
    SHARD1="$s1"; break
  done
fi

[ -x "$SERVER" ] || { ph "NO_SERVER $SERVER (build llama.cpp CUDA sm_80 first — see the cpu-offload staging sibling)"; exit 10; }
[ -n "$SHARD1" ] && [ -f "$SHARD1" ] || { ph "NO_SHARD (no 11-shard UD-Q4_K_M set found; stage it first)"; exit 11; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then ph "PORT_BUSY $PORT (already serving — not relaunching)"; exit 0; fi

ph "PREFLIGHT devices=$DEVICES free VRAM (MiB):"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader -i "$DEVICES" 2>/dev/null || true

# The one line that matters: NO --n-cpu-moe => experts stay on GPU. -ngl 999 offloads
# every layer; default split-mode=layer spreads them across the visible devices.
ph "LAUNCH ngl=999 (experts ON gpu) devices=$DEVICES port=$PORT ctx=$CTX model=$SHARD1"
CUDA_VISIBLE_DEVICES="$DEVICES" setsid "$SERVER" \
  --model "$SHARD1" --alias "$ALIAS" --jinja \
  --n-gpu-layers 999 \
  --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX" \
  > "$SRVLOG" 2>&1 < /dev/null &
SRV=$!
ph "SERVER_PID=$SRV (resident load of ~434 GiB from NVMe -> GPUs takes a few min)"

for i in $(seq 1 48); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -25 "$SRVLOG"; exit 40; fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    smoke=$(curl -s -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ok\"}],\"max_tokens\":8}")
    if printf '%s' "$smoke" | grep -q '"content"' && ! printf '%s' "$smoke" | grep -q '"error"'; then
      ph "GLM52_MGPU_READY port=$PORT"
      echo "== resident VRAM across the serve's GPUs (proof experts are on HBM) =="
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$DEVICES"
      exit 0
    fi
    ph "SMOKE_FAIL"; echo "$smoke"; exit 41
  fi
  [ $((i % 4)) -eq 0 ] && tail -2 "$SRVLOG" 2>/dev/null
  sleep 15
done
ph "HEALTH_TIMEOUT"; tail -25 "$SRVLOG"; exit 42
