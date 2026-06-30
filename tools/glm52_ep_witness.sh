#!/usr/bin/env bash
# glm52_ep_witness.sh — the resident-expert (expert-parallel) frontier witness for GLM-5.2
# (glm_moe_dsa, UD-Q4_K_M) on a multi-GPU datacenter node. It is the resident-path counterpart
# of tools/glm52_load_witness.sh: where that one keeps the experts host-offloaded
# (--cpu-offload-experts, the #971 wall) on ONE GPU, this one builds the -tags cuda,nccl binary
# and serves with --expert-parallel N so the routed-expert GEMMs go RESIDENT across N GPUs (no
# host offload), then times the load and decodes a turn — the witness that the experts moved off
# the host onto resident GPUs (the #971 escape). It writes a compact RESULT + a .done(rc) sentinel
# so it can be driven detached over a flaky control bridge:
#   setsid bash tools/glm52_ep_witness.sh </dev/null >boot.log 2>&1 &
#
# Requires: a CUDA toolchain, NCCL (libnccl.so on the loader path or NCCL_HOME), and >=N visible
# GPUs whose aggregate VRAM holds the sharded experts + per-rank replicated dense/attention + KV.
# The serve's own refuseEPPlanIfUnfit pre-check refuses an N that does not fit, before binding.
#
# Env (sane multi-A100 defaults):
#   RANKS        expert-parallel rank count = number of GPUs to shard experts across (default 7)
#   GLM_SHARD    first GGUF shard (default: first staged UD-Q4_K_M shard found on local NVMe)
#   FIRST_GPU    lowest visible GPU index (default 1, to leave GPU0 for a peer); ranks use FIRST_GPU..FIRST_GPU+RANKS-1
#   OUT          result file (default ./glm52-ep-witness-RESULT.txt)
#   PORT         serve port (default 8071)
#   FAK_CUDA_ARCH  GPU arch (default sm_80 = A100/Ampere)
#   SMOKE_S      decode wait bound (default 540); SMOKE_TOKENS max_tokens (default 16)
#   LOAD_MAX_S   max seconds to wait for load+ready (default 1200)
set -uo pipefail
export PATH="/usr/local/go/bin:/usr/local/cuda/bin:$PATH"
export GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}" GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}" FAK_CUDA_NCCL=1 CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
RANKS="${RANKS:-7}"
FIRST_GPU="${FIRST_GPU:-1}"
PORT="${PORT:-8071}"
OUT="${OUT:-./glm52-ep-witness-RESULT.txt}"
DONE="${OUT}.done"
SMOKE_S="${SMOKE_S:-540}"; SMOKE_TOKENS="${SMOKE_TOKENS:-16}"
LOAD_MAX_S="${LOAD_MAX_S:-1200}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "no repo root"; exit 90; }
mkdir -p "$GOCACHE" "$GOPATH"
rm -f "$DONE"; : > "$OUT"
log(){ echo "$*" | tee -a "$OUT"; }

# locate a staged shard
SHARD="${GLM_SHARD:-}"
if [ -z "$SHARD" ]; then
  for g in /mnt/*/glm52-q4/*-00001-of-*.gguf /projects/glm52-q4/UD-Q4_K_M/*-00001-of-*.gguf /opt/glm52-q4/UD-Q4_K_M/*-00001-of-*.gguf; do
    [ -f "$g" ] && { SHARD="$g"; break; }
  done
fi
VIS=$(seq "$FIRST_GPU" "$((FIRST_GPU + RANKS - 1))" | paste -sd,)
log "HEAD=$(git rev-parse --short HEAD 2>/dev/null) RANKS=$RANKS VIS=$VIS PORT=$PORT"
log "SHARD=$SHARD"
[ -f "$SHARD" ] || { log "NO_STAGED_SHARD — set GLM_SHARD"; echo 95 >"$DONE"; exit 95; }

log "== build -tags cuda,nccl =="
t=$(date +%s)
if bash internal/compute/build_cuda.sh binary ./cmd/fak "$ROOT/fakbin_nccl" >"$ROOT/build_nccl.log" 2>&1; then
  log "BUILD_OK $(( $(date +%s) - t ))s"
else
  log "BUILD_FAIL"; tail -30 "$ROOT/build_nccl.log" | tee -a "$OUT"; echo 94 >"$DONE"; exit 94
fi

log "== expert-parallel resident serve (experts resident across GPU $VIS, NO cpu-offload) =="
t=$(date +%s)
CUDA_VISIBLE_DEVICES="$VIS" "$ROOT/fakbin_nccl" serve --addr "127.0.0.1:$PORT" \
  --gguf "$SHARD" --backend cuda --expert-parallel "$RANKS" \
  --context-budget-tokens 4096 --model glm-5.2 >"$ROOT/serve_ep.log" 2>&1 &
SRV=$!
ready=0; iters=$(( LOAD_MAX_S / 15 ))
for _ in $(seq 1 "$iters"); do
  kill -0 "$SRV" 2>/dev/null || { log "SERVER_DIED"; tail -40 "$ROOT/serve_ep.log" | tee -a "$OUT"; echo 96 >"$DONE"; exit 96; }
  curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 15
done
L=$(( $(date +%s) - t ))
if [ "$ready" != 1 ]; then log "LOAD_TIMEOUT ${L}s"; tail -40 "$ROOT/serve_ep.log" | tee -a "$OUT"; kill "$SRV" 2>/dev/null; echo 97 >"$DONE"; exit 97; fi
log "LOAD_READY ${L}s"
log "-- expert-parallel / collective evidence --"
grep -iE "expert-parallel|collective|nccl|rank|resident=|allreduce" "$ROOT/serve_ep.log" | tail -14 | tee -a "$OUT"

log "-- EP decode smoke --"
ts=$(date +%s)
SM=$(curl -s -m "$SMOKE_S" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":$SMOKE_TOKENS}")
DT=$(( $(date +%s) - ts ))
log "SMOKE(${SMOKE_TOKENS}tok,${DT}s)=${SM:0:280}"
printf '%s' "$SM" | grep -q '"content"' && ! printf '%s' "$SM" | grep -q '"error"' && log "SMOKE_OK" || log "SMOKE_FAIL"
CT=$(printf '%s' "$SM" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$CT" ] && [ "$DT" -gt 0 ] && log "EP_DECODE tok=$CT wall=${DT}s rate=$(awk "BEGIN{printf \"%.4f\", $CT/$DT}")tok/s"
log "-- per-GPU state after decode (proves >1 GPU resident+used) --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$OUT"
kill "$SRV" 2>/dev/null
log "EP_WITNESS_DONE rc=0 load_s=$L"
echo 0 >"$DONE"
