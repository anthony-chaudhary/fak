#!/usr/bin/env bash
# glm52_load_witness.sh — measure GLM-5.2 (glm_moe_dsa, UD-Q4_K_M) NATIVE load time on an
# A100 (sm_80) node through fak's OWN kernel, and capture the per-quant-type load-path
# breakdown. It builds the -tags cuda fak binary from THIS checkout (so it witnesses the
# checked-out code), times the eager weight load of the staged 466 GB checkpoint, and smokes
# one turn to prove the resident k-quant experts decode.
#
# This is the load-speed counterpart of tools/glm52_fak_native_serve.sh: that one stands up a
# durable serve; this one MEASURES the load (the S1 parallel-load + S2 resident-Q5_K/Q6_K-expert
# levers in docs/notes/GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md) and exits. It writes a
# compact RESULT file + a .done(rc) sentinel so it can be driven detached over a flaky control
# bridge (run: setsid bash tools/glm52_load_witness.sh </dev/null >boot.log 2>&1 &).
#
# Env (all have sane A100/dgx3 defaults):
#   GLM_SHARD   first GGUF shard (default: first staged UD-Q4_K_M shard on local NVMe)
#   OUT         result file (default /mnt/sglang_dv3/glm52-load-witness-RESULT.txt)
#   PORT        serve port (default 8061)
#   FAK_GGUF_LOAD_WORKERS  load parallelism (default 64)
#   CUDA_VISIBLE_DEVICES   GPU to use (default 1)
#   LOAD_MAX_S  max seconds to wait for the load (default 1200 = 20 min)
set -uo pipefail
export PATH="/usr/local/go/bin:/usr/local/cuda/bin:$PATH"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export FAK_GGUF_LOAD_WORKERS="${FAK_GGUF_LOAD_WORKERS:-64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PORT="${PORT:-8061}"
OUT="${OUT:-/mnt/sglang_dv3/glm52-load-witness-RESULT.txt}"
DONE="${OUT}.done"
LOAD_MAX_S="${LOAD_MAX_S:-1200}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "no repo root" ; exit 90; }
mkdir -p "$GOCACHE" "$GOPATH"
rm -f "$DONE"
: > "$OUT"
log(){ echo "$*" | tee -a "$OUT"; }

# locate a staged shard
SHARD="${GLM_SHARD:-}"
if [ -z "$SHARD" ]; then
  for g in /mnt/sglang_dv3/glm52-q4/*-00001-of-*.gguf /projects/glm52-q4/UD-Q4_K_M/*-00001-of-*.gguf /opt/glm52-q4/UD-Q4_K_M/*-00001-of-*.gguf; do
    [ -f "$g" ] && { SHARD="$g"; break; }
  done
fi
log "HEAD=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format=%s 2>/dev/null | cut -c1-56)"
log "SHARD=$SHARD  WORKERS=$FAK_GGUF_LOAD_WORKERS  GPU=$CUDA_VISIBLE_DEVICES"
[ -f "$SHARD" ] || { log "NO_STAGED_SHARD — set GLM_SHARD"; echo 95 > "$DONE"; exit 95; }
# confirm the checkout carries the S1/S2 levers
if [ -f internal/model/quant_kquant.go ] && grep -q 'kqw' internal/model/kernel.go && [ -f internal/ggufload/gguf_parload.go ]; then
  log "CODE_OK resident-kquant + parallel-loader present"
else
  log "CODE_MISSING — checkout lacks S1/S2"; echo 93 > "$DONE"; exit 93
fi

log "== build -tags cuda fak binary =="
t=$(date +%s)
if bash internal/compute/build_cuda.sh binary ./cmd/fak "$ROOT/fakbin" >"$ROOT/build.log" 2>&1; then
  log "BUILD_OK $(( $(date +%s) - t ))s"
else
  log "BUILD_FAIL"; tail -25 "$ROOT/build.log" | tee -a "$OUT"; echo 94 > "$DONE"; exit 94
fi

log "== timed load =="
t=$(date +%s)
"$ROOT/fakbin" serve --addr "127.0.0.1:$PORT" --gguf "$SHARD" --backend cuda \
  --cpu-offload-experts --context-budget-tokens 8192 --model glm-5.2 >"$ROOT/serve.log" 2>&1 &
SRV=$!
ready=0; iters=$(( LOAD_MAX_S / 15 ))
for _ in $(seq 1 "$iters"); do
  kill -0 "$SRV" 2>/dev/null || { log "SERVER_DIED"; tail -30 "$ROOT/serve.log" | tee -a "$OUT"; echo 96 > "$DONE"; exit 96; }
  curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 15
done
L=$(( $(date +%s) - t ))
if [ "$ready" != 1 ]; then log "LOAD_TIMEOUT ${L}s"; tail -30 "$ROOT/serve.log" | tee -a "$OUT"; kill "$SRV" 2>/dev/null; echo 97 > "$DONE"; exit 97; fi
log "LOAD_READY ${L}s ($((L/60))m$((L%60))s)  under_10min=$([ "$L" -lt 600 ] && echo YES || echo NO)"

log "-- load-path breakdown (S4) --"
grep -E 'load-path breakdown|resident=|fak: resident:|loading model 100%' "$ROOT/serve.log" | tail -16 | tee -a "$OUT"
log "-- /metrics model-load --"
curl -s -m 10 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E '^fak_model_load_(duration_seconds|path_tensors|tensors|bytes) ' | head -40 | tee -a "$OUT"

log "-- serve smoke (resident k-quant experts decode) --"
SM=$(curl -s -m 180 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":8}')
log "SMOKE=${SM:0:300}"
printf '%s' "$SM" | grep -q '"content"' && ! printf '%s' "$SM" | grep -q '"error"' && log "SMOKE_OK" || log "SMOKE_FAIL"

kill "$SRV" 2>/dev/null
log "WITNESS_DONE rc=0 load_s=$L"
echo 0 > "$DONE"
