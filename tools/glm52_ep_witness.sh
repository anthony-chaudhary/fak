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
#   COORD_ADDR   local rank rendezvous address (default 127.0.0.1:$PORT+1000)
#   FAK_CUDA_ARCH  GPU arch (default sm_80 = A100/Ampere)
#   SMOKE_S      decode wait bound (default 540); SMOKE_TOKENS max_tokens (default 16)
#   LOAD_MAX_S   max seconds to wait for load+ready (default 1200)
set -uo pipefail
export PATH="/usr/local/go/bin:/usr/local/cuda/bin:$PATH"
export GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}" GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}" FAK_CUDA_NCCL=1 FAK_EP_REQUIRE_DEVICE_PG="${FAK_EP_REQUIRE_DEVICE_PG:-1}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
RANKS="${RANKS:-7}"
FIRST_GPU="${FIRST_GPU:-1}"
PORT="${PORT:-8071}"
COORD_ADDR="${COORD_ADDR:-127.0.0.1:$((PORT + 1000))}"
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

# --- pre-flight per-GPU free-VRAM gate (EP_PREFLIGHT, #4952) ---------------------------------
# Read every visible GPU's live free/total VRAM ONCE, here, before the CUDA build and the rank
# spawn. Without this the only signal is N opaque per-rank FitTooBig walls AFTER a ~20 minute
# build and a full load attempt — and because each rank reports only its own device, "needs 73.23
# GiB, device has 71.41 GiB" on all eight looked like a load-plan regression rather than what it
# was: leftover residency from a peer's process. #4952 was filed on that misreading.
#
# The threshold is the exact inverse of the load-time fit arithmetic — internal/compute's
# RequiredFreeBytes(per-rank plan bytes, headroom), the inverse of BudgetAfterHeadroom — so this
# gate cannot admit a run the in-process check then refuses, or refuse one it would have admitted.
# tools/glm52_ep_preflight_test.go pins the numbers below against that function.
#
# It FAILS OPEN by construction: no nvidia-smi, no reading, or no known plan total for this RANKS
# means SKIP and proceed. It never invents a refusal it cannot ground.
#
#   REQUIRE_FREE_GIB  per-GPU free VRAM to demand (overrides the derivation entirely)
#   PLAN_GIB          per-rank plan total in GiB (default: the published table below)
#   EP_HEADROOM       fit headroom fraction (default 0.05, the expert-parallel device headroom)
EP_HEADROOM="${EP_HEADROOM:-0.05}"
PLAN_GIB="${PLAN_GIB:-}"
if [ -z "$PLAN_GIB" ]; then
  # Per-rank plan totals: the EP device totals measured in
  # experiments/glm-gpu-witness/glm52-ep-load-plan-witness-2026-06-30.json (71.11 GiB at 8 ranks,
  # 79.21 GiB at 7) plus ~2.12 GiB of KV at --context-budget-tokens 4096. Only the rank counts with
  # a measured witness are listed; any other RANKS leaves the gate unarmed rather than guessing.
  case "$RANKS" in
    8) PLAN_GIB=73.23 ;;
    7) PLAN_GIB=81.33 ;;
  esac
fi
REQ_MIB=""
if [ -n "${REQUIRE_FREE_GIB:-}" ]; then
  REQ_MIB=$(awk "BEGIN{printf \"%d\", int($REQUIRE_FREE_GIB * 1024) + ($REQUIRE_FREE_GIB * 1024 > int($REQUIRE_FREE_GIB * 1024))}")
elif [ -n "$PLAN_GIB" ]; then
  # ceil(plan_mib / (1 - headroom)) — RequiredFreeBytes at MiB granularity, rounded the safe way.
  REQ_MIB=$(awk "BEGIN{r=$PLAN_GIB*1024/(1-$EP_HEADROOM); printf \"%d\", int(r) + (r > int(r))}")
fi
if [ -z "$REQ_MIB" ]; then
  log "EP_PREFLIGHT_SKIP no published per-rank plan total for RANKS=$RANKS — set REQUIRE_FREE_GIB or PLAN_GIB to arm the gate"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
  log "EP_PREFLIGHT_SKIP no nvidia-smi — cannot read free VRAM, proceeding (fail-open)"
else
  log "EP_PREFLIGHT require_free=$(awk "BEGIN{printf \"%.1f\", $REQ_MIB/1024}")GiB/gpu (plan=${PLAN_GIB:-override} headroom=$EP_HEADROOM) gpus=$VIS"
  ep_short=0; ep_seen=0; ep_toosmall=0; ep_detail=""
  for gpu in $(echo "$VIS" | tr ',' ' '); do
    read -r ep_free ep_total <<<"$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' | tr ',' ' ')"
    case "$ep_free" in ''|*[!0-9]*) continue ;; esac
    ep_seen=$((ep_seen + 1))
    [ "$ep_free" -ge "$REQ_MIB" ] && continue
    ep_short=$((ep_short + 1))
    case "$ep_total" in
      ''|*[!0-9]*) : ;;
      *) [ "$ep_total" -lt "$REQ_MIB" ] && ep_toosmall=$((ep_toosmall + 1)) ;;
    esac
    ep_detail="$ep_detail gpu$gpu(free=$(awk "BEGIN{printf \"%.1f\", $ep_free/1024}")GiB short=$(awk "BEGIN{printf \"%.1f\", ($REQ_MIB-$ep_free)/1024}")GiB)"
  done
  if [ "$ep_seen" = 0 ]; then
    log "EP_PREFLIGHT_SKIP nvidia-smi returned no usable reading for gpus $VIS — proceeding (fail-open)"
  elif [ "$ep_short" = 0 ]; then
    log "EP_PREFLIGHT_OK $ep_seen/$ep_seen gpu(s) hold >= $(awk "BEGIN{printf \"%.1f\", $REQ_MIB/1024}")GiB free — proceeds to build+serve"
  elif [ "$ep_toosmall" -gt 0 ]; then
    log "EP_PREFLIGHT_REFUSE ($ep_short/$ep_seen CARD TOO SMALL — need > card total):$ep_detail"
    log "  RANKS=$RANKS cannot fit this card at all. Raise RANKS to shard the experts further, lower --context-budget-tokens, or use a larger card."
    echo 93 >"$DONE"; exit 93
  else
    log "EP_PREFLIGHT_REFUSE ($ep_short/$ep_seen short):$ep_detail"
    log "  The card is big enough but the memory is not free — almost always stale residency from a peer process. Check 'nvidia-smi', free the GPUs, and re-run. Override with REQUIRE_FREE_GIB if this threshold is wrong for your config."
    echo 93 >"$DONE"; exit 93
  fi
fi
# --- end pre-flight per-GPU free-VRAM gate ---------------------------------------------------

log "== build -tags cuda,nccl =="
t=$(date +%s)
if bash internal/compute/build_cuda.sh binary ./cmd/fak "$ROOT/fakbin_nccl" >"$ROOT/build_nccl.log" 2>&1; then
  log "BUILD_OK $(( $(date +%s) - t ))s"
else
  log "BUILD_FAIL"; tail -30 "$ROOT/build_nccl.log" | tee -a "$OUT"; echo 94 >"$DONE"; exit 94
fi

log "== expert-parallel resident serve (sharded ranks across GPU $VIS, NO cpu-offload) =="
t=$(date +%s)
pids=""
for r in $(seq 0 "$((RANKS - 1))"); do
  gpu=$((FIRST_GPU + r))
  rank_port=$((PORT + r))
  rank_log="$ROOT/serve_ep_rank${r}.log"
  log "LAUNCH_RANK rank=$r gpu=$gpu port=$rank_port coord=$COORD_ADDR log=$rank_log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  FAK_EP_RANK="$r" \
  FAK_EP_COORD_ADDR="$COORD_ADDR" \
  FAK_EP_JOIN_TIMEOUT_S="${FAK_EP_JOIN_TIMEOUT_S:-1800}" \
  FAK_EP_REQUIRE_DEVICE_PG="$FAK_EP_REQUIRE_DEVICE_PG" \
  FAK_Q4K=1 \
  "$ROOT/fakbin_nccl" serve --addr "127.0.0.1:$rank_port" \
    --gguf "$SHARD" --backend cuda --expert-parallel "$RANKS" \
    --context-budget-tokens 4096 --model glm-5.2 >"$rank_log" 2>&1 &
  pids="$pids $!"
done
ready=0; iters=$(( LOAD_MAX_S / 15 ))
for _ in $(seq 1 "$iters"); do
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null || {
      log "SERVER_DIED pid=$pid"
      for lf in "$ROOT"/serve_ep_rank*.log; do echo "--- $lf ---" | tee -a "$OUT"; tail -40 "$lf" | tee -a "$OUT"; done
      for opid in $pids; do kill "$opid" 2>/dev/null || true; done
      echo 96 >"$DONE"; exit 96
    }
  done
  curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 15
done
L=$(( $(date +%s) - t ))
if [ "$ready" != 1 ]; then
  log "LOAD_TIMEOUT ${L}s"
  for lf in "$ROOT"/serve_ep_rank*.log; do echo "--- $lf ---" | tee -a "$OUT"; tail -40 "$lf" | tee -a "$OUT"; done
  for pid in $pids; do kill "$pid" 2>/dev/null || true; done
  echo 97 >"$DONE"; exit 97
fi
log "LOAD_READY ${L}s"
log "-- expert-parallel / collective evidence --"
grep -hiE "expert-parallel|collective|nccl|rank|resident=|allreduce|loads experts|joined" "$ROOT"/serve_ep_rank*.log | tail -30 | tee -a "$OUT"

log "-- EP decode smoke --"
ts=$(date +%s)
SM=$(curl -s -m "$SMOKE_S" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":$SMOKE_TOKENS}")
DT=$(( $(date +%s) - ts ))
log "SMOKE(${SMOKE_TOKENS}tok,${DT}s)=${SM:0:280}"
if printf '%s' "$SM" | grep -q '"content"' && ! printf '%s' "$SM" | grep -q '"error"'; then
  log "SMOKE_OK"
else
  log "SMOKE_FAIL"
  for pid in $pids; do kill "$pid" 2>/dev/null || true; done
  echo 98 >"$DONE"; exit 98
fi
CT=$(printf '%s' "$SM" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$CT" ] && [ "$DT" -gt 0 ] && log "EP_DECODE tok=$CT wall=${DT}s rate=$(awk "BEGIN{printf \"%.4f\", $CT/$DT}")tok/s"
log "-- per-GPU state after decode (proves >1 GPU resident+used) --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$OUT"
for pid in $pids; do kill "$pid" 2>/dev/null || true; done
log "EP_WITNESS_DONE rc=0 load_s=$L"
echo 0 >"$DONE"
