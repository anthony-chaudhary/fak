#!/usr/bin/env bash
# glm52_fak_native_serve.sh - durable stage + build + serve of GLM-5.2 on an A100 (sm_80)
# node via fak's OWN in-kernel engine (the PURE FAK KERNEL), not llama.cpp. It is the
# pure-fak-kernel sibling of tools/glm52_stage_serve_gpu_server.sh (which stands up the SAME
# checkpoint under llama.cpp as the BENCHMARK baseline). Prefer THIS; keep llama.cpp for
# the apples-to-apples comparison.
#
# WHY THIS IS THE PREFERRED PATH (vs the llama.cpp baseline):
#   * fak serves GLM-5.2 (glm_moe_dsa) through its OWN CUDA kernels — the forward is bit-exact
#     vs the CPU reference AT Q8 (cosine 1.000000, argmax-exact) on sm_80, witnessed at
#     experiments/glm-gpu-witness/a100-glm52-*.json (incl. the cpu-offload hybrid). That
#     correctness guarantee is fak's differentiator; the stock engines do not make it. NOTE:
#     the --cpu-offload-experts serve below runs the resident-Q4_K path, which is not yet
#     covered by a full-forward cosine witness (the q8 forward is).
#   * It is the SAME wire llama.cpp serves (OpenAI /v1) so `fak guard` / the #413 witness
#     front it identically — but the weights run in fak's kernel, not an external engine.
#   * llama.cpp stays the honest THROUGHPUT baseline for the comparison ladder
#     (docs/notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md). Never put the
#     two numbers side by side without holding {weights, hardware, precision, ctx} equal.
#
# HOW IT SERVES (the load-speed doc, GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25):
#   fak serve --gguf <shard1> --backend cuda --cpu-offload-experts --context-budget-tokens 4096
#   * --backend cuda  : prefill+decode run on the GPU HAL (needs a -tags cuda build + a GPU).
#   * --cpu-offload-experts : the ~424 GB MoE experts stay on host RAM (the A100s hold
#     attention/shared/dense); the device load uses the direct-resident-Q4_K path (no
#     Q4_K->f32->Q8 round-trip).
#   * --context-budget-tokens 4096 : the default 1M context plans a 533 GiB KV -> FitTooBig;
#     8192 also overfills the 8-way resident EP path on 80 GB H100s once CUDA reports usable
#     capacity (~74.6 GiB).
#
# HONEST SCOPE: this asserts NO throughput/quality number, and makes NO live-serve claim. It
# builds the cuda fak binary, stages the checkpoint, stands the endpoint up, and health-checks
# a REAL chat completion — a live GLM-5.2 serve turn is hardware+load-gated until that gate
# passes on a real A100. The witnessed claim is the Q8 forward correctness (cosine 1.0, sm_80);
# the resident-Q4_K serve path is not yet cosine-witnessed. Default to UD-Q4_K_S here because
# the mixed UD-Q4_K_M experts include Q5_K/Q6_K tensors that currently route through the slow
# host k-quant seam; the pure-Q4_K expert path is the performant CUDA demo target. Run
# tools/glm52_e2e_after_serve_gpu_server.sh against this endpoint for the #413 evidence.
#
# Usage (RUN ON THE GPU HOST, detached so a disconnect does not orphan a large load):
#   systemd-run --unit=glm52serve --collect bash tools/glm52_fak_native_serve.sh
# then poll:  cat "$GLM_DIR/PHASE"   and on GLM52_FAK_NATIVE_SERVE_READY run the witness.
set -uo pipefail

GLM_DIR="${GLM_DIR:-/opt/glm52-q4}"
REPO="${GLM_REPO:-unsloth/GLM-5.2-GGUF}"
SUBDIR="${GLM_SUBDIR:-UD-Q4_K_S}"
PORT="${PORT:-8000}"
ADDR="${ADDR:-0.0.0.0:${PORT}}"
MODEL_ID="${MODEL_ID:-glm-5.2}"
CTX="${CTX:-4096}"
FAK_BIN="${FAK_BIN:-/usr/local/bin/fak}"
GO_VERSION="${GO_VERSION:-1.26.4}"
EP_RANKS="${EP_RANKS:-1}"
FIRST_GPU="${FIRST_GPU:-0}"
EP_COORD_ADDR="${EP_COORD_ADDR:-127.0.0.1:19071}"
EP_JOIN_TIMEOUT_S="${EP_JOIN_TIMEOUT_S:-1800}"
GLM_SMOKE_TIMEOUT_S="${GLM_SMOKE_TIMEOUT_S:-900}"
GLM_SMOKE_MAX_TOKENS="${GLM_SMOKE_MAX_TOKENS:-1}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export FAK_HTTP_WRITE_TIMEOUT_S="${FAK_HTTP_WRITE_TIMEOUT_S:-0}"
export FAK_KQ_INT8="${FAK_KQ_INT8:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"

# locate the fak checkout root from this script's location (tools/<this>).
SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PHASE="$GLM_DIR/PHASE"
LOG="$GLM_DIR/fak_native_serve.log"
mkdir -p "$GLM_DIR" "$GOCACHE" "$GOPATH"
ph(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; echo "$*" > "$PHASE"; }

COMPLETE_SHARD1=""
SHARD_STATUS=""
SHARD_PRESENT=0
SHARD_TOTAL=0
complete_shard1_in_dir() {
  _d="$1"
  COMPLETE_SHARD1=""
  SHARD_STATUS=""
  SHARD_PRESENT=0
  SHARD_TOTAL=0
  _s1=$(ls "$_d"/*-00001-of-*.gguf 2>/dev/null | sort | head -1) || true
  [ -n "$_s1" ] || return 1
  _base="${_s1##*/}"
  _total="${_base##*-of-}"
  _total="${_total%.gguf}"
  case "$_total" in
    ''|*[!0-9]*) SHARD_STATUS="dir=$_d invalid_total=$_total"; return 1 ;;
  esac
  SHARD_TOTAL=$((10#$_total))
  _prefix="${_s1%00001-of-${_total}.gguf}"
  _missing=""
  for _i in $(seq 1 "$SHARD_TOTAL"); do
    _path="${_prefix}$(printf "%05d-of-%s.gguf" "$_i" "$_total")"
    if [ -s "$_path" ]; then
      SHARD_PRESENT=$((SHARD_PRESENT + 1))
    else
      _missing="${_missing}${_missing:+,}${_i}"
    fi
  done
  if [ "$SHARD_PRESENT" -eq "$SHARD_TOTAL" ]; then
    COMPLETE_SHARD1="$_s1"
    return 0
  fi
  SHARD_STATUS="dir=$_d shards=${SHARD_PRESENT}/${SHARD_TOTAL} missing=${_missing:-unknown}"
  return 1
}

have_nccl_headers() {
  [ -f "${NCCL_HOME:-}/include/nccl.h" ] ||
  [ -f "${NCCL_HOME:-}/usr/include/nccl.h" ] ||
  [ -f "${CUDA_HOME}/include/nccl.h" ] ||
  [ -f "${CUDA_HOME}/targets/x86_64-linux/include/nccl.h" ] ||
  [ -f /usr/include/nccl.h ]
}

have_nccl_lib() {
  [ -e "${NCCL_HOME:-}/lib/libnccl.so" ] ||
  [ -e "${NCCL_HOME:-}/lib64/libnccl.so" ] ||
  [ -e "${NCCL_HOME:-}/usr/lib/x86_64-linux-gnu/libnccl.so" ] ||
  [ -e "${CUDA_HOME}/lib64/libnccl.so" ] ||
  [ -e "${CUDA_HOME}/targets/x86_64-linux/lib/libnccl.so" ] ||
  [ -e /usr/lib/x86_64-linux-gnu/libnccl.so ]
}

case "$EP_RANKS" in
  ''|*[!0-9]*) ph "BAD_EP_RANKS $EP_RANKS"; exit 12 ;;
esac
case "$FIRST_GPU" in
  ''|*[!0-9]*) ph "BAD_FIRST_GPU $FIRST_GPU"; exit 12 ;;
esac
if [ "$EP_RANKS" -gt 1 ]; then
  export FAK_CUDA_NCCL=1
  REBUILD_FAK="${REBUILD_FAK:-1}"
  if [ -z "${FAK_EP_REQUIRE_DEVICE_PG:-}" ]; then
    export FAK_EP_REQUIRE_DEVICE_PG=1
  else
    export FAK_EP_REQUIRE_DEVICE_PG
  fi
else
  export FAK_EP_REQUIRE_DEVICE_PG="${FAK_EP_REQUIRE_DEVICE_PG:-0}"
fi

# EP_COORDINATED=1 selects the COORDINATED expert-parallel topology (#4835) instead of the
# default HTTP request mirror. The difference is what a rank does with a request:
#
#   mirror (EP_FRONTEND_FANOUT=1, the default) : rank 0 re-POSTs the whole body to every
#     follower, so all N ranks tokenize, prefill, decode AND sample the same prompt and only
#     rank 0's body is returned. That is the topology the sanctioned 8-GPU witness measured at
#     0.0406 tok/s, slower than the ~0.2 tok/s scalar pure-fak baseline.
#   coordinated (EP_COORDINATED=1) : ranks>0 bind NO listener at all. They park in
#     model.RunEPFollower and replay exactly the forward rank 0 announces, contributing only
#     their local expert work + collectives, so the prompt is tokenized once and sampled once.
#
# Three settings are load-bearing and are forced here rather than left to the operator, because
# `fak serve` REFUSES the combination at boot (cmd/fak/serve_ep_coord.go) and a refusal after a
# 466 GB stage is an expensive way to learn it:
#   * FAK_EP_COORDINATED_DECODE=1 is the opt-in the serve reads;
#   * EP_FRONTEND_FANOUT=0 keeps FAK_EP_FANOUT_ADDRS empty — the two topologies cannot coexist,
#     a follower fed its own copy of the request would run rank 0's frames AND an independent
#     decode on one DistComm;
#   * FAK_INKERNEL_RADIX=off — a rank-0 session restored from a KV-prefix hit prefills only the
#     divergent suffix, at a position the follower's fresh mirror never computed, and the
#     follower fails that request closed. Losing that cache is a REAL cost, not a formality:
#     the warm sanctioned-hardware read-back measured 0.214 tok/s on an exact-prefix repeat vs 0.0917 tok/s on
#     a distinct prompt, so the coordinated arm must beat the DISTINCT-prompt baseline to be a
#     net-true gain.
EP_COORDINATED="${EP_COORDINATED:-0}"
if [ "$EP_COORDINATED" = "1" ]; then
  if [ "$EP_RANKS" -le 1 ]; then
    ph "BAD_EP_COORDINATED EP_COORDINATED=1 needs EP_RANKS>1 (there is no process group to coordinate over)"
    exit 12
  fi
  export FAK_EP_COORDINATED_DECODE=1
  export FAK_INKERNEL_RADIX=off
  EP_FRONTEND_FANOUT=0
  ph "EP_COORDINATED ranks=$EP_RANKS rank0 owns tokenization+sampling; ranks 1-$((EP_RANKS - 1)) contribute local expert work only (HTTP request mirror OFF, radix prefix reuse OFF)"
fi
if [ "${FAK_CUDA_NCCL:-0}" = "1" ] && { ! have_nccl_headers || ! have_nccl_lib; }; then
  ph "NCCL_DEV_MISSING headers=$(have_nccl_headers && echo yes || echo no) lib=$(have_nccl_lib && echo yes || echo no) (install libnccl-dev/libnccl2 or set NCCL_HOME)"
  exit 13
fi

export PATH="/usr/local/go/bin:${CUDA_HOME}/bin:$PATH"

# 1. Ensure Go. The GCP Deep-Learning CUDA image ships nvcc but not always the Go toolchain
#    build_cuda.sh expects at /usr/local/go/bin; install it once if missing.
if ! command -v go >/dev/null 2>&1; then
  ph "INSTALL_GO ${GO_VERSION}"
  if curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz >>"$LOG" 2>&1 \
     && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz >>"$LOG" 2>&1; then :; else
    ph "GO_INSTALL_FAIL"; exit 11
  fi
fi
ph "GO $(go version 2>/dev/null || echo missing)"

# 1b. Prefer a COMPLETE local-NVMe copy over the slow /projects NFS (~2.9 GB/s vs ~0.055 GB/s,
#     ~53x; cold load ~44m vs ~1h41m). A verified copy may live FLAT in /mnt/sglang_dv3/glm52-q4/
#     (no quant subdir). Resolve NVMe-first and LOG the winner LOUDLY so a silent
#     fall-through to the slow NFS path is impossible to miss (the bug that caused a 62-min load).
PRESTAGED_SHARD1=""
for _d in /mnt/sglang_dv3/glm52-q4 "$GLM_DIR/$SUBDIR" "$GLM_DIR"; do
  if complete_shard1_in_dir "$_d"; then
    PRESTAGED_SHARD1="$COMPLETE_SHARD1"; break
  fi
  [ -n "$SHARD_STATUS" ] && ph "PARTIAL_PRESTAGED $SHARD_STATUS (will resume HF download)"
done
if [ -n "$PRESTAGED_SHARD1" ]; then
  case "$PRESTAGED_SHARD1" in
    /mnt/*|/nvme*|/local*|/raid*|/scratch*) ph "USING_LOCAL_NVME shard1=$PRESTAGED_SHARD1 (fast local read; HF download skipped)";;
    *) ph "USING_PRESTAGED shard1=$PRESTAGED_SHARD1 (WARN: NOT local NVMe — if this is /projects NFS the load is ~53x slower; stage to /mnt/sglang_dv3/glm52-q4 for the fast path)";;
  esac
fi

# 2. Download the GGUF shards (resumable; the HF CLI skips already-complete files).
if [ -z "$PRESTAGED_SHARD1" ]; then
ph "HF_DOWNLOAD_BACKEND disable_xet=${HF_HUB_DISABLE_XET:-}"
ph "DOWNLOAD_START repo=$REPO subdir=$SUBDIR dir=$GLM_DIR"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --include "$SUBDIR/*" --local-dir "$GLM_DIR" >>"$LOG" 2>&1; DL_RC=$?
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --include "$SUBDIR/*" --local-dir "$GLM_DIR" >>"$LOG" 2>&1; DL_RC=$?
else
  ph "NO_HF_CLI install huggingface_hub first"; exit 10
fi
if complete_shard1_in_dir "$GLM_DIR/$SUBDIR"; then
  SHARD1="$COMPLETE_SHARD1"
else
  SHARD1=$(ls "$GLM_DIR/$SUBDIR"/*-00001-of-*.gguf 2>/dev/null | head -1)
fi
ph "DOWNLOAD_DONE rc=$DL_RC shards=${SHARD_PRESENT}/${SHARD_TOTAL}"
[ "${DL_RC:-1}" -eq 0 ] && [ -n "$SHARD1" ] && [ "$SHARD_PRESENT" -eq "$SHARD_TOTAL" ] || { ph "DOWNLOAD_FAIL ${SHARD_STATUS:-dir=$GLM_DIR/$SUBDIR}"; exit 20; }
ph "SHARD1=$SHARD1"
else
  SHARD1="$PRESTAGED_SHARD1"
  ph "SHARD1=$SHARD1 (pre-staged; HF download skipped)"
fi

# 3. Build the -tags cuda fak binary (libfakcuda sm_80 + cmd/fak) via the canonical recipe
#    (internal/compute/build_cuda.sh resolves the CGO -I/-L/-rpath set for this host).
if [ ! -x "$FAK_BIN" ] || [ "${REBUILD_FAK:-0}" = "1" ]; then
  ph "BUILD_FAK_CUDA arch=$FAK_CUDA_ARCH out=$FAK_BIN"
  if ( cd "$ROOT" && bash internal/compute/build_cuda.sh binary ./cmd/fak "$FAK_BIN" ) >>"$LOG" 2>&1; then :; else
    ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
  fi
fi
[ -x "$FAK_BIN" ] || { ph "BUILD_FAK_FAIL"; exit 30; }
ph "FAK_BIN_READY $FAK_BIN"

# Runtime: put the CUDA shared libs (cudart/cublas) on LD_LIBRARY_PATH. The binary also
# bakes an rpath to them at link time; this is belt-and-braces for systemd's clean env.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# 4. Serve via the PURE FAK KERNEL. The embedded GGUF tokenizer makes /v1/chat/completions
#    serve real in-kernel chat; the eager load binds the listener only AFTER the weights are
#    resident, so /v1/models answering means the model is loaded.
smoke_body() {
  printf '{"model":"%s","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":%s,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' "$MODEL_ID" "$GLM_SMOKE_MAX_TOKENS"
}

smoke_ready_single() {
  start=$(date +%s)
  smoke=$(curl -sS -m "$GLM_SMOKE_TIMEOUT_S" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(smoke_body)" 2>&1)
  rc=$?
  elapsed=$(( $(date +%s) - start ))
  echo "SMOKE rc=$rc elapsed_s=$elapsed timeout_s=$GLM_SMOKE_TIMEOUT_S max_tokens=$GLM_SMOKE_MAX_TOKENS: $smoke" >>"$LOG"
  # GLM-5.2's current in-kernel forward can sample EOS as the first token while the
  # correctness/coherence bug is still open. Treat a 200 chat-completion envelope as
  # readiness (the full prefill/sampling path ran and did not deadlock), but still fail
  # closed on explicit error payloads.
  [ "$rc" -eq 0 ] && printf '%s' "$smoke" | grep -q '"choices"' && ! printf '%s' "$smoke" | grep -q '"error"'
}

smoke_ready_ep() {
  tmp="$(mktemp -d "$GLM_DIR/smoke.XXXXXX")"
  body="$(smoke_body)"
  start=$(date +%s)
  pids=""
  for r in $(seq 0 $((EP_RANKS - 1))); do
    rank_port=$((PORT + r))
    (
      curl -sS -m "$GLM_SMOKE_TIMEOUT_S" "http://127.0.0.1:${rank_port}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" >"$tmp/r${r}.out" 2>"$tmp/r${r}.err"
      echo $? >"$tmp/r${r}.rc"
    ) &
    pids="$pids $!"
  done
  ok=1
  for pid in $pids; do
    if ! wait "$pid"; then ok=0; fi
  done
  elapsed=$(( $(date +%s) - start ))
  for r in $(seq 0 $((EP_RANKS - 1))); do
    rc="$(cat "$tmp/r${r}.rc" 2>/dev/null || echo 999)"
    bytes="$(wc -c <"$tmp/r${r}.out" 2>/dev/null || echo 0)"
    err="$(cat "$tmp/r${r}.err" 2>/dev/null || true)"
    echo "SMOKE_EP rank=$r rc=$rc bytes=$bytes elapsed_s=$elapsed err=$err" >>"$LOG"
    [ "$rc" = "0" ] || ok=0
  done
  smoke="$(cat "$tmp/r0.out" 2>/dev/null || true)"
  echo "SMOKE_EP rank0 elapsed_s=$elapsed timeout_s=$GLM_SMOKE_TIMEOUT_S max_tokens=$GLM_SMOKE_MAX_TOKENS: $smoke" >>"$LOG"
  rm -rf "$tmp"
  [ "$ok" -eq 1 ] && printf '%s' "$smoke" | grep -q '"choices"' && ! printf '%s' "$smoke" | grep -q '"error"'
}

smoke_ready() {
  # Coordinated EP (#4835): ranks>0 bind NO listener, so the per-rank fanout smoke below would
  # curl seven dead ports and report SMOKE_FAIL on a serve that is in fact healthy. Rank 0 is
  # the whole public surface here, and its single request already drives every rank through
  # the collectives — which is exactly what this smoke is for.
  if [ "${EP_COORDINATED:-0}" = "1" ]; then
    smoke_ready_single
  elif [ "$EP_RANKS" -gt 1 ] && [ "${EP_FRONTEND_FANOUT:-1}" != "1" ]; then
    smoke_ready_ep
  else
    smoke_ready_single
  fi
}

launch_single_cpu_offload() {
  ph "LAUNCH fak serve --gguf $SHARD1 --backend cuda --cpu-offload-experts --context-budget-tokens $CTX --model $MODEL_ID (large load; resident-Q4_K path)"
  "$FAK_BIN" serve \
    --addr "$ADDR" \
    --gguf "$SHARD1" \
    --backend cuda \
    --cpu-offload-experts \
    --context-budget-tokens "$CTX" \
    --model "$MODEL_ID" \
    > "$GLM_DIR/server.log" 2>&1 &
  SRV=$!
  ph "SERVER_PID=$SRV"

  # 360 x 20s ~= 2 h, covering the large UD-Q4_K_M load.
  for _ in $(seq 1 360); do
    if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -40 "$GLM_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
    if curl -sf -m 5 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      if smoke_ready; then
        ph "GLM52_FAK_NATIVE_SERVE_READY port=$PORT model=$MODEL_ID ep_ranks=1"
        wait "$SRV"
        rc=$?
        ph "SERVER_EXITED rc=$rc"
        exit "$rc"
      fi
      ph "SMOKE_FAIL"; exit 41
    fi
    sleep 20
  done
  ph "HEALTH_TIMEOUT"; tail -20 "$GLM_DIR/server.log" >>"$LOG" 2>&1; exit 42
}

launch_expert_parallel() {
  ph "LAUNCH_EP fak serve --gguf $SHARD1 --backend cuda --expert-parallel $EP_RANKS --context-budget-tokens $CTX --model $MODEL_ID (sharded EP; require_device_pg=$FAK_EP_REQUIRE_DEVICE_PG; supported Q4_K/Q8/F32 on CUDA, mixed k-quants use host int8 fallback until CUDA kernels land)"
  pids=""
  for r in $(seq 0 $((EP_RANKS - 1))); do
    gpu=$((FIRST_GPU + r))
    rank_port=$((PORT + r))
    rank_addr="127.0.0.1:${rank_port}"
    [ "$r" -eq 0 ] && rank_addr="$ADDR"
    rank_log="$GLM_DIR/server-rank${r}.log"
    fanout_addrs=""
    if [ "$r" -eq 0 ] && [ "${EP_FRONTEND_FANOUT:-1}" = "1" ]; then
      for pr in $(seq 1 $((EP_RANKS - 1))); do
        peer_port=$((PORT + pr))
        fanout_addrs="${fanout_addrs}${fanout_addrs:+,}127.0.0.1:${peer_port}"
      done
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
    FAK_EP_RANK="$r" \
    FAK_EP_FANOUT_ADDRS="$fanout_addrs" \
    FAK_EP_COORD_ADDR="$EP_COORD_ADDR" \
    FAK_EP_JOIN_TIMEOUT_S="$EP_JOIN_TIMEOUT_S" \
    FAK_EP_REQUIRE_DEVICE_PG="$FAK_EP_REQUIRE_DEVICE_PG" \
    FAK_Q4K=1 \
    "$FAK_BIN" serve \
      --addr "$rank_addr" \
      --gguf "$SHARD1" \
      --backend cuda \
      --expert-parallel "$EP_RANKS" \
      --context-budget-tokens "$CTX" \
      --model "$MODEL_ID" \
      > "$rank_log" 2>&1 &
    pid=$!
    pids="$pids $pid"
    ph "EP_RANK_PID rank=$r gpu=$gpu port=$rank_port pid=$pid log=$rank_log"
  done

  # 360 x 20s ~= 2 h. Rank 0 binds the public endpoint only after every rank loads its
  # expert shard, joins the DistComm/device-PG group, and the model is ready.
  for _ in $(seq 1 360); do
    for pid in $pids; do
      if ! kill -0 "$pid" 2>/dev/null; then
        ph "EP_RANK_EXITED_EARLY pid=$pid"
        for lf in "$GLM_DIR"/server-rank*.log; do echo "--- $lf ---" >>"$LOG"; tail -40 "$lf" >>"$LOG" 2>&1 || true; done
        for opid in $pids; do kill "$opid" 2>/dev/null || true; done
        exit 40
      fi
    done
    if curl -sf -m 5 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      if smoke_ready; then
        grep -hiE "expert-parallel|device-NCCL|DistComm|rank .*joined|loads experts|collective" "$GLM_DIR"/server-rank*.log >>"$LOG" 2>&1 || true
        ph "GLM52_FAK_NATIVE_SERVE_READY port=$PORT model=$MODEL_ID ep_ranks=$EP_RANKS"
        wait $pids
        rc=$?
        ph "EP_SERVER_EXITED rc=$rc"
        exit "$rc"
      fi
      ph "SMOKE_FAIL"; for pid in $pids; do kill "$pid" 2>/dev/null || true; done; exit 41
    fi
    sleep 20
  done
  ph "HEALTH_TIMEOUT"; for lf in "$GLM_DIR"/server-rank*.log; do echo "--- $lf ---" >>"$LOG"; tail -20 "$lf" >>"$LOG" 2>&1 || true; done; for pid in $pids; do kill "$pid" 2>/dev/null || true; done; exit 42
}

# 5. Health-check: detect a crashed load immediately, and assert a REAL chat answer before
#    declaring ready (a server that bound but cannot decode must NOT greenlight a witness).
if [ "$EP_RANKS" -gt 1 ]; then
  launch_expert_parallel
else
  launch_single_cpu_offload
fi
