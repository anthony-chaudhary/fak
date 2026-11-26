#!/usr/bin/env bash
# Matched direct-engine MI300X witness runner. Runtime use is intentionally fail-closed.
set -Eeuo pipefail
umask 077

DRY_RUN=0
if [[ ${1:-} == "--dry-run" ]]; then DRY_RUN=1; shift; fi
if (($#)); then echo "usage: $0 [--dry-run]" >&2; exit 2; fi

readonly MODEL='Qwen/Qwen2.5-0.5B-Instruct'
readonly MODEL_REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
readonly VLLM_IMAGE='vllm/vllm-openai-rocm:v0.28.0'
readonly SGLANG_IMAGE='lmsysorg/sglang:v0.5.18-rocm700-mi30x'
readonly PROMPT='Explain in one sentence why matched benchmark geometry matters.'
readonly CONCURRENCY='16'
readonly REQUESTS_PER='64'
readonly MAX_TOKENS='128'
readonly ROOT="${MI300X_PACKET_OUT:-$PWD/mi300x-results}"
readonly LOADGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_PID=''
TELEMETRY_PID=''
ENGINE='setup'

say() { printf '%s\n' "$*"; }
fail() { say "FAIL: $*" >&2; exit 1; }
run() { say "+ $(printf '%q ' "$@")"; ((DRY_RUN)) || "$@"; }
cleanup() {
  local rc=$?
  set +e
  [[ -n $TELEMETRY_PID ]] && kill "$TELEMETRY_PID" 2>/dev/null
  [[ -n $SERVER_PID ]] && kill "$SERVER_PID" 2>/dev/null
  if ((!DRY_RUN)); then docker rm -f "mi300x-${ENGINE}" >/dev/null 2>&1; fi
  if ((rc != 0)) && ((!DRY_RUN)); then printf '%s\n' "engine=$ENGINE exit=$rc" >>"$ROOT/errors.log"; fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

if ((DRY_RUN)); then
  cat <<EOF
DRY-RUN ONLY: no GPU, network, Docker, model download, or output mutation is performed.
packet: MI300X; ROCm admission requires /dev/kfd, /dev/dri, ROCm >= 6.3, gfx942, and an MI300X name
model: $MODEL@$MODEL_REVISION (public, ungated)
images: $VLLM_IMAGE ; $SGLANG_IMAGE
ordering: vllm cold,warm; then sglang cold,warm
matched loadgen: concurrency=$CONCURRENCY requests-per=$REQUESTS_PER max-tokens=$MAX_TOKENS max-error-rate=0
captures: admission, image IDs/digests, versions, logs, amd-smi telemetry, loadgen JSON, errors, no-fallback checks
EOF
  exit 0
fi

command -v docker >/dev/null || fail 'docker is required'
command -v rocminfo >/dev/null || fail 'rocminfo is required on the host'
command -v amd-smi >/dev/null || fail 'amd-smi is required on the host'
command -v go >/dev/null || fail 'go is required for repository cmd/loadgen'
[[ -c /dev/kfd ]] || fail '/dev/kfd is not a character device'
[[ -d /dev/dri ]] || fail '/dev/dri is absent'
mkdir -p "$ROOT"
: >"$ROOT/errors.log"
rocminfo >"$ROOT/rocminfo.txt" 2>"$ROOT/rocminfo.stderr" || fail 'rocminfo failed'
amd-smi static --gpu all >"$ROOT/amd-smi-static.txt" 2>"$ROOT/amd-smi-static.stderr" || fail 'amd-smi static failed'
grep -qi 'gfx942' "$ROOT/rocminfo.txt" || fail 'gfx942 was not admitted'
grep -Eqi 'MI300X|Instinct MI300X' "$ROOT/amd-smi-static.txt" || fail 'AMD Instinct MI300X was not admitted'

ROCM_VER=""
if [[ -f /opt/rocm/.info/version ]]; then
  ROCM_VER="$(cat /opt/rocm/.info/version)"
elif command -v hipconfig >/dev/null 2>&1; then
  ROCM_VER="$(hipconfig --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
elif grep -qiE 'ROCm Version|ROCk module version' "$ROOT/rocminfo.txt" 2>/dev/null; then
  ROCM_VER="$(grep -iE 'ROCm Version|ROCk module version' "$ROOT/rocminfo.txt" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
fi
if [[ -z "$ROCM_VER" ]] && command -v dpkg >/dev/null 2>&1; then
  ROCM_VER="$(dpkg -l 2>/dev/null | grep -E 'rocm-core|rocm-hip-runtime' | awk '{print $3}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
fi
printf '%s\n' "${ROCM_VER:-unknown}" >"$ROOT/rocm-version.txt"
[[ -n "$ROCM_VER" ]] || fail 'host ROCm version could not be determined'
ROCM_MAJOR="$(printf '%s\n' "$ROCM_VER" | cut -d. -f1)"
ROCM_MINOR="$(printf '%s\n' "$ROCM_VER" | cut -d. -f2)"
if (( ROCM_MAJOR < 6 || (ROCM_MAJOR == 6 && ROCM_MINOR < 3) )); then
  fail "ROCm version $ROCM_VER is below required ROCm >= 6.3"
fi

: >"$ROOT/images.jsonl"
for image in "$VLLM_IMAGE" "$SGLANG_IMAGE"; do
  run docker pull "$image"
  docker image inspect "$image" --format '{"ref":{{json .RepoTags}},"id":{{json .Id}},"repo_digests":{{json .RepoDigests}}}' >>"$ROOT/images.jsonl"
  [[ $(docker image inspect "$image" --format '{{len .RepoDigests}}') -gt 0 ]] || fail "$image has no resolved repository digest"
done
readonly VLLM_ID="$(docker image inspect "$VLLM_IMAGE" --format '{{.Id}}')"
readonly SGLANG_ID="$(docker image inspect "$SGLANG_IMAGE" --format '{{.Id}}')"

docker run --rm "$VLLM_ID" --version >"$ROOT/vllm-version.txt" 2>"$ROOT/vllm-version.stderr" || fail 'cannot record vLLM version'
grep -Eq '(^|[^0-9])0\.28\.0([^0-9]|$)' "$ROOT/vllm-version.txt" || fail 'vLLM image is not 0.28.0'
docker run --rm --entrypoint python3 "$SGLANG_ID" -c 'import sglang; print(sglang.__version__)' >"$ROOT/sglang-version.txt" 2>"$ROOT/sglang-version.stderr" || fail 'cannot record SGLang version'
grep -Eq '^0\.5\.18([[:space:]]*)$' "$ROOT/sglang-version.txt" || fail 'SGLang image is not 0.5.18'

wait_ready() {
  local i
  for i in {1..180}; do
    if curl --fail --silent --show-error "http://127.0.0.1:18000/v1/models" >"$ROOT/$ENGINE-models.json" 2>>"$ROOT/$ENGINE-errors.log"; then return 0; fi
    kill -0 "$SERVER_PID" 2>/dev/null || fail "$ENGINE server exited before readiness"
    sleep 2
  done
  fail "$ENGINE readiness timed out"
}

no_fallback() {
  local log=$1
  grep -Fq "$MODEL_REVISION" "$log" || fail "$ENGINE log does not prove the exact model revision"
  if grep -Eqi 'fall(ing)? back|CPU fallback|CUDA|NVIDIA|no GPU|GPU unavailable' "$log"; then
    fail "$ENGINE log contains a fallback or wrong-backend marker"
  fi
  docker exec "mi300x-${ENGINE}" sh -c 'test -c /dev/kfd && test -d /dev/dri && rocminfo' >"$ROOT/$ENGINE-container-rocminfo.txt" 2>"$ROOT/$ENGINE-container-rocminfo.stderr" || fail "$ENGINE container ROCm admission failed"
  grep -qi gfx942 "$ROOT/$ENGINE-container-rocminfo.txt" || fail "$ENGINE container did not enumerate gfx942"
}

load_pair() {
  local stack=$1
  for phase in cold warm; do
    (cd "$LOADGEN_ROOT" && go run ./cmd/loadgen \
      -url http://127.0.0.1:18000/v1/chat/completions -model "$MODEL" -stack "$stack-$phase" \
      -prompt "$PROMPT" -max-tokens "$MAX_TOKENS" -concurrency "$CONCURRENCY" \
      -requests-per "$REQUESTS_PER" -max-error-rate 0 -timeout 20m \
      -out "$ROOT/$stack-$phase-loadgen.json") 2>"$ROOT/$stack-$phase-loadgen.stderr" || fail "$stack $phase loadgen failed"
  done
}

start_telemetry() {
  (while :; do date -u +'%Y-%m-%dT%H:%M:%SZ'; amd-smi metric --gpu all; sleep 1; done) >"$ROOT/$ENGINE-amd-smi.log" 2>"$ROOT/$ENGINE-amd-smi.stderr" &
  TELEMETRY_PID=$!
}
stop_engine() {
  kill "$TELEMETRY_PID"; wait "$TELEMETRY_PID" 2>/dev/null || true; TELEMETRY_PID=''
  docker stop "mi300x-${ENGINE}" >"$ROOT/$ENGINE-docker-stop.txt"
  wait "$SERVER_PID" || fail "$ENGINE docker process failed"; SERVER_PID=''
}

ENGINE='vllm'
printf 'engine=vllm version=0.28.0 model=%s revision=%s image_id=%s\n' "$MODEL" "$MODEL_REVISION" "$VLLM_ID" >"$ROOT/vllm-server.log"
start_telemetry
docker run --name mi300x-vllm --rm --network host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES='' "$VLLM_ID" \
  "$MODEL" --revision "$MODEL_REVISION" --served-model-name "$MODEL" --host 127.0.0.1 --port 18000 \
  --tensor-parallel-size 1 --dtype float16 --max-model-len 2048 >>"$ROOT/vllm-server.log" 2>&1 & SERVER_PID=$!
wait_ready
no_fallback "$ROOT/vllm-server.log"
load_pair vllm
stop_engine

ENGINE='sglang'
printf 'engine=sglang version=0.5.18 model=%s revision=%s image_id=%s\n' "$MODEL" "$MODEL_REVISION" "$SGLANG_ID" >"$ROOT/sglang-server.log"
start_telemetry
docker run --name mi300x-sglang --rm --network host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES='' --entrypoint python3 "$SGLANG_ID" \
  -m sglang.launch_server --model-path "$MODEL" --revision "$MODEL_REVISION" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port 18000 --tp-size 1 --dtype float16 --context-length 2048 \
  >>"$ROOT/sglang-server.log" 2>&1 & SERVER_PID=$!
wait_ready
no_fallback "$ROOT/sglang-server.log"
load_pair sglang
stop_engine

ENGINE='complete'
say "PASS: matched MI300X packet captured at $ROOT"
