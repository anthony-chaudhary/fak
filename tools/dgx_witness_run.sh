#!/usr/bin/env bash
# dgx_witness_run.sh — on-box GLM-5.2 GPU witness runner that emits a fetch-ready result.
#
# Run from a FRESH clone root (the bootstrap clones origin/main, then `cd src && bash this`):
#   bash tools/dgx_witness_run.sh <nonce> <gpu> <tag>
#
# It builds libfakcuda.a (sm_80), runs the 3 isolated `-tags cuda` GLM witnesses into a log,
# parses that log into a COMPACT one-line glm-gpu-witness/1 record (tools/glm_witness_record.py),
# and writes /tmp/fakgpu/<tag>.result wrapped in nonce sentinels:
#     <<<FAKRES nonce=<n> rc=<rc> sha=<sha256 of json> len=<bytes>>>>
#     {<compact json on a single line>}
#     <<<ENDFAKRES nonce=<n>>>
# so tools/dgx_witness_fetch.py (fak-private) can pull it in ONE robust round trip (the nonce
# defeats stale-tail/throttle, the sha catches a split/truncated transfer). It ALWAYS emits a
# result — on build/parse failure it writes an {"error":...} json with a non-zero rc — so the
# fetcher gets a typed terminal state instead of hanging.
#
# This is the small COMMITTED runner the fetch bootstrap invokes: the bootstrap `!send` stays
# tiny (a clone + this call), well under the control-hub's launch-args cap. Do NOT inline a big
# script into a single `!send` (it is silently dropped by the >2000-char arg cap).
set -uo pipefail
NONCE="${1:?usage: dgx_witness_run.sh <nonce> <gpu> <tag>}"
GPU="${2:-0}"
TAG="${3:-glmw$NONCE}"
WORK=/tmp/fakgpu
LOG=/tmp/fakglm_fetch/run.log
JSONF="$WORK/$TAG.witness.json"
RESULT="$WORK/$TAG.result"
export HOME="${HOME:-/root}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}" CUDA_VISIBLE_DEVICES="$GPU"
export PATH="/usr/local/go/bin:$CUDA_HOME/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"
mkdir -p "$WORK" "$GOCACHE" "$GOPATH" "$(dirname "$LOG")"

emit() { # ($1=rc) frame whatever is in $JSONF between nonce sentinels
  local sha len
  sha=$(sha256sum "$JSONF" 2>/dev/null | cut -d' ' -f1)
  len=$(wc -c < "$JSONF" 2>/dev/null || echo 0)
  {
    echo "<<<FAKRES nonce=$NONCE rc=$1 sha=${sha:-0} len=${len:-0}>>>"
    cat "$JSONF" 2>/dev/null
    echo
    echo "<<<ENDFAKRES nonce=$NONCE>>>"
  } > "$RESULT"
}

HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
{
  echo "=== node $(hostname) gpu=$GPU arch=$FAK_CUDA_ARCH nonce=$NONCE ==="
  echo "=== HEAD $HEAD_SHA ==="
  echo "=== build libfakcuda.a ($FAK_CUDA_ARCH) ==="
} > "$LOG"
if ! bash internal/compute/build_cuda.sh build >> "$LOG" 2>&1; then
  printf '%s' '{"schema":"glm-gpu-witness/1","error":"BUILD_FAIL"}' > "$JSONF"; emit 96; exit 96
fi
echo "[cuda] OK build" >> "$LOG"
PKG="$PWD/internal/compute"
export CGO_ENABLED=1 CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$PKG -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
for spec in \
  "TestCUDAGLMMoeDsaBackendForward|all-device GLM-5.2 DSA forward" \
  "TestCUDAGLMDsaCPUOffloadHybrid|cpu-offload hybrid" \
  "TestCUDAGLMMoeDsaIndexSelectMatches|device index score+topk"; do
  t="${spec%%|*}"; d="${spec##*|}"
  echo "=== go test -tags cuda -run $t ($d) ===" >> "$LOG"
  go test -tags cuda -count=1 -v -run "^$t\$" ./internal/model/ >> "$LOG" 2>&1
done
p() { grep -c "^--- PASS: $1" "$LOG" || true; }
rc1=$((1 - $(p TestCUDAGLMMoeDsaBackendForward)))
rc2=$((1 - $(p TestCUDAGLMDsaCPUOffloadHybrid)))
rc3=$((1 - $(p TestCUDAGLMMoeDsaIndexSelectMatches)))
RC=$rc1; [ "$rc2" -ne 0 ] && RC=$rc2; [ "$rc3" -ne 0 ] && RC=$rc3
echo "=== GLM GPU WITNESS DONE head=$HEAD_SHA rc1=$rc1 rc2=$rc2 rc3=$rc3 -> rc=$RC ===" >> "$LOG"
if JSON=$(python3 tools/glm_witness_record.py "$LOG" --utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --machine-id dgx 2>> "$LOG" \
           | python3 -c 'import sys,json;json.dump(json.load(sys.stdin),sys.stdout,sort_keys=True,separators=(",",":"))' 2>> "$LOG"); then
  printf '%s' "$JSON" > "$JSONF"; emit "$RC"
else
  printf '%s' '{"schema":"glm-gpu-witness/1","error":"PARSE_FAIL"}' > "$JSONF"; emit 1; exit 1
fi
