#!/usr/bin/env bash
# dgx_glm_throughput_run.sh — on-box GLM-5.2 NATIVE throughput runner; emits a fetch-ready result.
#
# The throughput analogue of dgx_witness_run.sh: that one records the cosine-CORRECTNESS witness,
# this one records the DECODE/PREFILL TOK/S of fak's native glm_moe_dsa kernels on the device.
#
# Run from a FRESH clone root (the bootstrap clones origin/main, then `cd src && bash this`):
#   bash tools/dgx_glm_throughput_run.sh <nonce> <gpu> <tag>
#
# It builds the CUDA backend (sm_80), runs cmd/glmdsatput -json across a config SWEEP into a log,
# folds the log into a COMPACT one-line glm-throughput/1 record (tools/glm_throughput_record.py),
# and writes /tmp/fakgpu/<tag>.result wrapped in nonce sentinels:
#     <<<FAKRES nonce=<n> rc=<rc> sha=<sha256 of json> len=<bytes>>>>
#     {<compact json on a single line>}
#     <<<ENDFAKRES nonce=<n>>>
# so tools/dgx_witness_fetch.py (fak-private) can pull it in ONE robust round trip. It ALWAYS
# emits a result — on build/parse failure it writes an {"error":...} json with a non-zero rc.
#
# HONEST SCOPE: glmdsatput builds a SYNTHETIC reduced-layer dense-FFN glm_moe_dsa (no MoE experts),
# so these are fak's native per-token device cost at a fits-one-GPU scale — an optimistic
# lower-bound, NOT the 753B serving rate (that is the llama.cpp CPU-offload baseline). The
# `scope` field rides in every record so the number cannot be quoted out of its caveat.
set -uo pipefail
NONCE="${1:?usage: dgx_glm_throughput_run.sh <nonce> <gpu> <tag>}"
GPU="${2:-0}"
TAG="${3:-glmt$NONCE}"
WORK=/tmp/fakgpu
LOG=/tmp/fakglm_fetch/throughput.log
JSONF="$WORK/$TAG.throughput.json"
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
  printf '%s' '{"schema":"glm-throughput/1","error":"BUILD_FAIL"}' > "$JSONF"; emit 96; exit 96
fi
echo "[cuda] OK build" >> "$LOG"

# CGO env for `go run -tags cuda`: derive the SAME resolved -I/-L/-rpath set that
# internal/compute/build_cuda.sh uses (it discovers whichever include/lib64/targets dirs
# actually exist, and pins CC/CXX). Hand-rolling a single -L$CUDA_HOME/lib64 was the prior
# bug — on a layout without lib64 (or where nvcc lives under targets/) the link silently
# missed -lfakcuda/-lcudart and glmdsatput started WITHOUT the cuda backend (exit 2,
# "not registered"). Mirror build_cuda.sh exactly so the throughput binary links like the
# witness tests do.
PKG="$PWD/internal/compute"
INC=""
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include"; do
  [ -d "$d" ] && INC="$INC -I$d"
done
LIB="-L$PKG"; RPATH=""; LDPATH=""
[ -d /usr/lib/wsl/lib ] && { RPATH="-Wl,-rpath,/usr/lib/wsl/lib"; LDPATH="/usr/lib/wsl/lib"; }
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/x86_64-linux/lib"; do
  if [ -d "$d" ]; then LIB="$LIB -L$d"; RPATH="${RPATH:+$RPATH }-Wl,-rpath,$d"; LDPATH="${LDPATH:+$LDPATH:}$d"; fi
done
export CGO_ENABLED=1
export CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}"
export CGO_CFLAGS="$INC"
export CGO_LDFLAGS="$LIB $RPATH"
export LD_LIBRARY_PATH="${LDPATH:+$LDPATH:}${LD_LIBRARY_PATH:-}"
echo "[cuda] CGO_LDFLAGS=$CGO_LDFLAGS" >> "$LOG"
# Build the binary ONCE (not go run per config) so a link failure surfaces here, loudly,
# before the sweep — and each config is a fast exec, not a recompile.
if ! go build -tags cuda -o /tmp/fakgpu/glmdsatput ./cmd/glmdsatput >> "$LOG" 2>&1; then
  echo "=== glmdsatput cuda BUILD/LINK FAILED (see above) ===" >> "$LOG"
  printf '%s' '{"schema":"glm-throughput/1","error":"GLMDSATPUT_LINK_FAIL"}' > "$JSONF"; emit 95; exit 95
fi
echo "[cuda] OK glmdsatput binary" >> "$LOG"

# The SWEEP ("all of it"): vary the dimensions that move the native per-token cost curve —
# depth (layers), width (hidden/inter), and DSA selection size (index-topk). Each row is one
# -json run; glmdsatput prints a GLMTPUT_JSON line the recorder folds together.
#   layers hidden heads inter  topk
SWEEP=(
  "8  2048 16 8192  256"
  "8  2048 16 8192  512"
  "16 2048 16 8192  256"
  "16 4096 32 14336 256"
  "32 4096 32 14336 512"
  "8  5120 40 12288 2048"
)
for row in "${SWEEP[@]}"; do
  read -r L H HE I TK <<< "$row"
  echo "=== glmdsatput -json layers=$L hidden=$H heads=$HE inter=$I topk=$TK ===" >> "$LOG"
  /tmp/fakgpu/glmdsatput -backend cuda -json \
    -layers "$L" -hidden "$H" -heads "$HE" -inter "$I" -index-topk "$TK" \
    -decode-prompt 512 -decode-steps 64 -decode-reps 5 >> "$LOG" 2>&1 \
    || echo "=== run FAILED layers=$L hidden=$H rc=$? (continuing) ===" >> "$LOG"
done

NRUNS=$(grep -c "GLMTPUT_JSON" "$LOG" || true)
echo "=== GLM THROUGHPUT DONE head=$HEAD_SHA configs=$NRUNS ===" >> "$LOG"
RC=0; [ "${NRUNS:-0}" -eq 0 ] && RC=1
if JSON=$(python3 tools/glm_throughput_record.py "$LOG" --utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --machine-id dgx 2>> "$LOG" \
           | python3 -c 'import sys,json;json.dump(json.load(sys.stdin),sys.stdout,sort_keys=True,separators=(",",":"))' 2>> "$LOG"); then
  printf '%s' "$JSON" > "$JSONF"; emit "$RC"
else
  printf '%s' '{"schema":"glm-throughput/1","error":"PARSE_FAIL"}' > "$JSONF"; emit 1; exit 1
fi
