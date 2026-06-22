#!/usr/bin/env bash
# run_482_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #482
# (async CUDA backend: ops enqueue on a stream and return unready Buffers; Read/Argmax are the
# ONLY host fences; Argmax runs on-device returning just the token id; Caps.Async=true).
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc) AND a reachable NVIDIA GPU:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the lab DGX, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA toolkit
#   and the GPU quota is walled, so the test RUN is the explicit residual handed off here. The
#   Go + cgo of the async backend already compiles and type-checks on the dev host
#   (`go build ./...` green, `go vet -tags cuda ./internal/compute/` green — the #479 bar);
#   only the nvcc link and the device execution + tok/s measure below need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` async witness:
#     TestCUDAAsyncArgmaxParityAndResidency  (internal/compute/cuda_async_test.go)
#   which, per prompt + generated token, asserts under the cuda backend's Approx class:
#     1. PARITY — the greedy-decode token id from the ASYNC path (on-device Argmax over the
#        device-resident logits) EQUALS the id from the SYNCHRONOUS path (Read the full logits
#        vector host-ward, then host argmax);
#     2. NO FULL-LOGITS HOST COPY — across an async step the device->host byte counter reads
#        exactly the token id (one int), NOT vocab*4, AND the logits Buffer is Ready()==false
#        (device-resident, unfenced) until the Argmax fence, then flips to Ready() after it.
#   Then it runs two decode benchmarks and prints the sync-vs-async tok/s delta:
#     BenchmarkCUDASyncDecode   (Read full logits + host argmax per token)
#     BenchmarkCUDAAsyncDecode  (logits stay resident; on-device Argmax returns just the id)
#   Exits 0 only if the witness PASSES on the device; non-zero on any build or test failure or
#   if the test SKIPPED (no reachable GPU — a skip is not a pass).
#
# USAGE
#   bash tools/run_482_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_NAME='TestCUDAAsyncArgmaxParityAndResidency'
BENCH_RE='^BenchmarkCUDA(Sync|Async)Decode$'

echo "== #482 async-backend on-GPU acceptance =="
echo "[482] repo root : $MOD_DIR"
echo "[482] witness   : $TEST_NAME (-tags cuda, ./internal/compute/)"
echo "[482] benchmarks: BenchmarkCUDASyncDecode / BenchmarkCUDAAsyncDecode"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[482] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[482] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[482] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
    exit 3
  fi
fi

# include / lib / rpath from whichever CUDA dirs actually exist (micromamba vs DLVM/DGX).
INC=""
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include"; do
  [ -d "$d" ] && INC="$INC -I$d"
done
LIB="-L$PKG_DIR"; RPATH=""; LDPATH=""
if [ -d /usr/lib/wsl/lib ]; then RPATH="-Wl,-rpath,/usr/lib/wsl/lib"; LDPATH="/usr/lib/wsl/lib"; fi
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/x86_64-linux/lib"; do
  if [ -d "$d" ]; then LIB="$LIB -L$d"; RPATH="${RPATH:+$RPATH }-Wl,-rpath,$d"; LDPATH="${LDPATH:+$LDPATH:}$d"; fi
done

ARCH="${FAK_CUDA_ARCH:-sm_89}"
case "$ARCH" in sm_*) ;; *) ARCH="sm_$ARCH";; esac

# ---- compile the CUDA kernels into libfakcuda.a -----------------------------------
echo "[482] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[482] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

# ---- cgo env for the `-tags cuda` test/bench link ---------------------------------
export PATH="/usr/local/go/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export CGO_ENABLED=1
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CGO_CFLAGS="$INC"
export CGO_LDFLAGS="$LIB $RPATH"
export LD_LIBRARY_PATH="${LDPATH:+$LDPATH:}${LD_LIBRARY_PATH:-}"

# ---- run the witness on the device ------------------------------------------------
LOG="$(mktemp -t fak482.XXXXXX.log)"
echo "[482] go test -tags cuda -run ^${TEST_NAME}\$ -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "^${TEST_NAME}\$" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== async-witness verdict =="
# surface the witness lines the test logs (parity, the per-token byte witness, any SKIP).
grep -aE "async witness|async==sync|B/token|Ready|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[482] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[482] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

if [ "$rc" -ne 0 ]; then
  echo "[482] FAIL: the async witness did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[482] PASS: async on-device Argmax ids == synchronous device-path ids, and only the token id crosses host-ward per step."

# ---- decode tok/s measure + sync-vs-async delta -----------------------------------
echo
echo "== decode tok/s (sync vs async) =="
BLOG="$(mktemp -t fak482bench.XXXXXX.log)"
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -run '^$' -bench "$BENCH_RE" -benchtime=2s ./internal/compute/ ) 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e

sync_ns=$(awk '/BenchmarkCUDASyncDecode/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
async_ns=$(awk '/BenchmarkCUDAAsyncDecode/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
if [ -n "$sync_ns" ] && [ -n "$async_ns" ]; then
  awk -v s="$sync_ns" -v a="$async_ns" 'BEGIN{
    st=1e9/s; at=1e9/a;
    printf "[482] sync  decode: %8.3f ms/token = %8.1f tok/s\n", s/1e6, st;
    printf "[482] async decode: %8.3f ms/token = %8.1f tok/s\n", a/1e6, at;
    printf "[482] delta (async - sync): %+.1f tok/s (%.2fx)\n", at-st, at/st;
  }'
else
  echo "[482] NOTE: could not parse benchmark ns/op (bench rc=$brc); tok/s delta unavailable." >&2
fi

rm -f "$LOG" "$BLOG"
echo "[482] DONE: #482 async witness PASSED on the device."
exit 0
