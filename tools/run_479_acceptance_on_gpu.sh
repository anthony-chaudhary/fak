#!/usr/bin/env bash
# run_479_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #479
# (device-resident KVStore with on-GPU Evict preserving the quarantine witness).
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc) AND a reachable NVIDIA GPU:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the lab DGX, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA
#   toolkit and the GPU quota is walled, so the test RUN is the explicit residual handed
#   off here. The Go + cgo of the on-GPU Evict already compiles and type-checks on the
#   dev host (`go build -tags cuda` / `go vet -tags cuda` green); only the nvcc link and
#   the device execution below need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` port of the
#   evict==never-saw witness over a MIDDLE span:
#     TestCUDAEvictMiddleSpanEqualsNeverSaw  (internal/compute/cuda_evict_test.go)
#   Under the cuda backend's Approx gate (max|Δ| ≤ tol, NOT bit-identity), it asserts:
#     1. a device middle-span Evict == a device cache that NEVER saw the span (K and V) —
#        the on-GPU compaction + single-rotation re-RoPE from device Kraw at the new index
#        reproduces the never-saw cache WITHOUT a host round-trip;
#     2. the suffix survivors actually moved (re-RoPE ran — the MIDDLE-span case, not an
#        end trim), while the prefix stayed byte-for-byte (the quarantine asymmetry,
#        MODEL-ARCH-SEAM §3 O1–O3);
#     3. Host() on a resident KV tensor stays (nil,false) — the cache never leaves VRAM.
#   Exits 0 only if the test PASSES on the device; non-zero on any build or test failure.
#
# USAGE
#   bash tools/run_479_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_NAME='TestCUDAEvictMiddleSpanEqualsNeverSaw'

echo "== #479 on-GPU Evict acceptance =="
echo "[479] repo root : $MOD_DIR"
echo "[479] test      : $TEST_NAME (-tags cuda, ./internal/compute/)"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[479] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[479] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[479] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
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
echo "[479] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[479] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

# ---- cgo env for the `-tags cuda` test link ---------------------------------------
export PATH="/usr/local/go/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export CGO_ENABLED=1
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CGO_CFLAGS="$INC"
export CGO_LDFLAGS="$LIB $RPATH"
export LD_LIBRARY_PATH="${LDPATH:+$LDPATH:}${LD_LIBRARY_PATH:-}"

# ---- run the witness on the device ------------------------------------------------
LOG="$(mktemp -t fak479.XXXXXX.log)"
echo "[479] go test -tags cuda -run ^${TEST_NAME}\$ -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "^${TEST_NAME}\$" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== Approx-gate verdict =="
# surface the witness lines the test logs (max|Δ|, the asymmetry note, and the SKIP if any).
grep -aE "max\|Δ\||evict==never-saw|repositioned|asymmetry|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[479] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[479] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

rm -f "$LOG"
if [ "$rc" -eq 0 ]; then
  echo "[479] PASS: on-GPU middle-span Evict == never-saw under the Approx gate (no host round-trip)."
else
  echo "[479] FAIL: the on-GPU Evict witness did not pass (go test rc=$rc)." >&2
fi
exit "$rc"
