#!/usr/bin/env bash
# run_926_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #926
# (the AWQ 4-bit device matmul cpuref-parity witness: AWQMatMul / AWQBatchedMatMul, kernels
# fcuda_awq_gemv / fcuda_awq_gemm. The weight stays NARROW in VRAM — nibble-packed 4-bit codes,
# 2/byte, with a per-channel f32 scale + symmetric zero-point 8 — and the GEMM dequant-fuses the
# nibble into the tile, F32 accumulate, so the rest of the op chain stays f32 and unchanged).
#
# This is the #905 selection: of every device op family, AWQ was the ONLY one with no recorded
# cosine floor and no acceptance witness — the single HARD defect in CUDA-DEV-SCORECARD
# cpuref_parity_coverage. cuda-dev.md "Honest residuals" named closing it as: add a
# cudaAWQCosineMin constant + a run_*-style acceptance script. This script is the latter; the
# constant in cuda.go records the floor, this run measures it.
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc + cuBLAS) AND a reachable NVIDIA GPU:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the GPU server, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA toolkit and
#   the GPU quota is walled, so the GEMM RUN + the cosine verdict + the tok/s are the explicit
#   residual handed off here. The Go + cgo of the AWQ path already compiles and type-checks on the
#   dev host (`go build ./...` green, `go vet -tags cuda ./internal/compute/` green — the
#   #479/#482/#483/#484/#485/#486 bar); only the nvcc link and the device execution below need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` #926 witness:
#     TestCUDAAWQMatMulApproxMatchesRef           (internal/compute/cuda_awq_test.go)
#     TestCUDAAWQBatchedMatMulApproxMatchesRef
#   which assert, against the cpuref f32 Reference and under the cuda backend's Approx class, that
#   the device AWQ 4-bit GEMM matches within its OWN RECORDED cosine floor — cudaAWQCosineMin
#   (0.995, the same 4-bit dequant-fused class as Q4_K; see the constant in cuda.go for WHY) —
#   with argmax-exact on the decode GEMV. A skip (no reachable GPU) is NOT a pass.
#   Then it runs the AWQ GEMM benchmark beside the F32 baseline and prints the throughput delta:
#     BenchmarkCUDABatchedMatMulF32  (F32 SGEMM device path — the baseline)
#     BenchmarkCUDAAWQBatchedMatMul (resident 4-bit AWQ weight, dequant fused into the tile)
#   Exits 0 only if the witness PASSES on the device; non-zero on any build/test failure or skip.
#
# WHAT IS RECORDED, NOT CLAIMED
#   This script is the authority for the AWQ cosine verdict and the AWQ-vs-f32 tok/s. The build
#   host only RECORDS the threshold (cudaAWQCosineMin) and confirms the path type-checks; it does
#   NOT and cannot assert the cosine passes or that the AWQ GEMM beat f32 — this run does. The
#   floor is a reasoned target derived from the analogous Q4_K dequant-fused lane; the first
#   successful run here records the realized value.
#
# NOTE ON tok/s: the bench GEMM (see fp16BenchDims) is a single square-ish matmul at a prefill tile,
#   sized to exercise the kernel — it isolates the AWQ-vs-SGEMM delta, not a full-model number.
#   The full-model decode tok/s gate lives with the model bench harness / the parity umbrella (#480).
#
# USAGE
#   bash tools/run_926_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_RE='^TestCUDAAWQ(MatMul|BatchedMatMul)ApproxMatchesRef$'
BENCH_RE='^BenchmarkCUDA(BatchedMatMulF32|AWQBatchedMatMul)$'

echo "== #926 AWQ 4-bit device matmul on-GPU acceptance =="
echo "[926] repo root : $MOD_DIR"
echo "[926] witness   : AWQ MatMul + BatchedMatMul gates (-tags cuda)"
echo "[926] benchmarks: BenchmarkCUDABatchedMatMulF32 / ...AWQBatchedMatMul"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[926] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[926] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[926] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
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
echo "[926] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[926] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

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
LOG="$(mktemp -t fak926.XXXXXX.log)"
echo "[926] go test -tags cuda -run '${TEST_RE}' -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "${TEST_RE}" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== #926 witness verdict =="
# surface the witness lines the tests log (per-dtype cosine, the gate, any SKIP).
grep -aE "AWQ MatMul|AWQ BatchedMatMul|cosine|argmax-exact|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[926] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[926] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

if [ "$rc" -ne 0 ]; then
  echo "[926] FAIL: the #926 witness did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[926] PASS: device AWQ 4-bit GEMM == cpuref f32 within the recorded cosine floor; weight resident at ~0.5 byte/elem."

# ---- GEMM throughput measure + AWQ-vs-f32 delta -----------------------------------
echo
echo "== GEMM throughput (F32 SGEMM vs AWQ 4-bit) =="
BLOG="$(mktemp -t fak926bench.XXXXXX.log)"
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -run '^$' -bench "$BENCH_RE" -benchtime=2s ./internal/compute/ ) 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e

f32_ns=$(awk '/BenchmarkCUDABatchedMatMulF32/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
awq_ns=$(awk '/BenchmarkCUDAAWQBatchedMatMul/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
if [ -n "$f32_ns" ]; then
  # fp16BenchDims() = out=4096 in=4096 P=512 -> 2*out*in*P flops per GEMM.
  awk -v f32="$f32_ns" -v awq="$awq_ns" 'BEGIN{
    flop=2.0*4096*4096*512;
    printf "[926] F32 SGEMM : %10.3f ms/GEMM = %8.1f GFLOP/s\n", f32/1e6, flop/f32;
    if (awq != "") printf "[926] AWQ  GEMM : %10.3f ms/GEMM = %8.1f GFLOP/s  (%.2fx vs F32)\n", awq/1e6, flop/awq, f32/awq;
  }'
  echo "[926] NOTE: these are correctness kernels (not tiled/tensor-core); the win is VRAM/bandwidth (a ~0.5-byte/elem resident weight), not raw GFLOP/s. Full-model decode tok/s lives with the parity umbrella (#480)."
else
  echo "[926] NOTE: could not parse benchmark ns/op (bench rc=$brc); throughput delta unavailable." >&2
fi

rm -f "$LOG" "$BLOG"
echo "[926] DONE: #926 AWQ 4-bit device matmul witness PASSED on the device."
exit 0
