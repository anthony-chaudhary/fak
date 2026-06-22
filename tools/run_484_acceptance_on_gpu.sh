#!/usr/bin/env bash
# run_484_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #484
# (fp16 compute path: cuBLAS tensor-core HGEMM with weights narrowed to F16 at H2D under
# Caps.UploadDtype, plus the `Layout` repack at H2D — RowMajor op_T vs the ColMajor transpose
# repack op_N; the GEMM accumulates in F32 so the rest of the op chain stays f32 and unchanged).
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc + cuBLAS) AND a reachable NVIDIA GPU with tensor cores:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the GPU server, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA toolkit
#   and the GPU quota is walled, so the HGEMM RUN + the cosine verdict + the tok/s are the
#   explicit residual handed off here. The Go + cgo of the fp16 path already compiles and
#   type-checks on the dev host (`go build ./...` green, `go vet -tags cuda ./internal/compute/`
#   green — the #479/#482/#483 bar); only the nvcc link and the device execution + tok/s below
#   need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` fp16 witness:
#     TestCUDAMatMulF16ApproxMatchesRef         (internal/compute/cuda_fp16_test.go)
#     TestCUDABatchedMatMulF16ApproxMatchesRef
#   which assert, against the cpuref f32 Reference and under the cuda backend's Approx class, that
#   the device fp16 GEMM (F16-resident weight, tensor-core HGEMM, F32 accumulate) matches within
#   the RECORDED fp16 cosine floor cudaFP16CosineMin (looser than the Q8 lane's 0.999 — see the
#   constant in cuda.go for WHY), for BOTH weight layouts (RowMajor op_T and the ColMajor
#   transpose-repack op_N — the `Layout` repack at H2D). A skip (no reachable GPU) is NOT a pass.
#   Then it runs two GEMM benchmarks and prints the F32-vs-F16 throughput delta:
#     BenchmarkCUDABatchedMatMulF32  (F32 SGEMM device path — the baseline)
#     BenchmarkCUDABatchedMatMulF16  (the same GEMM on tensor-core HGEMM)
#   Exits 0 only if the witness PASSES on the device; non-zero on any build/test failure or skip.
#
# WHAT IS RECORDED, NOT CLAIMED
#   This script is the authority for the fp16 cosine verdict and the fp16-vs-f32 tok/s. The build
#   host only RECORDS the threshold (cudaFP16CosineMin) and confirms the path type-checks; it does
#   NOT and cannot assert the cosine passes or that fp16 beat f32 — this run does.
#
# NOTE ON tok/s: the bench GEMM (see fp16BenchDims) is a single square-ish matmul at a prefill
#   tile, sized to exercise tensor cores — it isolates the HGEMM-vs-SGEMM delta, not a full-model
#   number. Compare the realized fp16 GEMM throughput against the F16 cell of
#   internal/model/bench_llamacpp.py (llama.cpp's F16 GGML kernels); the full-model decode tok/s
#   gate lives with the model bench harness / the parity umbrella (#480).
#
# USAGE
#   bash tools/run_484_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_RE='^TestCUDA(MatMul|BatchedMatMul)F16ApproxMatchesRef$'
BENCH_RE='^BenchmarkCUDABatchedMatMulF(32|16)$'

echo "== #484 fp16 (tensor-core HGEMM) on-GPU acceptance =="
echo "[484] repo root : $MOD_DIR"
echo "[484] witness   : TestCUDAMatMulF16ApproxMatchesRef + TestCUDABatchedMatMulF16ApproxMatchesRef (-tags cuda)"
echo "[484] benchmarks: BenchmarkCUDABatchedMatMulF32 / BenchmarkCUDABatchedMatMulF16"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[484] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[484] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[484] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
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
echo "[484] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[484] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

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
LOG="$(mktemp -t fak484.XXXXXX.log)"
echo "[484] go test -tags cuda -run '${TEST_RE}' -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "${TEST_RE}" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== fp16-witness verdict =="
# surface the witness lines the test logs (cosine per layout, the gate, any SKIP).
grep -aE "fp16 MatMul|fp16 BatchedMatMul|fp16 witness|cosine|UploadDtype|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[484] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[484] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

if [ "$rc" -ne 0 ]; then
  echo "[484] FAIL: the fp16 witness did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[484] PASS: device fp16 HGEMM == cpuref f32 within the recorded fp16 cosine floor, both layouts."

# ---- GEMM throughput measure + F32-vs-F16 delta -----------------------------------
echo
echo "== GEMM throughput (F32 SGEMM vs F16 HGEMM) =="
BLOG="$(mktemp -t fak484bench.XXXXXX.log)"
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -run '^$' -bench "$BENCH_RE" -benchtime=2s ./internal/compute/ ) 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e

f32_ns=$(awk '/BenchmarkCUDABatchedMatMulF32/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
f16_ns=$(awk '/BenchmarkCUDABatchedMatMulF16/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
if [ -n "$f32_ns" ] && [ -n "$f16_ns" ]; then
  # fp16BenchDims() = out=4096 in=4096 P=512 -> 2*out*in*P flops per GEMM.
  awk -v f32="$f32_ns" -v f16="$f16_ns" 'BEGIN{
    flop=2.0*4096*4096*512;
    g32=flop/f32; g16=flop/f16;   # ns -> GFLOP/s (flop / ns == GFLOP/s)
    printf "[484] F32 SGEMM : %10.3f ms/GEMM = %8.1f GFLOP/s\n", f32/1e6, g32;
    printf "[484] F16 HGEMM : %10.3f ms/GEMM = %8.1f GFLOP/s\n", f16/1e6, g16;
    printf "[484] speedup (F16 vs F32): %.2fx\n", f32/f16;
  }'
  echo "[484] compare the F16 GFLOP/s above against the F16 cell of internal/model/bench_llamacpp.py (llama.cpp F16 GGML)."
else
  echo "[484] NOTE: could not parse benchmark ns/op (bench rc=$brc); throughput delta unavailable." >&2
fi

rm -f "$LOG" "$BLOG"
echo "[484] DONE: #484 fp16 HGEMM witness PASSED on the device."
exit 0
