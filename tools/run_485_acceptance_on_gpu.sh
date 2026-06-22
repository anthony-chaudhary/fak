#!/usr/bin/env bash
# run_485_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #485
# (native Q8_0 / Q4_K device matmul, no dequant-to-f32: the weight stays NARROW in VRAM — int8
# codes / Q4_K super-block bytes — and the GEMM consumes it directly; the activation is quantized
# on-device for Q8_0, the dequant is fused into the GEMM tile for Q4_K. F32 accumulate, so the rest
# of the op chain stays f32 and unchanged).
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc + cuBLAS) AND a reachable NVIDIA GPU:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the GPU server, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA toolkit and
#   the GPU quota is walled, so the GEMM RUN + the per-dtype cosine verdicts + the VRAM numbers + the
#   tok/s are the explicit residual handed off here. The Go + cgo of the quantized path already
#   compiles and type-checks on the dev host (`go build ./...` green, `go vet -tags cuda
#   ./internal/compute/` green — the #479/#482/#483/#484 bar); only the nvcc link and the device
#   execution below need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` #485 witness:
#     TestCUDAQ8MatMulApproxMatchesRef          (internal/compute/cuda_quant_test.go)
#     TestCUDAQ8BatchedMatMulApproxMatchesRef
#     TestCUDAQ4KMatMulApproxMatchesRef         (its OWN gate instance — distinct numeric path)
#     TestCUDAQ4KBatchedMatMulApproxMatchesRef
#     TestCUDAQuantVRAMWitness                  (resident weight ≈ int8/int4 size, not f32)
#   which assert, against the cpuref f32 Reference and under the cuda backend's Approx class, that
#   each device quantized GEMM matches within its OWN RECORDED cosine floor — cudaQ8CosineMin (0.999)
#   for Q8_0, cudaQ4KCosineMin (0.995) for Q4_K (looser, see the constants in cuda.go for WHY) — with
#   argmax-exact on the decode GEMV, and that the resident weight is int8/int4-sized. A skip (no
#   reachable GPU) is NOT a pass.
#   Then it runs the quantized GEMM benchmarks beside the F32 baseline and prints the throughput
#   delta:
#     BenchmarkCUDABatchedMatMulF32  (F32 SGEMM device path — the baseline)
#     BenchmarkCUDAQ8BatchedMatMul   (resident int8 weight, on-device activation quant)
#     BenchmarkCUDAQ4KBatchedMatMul  (resident Q4_K weight, dequant fused into the tile)
#   Exits 0 only if the witness PASSES on the device; non-zero on any build/test failure or skip.
#
# WHAT IS RECORDED, NOT CLAIMED
#   This script is the authority for the per-dtype cosine verdicts, the VRAM numbers, and the
#   quantized-vs-f32 tok/s. The build host only RECORDS the thresholds (cudaQ8CosineMin /
#   cudaQ4KCosineMin) and confirms the path type-checks; it does NOT and cannot assert the cosines
#   pass, that VRAM shrank, or that the quantized GEMM beat f32 — this run does.
#
# NOTE ON tok/s: the bench GEMM (see fp16BenchDims) is a single square-ish matmul at a prefill tile,
#   sized to exercise the kernels — it isolates the quantized-vs-SGEMM delta, not a full-model number.
#   The full-model decode tok/s gate lives with the model bench harness / the parity umbrella (#480).
#
# USAGE
#   bash tools/run_485_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_RE='^TestCUDA(Q8MatMul|Q8BatchedMatMul|Q4KMatMul|Q4KBatchedMatMul)ApproxMatchesRef$|^TestCUDAQuantVRAMWitness$'
BENCH_RE='^BenchmarkCUDA(BatchedMatMulF32|Q8BatchedMatMul|Q4KBatchedMatMul)$'

echo "== #485 native Q8_0 / Q4_K device matmul on-GPU acceptance =="
echo "[485] repo root : $MOD_DIR"
echo "[485] witness   : Q8_0 + Q4_K MatMul/BatchedMatMul gates + VRAM witness (-tags cuda)"
echo "[485] benchmarks: BenchmarkCUDABatchedMatMulF32 / ...Q8BatchedMatMul / ...Q4KBatchedMatMul"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[485] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[485] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[485] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
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
echo "[485] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[485] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

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
LOG="$(mktemp -t fak485.XXXXXX.log)"
echo "[485] go test -tags cuda -run '${TEST_RE}' -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "${TEST_RE}" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== #485 per-dtype witness verdict =="
# surface the witness lines the tests log (per-dtype cosine, the gate, VRAM bytes, any SKIP).
grep -aE "Q8_0 MatMul|Q8_0 BatchedMatMul|Q4_K MatMul|Q4_K BatchedMatMul|VRAM witness|cosine|argmax-exact|UploadDtype|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[485] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[485] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

if [ "$rc" -ne 0 ]; then
  echo "[485] FAIL: the #485 witness did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[485] PASS: device Q8_0 + Q4_K GEMM == cpuref f32 within each recorded cosine floor; weights resident at int8/int4 size."

# ---- GEMM throughput measure + quantized-vs-f32 delta -----------------------------
echo
echo "== GEMM throughput (F32 SGEMM vs Q8_0 vs Q4_K) =="
BLOG="$(mktemp -t fak485bench.XXXXXX.log)"
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -run '^$' -bench "$BENCH_RE" -benchtime=2s ./internal/compute/ ) 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e

f32_ns=$(awk '/BenchmarkCUDABatchedMatMulF32/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
q8_ns=$(awk '/BenchmarkCUDAQ8BatchedMatMul/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
q4k_ns=$(awk '/BenchmarkCUDAQ4KBatchedMatMul/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
if [ -n "$f32_ns" ]; then
  # fp16BenchDims() = out=4096 in=4096 P=512 -> 2*out*in*P flops per GEMM.
  awk -v f32="$f32_ns" -v q8="$q8_ns" -v q4k="$q4k_ns" 'BEGIN{
    flop=2.0*4096*4096*512;
    printf "[485] F32 SGEMM : %10.3f ms/GEMM = %8.1f GFLOP/s\n", f32/1e6, flop/f32;
    if (q8  != "") printf "[485] Q8_0 GEMM : %10.3f ms/GEMM = %8.1f GFLOP/s  (%.2fx vs F32)\n", q8/1e6,  flop/q8,  f32/q8;
    if (q4k != "") printf "[485] Q4_K GEMM : %10.3f ms/GEMM = %8.1f GFLOP/s  (%.2fx vs F32)\n", q4k/1e6, flop/q4k, f32/q4k;
  }'
  echo "[485] NOTE: these are correctness-kernel GEMMs (not tiled/tensor-core); the win is VRAM/bandwidth (smaller resident weight), not raw GFLOP/s. Full-model decode tok/s lives with the parity umbrella (#480)."
else
  echo "[485] NOTE: could not parse benchmark ns/op (bench rc=$brc); throughput delta unavailable." >&2
fi

rm -f "$LOG" "$BLOG"
echo "[485] DONE: #485 native Q8_0 / Q4_K device matmul witness PASSED on the device."
exit 0
