#!/usr/bin/env bash
# run_486_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #486
# (fused flash/paged-attention CUDA kernel: replace the naive one-block-per-head decode kernel —
# which materializes a full scores[nPos] row in global scratch and makes four passes over it — with
# a FlashAttention online-softmax kernel that streams the KV window with a running (max, sum, acc)
# so NO scores row is materialized and no per-call global scratch is allocated; causal/grp/scale are
# consumed as kernel params; Caps.FusedAttn=true).
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc + cuBLAS) AND a reachable NVIDIA GPU:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the lab DGX, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA toolkit and
#   the GPU quota is walled, so the flash-attn RUN + the logit-cosine verdict + the fused-vs-naive
#   speedup are the explicit residual handed off here. The Go + cgo of the flash path already
#   compiles and type-checks on the dev host (`go build ./...` green for the compute lane, `go vet
#   -tags cuda ./internal/compute/` green — the #479/#482/#483/#484/#485 bar); only the nvcc link and
#   the device execution below need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` #486 witness:
#     TestCUDAFlashAttentionMatchesRef   (internal/compute/cuda_flash_test.go) — fused flash vs the
#                                         cpuref f32 Reference AND vs the retained naive device kernel,
#                                         across MHA / GQA / MQA, cosine >= cudaFlashAttnCosineMin.
#     TestCUDAForwardMatchesRef          (internal/compute/cuda_test.go) — the multi-layer Llama decode
#                                         forward whose per-layer Attention IS now the flash kernel:
#                                         argmax-exact + logit cosine >= cudaFlashAttnCosineMin.
#   which assert, against the cpuref f32 Reference and under the cuda backend's Approx class, that the
#   fused flash kernel reproduces the reference softmax(scale*q.k)*V within the RECORDED cosine floor
#   cudaFlashAttnCosineMin (0.999 — see the constant's comment in cuda.go for WHY: the online-softmax
#   reorder differs from the reference only in f32 reduction order, no narrowed operand), with the
#   decode argmax EXACT. A skip (no reachable GPU) is NOT a pass.
#   Then it runs the fused-vs-naive microbench at a representative decode shape and prints the speedup:
#     BenchmarkCUDAFlashAttention   (fused flash/online-softmax — the live Attention path)
#     BenchmarkCUDANaiveAttention   (the retained naive baseline — full global scores row, four passes)
#   Exits 0 only if the witness PASSES on the device; non-zero on any build/test failure or skip.
#
# WHAT IS RECORDED, NOT CLAIMED
#   This script is the authority for the logit-cosine verdict and the fused-vs-naive speedup. The
#   build host only RECORDS the threshold (cudaFlashAttnCosineMin) and confirms the path type-checks;
#   it does NOT and cannot assert the cosine passes or that the flash kernel beat the naive one —
#   this run does.
#
# NOTE ON the microbench: BenchmarkCUDA{Flash,Naive}Attention time ONE decode Attention op at the
#   representative shape (32 query heads / 8 KV heads, head dim 128, 1024-token KV window) — they
#   isolate the kernel delta, not a full-model number. The full-model decode tok/s gate lives with the
#   model bench harness / the parity umbrella (#480).
#
# USAGE
#   bash tools/run_486_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_RE='^TestCUDA(FlashAttentionMatchesRef|ForwardMatchesRef)$'
BENCH_RE='^BenchmarkCUDA(FlashAttention|NaiveAttention)$'

echo "== #486 fused flash/online-softmax attention on-GPU acceptance =="
echo "[486] repo root : $MOD_DIR"
echo "[486] witness   : TestCUDAFlashAttentionMatchesRef + TestCUDAForwardMatchesRef (-tags cuda)"
echo "[486] benchmarks: BenchmarkCUDAFlashAttention vs BenchmarkCUDANaiveAttention"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[486] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[486] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[486] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
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
echo "[486] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[486] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

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
LOG="$(mktemp -t fak486.XXXXXX.log)"
echo "[486] go test -tags cuda -run '${TEST_RE}' -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "${TEST_RE}" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== #486 flash-attn witness verdict =="
# surface the witness lines the tests log (per-config cosine, the gate, argmax-exact, any SKIP).
grep -aE "flash attention parity|flash witness|forward parity|cosine|argmax|FusedAttn|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[486] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[486] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

if [ "$rc" -ne 0 ]; then
  echo "[486] FAIL: the #486 flash-attn witness did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[486] PASS: fused flash attention == cpuref within the recorded cosine floor; decode argmax exact."

# ---- fused-vs-naive throughput delta ----------------------------------------------
echo
echo "== decode-attention throughput (fused flash vs naive) =="
BLOG="$(mktemp -t fak486bench.XXXXXX.log)"
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -run '^$' -bench "$BENCH_RE" -benchtime=2s ./internal/compute/ ) 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e

flash_ns=$(awk '/BenchmarkCUDAFlashAttention/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
naive_ns=$(awk '/BenchmarkCUDANaiveAttention/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
if [ -n "$flash_ns" ] && [ -n "$naive_ns" ]; then
  awk -v flash="$flash_ns" -v naive="$naive_ns" 'BEGIN{
    printf "[486] naive attention : %10.3f us/op\n", naive/1e3;
    printf "[486] flash attention : %10.3f us/op  (%.2fx vs naive)\n", flash/1e3, naive/flash;
    if (flash < naive) printf "[486] VERDICT: fused flash BEATS naive at the representative shape (%.2fx faster).\n", naive/flash;
    else               printf "[486] VERDICT: fused flash did NOT beat naive (%.2fx) — investigate before claiming the speedup.\n", naive/flash;
  }'
  echo "[486] NOTE: this is the single-op decode-attention delta (32 heads / 8 KV / hd 128 / 1024-pos window); full-model decode tok/s lives with the parity umbrella (#480)."
else
  echo "[486] NOTE: could not parse benchmark ns/op (bench rc=$brc); throughput delta unavailable." >&2
fi

rm -f "$LOG" "$BLOG"
echo "[486] DONE: #486 fused flash/online-softmax attention witness PASSED on the device."
exit 0
