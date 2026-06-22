#!/usr/bin/env bash
# run_483_acceptance_on_gpu.sh — the on-GPU acceptance gate for issue #483
# (CUDA Graphs / capture for the batch-1 decode step: the FIXED per-token decode op sequence
# RMSNorm→QKV→RoPE→Attention→o_proj→FFN is captured ONCE into a cudaGraph_t on g_stream and
# replayed each step as a single cudaGraphLaunch instead of N kernel launches; the cuda backend
# advertises Caps.GraphCompile=true when that path is live, falling back to the synchronous
# per-op core otherwise).
#
# WHAT NODE THIS RUNS ON
#   Any host with a CUDA toolkit (nvcc) AND a reachable NVIDIA GPU:
#     • a dev box / WSL with a consumer card (e.g. RTX 4070 — enough for correctness),
#     • the lab DGX, or
#     • a GCP Deep-Learning-VM GPU instance (CUDA at /usr/local/cuda).
#   It CANNOT run on the win32 dev host that produced this code: that host has no CUDA toolkit
#   and the GPU quota is walled, so the capture+replay RUN is the explicit residual handed off
#   here. The Go + cgo of the graph path already compiles and type-checks on the dev host
#   (`go build ./...` green, `go vet -tags cuda ./internal/compute/` green — the #479/#482 bar);
#   only the nvcc link and the device execution + tok/s measure below need a GPU.
#
# WHAT IT PROVES
#   Builds the CUDA kernels (nvcc -> libfakcuda.a) and runs the `-tags cuda` graph witness:
#     TestCUDAGraphDecodeParity  (internal/compute/cuda_graph_test.go)
#   which, per prompt + generated token, asserts under the cuda backend's Approx class that the
#   CAPTURED decode path (one graph launch/token) is numerically UNCHANGED vs the EAGER device
#   path (N launches/token):
#     1. ARGMAX-EXACT — the graph-replayed next token id EQUALS the eager device next token id;
#     2. LOGIT COSINE ≥ 0.999 — the graph-replayed logits match the eager logits within the gate;
#   and that capture actually ENGAGED (GraphBegin consented) so the witness is not vacuous. It
#   also checks Caps.GraphCompile tracks graphEnabled (advertise/fallback contract,
#   TestCUDAGraphCompileCapGated).
#   Then it runs two decode benchmarks and prints the no-capture-vs-capture tok/s delta:
#     BenchmarkCUDADecodeNoCapture  (each op launched individually)
#     BenchmarkCUDADecodeCapture    (the op stream captured once, replayed as one launch)
#   Exits 0 only if the witness PASSES on the device; non-zero on any build or test failure or
#   if the test SKIPPED (no reachable GPU — a skip is not a pass).
#
# NOTE ON tok/s: the synthetic witness model is tiny (a few small layers), so its absolute tok/s
#   are not the headline Qwen2.5-7B number from the issue — they isolate the launch-overhead
#   delta the graph removes. The full-model decode tok/s gate lives with the model bench harness.
#
# USAGE
#   bash tools/run_483_acceptance_on_gpu.sh
#   env knobs (same as internal/compute/build_cuda.sh):
#     FAK_CUDA_ARCH=sm_89|sm_90|sm_100   GPU arch (default sm_89; "89" also accepted)
#     CUDA_HOME=/usr/local/cuda          CUDA toolkit root (default ~/cudaenv, else PATH nvcc)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$MOD_DIR/internal/compute"
TEST_RE='^TestCUDAGraph(DecodeParity|CompileCapGated)$'
WITNESS='TestCUDAGraphDecodeParity'
BENCH_RE='^BenchmarkCUDADecode(NoCapture|Capture)$'

echo "== #483 CUDA-graph decode on-GPU acceptance =="
echo "[483] repo root : $MOD_DIR"
echo "[483] witness   : $WITNESS (-tags cuda, ./internal/compute/)"
echo "[483] benchmarks: BenchmarkCUDADecodeNoCapture / BenchmarkCUDADecodeCapture"

# ---- resolve the CUDA toolchain (mirrors internal/compute/build_cuda.sh, portable) -
CUDA_HOME="${CUDA_HOME:-$HOME/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    echo "[483] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "[483] FAIL: no nvcc at $NVCC and none on PATH — this is not a CUDA node." >&2
    echo "[483] Run the CUDA-toolchain setup first (see internal/compute/setup_cuda_wsl.sh)." >&2
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
echo "[483] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[483] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

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
LOG="$(mktemp -t fak483.XXXXXX.log)"
echo "[483] go test -tags cuda -run '${TEST_RE}' -v ./internal/compute/ ..."
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -v -run "${TEST_RE}" ./internal/compute/ ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "== graph-witness verdict =="
# surface the witness lines the test logs (parity, capture engagement, any SKIP).
grep -aE "graph witness|argmax-exact|cosine|GraphCompile|RUN |PASS|FAIL|ok |--- (FAIL|SKIP|PASS)|no reachable CUDA" "$LOG" || true
echo

if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$LOG"; then
  echo "[483] INCONCLUSIVE: the test SKIPPED — no reachable CUDA device on this node." >&2
  echo "[483] A skip is not a pass. Run on a node with a live GPU." >&2
  rm -f "$LOG"
  exit 4
fi

if [ "$rc" -ne 0 ]; then
  echo "[483] FAIL: the graph witness did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[483] PASS: graph-replayed decode == eager device decode (argmax-exact, logit cosine ≥ 0.999); capture engaged."

# ---- decode tok/s measure + no-capture-vs-capture delta ---------------------------
echo
echo "== decode tok/s (no-capture vs capture) =="
BLOG="$(mktemp -t fak483bench.XXXXXX.log)"
set +e
( cd "$MOD_DIR"
  go test -tags cuda -count=1 -run '^$' -bench "$BENCH_RE" -benchtime=2s ./internal/compute/ ) 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e

nocap_ns=$(awk '/BenchmarkCUDADecodeNoCapture/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
cap_ns=$(awk '/BenchmarkCUDADecodeCapture/{for(i=1;i<=NF;i++) if($i=="ns/op") print $(i-1)}' "$BLOG" | tail -1)
if [ -n "$nocap_ns" ] && [ -n "$cap_ns" ]; then
  awk -v nc="$nocap_ns" -v c="$cap_ns" 'BEGIN{
    nct=1e9/nc; ct=1e9/c;
    printf "[483] no-capture decode: %8.3f ms/token = %8.1f tok/s\n", nc/1e6, nct;
    printf "[483] capture    decode: %8.3f ms/token = %8.1f tok/s\n", c/1e6, ct;
    printf "[483] delta (capture - no-capture): %+.1f tok/s (%.2fx)\n", ct-nct, ct/nct;
  }'
else
  echo "[483] NOTE: could not parse benchmark ns/op (bench rc=$brc); tok/s delta unavailable." >&2
fi

rm -f "$LOG" "$BLOG"
echo "[483] DONE: #483 CUDA-graph decode witness PASSED on the device."
exit 0
