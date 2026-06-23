#!/usr/bin/env bash
# dgx_glm_gpu_witness.sh — run the on-device #86 (partial) witness: GLM-5.2's (glm_moe_dsa)
# MoE/FFN + head GEMMs executing on the GPU pure kernel (k_q8_gemm) via the compute.Backend, vs the
# all-host CPU Q8 forward (argmax-exact within the Approx cosine floor). Isolated run of
# TestCUDAGLMMoeDsaBackendForward — NOT the full -tags cuda suite (which has a separate combined-run
# graph-path panic). Self-backgrounds; poll /tmp/fakglm/run.log + /tmp/fakglm/DONE.<rc>.
# Env: FAK_CUDA_ARCH=sm_80  CUDA_HOME=/usr/local/cuda  FAK_GPU=1
set -uo pipefail
WORK=/tmp/fakglm
SELF="$0"
if [ "${FAKGLM_BG:-}" != "1" ]; then
  mkdir -p "$WORK"; rm -f "$WORK"/DONE.* 2>/dev/null || true
  cp -f "$SELF" "$WORK/run.sh" 2>/dev/null || true
  FAKGLM_BG=1 setsid bash "$WORK/run.sh" </dev/null >"$WORK/run.log" 2>&1 &
  echo "LAUNCHED pid $! -> $WORK/run.log"; exit 0
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export CUDA_VISIBLE_DEVICES="${FAK_GPU:-1}"
export PATH="/usr/local/go/bin:$CUDA_HOME/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export HOME="${HOME:-/root}"; export GOCACHE="${GOCACHE:-/tmp/gocache}"; export GOPATH="${GOPATH:-/tmp/gopath}"
mkdir -p "$GOCACHE" "$GOPATH"
SRC="$WORK/src"
say() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }
say "clone origin/main"
rm -rf "$SRC"
git clone --depth 1 https://github.com/anthony-chaudhary/fak.git "$SRC" || { echo clone >"$WORK/DONE.97"; exit 97; }
cd "$SRC" || { echo nosrc >"$WORK/DONE.97"; exit 97; }
say "HEAD $(git rev-parse --short HEAD)"
say "build libfakcuda.a ($FAK_CUDA_ARCH)"
bash internal/compute/build_cuda.sh build
PKG="$SRC/internal/compute"
export CGO_ENABLED=1
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$PKG -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Two isolated runs (separate processes, so the global cuda graph/exec state never leaks between
# them — the reason this witness avoids the full -tags cuda suite's combined-run graph-path panic):
#   1) the all-device GLM-DSA forward (#86/#413): MoE/FFN + head + DSA projections + sparse attend.
#   2) the --n-cpu-moe CPU-offload hybrid: experts host-resident, dense + router + DSA attention on
#      the GPU (k_q8_gemm + k_dsa_sparse_attend), argmax-exact vs the all-host CPU Q8 reference.
say "go test -tags cuda -run TestCUDAGLMMoeDsaBackendForward ./internal/model/ -v"
go test -tags cuda -count=1 -v -run '^TestCUDAGLMMoeDsaBackendForward$' ./internal/model/
rc1=$?
say "go test -tags cuda -run TestCUDAGLMDsaCPUOffloadHybrid ./internal/model/ -v"
go test -tags cuda -count=1 -v -run '^TestCUDAGLMDsaCPUOffloadHybrid$' ./internal/model/
rc2=$?
rc=$rc1; [ "$rc2" -ne 0 ] && rc=$rc2
say "GLM GPU WITNESS DONE rc1=$rc1 (all-device forward) rc2=$rc2 (cpu-offload hybrid) -> rc=$rc"
echo done >"$WORK/DONE.$rc"
exit "$rc"
