#!/usr/bin/env bash
# dgx_pure_kernel_run.sh — turnkey "run the PURE fak CUDA kernel on a datacenter GPU"
# bootstrap. Clones the current public repo HEAD onto the GPU node, builds the CUDA
# kernels (nvcc -> libfakcuda.a) for the node's arch, and runs the pure-kernel device
# acceptance witnesses (#485 Q8/Q4K GEMM, #486 fused flash attention) + the multi-layer
# decode forward witness (TestCUDAForwardMatchesRef) on the live GPU. A SKIP is not a PASS.
#
# WHY A BOOTSTRAP: the GPU node (the lab 8x A100 server) carries an OLDER rsync of the
# tree whose cuda_kernels.cu predates the Q8/Q4K/flash work; the win32 build host that
# drives this has no CUDA toolkit. So this script fetches the pushed HEAD fresh and runs
# it on the device — the only place the device numbers can be witnessed.
#
# WHY SELF-BACKGROUNDING: the private control bridge that fronts the node times out on a
# multi-minute synchronous exec. This script re-execs itself under nohup on first call
# (writing /tmp/fakpure/run.log), so the driving `exec 'bash run.sh'` returns at once and
# the heavy build/test keeps running on the node; the driver then polls the log + the
# /tmp/fakpure/DONE.<rc> sentinel.
#
# WHAT IS PURE vs NOT (the headline this script witnesses):
#   PURE fak kernels  : Q8_0 / Q4_K / AWQ GEMM (k_q8_gemm/k_q4k_gemm/k_awq_gemm),
#                       flash attention (k_flash_attention), RMSNorm/RoPE/SwiGLU/Add/
#                       Argmax/KV-write. A Q8-quantized decode touches ZERO cuBLAS.
#   NOT pure (cuBLAS) : only the F32 (cublasSgemm) and F16 (cublasGemmEx) GEMM paths.
#   NOT pure (NVIDIA) : the CUDA runtime/driver (cudaMalloc/memcpy/launch) — unavoidable.
#
# Usage (on the node, via the bridge):
#   ship tools/dgx_pure_kernel_run.sh /tmp/fakpure/run.sh
#   exec 'bash /tmp/fakpure/run.sh'
#   exec 'tail -c 2000 /tmp/fakpure/run.log; ls /tmp/fakpure/DONE.* 2>/dev/null'
# Env knobs:
#   FAK_CUDA_ARCH=sm_80   (A100; default sm_80 here)        CUDA_HOME=/usr/local/cuda
#   FAK_GPU=1             (CUDA_VISIBLE_DEVICES; default 1 to dodge a busy GPU0)
#   FAK_REPO_URL=https://github.com/anthony-chaudhary/fak.git
set -uo pipefail

WORK=/tmp/fakpure
SELF="$0"

# ---- self-background on first entry so the bridge exec returns immediately ----------
if [ "${FAKPURE_BG:-}" != "1" ]; then
  mkdir -p "$WORK"
  rm -f "$WORK"/DONE.* 2>/dev/null || true
  cp -f "$SELF" "$WORK/run.sh" 2>/dev/null || true
  # setsid + </dev/null FULLY detaches the worker into its own session, so the control
  # bridge that typed this command does NOT wait on the 20-min build and the session does
  # not stay busy (a plain `nohup ... &` leaves the child in the bridge's process group,
  # which wedges the session's readback until the build ends — the failure that cost hours).
  FAKPURE_BG=1 setsid bash "$WORK/run.sh" </dev/null >"$WORK/run.log" 2>&1 &
  echo "LAUNCHED pid $! -> $WORK/run.log"
  exit 0
fi

# ---- from here on we are the detached worker ----------------------------------------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export CUDA_VISIBLE_DEVICES="${FAK_GPU:-1}"
export PATH="/usr/local/go/bin:$CUDA_HOME/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
# setsid + the bridge's non-interactive shell drop $HOME, so `go` aborts with
# "GOCACHE is not defined and neither $XDG_CACHE_HOME nor $HOME are defined". Pin them.
export HOME="${HOME:-/root}"
export GOCACHE="${GOCACHE:-/tmp/gocache}"
export GOPATH="${GOPATH:-/tmp/gopath}"
mkdir -p "$GOCACHE" "$GOPATH"
REPO_URL="${FAK_REPO_URL:-https://github.com/anthony-chaudhary/fak.git}"
SRC="$WORK/src"

say() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }
rc_global=0
step() { # step <name> <cmd...>  (errexit is intentionally OFF so one failure doesn't abort the rest)
  local name="$1"; shift
  say "BEGIN $name"
  "$@"; local rc=$?
  say "END $name rc=$rc"
  if [ "$rc" -ne 0 ]; then rc_global=$rc; fi
  return 0
}

say "node $(hostname) | arch $FAK_CUDA_ARCH | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES | CUDA_HOME=$CUDA_HOME"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv,noheader || true
"$CUDA_HOME/bin/nvcc" --version | tail -2 || true
/usr/local/go/bin/go version || true

# ---- fetch the pushed HEAD fresh ----------------------------------------------------
say "clone $REPO_URL (depth 1)"
rm -rf "$SRC"
if ! git clone --depth 1 "$REPO_URL" "$SRC"; then
  say "CLONE FAILED — no internet on node? falling back to /srv/fleet/fak (may be stale)"
  SRC=/srv/fleet/fak
fi
cd "$SRC" || { say "no source tree"; echo done >"$WORK/DONE.97"; exit 97; }
KCNT=$(grep -c 'fcuda_q8_matmul_f32\|fcuda_q4k_matmul_f32\|fcuda_flash_attention_f32' internal/compute/cuda_kernels.cu 2>/dev/null) || true
KCNT=${KCNT:-0}
say "HEAD $(git rev-parse --short HEAD 2>/dev/null || echo '(non-git)')  kernels=$KCNT lines=$(wc -l < internal/compute/cuda_kernels.cu 2>/dev/null)"
if [ "${KCNT:-0}" -lt 3 ]; then
  say "ABORT: source tree at $SRC lacks the Q8/Q4K/flash kernels (clone failed + stale fallback). Ship the current internal/compute/* instead."
  echo "stale-tree" >"$WORK/DONE.96"; exit 96
fi

# ---- build the kernels + the -tags cuda compute package -----------------------------
step "build_cuda.sh build (nvcc sm_80 + go build -tags cuda)" bash internal/compute/build_cuda.sh build

# ---- pure-kernel device acceptance gates --------------------------------------------
step "run_485 (native Q8_0 / Q4_K device GEMM, no dequant-to-f32)" bash tools/run_485_acceptance_on_gpu.sh
step "run_486 (fused flash/online-softmax attention)"             bash tools/run_486_acceptance_on_gpu.sh

# ---- the full -tags cuda witness suite incl. the multi-layer decode forward ---------
# build_cuda.sh test runs CUDA|HALDevice across compute + model, graphs off then on.
step "build_cuda.sh test (CUDA witnesses incl. TestCUDAForwardMatchesRef)" bash internal/compute/build_cuda.sh test

say "ALL STEPS DONE rc_global=$rc_global"
echo "done" >"$WORK/DONE.$rc_global"
exit "$rc_global"
