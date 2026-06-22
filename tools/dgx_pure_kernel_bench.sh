#!/usr/bin/env bash
# dgx_pure_kernel_bench.sh — end-to-end DECODE THROUGHPUT of the pure fak CUDA kernel on a
# real GPU. Companion to dgx_pure_kernel_run.sh (which proves correctness). Downloads a small
# real checkpoint (SmolLM2-135M, a Llama-family model that fits one A100), then runs
# `modelbench -backend cuda -lean` so the Q8 decode path goes through fak's OWN device kernels
# (k_q8_gemm + k_flash_attention + ...), NOT cuBLAS — and reports real tok/s on the device.
#
# WHY SmolLM2 and not GLM-5.2: GLM-MoE-DSA refuses any compute.Backend (#86, requireGLMDsaSession
# panics), so it cannot take the GPU path at all. SmolLM2 is the closest honest "pure fak kernel
# generating tokens on a real datacenter GPU, end-to-end" we can run today.
#
# Self-backgrounds like the run script; poll /tmp/fakbench/bench.log + /tmp/fakbench/DONE.<rc>.
# Env: FAK_CUDA_ARCH=sm_80  CUDA_HOME=/usr/local/cuda  FAK_GPU=1  STEPS=128
set -uo pipefail
WORK=/tmp/fakbench
SELF="$0"
if [ "${FAKBENCH_BG:-}" != "1" ]; then
  mkdir -p "$WORK"; rm -f "$WORK"/DONE.* 2>/dev/null || true
  cp -f "$SELF" "$WORK/bench.sh" 2>/dev/null || true
  FAKBENCH_BG=1 setsid bash "$WORK/bench.sh" </dev/null >"$WORK/bench.log" 2>&1 &
  echo "LAUNCHED pid $! -> $WORK/bench.log"; exit 0
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export CUDA_VISIBLE_DEVICES="${FAK_GPU:-1}"
export PATH="/usr/local/go/bin:$CUDA_HOME/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export HOME="${HOME:-/root}"
export GOCACHE="${GOCACHE:-/tmp/gocache}"
export GOPATH="${GOPATH:-/tmp/gopath}"
mkdir -p "$GOCACHE" "$GOPATH"
STEPS="${STEPS:-128}"
SRC="$WORK/src"
MODEL="$WORK/smollm2-135m"
HF=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main
say() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# reuse the main run's clone if present; else clone fresh
if [ -d /tmp/fakpure/src/internal/compute ]; then SRC=/tmp/fakpure/src; say "reusing clone at $SRC"
else say "clone fresh"; rm -rf "$SRC"; git clone --depth 1 https://github.com/anthony-chaudhary/fak.git "$SRC"; fi
cd "$SRC" || { echo nosrc >"$WORK/DONE.97"; exit 97; }

# ensure libfakcuda.a is built for the cuda tag (idempotent; the run script may have built it)
if [ ! -f internal/compute/libfakcuda.a ]; then
  say "build libfakcuda.a ($FAK_CUDA_ARCH)"; bash internal/compute/build_cuda.sh build || true
fi

# fetch the model (config + safetensors; modelbench drives LCG token ids, no tokenizer needed)
mkdir -p "$MODEL"
say "download SmolLM2-135M (config + safetensors)"
curl -fsSL -o "$MODEL/config.json"        "$HF/config.json"        || { say "config download FAILED (no HF?)"; echo dl >"$WORK/DONE.95"; exit 95; }
curl -fsSL -o "$MODEL/model.safetensors"  "$HF/model.safetensors"  || { say "safetensors download FAILED"; echo dl >"$WORK/DONE.95"; exit 95; }
say "model bytes: $(wc -c < "$MODEL/model.safetensors")"

# cgo link flags for the -tags cuda modelbench (build_cuda.sh sets these for its own go
# commands but `go run -tags cuda` here needs them too, else ld can't find -lcudart/-lcublas).
PKG="$SRC/internal/compute"
export CGO_ENABLED=1
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$PKG -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# pure-kernel decode on the A100: -lean (Q8 quantize-at-load) + -backend cuda -> k_q8_gemm path
say "modelbench -backend cuda -lean -hf $MODEL -decode-steps $STEPS"
go run -tags cuda ./cmd/modelbench -hf "$MODEL" -lean -backend cuda -require-non-reference \
   -decode-steps "$STEPS" -decode-reps 5 -decode-prompt 16 -prefill-sizes 16,64,256
rc=$?
say "BENCH DONE rc=$rc"
echo done >"$WORK/DONE.$rc"
exit "$rc"
