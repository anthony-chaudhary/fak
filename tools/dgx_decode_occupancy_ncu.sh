#!/usr/bin/env bash
# dgx_decode_occupancy_ncu.sh — MEASURE the decode-kernel occupancy + HBM-traffic that
# internal/compute/decode_occupancy.go PREDICTS, on a real A100, with Nsight Compute (ncu).
# The witness is exact-but-analytic (block counts vs SMs, operand bytes); this script collects
# the device numbers that corroborate it: achieved occupancy, registers/thread, grid size, SM
# and DRAM throughput — for the three decode kernels the KernelWiki shortlist names
# (k_flash_attention, k_q8_gemm, k_awq_gemv). Reconciliation vs the witness happens back on the
# host (docs/notes reconcile step); this script's only job is to produce ncu.csv.
#
# WHY SmolLM2-135M: same as dgx_pure_kernel_bench.sh — GLM-MoE-DSA refuses a compute.Backend
# (#86) so cannot take the GPU decode path; SmolLM2 (9 query heads, 3 KV heads, hd 64) is the
# honest "pure fak decode kernels on a real A100" run. Its flash decode grid is exactly nH=9
# blocks, which the witness predicts leaves 99 of 108 SMs idle — the headline this ncu run checks.
#
# Self-backgrounds like the sibling dgx scripts; poll /tmp/fakncu/ncu.log + /tmp/fakncu/DONE.<rc>,
# then fetch /tmp/fakncu/ncu.csv (the per-kernel metric table) for reconciliation.
# Env: FAK_CUDA_ARCH=sm_80  CUDA_HOME=/usr/local/cuda  FAK_GPU=1  STEPS=8  LAUNCHES=24
set -uo pipefail
WORK=/tmp/fakncu
SELF="$0"
if [ "${FAKNCU_BG:-}" != "1" ]; then
  mkdir -p "$WORK"; rm -f "$WORK"/DONE.* 2>/dev/null || true
  cp -f "$SELF" "$WORK/ncu.sh" 2>/dev/null || true
  FAKNCU_BG=1 setsid bash "$WORK/ncu.sh" </dev/null >"$WORK/ncu.log" 2>&1 &
  echo "LAUNCHED pid $! -> $WORK/ncu.log (poll $WORK/DONE.<rc>, fetch $WORK/ncu.csv)"; exit 0
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
STEPS="${STEPS:-8}"
LAUNCHES="${LAUNCHES:-24}"      # cap ncu replays: a few decode steps × the 3 kernels of interest
SRC="$WORK/src"
MODEL="$WORK/smollm2-135m"
HF=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main
say() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# ncu must exist, or there is nothing to measure — fail loud, do not fabricate numbers.
if ! command -v ncu >/dev/null 2>&1; then
  say "ncu (Nsight Compute) NOT on PATH — cannot measure; install nsight-compute or add to PATH"
  echo noncu >"$WORK/DONE.96"; exit 96
fi
say "ncu: $(ncu --version 2>/dev/null | head -1)"
say "gpu: $(nvidia-smi --query-gpu=name,compute_cap,count --format=csv,noheader 2>/dev/null | head -1)"

# reuse the pure-kernel clone if present, else clone fresh
if [ -d /tmp/fakpure/src/internal/compute ]; then SRC=/tmp/fakpure/src; say "reusing clone at $SRC"
elif [ -d "$SRC/internal/compute" ]; then say "reusing clone at $SRC"
else say "clone fresh"; rm -rf "$SRC"; git clone --depth 1 https://github.com/anthony-chaudhary/fak.git "$SRC"; fi
cd "$SRC" || { echo nosrc >"$WORK/DONE.97"; exit 97; }

# build libfakcuda.a for sm_80 (idempotent)
if [ ! -f internal/compute/libfakcuda.a ]; then
  say "build libfakcuda.a ($FAK_CUDA_ARCH)"; bash internal/compute/build_cuda.sh build || true
fi

# fetch the model
mkdir -p "$MODEL"
if [ ! -s "$MODEL/model.safetensors" ]; then
  say "download SmolLM2-135M (config + safetensors)"
  curl -fsSL -o "$MODEL/config.json"       "$HF/config.json"       || { say "config dl FAILED"; echo dl >"$WORK/DONE.95"; exit 95; }
  curl -fsSL -o "$MODEL/model.safetensors" "$HF/model.safetensors" || { say "safetensors dl FAILED"; echo dl >"$WORK/DONE.95"; exit 95; }
fi
say "model bytes: $(wc -c < "$MODEL/model.safetensors")"

# cgo link flags for the -tags cuda build (mirror dgx_pure_kernel_bench.sh)
PKG="$SRC/internal/compute"
export CGO_ENABLED=1
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$PKG -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# build the modelbench binary ONCE so ncu profiles the kernels, not the go toolchain.
BIN="$WORK/modelbench_cuda"
say "build modelbench (-tags cuda)"
go build -tags cuda -o "$BIN" ./cmd/modelbench || { say "go build FAILED"; echo build >"$WORK/DONE.94"; exit 94; }

# The metrics that corroborate decode_occupancy.go, per kernel:
#   launch__grid_size            -> GridBlocks   (flash must read 9 = nH; GEMV reads out)
#   launch__block_size           -> ThreadsPerBlock (128 flash / 256 gemv)
#   launch__registers_per_thread -> the flashRegs input the witness takes (feeds the reg limiter)
#   launch__shared_mem_per_block_{static,dynamic} -> SmemBytesPerBlock ((hd+128)*4 for flash)
#   sm__ctas_launched.sum        -> total CTAs (== grid; with 108 SMs, 9 CTAs => >=99 idle SMs)
#   sm__warps_active.avg.pct...  -> achieved occupancy  (device-wide; flash ~0.5%, GEMV ~high)
#   sm__throughput.avg.pct...    -> SM utilization      (low on flash = the underfill)
#   gpu__dram_throughput.avg.pct -> DRAM %              (high = memory-bound, the ~0.5 FLOP/byte)
METRICS=launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,sm__ctas_launched.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed

say "ncu profile: flash + q8_gemm + awq_gemv (launch-count $LAUNCHES, decode-steps $STEPS)"
# --replay-mode kernel: re-run each kernel in place for metric collection (no app rerun).
# --kernel-name regex: only the decode kernels of interest. --launch-count bounds total profiled.
ncu --target-processes all \
    --replay-mode kernel \
    --kernel-name 'regex:k_flash_attention|k_q8_gemm|k_awq_gemv' \
    --launch-count "$LAUNCHES" \
    --metrics "$METRICS" \
    --csv --page raw --log-file "$WORK/ncu.csv" \
    "$BIN" -hf "$MODEL" -lean -backend cuda -require-non-reference \
           -decode-steps "$STEPS" -decode-reps 1 -decode-prompt 16 -prefill-sizes 16
rc=$?
say "NCU DONE rc=$rc"
if [ -s "$WORK/ncu.csv" ]; then
  say "wrote $WORK/ncu.csv ($(wc -l < "$WORK/ncu.csv") rows) — fetch + reconcile vs decode_occupancy.go"
else
  say "WARNING: ncu.csv empty — check ncu.log for profiling errors (perms? --launch-count too low?)"
fi
echo done >"$WORK/DONE.$rc"
exit "$rc"
