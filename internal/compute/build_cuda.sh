#!/usr/bin/env bash
# build_cuda.sh — compile the CUDA kernels (nvcc -> libfakcuda.a) and build/test the
# `-tags cuda` variant of the compute package. Portable across three hosts with no
# edits: WSL (user-space micromamba CUDA env at ~/cudaenv, no sudo), the GPU server, and
# a GCP GPU VM (Deep-Learning-VM image, CUDA at /usr/local/cuda). The default
# `go build` (no tags) needs none of this and stays pure-Go.
#
#   usage:  bash internal/compute/build_cuda.sh [check|build|test|bench]   (default: test)
#   env:    FAK_CUDA_ARCH=sm_89|sm_90|sm_100  (default sm_89; "89" also accepted)
#           CUDA_HOME=/usr/local/cuda          (default ~/cudaenv, else system nvcc)
set -euo pipefail

# locate the module root (dir containing go.mod) from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR"                 # internal/compute
MOD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # fak/

cmd="${1:-test}"
if [ "$cmd" = "check" ]; then
  PY="${PYTHON:-}"
  if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
      PY="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
      PY="$(command -v python)"
    else
      echo "no python3/python on PATH — cannot run tools/cuda_abi_parity.py"; exit 1
    fi
  fi

  echo "[cuda] GPU-free ABI/header portability check ..."
  ( cd "$MOD_DIR" && "$PY" tools/cuda_abi_parity.py --check )

  HEADER_CC="${CC:-}"
  if [ -z "$HEADER_CC" ]; then
    for cand in gcc clang cc; do
      if command -v "$cand" >/dev/null 2>&1; then
        HEADER_CC="$(command -v "$cand")"
        break
      fi
    done
  fi
  if [ -n "$HEADER_CC" ]; then
    echo "[cuda] strict standalone parse cuda_backend.h ($HEADER_CC) ..."
    "$HEADER_CC" -x c -std=c11 -fsyntax-only -Wall -Werror "$PKG_DIR/cuda_backend.h"
  else
    echo "[cuda] strict standalone parse skipped: no C compiler on PATH; explicit header deps checked"
  fi
  echo "[cuda] OK check"
  exit 0
fi

# CUDA toolchain location. Default is the WSL user-space micromamba env (~/cudaenv,
# no sudo). On a datacenter image (GCP DLVM, DGX) CUDA lives at /usr/local/cuda and
# nvcc is already on PATH — fall back to that so the SAME script builds everywhere.
# Guard $HOME: under `set -u` a detached/CI/bg context can have HOME unset, which
# would abort here before we ever reach the system-nvcc fallback. Default it so the
# fallback (system nvcc at /usr/local/cuda on a DGX/DLVM) still runs.
CUDA_HOME="${CUDA_HOME:-${HOME:-/opt}/cudaenv}"
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"   # .../bin/nvcc -> CUDA_HOME
    echo "[cuda] using system nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"
  else
    echo "no nvcc at $NVCC and none on PATH — run the CUDA-toolchain setup first"; exit 1
  fi
fi

# Build -I / -L / -rpath from whichever CUDA include/lib dirs actually exist: the
# micromamba env uses include + lib + targets/x86_64-linux/{include,lib}; a
# system/DLVM install uses include + lib64. Resolving only real dirs keeps the link
# line clean and makes this portable across both layouts.
INC=""
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include"; do
  [ -d "$d" ] && INC="$INC -I$d"
done
LIB="-L$PKG_DIR"; RPATH=""; LDPATH=""
# WSL keeps libcuda.so under /usr/lib/wsl/lib; only add it where it exists (a DLVM/DGX
# has no such path, and an rpath to a missing dir is just noise on the link line).
if [ -d /usr/lib/wsl/lib ]; then RPATH="-Wl,-rpath,/usr/lib/wsl/lib"; LDPATH="/usr/lib/wsl/lib"; fi
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/x86_64-linux/lib"; do
  if [ -d "$d" ]; then LIB="$LIB -L$d"; RPATH="${RPATH:+$RPATH }-Wl,-rpath,$d"; LDPATH="${LDPATH:+$LDPATH:}$d"; fi
done

# GPU arch: default sm_89 (Ada / L4), override via FAK_CUDA_ARCH for A100 (sm_80),
# H100/H200 (sm_90), or B200/GB200 (sm_100). Accept either "89" or "sm_89".
ARCH="${FAK_CUDA_ARCH:-sm_89}"
case "$ARCH" in sm_*) ;; *) ARCH="sm_$ARCH";; esac
echo "[cuda] nvcc compile kernels ($ARCH) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 -arch="$ARCH" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  ar rcs libfakcuda.a cuda_kernels.o
  echo "[cuda] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

export PATH="/usr/local/go/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
export CGO_ENABLED=1
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CGO_CFLAGS="$INC"
export CGO_LDFLAGS="$LIB $RPATH"
export LD_LIBRARY_PATH="${LDPATH:+$LDPATH:}${LD_LIBRARY_PATH:-}"

cd "$MOD_DIR"
case "$cmd" in
  build)
    echo "[cuda] go build -tags cuda ./internal/compute/ ..."
    go build -tags cuda ./internal/compute/
    echo "[cuda] OK build"
    ;;
  binary)
    # Build a -tags cuda binary of an arbitrary package with the SAME resolved CGO env
    # (the -I/-L/-rpath set discovered above) + the freshly compiled libfakcuda.a, so a
    # node-side serve/bench script links the cuda backend exactly like the witness tests
    # do — no hand-rolled -L$CUDA_HOME/lib64 that silently drops -lfakcuda on a layout
    # without lib64 (the prior dgx_glm_throughput_run.sh bug). Pairs with
    # tools/glm52_fak_native_serve.sh.
    #   usage: build_cuda.sh binary <pkg> <out-abs-path>
    pkg="${2:?usage: build_cuda.sh binary <pkg> <out>}"
    out="${3:?usage: build_cuda.sh binary <pkg> <out>}"
    echo "[cuda] go build -tags cuda -o $out $pkg ..."
    go build -tags cuda -o "$out" "$pkg"
    echo "[cuda] OK binary $out"
    ;;
  test)
    echo "[cuda] go test -tags cuda (default: graphs off) ..."
    go test -tags cuda -count=1 -run 'CUDA|HALDevice' ./internal/compute/ ./internal/model/
    echo "[cuda] go test -tags cuda (FAK_CUDA_GRAPH=1: graph capture path) ..."
    FAK_CUDA_GRAPH=1 go test -tags cuda -count=1 -run 'CUDA|HALDevice' ./internal/compute/ ./internal/model/
    ;;
  bench)
    # bench the cuda backend's decode throughput on a real model via modelbench.
    #   usage: build_cuda.sh bench [model-dir] [decode-steps]
    dir="${2:-internal/model/.cache/smollm2-135m}"
    steps="${3:-128}"
    echo "[cuda] modelbench -backend cuda -dir $dir -decode-steps $steps ..."
    go run -tags cuda ./cmd/modelbench -dir "$dir" -backend cuda \
        -decode-steps "$steps" -decode-reps 5 -decode-prompt 16 2>&1 \
      | grep -aiE "prefill P=|decode:|tok_per_sec|panic|error|fak-cuda:" | tail -45
    ;;
  *)
    echo "unknown subcommand: $cmd"; exit 2;;
esac
