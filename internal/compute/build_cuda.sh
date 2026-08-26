#!/usr/bin/env bash
# build_cuda.sh — compile the CUDA kernels (nvcc -> libfakcuda.a) and build/test the
# `-tags cuda` variant of the compute package. Portable across three hosts with no
# edits: WSL (user-space micromamba CUDA env at ~/cudaenv, no sudo), the GPU server, and
# a GCP GPU VM (Deep-Learning-VM image, CUDA at /usr/local/cuda). The default
# `go build` (no tags) needs none of this and stays pure-Go.
#
#   usage:  bash internal/compute/build_cuda.sh [check|build|test|bench]   (default: test)
#   env:    FAK_CUDA_ARCH=sm_89|sm_90|sm_100  (default sm_89; "89" also accepted)
#           FAK_CUDA_NCCL=1                    (also compile cuda_nccl.cu; Go tags cuda,nccl)
#           CUDA_HOME=/usr/local/cuda          (default ~/cudaenv, else system nvcc)
#           NCCL_HOME=/tmp/nccl-root           (optional user-space nccl.h/libnccl.so root)
set -euo pipefail

# locate the module root (dir containing go.mod) from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR"                 # internal/compute
MOD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # fak/

cmd="${1:-test}"
ARCH_FILE="$SCRIPT_DIR/cuda_arch.txt"
CUDA_ARCHES=()
while IFS= read -r arch || [ -n "$arch" ]; do
  arch="${arch%$'\r'}"
  [ -n "$arch" ] || continue
  CUDA_ARCHES+=("$arch")
done < "$ARCH_FILE"
ARCH="${FAK_CUDA_ARCH:-}"
GENCODE=()
if [ -n "$ARCH" ]; then
  case "$ARCH" in sm_*) ;; *) ARCH="sm_$ARCH";; esac
  arch_supported=false
  for supported_arch in "${CUDA_ARCHES[@]}"; do
    if [ "$ARCH" = "$supported_arch" ]; then
      arch_supported=true
      break
    fi
  done
  if [ "$arch_supported" != true ]; then
    echo "ERROR: unsupported CUDA arch '$ARCH'; choose one from $ARCH_FILE: ${CUDA_ARCHES[*]}" >&2
    exit 2
  fi
  cc="${ARCH#sm_}"
  GENCODE=(-gencode "arch=compute_${cc},code=${ARCH}")
  BUILD_ARCHS="$ARCH (single-arch dev build)"
else
  for arch in "${CUDA_ARCHES[@]}"; do
    cc="${arch#sm_}"
    GENCODE+=(-gencode "arch=compute_${cc},code=${arch}")
    PTX_CC="$cc"
  done
  GENCODE+=(-gencode "arch=compute_${PTX_CC},code=compute_${PTX_CC}")
  BUILD_ARCHS="${CUDA_ARCHES[*]} + compute_${PTX_CC} PTX"
fi
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
  echo "[cuda] GPU-free architecture matrix check ..."
  ( cd "$MOD_DIR" && "$PY" tools/cuda_arch_matrix.py )

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
if [ -n "${NCCL_HOME:-}" ]; then
  for d in "$NCCL_HOME/include" "$NCCL_HOME/usr/include"; do
    [ -d "$d" ] && INC="$INC -I$d"
  done
fi
sanitize_ld_path() {
  local raw="${1:-}" out="" part
  local oldifs="$IFS"
  IFS=:
  for part in $raw; do
    IFS="$oldifs"
    [ -z "$part" ] && continue
    case "$part" in
      */stubs|*/stubs/*) ;;
      *) out="${out:+$out:}$part" ;;
    esac
    IFS=:
  done
  IFS="$oldifs"
  printf '%s' "$out"
}

path_contains_libcuda() {
  local d="$1"
  compgen -G "$d/libcuda.so*" >/dev/null
}

append_colon_path() {
  local cur="$1" add="$2"
  [ -n "$add" ] || { printf '%s' "$cur"; return; }
  case ":$cur:" in
    *":$add:"*) printf '%s' "$cur" ;;
    *) printf '%s' "${cur:+$cur:}$add" ;;
  esac
}

stage_nccl_link_dir() {
  local src_dir="$1" link_dir="$PKG_DIR/.cuda-nccl-lib" cand=""
  [ -e "$src_dir/libnccl.so" ] && return 0
  for cand in "$src_dir"/libnccl.so.*; do
    [ -e "$cand" ] || continue
    mkdir -p "$link_dir"
    ln -sf "$cand" "$link_dir/libnccl.so"
    printf '%s' "$link_dir"
    return 0
  done
}

LIB="-L$PKG_DIR"; RPATH=""; LDPATH=""
# WSL keeps libcuda.so under /usr/lib/wsl/lib; only add it where it exists (a DLVM/DGX
# has no such path, and an rpath to a missing dir is just noise on the link line).
if [ -d /usr/lib/wsl/lib ]; then RPATH="-Wl,-rpath,/usr/lib/wsl/lib"; LDPATH="/usr/lib/wsl/lib"; fi
for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
  [ -e "$d/libcuda.so.1" ] && LDPATH="$(append_colon_path "$LDPATH" "$d")"
done
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/x86_64-linux/lib"; do
  if [ -d "$d" ]; then
    LIB="$LIB -L$d"
    LDPATH="$(append_colon_path "$LDPATH" "$d")"
    # CUDA toolkit dirs may carry libcuda stubs beside libcudart/cublas. Keep such dirs in
    # LD_LIBRARY_PATH after the real driver dirs, but do not bake them into RUNPATH/RPATH.
    if ! path_contains_libcuda "$d"; then
      RPATH="${RPATH:+$RPATH }-Wl,-rpath,$d"
    fi
  fi
done
if [ -n "${NCCL_HOME:-}" ]; then
  for d in "$NCCL_HOME/lib64" "$NCCL_HOME/lib" "$NCCL_HOME/usr/lib/x86_64-linux-gnu" "$NCCL_HOME/usr/lib"; do
    if [ -d "$d" ]; then
      LIB="$LIB -L$d"
      RPATH="${RPATH:+$RPATH }-Wl,-rpath,$d"
      LDPATH="$(append_colon_path "$LDPATH" "$d")"
      link_dir="$(stage_nccl_link_dir "$d")"
      if [ -n "$link_dir" ]; then
        LIB="$LIB -L$link_dir"
        RPATH="${RPATH:+$RPATH }-Wl,-rpath,$link_dir"
        LDPATH="$(append_colon_path "$LDPATH" "$link_dir")"
      fi
    fi
  done
fi

# Refuse an executable-but-incomplete cached Go toolchain before the expensive nvcc build. A
# bridge shell can retain $GOROOT/bin/go after a partial module-cache extraction while stdlib
# source trees are missing; `go version` then succeeds but a later cmd/fak build fails on ordinary
# packages such as encoding/base32 and go/types (#6128).
export PATH="/usr/local/go/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
if ! command -v go >/dev/null 2>&1; then
  echo "[cuda] Go 1.26+ is required (no go executable on PATH)" >&2
  exit 2
fi
GO_ROOT="$(go env GOROOT 2>/dev/null || true)"
for required in src/encoding/base32/base32.go src/go/types/api.go; do
  if [ -z "$GO_ROOT" ] || [ ! -f "$GO_ROOT/$required" ]; then
    echo "[cuda] incomplete Go toolchain at ${GO_ROOT:-<unknown>}: missing $required; install a complete Go 1.26+ distribution before the CUDA build" >&2
    exit 2
  fi
done

# GPU arch: default sm_89 (Ada / L4), override via FAK_CUDA_ARCH for A100 (sm_80),
# H100/H200 (sm_90), or B200/GB200 (sm_100). Accept either "89" or "sm_89".
echo "[cuda] nvcc compile kernels ($BUILD_ARCHS) ..."
( cd "$PKG_DIR"
  "$NVCC" -O3 -std=c++14 "${GENCODE[@]}" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
  objs="cuda_kernels.o"
  if [ "${FAK_CUDA_NCCL:-0}" = "1" ]; then
    echo "[cuda] nvcc compile NCCL collectives ($BUILD_ARCHS) ..."
    "$NVCC" -O3 -std=c++14 "${GENCODE[@]}" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
        -Xcompiler -fPIC -c cuda_nccl.cu -o cuda_nccl.o
    objs="$objs cuda_nccl.o"
    echo "[cuda] nvcc compile multi-process NCCL process group ($BUILD_ARCHS) ..."
    "$NVCC" -O3 -std=c++14 "${GENCODE[@]}" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $INC \
        -Xcompiler -fPIC -c cuda_nccl_pg.cu -o cuda_nccl_pg.o
    objs="$objs cuda_nccl_pg.o"
  fi
  ar rcs libfakcuda.a $objs
  echo "[cuda] built $(ls -la libfakcuda.a | awk '{print $5}') byte libfakcuda.a" )

export CGO_ENABLED=1
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CGO_CFLAGS="$INC"
GO_TAGS="cuda"
if [ "${FAK_CUDA_NCCL:-0}" = "1" ]; then
  GO_TAGS="cuda,nccl"
  LIB="$LIB -lnccl"
fi
export CGO_LDFLAGS="$LIB $RPATH"
# Keep CUDA stub-driver directories link-only. If an inherited LD_LIBRARY_PATH contains
# .../stubs, NCCL can resolve the fake libcuda and fail at first collective with
# "CUDA driver is a stub library" even though cudaMalloc/cudart calls already worked.
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$(sanitize_ld_path "${LDPATH:+$LDPATH:}${LD_LIBRARY_PATH:-}")"

cd "$MOD_DIR"
case "$cmd" in
  build)
    echo "[cuda] go build -tags $GO_TAGS ./internal/compute/ ..."
    go build -tags "$GO_TAGS" ./internal/compute/
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
    echo "[cuda] go build -tags $GO_TAGS -o $out $pkg ..."
    go build -tags "$GO_TAGS" -o "$out" "$pkg"
    echo "[cuda] OK binary $out"
    ;;
  test)
    echo "[cuda] go test -tags $GO_TAGS (default: graphs off) ..."
    go test -tags "$GO_TAGS" -count=1 -run 'CUDA|HALDevice' ./internal/compute/ ./internal/model/
    echo "[cuda] go test -tags $GO_TAGS (FAK_CUDA_GRAPH=1: graph capture path) ..."
    FAK_CUDA_GRAPH=1 go test -tags "$GO_TAGS" -count=1 -run 'CUDA|HALDevice' ./internal/compute/ ./internal/model/
    ;;
  bench)
    # bench the cuda backend's decode throughput on a real model via modelbench.
    #   usage: build_cuda.sh bench [model-dir] [decode-steps]
    dir="${2:-internal/model/.cache/smollm2-135m}"
    steps="${3:-128}"
    echo "[cuda] modelbench -backend cuda -dir $dir -decode-steps $steps ..."
    go run -tags "$GO_TAGS" ./cmd/modelbench -dir "$dir" -backend cuda \
        -decode-steps "$steps" -decode-reps 5 -decode-prompt 16 2>&1 \
      | grep -aiE "prefill P=|decode:|tok_per_sec|panic|error|fak-cuda:" | tail -45
    ;;
  *)
    echo "unknown subcommand: $cmd"; exit 2;;
esac
