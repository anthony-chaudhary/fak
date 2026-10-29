#!/usr/bin/env bash
# build_cuda.sh — entrypoint for the CUDA build/test flow. The GPU-free `check` contract
# (ABI/header portability, architecture-matrix check, strict standalone header parse) and
# the complete-Go-toolchain preflight run here in bash; kernel compilation (nvcc ->
# libfakcuda.a), archiving, stamping and the `-tags cuda` go build/test/bench dispatch are
# owned by the native Go driver (the fak-dev CUDA legacy route), which this script
# delegates to after the preflight. Portable across three hosts with no edits: WSL (user-space
# micromamba CUDA env at ~/cudaenv, no sudo), the GPU server, and a GCP GPU VM
# (Deep-Learning-VM image, CUDA at /usr/local/cuda). The default `go build` (no tags) needs
# none of this and stays pure-Go.
#
#   usage:  bash internal/compute/build_cuda.sh [check|build|test|bench]   (default: test)
#           bash internal/compute/build_cuda.sh binary <pkg> <out-abs-path>
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
  ( cd "$MOD_DIR" && go test ./internal/cudaarch )

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

# Refuse an executable-but-incomplete cached Go toolchain before the expensive nvcc build. A
# bridge shell can retain $GOROOT/bin/go after a partial module-cache extraction while stdlib
# source trees are missing; `go version` then succeeds but a later cmd/fak build fails on ordinary
# packages such as encoding/base32 and go/types (#6128). The native legacy driver re-verifies
# the toolchain after the handoff; refusing here keeps the failure cheap and in the documented
# place, before any kernel work starts.
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

# Kernel compilation, archiving, stamping and the tagged go build/test/bench dispatch are
# owned by the native Go driver. Announce the boundary on stderr only: the delegation
# contract is pure JSON on stdout, so anything written to stdout here would corrupt it.
echo "[cuda] nvcc compile kernels ($BUILD_ARCHS) ..." >&2
cd "$MOD_DIR"
exec "${GO:-go}" run ./cmd/fak-dev build-cuda legacy "$@"
