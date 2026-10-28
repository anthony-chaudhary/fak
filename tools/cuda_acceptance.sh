#!/usr/bin/env bash
# cuda_acceptance.sh — run EVERY on-GPU CUDA witness with ONE verdict.
#
# The CUDA correctness story is spread across the per-family acceptance scripts
# (tools/run_479..486_acceptance_on_gpu.sh + tools/run_926_acceptance_on_gpu.sh) plus the GLM-DSA
# device witness (tools/dgx_glm_gpu_witness.sh). Today "is the cuda path green?" means running each
# by hand and eyeballing the logs. This aggregator runs them all and prints ONE per-family manifest
# with a three-state verdict, so a single command answers the question.
#
# THREE STATES — and SKIP-IS-NOT-PASS:
#   PASS  the witness ran on a real GPU and cleared its recorded floor.
#   FAIL  the witness ran but missed its floor (or failed to build/link) — a real defect.
#   SKIP  no reachable CUDA device on this node — the witness could NOT be proven.
# A SKIP is NOT a pass. On a host with no GPU (the win32 dev host, a CPU-only CI box) every
# family SKIPs and this script exits NON-ZERO (code 3), so a no-GPU run can never read as
# green. It exits 0 only when every runnable family actually PASSED on a device.
#
# WHAT NODE THIS RUNS ON: any host with a CUDA toolkit (nvcc + cuBLAS) AND a reachable
# NVIDIA GPU — a dev box / WSL with a consumer card, the GPU server, or a GCP DLVM GPU VM.
# It cannot prove anything on the win32 build host (no toolkit, walled GPU); there it SKIPs
# every family and exits 3, honestly reporting "unverified", never a false green.
#
#   usage:  bash tools/cuda_acceptance.sh
#   env:    FAK_CUDA_ARCH=sm_89|sm_80|sm_90|sm_100   (passed through to each witness)
#           CUDA_HOME=/usr/local/cuda                (default ~/cudaenv, else PATH nvcc)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOD_DIR"

# The witness family roster: a label, the script/command that proves it, and the floor it
# clears. Each `run_48X` script already enforces SKIP-is-not-PASS (exit 3/4 on no-GPU); the
# GLM-DSA family runs its device Go witnesses through tools/dgx_glm_gpu_witness.sh; the
# NVFP4 W4A4 native family is an in-script witness (run_nvfp4_w4a4_family below) because the
# no-new-.sh gate keeps new shell scripts out of tools/.
#   479 = base f32 SGEMM forward · 482 = async host-transfer · 483 = CUDA-graph decode
#   484 = fp16 tensor-core HGEMM · 485 = native Q8_0/Q4_K GEMM · 486 = fused flash attention
#   NVFP4 W4A4 native = cuBLASLt W4A4 GEMM on qualified Blackwell (sm_100/103/120, Lt >= 12.8.4)
declare -a FAM_LABEL=(
  "base-forward (#479, f32 SGEMM)"
  "async-hostxfer (#482)"
  "cuda-graph (#483)"
  "fp16 HGEMM (#484, cudaFP16CosineMin)"
  "quant Q8_0/Q4_K (#485, cudaQ8/Q4KCosineMin)"
  "flash-attention (#486, cudaFlashAttnCosineMin)"
  "GLM-DSA (cudaDsaSparseAttnCosineMin + cudaDsaIndexSelectionExact)"
  "AWQ 4-bit (#926, cudaAWQCosineMin)"
  "NVFP4 W4A4 native (sm_100/103/120, cuBLASLt>=12.8.4)"
)
declare -a FAM_CMD=(
  "tools/run_479_acceptance_on_gpu.sh"
  "tools/run_482_acceptance_on_gpu.sh"
  "tools/run_483_acceptance_on_gpu.sh"
  "tools/run_484_acceptance_on_gpu.sh"
  "tools/run_485_acceptance_on_gpu.sh"
  "tools/run_486_acceptance_on_gpu.sh"
  "tools/dgx_glm_gpu_witness.sh"
  "tools/run_926_acceptance_on_gpu.sh"
  "run_nvfp4_w4a4_family"
)

# ---- NVFP4 W4A4 native witness (in-script family) ----------------------------------
# Proves the native cuBLASLt NVFP4 W4A4 GEMM (internal/compute/cuda.go — policy-admitted at
# sm in {100,103,120} with cuBLASLt >= 12.8.4) against the CPU oracle by running
# FAK_REQUIRE_CUDA_NVFP4_NATIVE=1 go test -tags cuda over TestCUDANVFP4NativeScaleTileCrossings
# (internal/compute/cuda_nvfp4_native_test.go). The require-env turns every "could not prove"
# path inside the test into a hard failure instead of a silent skip, so this wrapper pre-gates
# the node and reads SKIP (rc 4, never FAIL) wherever the witness cannot be proven here:
# no toolkit, no reachable GPU, non-Blackwell tier, or cuBLASLt below the 12.8.4 floor.
# On a qualified Blackwell node the test must PASS (bit-exact vs the CPU oracle).
run_nvfp4_w4a4_family() {
  local pkg_dir="$MOD_DIR/internal/compute"
  local cuda_home="${CUDA_HOME:-${HOME:-/tmp/fakgpu/home}/cudaenv}"
  local nvcc="$cuda_home/bin/nvcc"
  if [ ! -x "$nvcc" ]; then
    if command -v nvcc >/dev/null 2>&1; then
      nvcc="$(command -v nvcc)"
      cuda_home="$(dirname "$(dirname "$nvcc")")"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
      nvcc=/usr/local/cuda/bin/nvcc
      cuda_home=/usr/local/cuda
    else
      echo "[nvfp4] SKIP: no reachable CUDA toolkit (no nvcc at $cuda_home, /usr/local/cuda, or PATH)."
      return 4
    fi
  fi

  # Device gate: only qualified Blackwell can prove this witness; anywhere else it is
  # unprovable, and FAK_REQUIRE_CUDA_NVFP4_NATIVE=1 would turn that into a hard FAIL.
  local cap major minor sm
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)"
  if [[ ! "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "[nvfp4] SKIP: no reachable CUDA device on this node (nvidia-smi reported no compute capability)."
    return 4
  fi
  major="${cap%%.*}"; minor="${cap##*.}"
  sm=$((major * 10 + minor))
  case "$sm" in
    100|103|120) : ;;
    *)
      echo "[nvfp4] SKIP: native CUDA NVFP4 requires a qualified Blackwell GPU; found sm_$sm."
      return 4
      ;;
  esac

  # cuBLASLt floor gate: cuBLASLt ships with the CUDA toolkit, so the toolkit build number
  # is the shell-side proxy for the Lt version; the authoritative check stays the in-test
  # cuBLASLt preflight (fcuda_nvfp4_w4a4_preflight + nvfp4CUBLASLtPolicy).
  local rel lt_major lt_minor lt_patch lt_ok
  rel="$( "$nvcc" --version 2>/dev/null | grep -oE 'V[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 | cut -c2- )"
  [ -z "$rel" ] && rel="$(awk '$1 == "CUDA" && $2 == "Version" {print $3}' "$cuda_home/version.txt" 2>/dev/null)"
  if [[ ! "$rel" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "[nvfp4] SKIP: cuBLASLt version undetermined on this node; the W4A4 native witness cannot be proven."
    return 4
  fi
  IFS=. read -r lt_major lt_minor lt_patch <<<"$rel"
  lt_patch="${lt_patch:-0}"
  lt_ok=0
  if [ "$lt_major" -gt 12 ]; then
    lt_ok=1
  elif [ "$lt_major" -eq 12 ]; then
    if [ "$lt_minor" -gt 8 ]; then
      lt_ok=1
    elif [ "$lt_minor" -eq 8 ] && [ "$lt_patch" -ge 4 ]; then
      lt_ok=1
    fi
  fi
  if [ "$lt_ok" -ne 1 ]; then
    echo "[nvfp4] SKIP: cuBLASLt $rel is below the 12.8.4 policy floor; the NVFP4 W4A4 native path is ineligible on this node."
    return 4
  fi

  # Build the kernels exactly like the run_48X witnesses (nvcc -> libfakcuda.a), for the
  # detected device arch unless the operator pinned FAK_CUDA_ARCH.
  local arch="${FAK_CUDA_ARCH:-sm_$sm}" inc="" lib="-L$pkg_dir" rpath="" ldpath=""
  case "$arch" in sm_*) ;; *) arch="sm_$arch" ;; esac
  for d in "$cuda_home/include" "$cuda_home/targets/x86_64-linux/include"; do
    [ -d "$d" ] && inc="$inc -I$d"
  done
  [ -d /usr/lib/wsl/lib ] && { rpath="-Wl,-rpath,/usr/lib/wsl/lib"; ldpath="/usr/lib/wsl/lib"; }
  for d in "$cuda_home/lib64" "$cuda_home/lib" "$cuda_home/targets/x86_64-linux/lib"; do
    if [ -d "$d" ]; then
      lib="$lib -L$d"; rpath="${rpath:+$rpath }-Wl,-rpath,$d"; ldpath="${ldpath:+$ldpath:}$d"
    fi
  done
  echo "[nvfp4] nvcc compile kernels ($arch) ..."
  if ! ( cd "$pkg_dir"
    "$nvcc" -O3 -std=c++14 -arch="$arch" -ccbin "${FAK_NVCC_CCBIN:-/usr/bin/g++}" $inc \
      -Xcompiler -fPIC -c cuda_kernels.cu -o cuda_kernels.o
    ar rcs libfakcuda.a cuda_kernels.o ); then
    echo "[nvfp4] FAIL: kernel build failed for $arch."
    return 1
  fi

  # Run the witness with the require-env ON: on this qualified node any non-pass is a defect.
  local go_scratch log rc
  go_scratch="$(mktemp -d -t fakanvfp4.XXXXXX)"
  log="$(mktemp -t fakanvfp4.XXXXXX.log)"
  ( cd "$MOD_DIR"
    export GOCACHE="${GOCACHE:-$go_scratch/build}"
    export GOMODCACHE="${GOMODCACHE:-$go_scratch/mod}"
    export GOPATH="${GOPATH:-$go_scratch/gopath}"
    mkdir -p "$GOCACHE" "$GOMODCACHE" "$GOPATH"
    export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
    export CGO_ENABLED=1
    export CC="${CC:-/usr/bin/gcc}"
    export CXX="${CXX:-/usr/bin/g++}"
    export CGO_CFLAGS="$inc"
    export CGO_LDFLAGS="$lib $rpath"
    export LD_LIBRARY_PATH="${ldpath:+$ldpath:}${LD_LIBRARY_PATH:-}"
    export FAK_REQUIRE_CUDA_NVFP4_NATIVE=1
    go test -tags cuda -count=1 -v -run '^TestCUDANVFP4NativeScaleTileCrossings$' ./internal/compute/ ) >"$log" 2>&1
  rc=$?
  rm -rf "$go_scratch"
  if grep -aq "no CUDA device\|not registered (no reachable CUDA device)" "$log"; then
    tail -4 "$log" | sed 's/^/    | /'
    echo "[nvfp4] SKIP: no reachable CUDA device on this node — NOT a pass"
    rm -f "$log"
    return 4
  fi
  if [ "$rc" -ne 0 ] || ! grep -aq -- "--- PASS: TestCUDANVFP4NativeScaleTileCrossings" "$log"; then
    tail -6 "$log" | sed 's/^/    | /'
    echo "[nvfp4] FAIL: the NVFP4 W4A4 native witness did not pass (go test rc=$rc)."
    rm -f "$log"
    return 1
  fi
  echo "[nvfp4] PASS: native cuBLASLt NVFP4 W4A4 GEMM is bit-exact vs the CPU oracle on sm_$sm (cuBLASLt $rel >= 12.8.4)."
  rm -f "$log"
  return 0
}

echo "== CUDA acceptance — every GPU witness, one verdict (SKIP is NOT pass) =="
echo "[accept] repo root : $MOD_DIR"
echo "[accept] families  : ${#FAM_LABEL[@]} (run_479/482/483/484/485/486/926 + GLM-DSA + NVFP4 W4A4 native)"
echo

declare -a STATE
n_pass=0; n_fail=0; n_skip=0; n_missing=0

for i in "${!FAM_LABEL[@]}"; do
  label="${FAM_LABEL[$i]}"; cmd="${FAM_CMD[$i]}"
  printf -- "---- %-2s %s\n" "$((i + 1))" "$label"
  log="$(mktemp -t fakaccept.XXXXXX.log)"
  if declare -F "$cmd" >/dev/null 2>&1; then
    # In-script witness: the no-new-.sh gate keeps the NVFP4 family in this file rather than
    # as another tools/*.sh, so it registers as a function name, not a path.
    "$cmd" >"$log" 2>&1
  elif [ -f "$cmd" ]; then
    bash "$cmd" >"$log" 2>&1
  else
    echo "    MISSING: $cmd not on disk — cannot run this witness"
    rm -f "$log"
    STATE[$i]="MISSING"; n_missing=$((n_missing + 1)); continue
  fi
  rc=$?
  # SKIP detection: the run_48X scripts exit 3/4 and log a no-GPU marker; mirror that here so
  # a CPU-only box reads SKIP, never PASS. Anything else non-zero is a real FAIL.
  if [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ] || grep -qaiE "no reachable CUDA|no CUDA device|not registered \(no reachable|SKIP" "$log"; then
    echo "    SKIP: no reachable CUDA device on this node (rc=$rc) — NOT a pass"
    STATE[$i]="SKIP"; n_skip=$((n_skip + 1))
  elif [ "$rc" -eq 0 ]; then
    echo "    PASS: witness cleared its floor on the device"
    STATE[$i]="PASS"; n_pass=$((n_pass + 1))
  else
    echo "    FAIL: witness did not pass (rc=$rc) — tail:"
    tail -6 "$log" | sed 's/^/      | /'
    STATE[$i]="FAIL"; n_fail=$((n_fail + 1))
  fi
  rm -f "$log"
done

echo
echo "== manifest =="
for i in "${!FAM_LABEL[@]}"; do
  printf -- "  %-6s  %s\n" "${STATE[$i]}" "${FAM_LABEL[$i]}"
done
echo
echo "[accept] PASS=$n_pass  FAIL=$n_fail  SKIP=$n_skip  MISSING=$n_missing  of ${#FAM_LABEL[@]} families"

# Verdict + exit semantics (SKIP-is-not-PASS):
if [ "$n_fail" -gt 0 ] || [ "$n_missing" -gt 0 ]; then
  echo "[accept] VERDICT: FAIL — at least one witness missed its floor or is missing."
  exit 1
fi
if [ "$n_pass" -eq 0 ]; then
  echo "[accept] VERDICT: UNVERIFIED — no GPU on this node, every family SKIPPED. A skip is not a"
  echo "[accept]          pass; run on a node with a live CUDA device to prove the floors."
  exit 3
fi
if [ "$n_skip" -gt 0 ]; then
  echo "[accept] VERDICT: PARTIAL — $n_pass family(ies) PASSED, $n_skip SKIPPED (no GPU for those)."
  echo "[accept]          Re-run on a node that can reach every family to get a full green."
  exit 2
fi
echo "[accept] VERDICT: PASS — every CUDA witness cleared its floor on the device."
exit 0
