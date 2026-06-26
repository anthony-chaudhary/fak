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
# GLM-DSA family runs its device Go witnesses through tools/dgx_glm_gpu_witness.sh.
#   479 = base f32 SGEMM forward · 482 = async host-transfer · 483 = CUDA-graph decode
#   484 = fp16 tensor-core HGEMM · 485 = native Q8_0/Q4_K GEMM · 486 = fused flash attention
declare -a FAM_LABEL=(
  "base-forward (#479, f32 SGEMM)"
  "async-hostxfer (#482)"
  "cuda-graph (#483)"
  "fp16 HGEMM (#484, cudaFP16CosineMin)"
  "quant Q8_0/Q4_K (#485, cudaQ8/Q4KCosineMin)"
  "flash-attention (#486, cudaFlashAttnCosineMin)"
  "GLM-DSA (cudaDsaSparseAttnCosineMin + cudaDsaIndexSelectionExact)"
  "AWQ 4-bit (#926, cudaAWQCosineMin)"
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
)

echo "== CUDA acceptance — every GPU witness, one verdict (SKIP is NOT pass) =="
echo "[accept] repo root : $MOD_DIR"
echo "[accept] families  : ${#FAM_LABEL[@]} (run_479/482/483/484/485/486/926 + GLM-DSA witness)"
echo

declare -a STATE
n_pass=0; n_fail=0; n_skip=0; n_missing=0

for i in "${!FAM_LABEL[@]}"; do
  label="${FAM_LABEL[$i]}"; cmd="${FAM_CMD[$i]}"
  printf -- "---- %-2s %s\n" "$((i + 1))" "$label"
  if [ ! -f "$cmd" ]; then
    echo "    MISSING: $cmd not on disk — cannot run this witness"
    STATE[$i]="MISSING"; n_missing=$((n_missing + 1)); continue
  fi
  log="$(mktemp -t fakaccept.XXXXXX.log)"
  bash "$cmd" >"$log" 2>&1
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
