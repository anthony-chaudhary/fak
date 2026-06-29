#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "gpu-smoke(vulkan): missing required command: $1" >&2
    exit 2
  fi
}

find_pwsh() {
  if [[ -n "${FAK_PWSH:-}" ]]; then
    printf '%s\n' "$FAK_PWSH"
  elif command -v pwsh >/dev/null 2>&1; then
    command -v pwsh
  elif command -v powershell.exe >/dev/null 2>&1; then
    command -v powershell.exe
  else
    printf '%s\n' ""
  fi
}

find_hf_snapshot() {
  if [[ -n "${FAK_GPU_SMOKE_HF:-}" ]]; then
    printf '%s\n' "$FAK_GPU_SMOKE_HF"
    return
  fi
  local hub="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots"
  local d
  for d in "$hub"/*; do
    [[ -d "$d" && -f "$d/config.json" && -f "$d/model.safetensors" ]] || continue
    printf '%s\n' "$d"
    return
  done
  printf '%s\n' ""
}

require_hf_snapshot() {
  local hf="$1"
  if [[ -z "$hf" || ! -f "$hf/config.json" || ! -f "$hf/model.safetensors" ]]; then
    cat >&2 <<'MSG'
gpu-smoke(vulkan): missing HuggingFace safetensors snapshot.
Set FAK_GPU_SMOKE_HF to a directory containing config.json and model.safetensors.
Canonical small model: HuggingFaceTB/SmolLM2-135M-Instruct.
MSG
    exit 2
  fi
}

run_checked() {
  local label="$1"
  local logfile="$2"
  shift 2
  echo
  echo "== $label =="
  set +e
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    if grep -Eqi 'vulkan|radeon|amd|VK_ERROR|not registered|no device|driver|glslc' "$logfile"; then
      echo "gpu-smoke(vulkan): NO AMD Vulkan GPU detected or Vulkan backend unavailable (see $logfile)" >&2
    fi
    exit "$rc"
  fi
}

extract_decode_tps() {
  awk '
    /"decode": *\{/ { in_decode=1; next }
    in_decode && /"tok_per_sec":/ { gsub(/,/, "", $2); print $2; exit }
  ' "$1"
}

extract_decode_ms() {
  awk '
    /"decode": *\{/ { in_decode=1; next }
    in_decode && /"per_token_median_ms":/ { gsub(/,/, "", $2); print $2; exit }
  ' "$1"
}

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *)
    echo "gpu-smoke(vulkan): Vulkan backend is currently native-Windows only; run from Git Bash/MSYS on the AMD Windows host." >&2
    exit 2
    ;;
esac

need go
need awk

PWSH="$(find_pwsh)"
if [[ -z "$PWSH" ]]; then
  echo "gpu-smoke(vulkan): missing PowerShell (pwsh or powershell.exe)" >&2
  exit 2
fi

HF="$(find_hf_snapshot)"
require_hf_snapshot "$HF"

if command -v vulkaninfo >/dev/null 2>&1; then
  if ! vulkaninfo --summary 2>/dev/null | grep -Eqi 'AMD|Radeon|RADV'; then
    echo "gpu-smoke(vulkan): NO AMD Vulkan GPU detected by vulkaninfo" >&2
    exit 1
  fi
fi

OUT="${FAK_GPU_SMOKE_OUT:-$ROOT/.cache/gpu-smoke/vulkan}"
mkdir -p "$OUT"

GPUCHECK="$OUT/gpucheck-vulkan.exe"
MODELBENCH="$OUT/modelbench-vulkan.exe"
TOKENS="${FAK_GPU_SMOKE_TOKENS:-10}"
STEPS="${FAK_GPU_SMOKE_STEPS:-16}"
PREFILL="${FAK_GPU_SMOKE_PREFILL:-16}"
DECODE_REPS="${FAK_GPU_SMOKE_DECODE_REPS:-1}"
PREFILL_REPS="${FAK_GPU_SMOKE_PREFILL_REPS:-1}"
REPORT="$OUT/modelbench-vulkan.json"

echo "gpu-smoke(vulkan): hf=$HF"
echo "gpu-smoke(vulkan): powershell=$PWSH"

"$PWSH" -NoProfile -ExecutionPolicy Bypass -File internal/compute/build_vulkan.ps1 binary ./cmd/gpucheck "$GPUCHECK"
"$PWSH" -NoProfile -ExecutionPolicy Bypass -File internal/compute/build_vulkan.ps1 binary ./cmd/modelbench "$MODELBENCH"

export FAK_VULKAN_SPIRV="$ROOT/internal/compute/spirv"

run_checked "argmax-exact witness" "$OUT/gpucheck.log" \
  "$GPUCHECK" -hf "$HF" -backend vulkan -n "$TOKENS"
echo "gpu-smoke(vulkan): argmax-exact PASS vs cpu-ref"

run_checked "decode throughput" "$OUT/modelbench.log" \
  "$MODELBENCH" -hf "$HF" -backend vulkan -require-non-reference \
    -prefill-sizes "$PREFILL" -prefill-reps "$PREFILL_REPS" \
    -decode-steps "$STEPS" -decode-reps "$DECODE_REPS" \
    -out "$REPORT"

TPS="$(extract_decode_tps "$REPORT")"
MS="$(extract_decode_ms "$REPORT")"
echo "gpu-smoke(vulkan): decode_tok_per_sec=$TPS per_token_median_ms=$MS report=$REPORT"
