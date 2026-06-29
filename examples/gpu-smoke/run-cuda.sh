#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "gpu-smoke(cuda): missing required command: $1" >&2
    exit 2
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
gpu-smoke(cuda): missing HuggingFace safetensors snapshot.
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
    if grep -Eqi 'cuda|nvidia|no device|not registered|driver|nvcc|cuBLAS|CUDA_ERROR' "$logfile"; then
      echo "gpu-smoke(cuda): NO NVIDIA CUDA GPU detected or CUDA backend unavailable (see $logfile)" >&2
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
  Linux*) ;;
  *)
    echo "gpu-smoke(cuda): CUDA smoke expects Linux or WSL; use run-vulkan.sh on native Windows AMD." >&2
    exit 2
    ;;
esac

need bash
need go

HF="$(find_hf_snapshot)"
require_hf_snapshot "$HF"

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi -L 2>/dev/null | grep -qi 'NVIDIA'; then
    echo "gpu-smoke(cuda): NO NVIDIA CUDA GPU detected by nvidia-smi" >&2
    exit 1
  fi
else
  echo "gpu-smoke(cuda): nvidia-smi not found; continuing to the CUDA runtime check" >&2
fi

OUT="${FAK_GPU_SMOKE_OUT:-$ROOT/.cache/gpu-smoke/cuda}"
mkdir -p "$OUT"

GPUCHECK="$OUT/gpucheck-cuda"
MODELBENCH="$OUT/modelbench-cuda"
TOKENS="${FAK_GPU_SMOKE_TOKENS:-10}"
STEPS="${FAK_GPU_SMOKE_STEPS:-32}"
PREFILL="${FAK_GPU_SMOKE_PREFILL:-16}"
DECODE_REPS="${FAK_GPU_SMOKE_DECODE_REPS:-1}"
PREFILL_REPS="${FAK_GPU_SMOKE_PREFILL_REPS:-1}"
REPORT="$OUT/modelbench-cuda.json"

echo "gpu-smoke(cuda): hf=$HF"
echo "gpu-smoke(cuda): FAK_CUDA_GRAPH=${FAK_CUDA_GRAPH:-1}"

bash internal/compute/build_cuda.sh binary ./cmd/gpucheck "$GPUCHECK"
bash internal/compute/build_cuda.sh binary ./cmd/modelbench "$MODELBENCH"

export FAK_CUDA_GRAPH="${FAK_CUDA_GRAPH:-1}"

run_checked "argmax-exact witness" "$OUT/gpucheck.log" \
  "$GPUCHECK" -hf "$HF" -backend cuda -n "$TOKENS"
echo "gpu-smoke(cuda): argmax-exact PASS vs cpu-ref"

run_checked "decode throughput" "$OUT/modelbench.log" \
  "$MODELBENCH" -hf "$HF" -backend cuda -require-non-reference \
    -prefill-sizes "$PREFILL" -prefill-reps "$PREFILL_REPS" \
    -decode-steps "$STEPS" -decode-reps "$DECODE_REPS" \
    -out "$REPORT"

TPS="$(extract_decode_tps "$REPORT")"
MS="$(extract_decode_ms "$REPORT")"
echo "gpu-smoke(cuda): decode_tok_per_sec=$TPS per_token_median_ms=$MS report=$REPORT"
