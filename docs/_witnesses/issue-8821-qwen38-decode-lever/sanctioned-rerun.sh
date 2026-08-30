#!/usr/bin/env bash
set -euo pipefail

readonly BASE_REVISION="97ab1e4e3b34b26fd9f901c0a7d12f55b6bd3722"
readonly EXPECTED_FILE="Qwen3.8-27B-Q4_K_M.gguf"
readonly EXPECTED_SHA256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
readonly HF_REPOSITORY="unsloth/Qwen3.8-27B-GGUF"
readonly A100_TIER="a2-high-a100-40gb-1g"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

fail() {
  printf 'HOLD: %s\n' "$1" >&2
  exit 2
}

: "${GCP_PROJECT:?GCP_PROJECT must be set explicitly to the sanctioned project identifier}"
: "${GGUF_PATH:?GGUF_PATH must be set explicitly to the exact GGUF path}"

case "$GGUF_PATH" in
  /*) ;;
  *) fail "GGUF_PATH must be absolute" ;;
esac

[[ -f "$GGUF_PATH" ]] || fail "the exact GGUF file is not readable"
[[ "$(basename "$GGUF_PATH")" == "$EXPECTED_FILE" ]] || fail "GGUF_PATH must name $EXPECTED_FILE"

if command -v sha256sum >/dev/null 2>&1; then
  actual_sha256="$(sha256sum "$GGUF_PATH" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  actual_sha256="$(shasum -a 256 "$GGUF_PATH" | awk '{print $1}')"
else
  fail "sha256sum or shasum is required"
fi
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || fail "artifact SHA-256 does not match the frozen receipt"

probe_json="$(mktemp "${TMPDIR:-/tmp}/fak-8821-gcp-probe.XXXXXX.json")"
trap 'rm -f "$probe_json"' EXIT

python3 "$REPO_ROOT/tools/gcp_gpu_probe.py" \
  --project "$GCP_PROJECT" \
  --all-tiers \
  --json "$probe_json"

python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    probe = json.load(handle)
tier = sys.argv[2]
matches = [row for row in probe.get("tiers", []) if row.get("tier") == tier]
if len(matches) != 1 or matches[0].get("verdict") != "PROVISIONABLE":
    raise SystemExit("HOLD: pinned A100-40GB tier is not proven provisionable")
' "$probe_json" "$A100_TIER"

python3 "$REPO_ROOT/tools/gcp_bench.py" \
  --dry-run \
  --project "$GCP_PROJECT" \
  --tier "$A100_TIER" \
  --engine fak-cuda \
  --hf-repo "$HF_REPOSITORY" \
  --hf-file "$EXPECTED_FILE" \
  --fak-ref "$BASE_REVISION"

printf '%s\n' \
  'SANCTIONED ON-NODE CUDA ACCEPTANCE/BUILD/SERVE SEQUENCE:' \
  'test "$(git rev-parse HEAD)" = "97ab1e4e3b34b26fd9f901c0a7d12f55b6bd3722"' \
  'test "$(basename "$GGUF_PATH")" = "Qwen3.8-27B-Q4_K_M.gguf"' \
  'test "$(sha256sum "$GGUF_PATH" | awk '\''{print $1}'\'')" = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"' \
  'env FAK_CUDA_ARCH=sm_80 CUDA_HOME=/usr/local/cuda bash tools/cuda_acceptance.sh' \
  'env FAK_CUDA_ARCH=sm_80 CUDA_HOME=/usr/local/cuda bash internal/compute/build_cuda.sh' \
  'env CGO_ENABLED=1 FAK_CUDA_ARCH=sm_80 CUDA_HOME=/usr/local/cuda go build -tags cuda -o /tmp/fak-issue-8821 ./cmd/fak' \
  'env FAK_Q4K=1 /tmp/fak-issue-8821 serve --addr 127.0.0.1:8137 --engine inkernel --gguf "$GGUF_PATH" --model qwen38:27b --backend cuda --context-budget-tokens 22'

printf '%s\n' \
  'HOLD: current public tooling lacks full-model GDN per-stage CUDA-event capture plus the strict cosine/argmax/max-absolute-logit triad.' \
  'HOLD: no lever may be selected until those artifacts bind engine=fak-native, runtime=inkernel, path=cuda/qwen35-gdn-ssm-decode-v1, fallback=0, P/C/O=22/22/6, and inclusive setup/recovery/verification accounting.' >&2
exit 2
