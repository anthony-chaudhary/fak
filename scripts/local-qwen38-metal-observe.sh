#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="qwen38:27b"
MAX_AGE_SECONDS=900

usage() {
  echo "usage: $0 [--output RECEIPT] [--base-url URL] | --validate RECEIPT [--now RFC3339]" >&2
}

epoch() {
  date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$1" '+%s' 2>/dev/null ||
    date -u -d "$1" '+%s' 2>/dev/null
}

validate_receipt() {
  local receipt="$1" now="${2:-$(date -u '+%Y-%m-%dT%H:%M:%SZ')}"
  jq -e '
    .schema == "fak-native-qwen38-metal-observation/v1" and
    .issue == 10005 and
    .status == "success" and
    .engine == "inkernel" and
    .model == "qwen38:27b" and
    .runtime == "native" and
    .backend == "metal" and
    (.device | type == "string" and length > 0 and test("^[A-Za-z0-9 ._+()-]+$")) and
    .fallback_count == 0 and
    .fallback_active == false and
    .llama_cpp_used == false and
    (.run_id | type == "string" and test("^npc1_[0-9a-f]{32}$")) and
    (.build.commit | test("^[0-9a-f]{40}$")) and
    .build.dirty == false and
    .build.stamped == true and
    .build.os == "darwin" and
    .build.arch == "arm64" and
    (.captured_at_utc | type == "string") and
    (.completed_at_utc | type == "string") and
    .live_execution_obtained == true and
    .raw_logs_committed == false and
    .private_identifiers_committed == false and
    .required_execution.engine == "fak-native" and
    .required_execution.runtime_engine == "inkernel" and
    .required_execution.planner == "inkernel" and
    .required_execution.model_owner == "fak" and
    .required_execution.fallback_count == 0 and
    .required_execution.fallback_active == false and
    .required_execution.llama_cpp_used == false and
    .observed_execution.engine == "fak-native" and
    .observed_execution.runtime_engine == "inkernel" and
    .observed_execution.planner == "inkernel" and
    .observed_execution.model == "Qwen3.8" and
    .observed_execution.model_owner == "fak" and
    .observed_execution.fallback_count == 0 and
    .observed_execution.fallback_active == false and
    .observed_execution.llama_cpp_used == false and
    .observed_execution.completed == true and
    (.observed_execution.output_tokens | type == "number" and . > 0) and
    .observed_execution.correlation_key == .run_id and
    (.native_receipt.forward_path | type == "string" and startswith("metal/")) and
    .native_receipt.q4k == true and
    (.native_receipt.sha256 | test("^[0-9a-f]{64}$"))
  ' "$receipt" >/dev/null

  local completed_epoch now_epoch age
  completed_epoch="$(epoch "$(jq -er '.completed_at_utc' "$receipt")")"
  now_epoch="$(epoch "$now")"
  age=$((now_epoch - completed_epoch))
  if (( age < 0 || age > MAX_AGE_SECONDS )); then
    echo "receipt is stale or future-dated: age=${age}s maximum=${MAX_AGE_SECONDS}s" >&2
    return 1
  fi
}

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
output="$repo_root/tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json"
base_url="${FAK_QWEN38_METAL_BASE_URL:-}"
validate=""
validate_now=""

while (($#)); do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --base-url) base_url="$2"; shift 2 ;;
    --validate) validate="$2"; shift 2 ;;
    --now) validate_now="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

for command_name in jq curl shasum system_profiler; do
  command -v "$command_name" >/dev/null || { echo "missing required command: $command_name" >&2; exit 1; }
done

if [[ -n "$validate" ]]; then
  validate_receipt "$validate" "$validate_now"
  echo "local Qwen3.8 Metal receipt: valid"
  exit 0
fi

fak_bin="${FAK_BIN:-$(command -v fak || true)}"
[[ -n "$fak_bin" && -x "$fak_bin" ]] || { echo "installed fak binary not found" >&2; exit 1; }
build_identity="$("$fak_bin" version --json)"
jq -e '.stamped == true and .dirty == false and .os == "darwin" and .arch == "arm64" and (.commit | test("^[0-9a-f]{40}$"))' <<<"$build_identity" >/dev/null || {
  echo "refusing unstamped, dirty, or non-Darwin/arm64 fak build evidence" >&2
  exit 1
}
build_commit="$(jq -er '.commit' <<<"$build_identity")"
build_version="$(jq -er '.app_version' <<<"$build_identity")"
serve_help="$("$fak_bin" serve help all 2>&1)"
for required in '-engine string' '-gguf string' '-model string' '-qwen38-runtime string' '-metal' '-debug-stats'; do
  grep -F -- "$required" <<<"$serve_help" >/dev/null || {
    echo "installed fak serve contract lacks $required" >&2
    exit 1
  }
done
grep -F -- 'removed auto value is rejected' <<<"$serve_help" >/dev/null || {
  echo "installed fak does not expose the fail-closed Qwen3.8 runtime contract" >&2
  exit 1
}

run_dir="$(mktemp -d "${TMPDIR:-/tmp}/fak-qwen38-metal-observe.XXXXXX")"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf -- "$run_dir"
}
trap cleanup EXIT INT TERM

if [[ -z "$base_url" ]]; then
  addr="${FAK_QWEN38_METAL_ADDR:-127.0.0.1:18085}"
  base_url="http://$addr"
  FAK_METAL_STREAM_Q4K=1 FAK_Q4K_FREE_CPU=1 FAK_Q4K=1 "$fak_bin" serve \
    --addr "$addr" \
    --engine inkernel \
    --gguf "$MODEL_ALIAS" \
    --model "$MODEL_ALIAS" \
    --qwen38-runtime native \
    --metal \
    --debug-stats \
    --session-registry off \
    --context-budget-tokens 4096 >"$run_dir/serve.log" 2>&1 &
  server_pid=$!
fi

case "$base_url" in
  http://127.0.0.1:*|http://localhost:*) ;;
  *) echo "base URL must be a loopback HTTP endpoint" >&2; exit 1 ;;
esac

ready=""
for _ in $(seq 1 "${FAK_QWEN38_METAL_READY_POLLS:-180}"); do
  if ready="$(curl -fsS --max-time 2 "$base_url/readyz" 2>/dev/null)" &&
     jq -e --arg model "$MODEL_ALIAS" '.ok == true and .startup_ready == true and .engine == "inkernel" and .planner == "inkernel" and .model == $model' <<<"$ready" >/dev/null; then
    break
  fi
  if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
    echo "fak serve stopped before native Metal readiness" >&2
    exit 1
  fi
  sleep 2
  ready=""
done
[[ -n "$ready" ]] || { echo "timed out waiting for native Qwen3.8 readiness" >&2; exit 1; }

device="$(system_profiler SPDisplaysDataType -json | jq -er '.SPDisplaysDataType[] | select(._name | test("Apple")) | .sppci_model' | head -1)"
[[ "$device" =~ ^[A-Za-z0-9\ ._+\(\)-]+$ ]] || { echo "Metal device identity is not public-safe" >&2; exit 1; }

started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
run_id="npc1_$(printf '%s' "$$:$started_at:$RANDOM" | shasum -a 256 | awk '{print substr($1,1,32)}')"
jq -nc --arg model "$MODEL_ALIAS" '{model:$model,messages:[{role:"user",content:"Reply with exactly: Q38"}],max_tokens:1,temperature:0,fak:{native_inference_receipt:true}}' >"$run_dir/request.json"
curl -fsS --max-time "${FAK_QWEN38_METAL_REQUEST_TIMEOUT:-180}" \
  -H 'content-type: application/json' -H "X-Trace-Id: $run_id" \
  --data-binary "@$run_dir/request.json" "$base_url/v1/chat/completions" >"$run_dir/response.json"
curl -fsS --max-time 5 "$base_url/metrics" >"$run_dir/metrics.txt"
completed_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

jq -e --arg model "$MODEL_ALIAS" '
  .model == $model and
  (.usage.completion_tokens | type == "number" and . > 0) and
  .fak.native_inference_receipt.engine == "inkernel" and
  .fak.native_inference_receipt.planner == "inkernel" and
  .fak.native_inference_receipt.owner == "fak" and
  .fak.native_inference_receipt.backend == "metal" and
  (.fak.native_inference_receipt.forward_path | startswith("metal/")) and
  .fak.native_inference_receipt.q4k == true and
  .fak.native_inference_receipt.fallback_active == false
' "$run_dir/response.json" >/dev/null
grep -E 'fak_native_runtime_info\{[^}]*engine="inkernel"[^}]*backend="metal"[^}]*model="qwen3\.8"[^}]*planner="inkernel"[^}]*owner="fak"' "$run_dir/metrics.txt" >/dev/null || {
  echo "metrics do not prove the exact fak-native Qwen3.8 Metal identity" >&2
  exit 1
}

response_sha="$(shasum -a 256 "$run_dir/response.json" | awk '{print $1}')"
output_tokens="$(jq -er '.usage.completion_tokens' "$run_dir/response.json")"
forward_path="$(jq -er '.fak.native_inference_receipt.forward_path' "$run_dir/response.json")"
receipt_tmp="$run_dir/public-receipt.json"
jq -n \
  --arg captured "$completed_at" --arg started "$started_at" --arg completed "$completed_at" \
  --arg run_id "$run_id" --arg device "$device" --arg response_sha "$response_sha" \
  --arg forward_path "$forward_path" --arg build_commit "$build_commit" \
  --arg build_version "$build_version" --argjson output_tokens "$output_tokens" '
  {
    schema:"fak-native-qwen38-metal-observation/v1", issue:10005, status:"success",
    captured_at_utc:$captured, completed_at_utc:$completed, run_id:$run_id,
    engine:"inkernel", model:"qwen38:27b", runtime:"native", backend:"metal",
    device:$device, fallback_count:0, fallback_active:false, llama_cpp_used:false,
    build:{version:$build_version,commit:$build_commit,dirty:false,stamped:true,os:"darwin",arch:"arm64"},
    scope:"One bounded local fak-native Qwen3.8 Metal execution; no comparative performance claim.",
    model_identity:{alias:"qwen38:27b",family:"Qwen3.8",quantization:"Q4_K_M"},
    required_execution:{engine:"fak-native",runtime_engine:"inkernel",planner:"inkernel",model_owner:"fak",fallback_count:0,fallback_active:false,llama_cpp_used:false},
    observed_execution:{engine:"fak-native",runtime_engine:"inkernel",planner:"inkernel",model:"Qwen3.8",model_owner:"fak",fallback_count:0,fallback_active:false,llama_cpp_used:false,completed:true,output_tokens:$output_tokens,correlation_key:$run_id},
    native_receipt:{forward_path:$forward_path,q4k:true,sha256:$response_sha},
    live_execution_obtained:true, raw_logs_committed:false, private_identifiers_committed:false
  }
' >"$receipt_tmp"
validate_receipt "$receipt_tmp" "$completed_at"
mkdir -p "$(dirname -- "$output")"
mv -- "$receipt_tmp" "$output"
echo "wrote scrubbed native Metal receipt: $output"
