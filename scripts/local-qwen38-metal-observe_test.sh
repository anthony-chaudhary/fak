#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
launcher="$script_dir/local-qwen38-metal-observe.sh"
receipt="$script_dir/../tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/fak-qwen38-metal-observe-test.XXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT INT TERM

completed="$(jq -er '.completed_at_utc' "$receipt")"
"$launcher" --validate "$receipt" --now "$completed" >/dev/null
"$launcher" --validate-dashboard-queries "$receipt" >/dev/null

epoch() {
  date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$1" '+%s' 2>/dev/null ||
    date -u -d "$1" '+%s' 2>/dev/null
}

rfc3339_from_epoch() {
  date -u -r "$1" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null ||
    date -u -d "@$1" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null
}

reject() {
  local name="$1" filter="$2"
  jq "$filter" "$receipt" >"$tmp/$name.json"
  if "$launcher" --validate "$tmp/$name.json" --now "$completed" >/dev/null 2>&1; then
    echo "validator accepted invalid receipt mutation: $name" >&2
    exit 1
  fi
}

reject_dashboard_receipt() {
  local name="$1" filter="$2"
  jq "$filter" "$receipt" >"$tmp/$name-query.json"
  if "$launcher" --validate-dashboard-queries "$tmp/$name-query.json" >/dev/null 2>&1; then
    echo "dashboard query validator accepted invalid receipt mutation: $name" >&2
    exit 1
  fi
}

reject_metrics() {
  local name="$1" content="$2"
  printf '%s\n' "$content" >"$tmp/$name.prom"
  if "$launcher" --validate-dashboard-queries "$tmp/$name.prom" >/dev/null 2>&1; then
    echo "dashboard query validator accepted invalid metrics: $name" >&2
    exit 1
  fi
}

# 1. Enforce engine=fak-native
reject engine '.engine = "llama.cpp"'
reject required_engine '.required_execution.engine = "llama.cpp"'
reject nested_engine '.observed_execution.engine = "gateway"'
reject observed_engine_not_fak_native '.observed_execution.engine = "vllm"'

# 2. Enforce model=qwen38:27b
reject model '.model = "qwen36:30b"'

# 3. Enforce qwen38_runtime=native
reject runtime '.runtime = "auto"'
reject qwen38_runtime '.qwen38_runtime = "llama.cpp"'
reject qwen38_runtime_auto '.qwen38_runtime = "auto"'
reject required_qwen38_runtime '.required_execution.qwen38_runtime = "auto"'
reject nested_qwen38_runtime '.observed_execution.qwen38_runtime = "auto"'

# 4. Enforce backend=metal
reject backend '.backend = "cpu"'
reject required_backend '.required_execution.backend = "cuda"'
reject nested_backend '.observed_execution.backend = "cpu"'
reject forward_path_not_metal '.native_receipt.forward_path = "cpu/qwen"'
reject device '.device = ""'

# 5. Enforce fallback_count=0
reject fallback_count '.fallback_count = 1'
reject required_fallback_count '.required_execution.fallback_count = 1'
reject nested_fallback_count '.observed_execution.fallback_count = 1'
reject fallback_active '.fallback_active = true'
reject required_fallback_active '.required_execution.fallback_active = true'
reject nested_fallback_active '.observed_execution.fallback_active = true'

# 6. Enforce llama_cpp_used=false
reject llama_cpp '.llama_cpp_used = true'
reject required_llama_cpp '.required_execution.llama_cpp_used = true'
reject nested_llama_cpp '.observed_execution.llama_cpp_used = true'

# 7. Enforce execution tracking and privacy
reject run_id '.run_id = "private-host-request-123"'
reject build_commit '.build.commit = "unknown"'
reject dirty_build '.build.dirty = true'
reject missing_output '.observed_execution.output_tokens = 0'
reject raw_logs '.raw_logs_committed = true'

# 8. Live dashboard query validation in receipt
reject dashboard_queries_failed '.dashboard_queries = {status: "failed", queries_passed: 0, failed_queries: 1, zero_coerced: false}'
reject dashboard_queries_zero_coerced '.dashboard_queries = {status: "valid", queries_passed: 4, failed_queries: 0, zero_coerced: true}'
reject dashboard_queries_no_pass '.dashboard_queries = {status: "valid", queries_passed: 0, failed_queries: 0, zero_coerced: false}'

# 9. Live dashboard query validation via --validate-dashboard-queries on receipt
reject_dashboard_receipt backend_not_metal '.backend = "cpu"'
reject_dashboard_receipt fallback_count '.fallback_count = 1'
reject_dashboard_receipt fallback_active '.fallback_active = true'
reject_dashboard_receipt llama_cpp '.llama_cpp_used = true'
reject_dashboard_receipt query_failed '.dashboard_queries = {status: "failed", queries_passed: 0, failed_queries: 1, zero_coerced: false}'

# 10. Live dashboard query validation on metrics
cat >"$tmp/valid_metrics.prom" <<'EOF'
fak_native_runtime_info{engine="inkernel",backend="metal",forward_path="metal/qwen35-hybrid-session-v1",model="qwen3.8",planner="inkernel",owner="fak"} 1
fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="metal/qwen35-hybrid-session-v1"} 1
fak_native_receipt_latest_age_seconds 1.2
fak_native_receipt_latest_stale 0
fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="metal/qwen35-hybrid-session-v1",phase="prefill"} 0.05
fak_native_receipt_signal_supported{signal="kernel"} 1
EOF
"$launcher" --validate-dashboard-queries "$tmp/valid_metrics.prom" >/dev/null

reject_metrics missing_runtime_info 'fak_native_receipt_requests_total{engine="inkernel",backend="metal"} 1'
reject_metrics non_metal_backend 'fak_native_runtime_info{engine="inkernel",backend="cpu",model="qwen3.8",planner="inkernel",owner="fak"} 1'
reject_metrics llama_cpp_engine 'fak_native_runtime_info{engine="llama.cpp",backend="metal",model="qwen3.8",planner="inkernel",owner="fak"} 1'
reject_metrics stale_metrics 'fak_native_runtime_info{engine="inkernel",backend="metal",model="qwen3.8",planner="inkernel",owner="fak"} 1
fak_native_receipt_requests_total{engine="inkernel",backend="metal"} 1
fak_native_receipt_latest_stale 1'
reject_metrics missing_phase 'fak_native_runtime_info{engine="inkernel",backend="metal",model="qwen3.8",planner="inkernel",owner="fak"} 1
fak_native_receipt_requests_total{engine="inkernel",backend="metal"} 1
fak_native_receipt_signal_supported{signal="kernel"} 1'
reject_metrics missing_signal 'fak_native_runtime_info{engine="inkernel",backend="metal",model="qwen3.8",planner="inkernel",owner="fak"} 1
fak_native_receipt_requests_total{engine="inkernel",backend="metal"} 1
fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",phase="prefill"} 0.05'

stale="$(rfc3339_from_epoch $(( $(epoch "$completed") + 901 )))"
if "$launcher" --validate "$receipt" --now "$stale" >/dev/null 2>&1; then
  echo "validator accepted a stale receipt" >&2
  exit 1
fi

if grep -E -n '/Users/|/home/|hostname|serve\.log|raw_response' "$receipt" >/dev/null; then
  echo "receipt contains a private path or raw-log marker" >&2
  exit 1
fi

echo "local Qwen3.8 Metal observer tests: PASS"
