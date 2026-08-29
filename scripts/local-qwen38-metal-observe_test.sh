#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
launcher="$script_dir/local-qwen38-metal-observe.sh"
receipt="$script_dir/../tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/fak-qwen38-metal-observe-test.XXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT INT TERM

completed="$(jq -er '.completed_at_utc' "$receipt")"
"$launcher" --validate "$receipt" --now "$completed" >/dev/null

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

reject engine '.engine = "llama.cpp"'
reject model '.model = "qwen36:30b"'
reject runtime '.runtime = "auto"'
reject backend '.backend = "cpu"'
reject device '.device = ""'
reject fallback_count '.fallback_count = 1'
reject fallback_active '.fallback_active = true'
reject llama_cpp '.llama_cpp_used = true'
reject nested_fallback_count '.observed_execution.fallback_count = 1'
reject nested_fallback_active '.observed_execution.fallback_active = true'
reject run_id '.run_id = "private-host-request-123"'
reject build_commit '.build.commit = "unknown"'
reject dirty_build '.build.dirty = true'
reject nested_engine '.observed_execution.engine = "gateway"'
reject missing_output '.observed_execution.output_tokens = 0'
reject raw_logs '.raw_logs_committed = true'

stale="$(rfc3339_from_epoch $(( $(epoch "$completed") + 901 )))"
if "$launcher" --validate "$receipt" --now "$stale" >/dev/null 2>&1; then
  echo "validator accepted a stale receipt" >&2
  exit 1
fi

if rg -n '/Users/|/home/|hostname|serve\.log|raw_response' "$receipt" >/dev/null; then
  echo "receipt contains a private path or raw-log marker" >&2
  exit 1
fi

echo "local Qwen3.8 Metal observer tests: PASS"
