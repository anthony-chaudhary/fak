#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FAK_BIN="${FAK_BIN:-fak}"
POLICY="examples/deny-as-value/policy.json"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "deny-as-value: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  exit 2
fi

run_case() {
  local label="$1"
  local tool="$2"
  local args="$3"
  echo "== $label =="
  "$FAK_BIN" preflight --policy "$POLICY" --tool "$tool" --args "$args" --explain
  echo
}

run_case "RETRYABLE / MISROUTE" "fix_my_args" '{}'
run_case "WAIT / RATE_LIMITED" "rate_limited_call" '{}'
run_case "ESCALATE / SELF_MODIFY" "write_kernel" '{"path":"internal/kernel/kernel.go"}'
run_case "TERMINAL / POLICY_BLOCK" "delete_account" '{}'
