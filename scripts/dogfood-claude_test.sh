#!/usr/bin/env bash
# dogfood-claude_test.sh - offline regression for the bounded backend-readiness
# waits in dogfood-claude.sh (#3499).
#
# No network, no ollama, no python: the test EXTRACTS the real `until curl ...`
# loop bodies from dogfood-claude.sh and runs them with a stub curl that never
# succeeds. It proves the loop now terminates two ways instead of spinning
# forever — via a dead-PID liveness check (kill -0) and via a wall-clock
# deadline — which is exactly the defect #3499 reported (unbounded until-curl
# with no deadline and no kill -0).
#
# Run: bash scripts/dogfood-claude_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/dogfood-claude.sh"

fails=0
pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; fails=$((fails + 1)); }

# extract_loop <marker-substring> prints the `until ... done` block whose `until`
# line contains the marker. Depends on the loops being single, un-nested blocks
# terminated by a `done` at their own indentation (as written in the script).
extract_loop() {
  awk -v marker="$1" '
    $0 ~ "until curl" && index($0, marker) { grab=1 }
    grab { print }
    grab && $0 ~ /^[[:space:]]*done[[:space:]]*$/ { exit }
  ' "$SRC"
}

# Stubs shared by every case: curl always fails (backend never becomes ready),
# die records the reason and breaks the loop by returning non-zero from a
# subshell, cat/log are silenced. We run each extracted block in its own bash -c
# so a `die` (which calls `exit 1` in the real script) cannot kill this harness.
run_case() {
  local block="$1" pid_expr="$2" timeout_s="$3"
  local harness
  harness=$(cat <<EOF
set -uo pipefail
curl() { return 1; }        # backend endpoint never answers
cat()  { return 0; }        # swallow log dumps
die()  { echo "DIED: \$*"; exit 9; }
OLLAMA_HOST="127.0.0.1:1"
SHIM_PORT="1"
MODEL="stub-model"        # a precondition of start_shim_backend in the real script
FAK_DOGFOOD_OLLAMA_TIMEOUT_S="$timeout_s"
FAK_DOGFOOD_SHIM_TIMEOUT_S="$timeout_s"
OLLAMA_PID=$pid_expr
SHIM_PID=$pid_expr
ollama_deadline=\$(( \$(date +%s) + $timeout_s ))
shim_deadline=\$(( \$(date +%s) + $timeout_s ))
$block
echo "NODIE"   # only reached if the loop exited normally (it never should here)
EOF
)
  bash -c "$harness"
}

for name in ollama shim; do
  case "$name" in
    ollama) block="$(extract_loop '$OLLAMA_HOST/api/tags')" ;;
    shim)   block="$(extract_loop '$SHIM_PORT/v1/models')" ;;
  esac

  if [[ -z "$block" || "$block" != *"kill -0"* ]]; then
    fail "$name: extracted a bounded loop with a kill -0 liveness check"
    continue
  fi
  pass "$name: loop has a kill -0 liveness check"
  if [[ "$block" != *deadline* ]]; then
    fail "$name: extracted loop enforces a deadline"
    continue
  fi
  pass "$name: loop enforces a deadline"

  # Case A: dead PID -> the liveness check must fire immediately (well under 5s).
  start=$(date +%s)
  out="$(run_case "$block" '$(bash -c "echo \$\$")' 30 2>&1)"   # a PID that is already gone
  elapsed=$(( $(date +%s) - start ))
  # < 8s absorbs Windows/WSL bash spawn overhead while staying far under the 30s
  # deadline set below — so passing here proves the kill -0 rung fired, not the timer.
  if [[ "$out" == *DIED* && "$out" != *NODIE* && "$elapsed" -lt 8 ]]; then
    pass "$name: dead backend PID -> dies fast via kill -0 (${elapsed}s)"
  else
    fail "$name: dead PID should die fast via kill -0 (elapsed=${elapsed}s out=$out)"
  fi

  # Case B: live PID but endpoint never answers -> the deadline must fire.
  sleep 20 &
  live=$!
  start=$(date +%s)
  out="$(run_case "$block" "$live" 2 2>&1)"
  elapsed=$(( $(date +%s) - start ))
  kill "$live" 2>/dev/null || true
  if [[ "$out" == *DIED* && "$out" != *NODIE* && "$elapsed" -lt 6 ]]; then
    pass "$name: live PID, dead endpoint -> dies via deadline (${elapsed}s)"
  else
    fail "$name: live-PID loop should hit the deadline, not spin (elapsed=${elapsed}s out=$out)"
  fi
done

if [[ "$fails" -eq 0 ]]; then
  echo "PASS: all dogfood-claude bounded-wait checks green"
  exit 0
fi
echo "FAIL: $fails dogfood-claude check(s) failed"
exit 1
