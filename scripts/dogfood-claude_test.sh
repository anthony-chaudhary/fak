#!/usr/bin/env bash
# dogfood-claude_test.sh - offline regressions for dogfood-claude.sh.
#
# (1) bounded backend-readiness waits (#3499, extended to the kernel /healthz wait
#     by #3479): EXTRACTS the real `until curl ...` loop bodies and runs them with
#     a stub curl that never succeeds, proving each loop terminates via a dead-PID
#     liveness check (kill -0) AND a wall-clock deadline — the defect #3499/#3479
#     reported (unbounded until-curl). The `serve` case is the third loop: it always
#     had the kill -0 rung (#3479 cites it as the proof the others omitted one) but
#     no deadline, so an alive-yet-never-healthy kernel still spun forever.
# (2) the external-provider graduation gate (#3034): drives the real `--install`
#     decision path (build-free, via FAK_DOGFOOD_INSTALL_DRYRUN) and `--graduation`
#     to witness BOTH states — an opt-in wrapper that is NOT installed by default,
#     and a graduated provider that IS — plus the --install-all operator override.
#
# No network, no ollama, no python, no toolchain build.
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
PORT="1"
MODEL="stub-model"        # a precondition of start_shim_backend in the real script
FAK_DOGFOOD_OLLAMA_TIMEOUT_S="$timeout_s"
FAK_DOGFOOD_SHIM_TIMEOUT_S="$timeout_s"
FAK_DOGFOOD_HEALTH_TIMEOUT_S="$timeout_s"
OLLAMA_PID=$pid_expr
SHIM_PID=$pid_expr
SERVE_PID=$pid_expr
ollama_deadline=\$(( \$(date +%s) + $timeout_s ))
shim_deadline=\$(( \$(date +%s) + $timeout_s ))
serve_deadline=\$(( \$(date +%s) + $timeout_s ))
$block
echo "NODIE"   # only reached if the loop exited normally (it never should here)
EOF
)
  bash -c "$harness"
}

for name in ollama shim serve; do
  case "$name" in
    ollama) block="$(extract_loop '$OLLAMA_HOST/api/tags')" ;;
    shim)   block="$(extract_loop '$SHIM_PORT/v1/models')" ;;
    serve)  block="$(extract_loop '$PORT/healthz')" ;;
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

# --- external-provider graduation gate (#3034) -------------------------------
# The rule: `--install` symlinks a launcher only if it has GRADUATED; opt-in
# (not-yet-graduated) external-provider launchers stay wrappers unless
# --install-all is passed. We witness BOTH states through the real decision path
# (dry-run, so no toolchain build and no PATH mutation): fak-qwen36-claude is a
# graduated local preset; claude-nim-kimi is the opt-in external-provider wrapper.

# --graduation prints the intrinsic verdicts the installer gates on.
grad_out="$(bash "$SRC" --graduation 2>/dev/null)"
if printf '%s\n' "$grad_out" | grep -Eq '^graduated[[:space:]]+fak-qwen36-claude[[:space:]]'; then
  pass "graduation: fak-qwen36-claude verdict is 'graduated'"
else
  fail "graduation: fak-qwen36-claude should be graduated (got: $(printf '%s\n' "$grad_out" | grep fak-qwen36-claude))"
fi
if printf '%s\n' "$grad_out" | grep -Eq '^opt-in[[:space:]]+claude-nim-kimi[[:space:]]'; then
  pass "graduation: claude-nim-kimi verdict is 'opt-in'"
else
  fail "graduation: claude-nim-kimi should be opt-in (got: $(printf '%s\n' "$grad_out" | grep claude-nim-kimi))"
fi

# --install (default) must install the graduated provider and NOT the opt-in one.
inst_out="$(FAK_DOGFOOD_INSTALL_DRYRUN=1 bash "$SRC" --install 2>/dev/null)"
if printf '%s\n' "$inst_out" | grep -qx 'would-install: fak-qwen36-claude'; then
  pass "install: graduated fak-qwen36-claude WOULD be installed by default"
else
  fail "install: graduated fak-qwen36-claude should be installed by default"
fi
if printf '%s\n' "$inst_out" | grep -qx 'would-install: claude-nim-kimi'; then
  fail "install: opt-in claude-nim-kimi must NOT be installed by default"
else
  pass "install: opt-in claude-nim-kimi is NOT installed by default"
fi
if printf '%s\n' "$inst_out" | grep -q '^opt-in-skip: claude-nim-kimi '; then
  pass "install: opt-in claude-nim-kimi reported with a one-command probe recipe"
else
  fail "install: opt-in claude-nim-kimi should be reported as opt-in-skip with a probe recipe"
fi

# --install-all is the operator override: it installs the opt-in launchers too.
all_out="$(FAK_DOGFOOD_INSTALL_DRYRUN=1 bash "$SRC" --install-all 2>/dev/null)"
if printf '%s\n' "$all_out" | grep -qx 'would-install: claude-nim-kimi'; then
  pass "install-all: override installs the opt-in claude-nim-kimi"
else
  fail "install-all: --install-all should install the opt-in claude-nim-kimi"
fi

if [[ "$fails" -eq 0 ]]; then
  echo "PASS: all dogfood-claude checks green (bounded-wait #3499/#3479 + graduation gate #3034)"
  exit 0
fi
echo "FAIL: $fails dogfood-claude check(s) failed"
exit 1
