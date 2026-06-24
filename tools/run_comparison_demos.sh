#!/usr/bin/env bash
# run_comparison_demos.sh — "fak in 30 seconds": the four side-by-side comparisons,
# one after another, in your terminal — and a cross-platform acceptance gate for them.
#
# WHAT THIS IS
#   fak has two value points (a security gate and a performance gate). This script plays
#   the FOUR self-contained, terminal side-by-side comparisons that make them visible —
#   one per value axis — then verifies each still reproduces its documented headline:
#
#     safety      cmd/guarddemo  -print   WITHOUT fak vs WITH fak on the same attack
#                                         (poison admitted + account deleted  →  0 / 0)
#     efficiency  cmd/turntaxdemo -print   a tuned 2026 SOTA agent's forced round-trips
#                                         vs fak's flat 0
#     reuse       cmd/ctxdemo    -bars     prefill tokens the model must re-read:
#                                         tuned warm-cache vs fak
#     tokens      cmd/tokendemo  -print   two meters: model-context tokens kept OUT
#                                         (a prefiltered /bad call) + tool round-trips
#                                         collapsed (a re-read served from cache)
#
#   Every one is MODEL-AGNOSTIC: no weights, no GPU, no API key, no network. The numbers
#   are deterministic (replayed through the real kernel, or exact timing-free token
#   accounting), so a PASS here means the same on a laptop and a MacBook. Colors render
#   when stdout is a TTY and honor NO_COLOR.
#
# WHERE IT RUNS
#   Any box with Go 1.26+: macOS, Linux, or Windows under WSL / Git Bash. `go run` works
#   natively on Windows (it is `go test` that an OS Application-Control policy blocks), so
#   this whole script runs anywhere `go` does.
#
# USAGE
#   bash tools/run_comparison_demos.sh            # play all four, then verify
#   bash tools/run_comparison_demos.sh -q         # quiet: verify only (the acceptance gate)
#
# Exit code: 0 = every comparison reproduced its documented headline; non-zero otherwise.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || { echo "cannot cd to repo root $REPO_ROOT" >&2; exit 2; }

QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

PASS=0
FAIL=0

# gate LABEL  MUST_CONTAIN  -- command...
#   Runs the command (capturing output, so the demo sees a non-TTY stdout and emits no
#   color → clean fixed-string matching) and checks it exits 0 AND prints MUST_CONTAIN.
gate() {
	label="$1"; want="$2"; shift 2
	out="$("$@" 2>&1)"; rc=$?
	ok=1
	[ "$rc" -ne 0 ] && ok=0
	if [ -n "$want" ] && ! printf '%s' "$out" | grep -qF "$want"; then ok=0; fi
	printf '  %-40s ... ' "$label"
	if [ "$ok" -eq 1 ]; then
		printf 'PASS\n'; PASS=$((PASS + 1))
	else
		printf 'FAIL (exit=%d)\n' "$rc"; FAIL=$((FAIL + 1))
		printf '%s\n' "$out" | sed 's/^/      | /' | tail -20
	fi
}

# show LABEL -- command...
#   Runs the command with stdout inherited (a TTY → colored side-by-side) so a human sees
#   the comparison. Skipped in -q mode. Failures are caught by the gate() pass below.
show() {
	[ "$QUIET" -eq 1 ] && return 0
	"$@" || true
}

if [ "$QUIET" -eq 0 ]; then
	echo "==================================================================="
	echo " fak in 30 seconds — four side-by-side comparisons, no model, no GPU"
	echo "==================================================================="
	echo " go:    $(go version 2>/dev/null || echo 'go NOT FOUND')"
	echo " uname: $(uname -sm 2>/dev/null || echo unknown)"
	echo
	echo "### 1/4 · SAFETY — WITHOUT fak vs WITH fak (same attack) ###########"
	show go run ./cmd/guarddemo -print
	echo
	echo "### 2/4 · EFFICIENCY — a tuned SOTA agent's wasted turns vs fak #####"
	show go run ./cmd/turntaxdemo -print
	echo
	echo "### 3/4 · REUSE — tokens the model must re-read (warm/fak) #####"
	show go run ./cmd/ctxdemo -bars
	echo
	echo "### 4/4 · TOKENS — model-context kept out + tool round-trips collapsed ##"
	show go run ./cmd/tokendemo -print
	echo
fi

echo "-- acceptance: each comparison reproduces its documented headline --"
# safety: the red-team scenario breaches 4 WITHOUT fak; the -selfcheck path pins all
# scenarios' safety-floor invariants (fak NEVER breaches).
gate "guarddemo safety floor (4 -> 0)"  "WITHOUT fak: 4 breaches" \
	go run ./cmd/guarddemo -print
gate "guarddemo -selfcheck (all scenarios)" "reproduced the documented safety-floor" \
	go run ./cmd/guarddemo -selfcheck
# efficiency: airline forces 5 round-trips on even a tuned agent; fak deletes them.
gate "turntaxdemo turn-tax (5 forced -> 0)" "tuned SOTA agent: 5 forced round-trips" \
	go run ./cmd/turntaxdemo -print
gate "turntaxdemo -selfcheck (all suites)"  "reproduced the documented turn-tax" \
	go run ./cmd/turntaxdemo -selfcheck
# reuse: the exact, timing-free fak-vs-tuned token win.
gate "ctxdemo reuse bars (fak re-reads less)" "fak makes the model re-read" \
	go run ./cmd/ctxdemo -bars
# tokens: a prefiltered /bad call keeps 1,452 model-context tokens out (win 1); a re-read
# collapses tool round-trips (win 2). -selfcheck pins every suite's ledger invariants.
gate "tokendemo prefilter win (1,452 model tok)" "fak keeps 1,452 tokens out of the model" \
	go run ./cmd/tokendemo -print -suite prefilter-bad-calls
gate "tokendemo -selfcheck (all suites)" "reproduced the documented ledger" \
	go run ./cmd/tokendemo -selfcheck
echo

echo "== summary: $PASS passed, $FAIL failed =="
if [ "$FAIL" -ne 0 ]; then
	echo "ACCEPTANCE FAILED"
	exit 1
fi
echo "ACCEPTANCE PASSED — all four side-by-side comparisons reproduced their headline"
