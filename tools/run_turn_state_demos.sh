#!/usr/bin/env bash
# run_turn_state_demos.sh — cross-platform acceptance for the OTHER key turn/state demos.
#
# WHAT THIS IS
#   The five fleet demos (fanbench, fleetbench, fak turntax, radixbench, ctxdemo) are
#   witnessed in GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md §3. This
#   script is the dog-food acceptance gate for the OTHER turn/state demos — the ones that
#   prove the kernel's STATE properties (provable deletion, causal eviction, context
#   admission) and the browser TURN demo's headless data path:
#
#     deletioncert  (STATE) bit-exact KV eviction + tamper-evident certificate
#     causalbench   (STATE) external-write causal invalidation, siblings warm
#     ctxbench      (STATE) write-time context-admission gate over the poison corpus
#     ctxbench -chain      the normgate canonicalize-and-rescan admission driver
#     turntaxdemo   (TURN)  the browser turn-tax race, replayed headless via -selfcheck
#     fak turntax          the canonical headless turn-tax A/B (airline + happy control)
#
#   Every one is MODEL-AGNOSTIC: no weights, no GPU, no API key, no network. The numbers
#   are deterministic and seeded, so a PASS here means the same on a laptop and a MacBook.
#
# WHERE IT RUNS
#   Any box with Go 1.26+: macOS (Apple Silicon arm64 or Intel), Linux, or Windows under
#   WSL / Git Bash. The point of this script is the "run on laptop AND macbook" check: the
#   deterministic invariants must reproduce byte-for-byte across win/amd64 and mac/arm64.
#
#   NOTE on `go test` on native Windows: an OS Application-Control policy blocks freshly
#   compiled native test binaries, so the GO-TEST phase is skipped there — run it under WSL.
#   Set SKIP_GO_TEST=1 to skip it anywhere; the demo phase (the dog-food) always runs.
#
# USAGE
#   bash tools/run_turn_state_demos.sh
#   SKIP_GO_TEST=1 bash tools/run_turn_state_demos.sh     # demos only
#
# Exit code: 0 = every demo + test reproduced its documented invariant; non-zero otherwise.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || { echo "cannot cd to repo root $REPO_ROOT" >&2; exit 2; }

PASS=0
FAIL=0

# check LABEL  MUST_CONTAIN  MUST_NOT_CONTAIN  -- command...
#   MUST_CONTAIN / MUST_NOT_CONTAIN are plain (fixed-string) substrings; pass "" to skip.
#   ASCII-only patterns by design, so grep behaves under a C locale on macOS.
check() {
	label="$1"; want="$2"; notwant="$3"; shift 3
	printf '  %-32s ... ' "$label"
	out="$("$@" 2>&1)"; rc=$?
	ok=1
	[ "$rc" -ne 0 ] && ok=0
	if [ -n "$want" ] && ! printf '%s' "$out" | grep -qF "$want"; then ok=0; fi
	if [ -n "$notwant" ] && printf '%s' "$out" | grep -qF "$notwant"; then ok=0; fi
	if [ "$ok" -eq 1 ]; then
		printf 'PASS\n'; PASS=$((PASS + 1))
	else
		printf 'FAIL (exit=%d)\n' "$rc"; FAIL=$((FAIL + 1))
		printf '%s\n' "$out" | sed 's/^/      | /' | tail -24
	fi
}

echo "== fak turn/state demos — cross-platform dog-food =="
echo "repo:   $REPO_ROOT"
echo "go:     $(go version 2>/dev/null || echo 'go NOT FOUND')"
echo "uname:  $(uname -sm 2>/dev/null || echo unknown)"
echo

echo "-- STATE demos (addressable KV cache: deletion, causal eviction, admission) --"
# deletioncert: the evicted span must leave the context byte-identical to a run that never
# saw the secret, and the tamper-rejection branch must fail closed.
check "deletioncert -selfcheck"      "evicted == never-saw"                  "" \
	go run ./cmd/deletioncert -selfcheck
check "deletioncert (cert minted)"   "provable-deletion certificate minted"  "" \
	go run ./cmd/deletioncert -selfcheck
# causalbench: an external write evicts exactly the dependent read, byte-exact, siblings warm.
check "causalbench"                  "causally evicted exactly the dependent" "" \
	go run ./cmd/causalbench
# ctxbench: the poison corpus must leave 0 trigger bytes in context (the security invariant)
# AND must NOT print "(0 bytes total)" — the byte-accounting regression guard.
check "ctxbench (LEAK == 0)"         "LEAK): 0"                "(0 bytes total)" \
	go run ./cmd/ctxbench
check "ctxbench (2 quarantined)"     "QUARANTINE  2"                          "" \
	go run ./cmd/ctxbench
check "ctxbench -chain (normgate)"   "ctxbench: fak security gates"           "" \
	go run ./cmd/ctxbench -chain
echo

echo "-- TURN demos (turn-tax elimination, headless) --"
# turntaxdemo -selfcheck pins the browser demo's data path to the documented invariants:
# airline turns_saved=9 (forced 5 + elision 4), happy=0, safety inj/destr 1->0, exit 0.
check "turntaxdemo -selfcheck"       "reproduced the documented turn-tax"     "" \
	go run ./cmd/turntaxdemo -selfcheck
# fak turntax: the canonical headless A/B. Airline saves 9; happy (control) inflates nothing.
check "fak turntax (airline = 9)"    "turns=9"                                "" \
	go run ./cmd/fak turntax --suite turntax-airline
check "fak turntax (happy control)"  "turntax-happy"                          "" \
	go run ./cmd/fak turntax --suite turntax-happy
echo

if [ "${SKIP_GO_TEST:-0}" = "1" ]; then
	echo "-- GO-TEST phase: SKIPPED (SKIP_GO_TEST=1) --"
elif [ "$(uname -s 2>/dev/null)" = "Windows_NT" ] || printf '%s' "${OS:-}" | grep -qi 'windows'; then
	echo "-- GO-TEST phase: SKIPPED (native Windows app-control blocks test binaries; run under WSL) --"
else
	echo "-- GO-TEST phase: the package witnesses behind the demos --"
	printf '  %-32s ... ' "go test (turn/state pkgs)"
	tout="$(go test -short -count=1 \
		./internal/turnbench/ ./internal/ctxmmu/ ./internal/recall/ \
		./cmd/causalbench/ ./cmd/turntaxdemo/ ./cmd/sessionbench/ 2>&1)"; trc=$?
	if [ "$trc" -eq 0 ]; then
		printf 'PASS\n'; PASS=$((PASS + 1))
	else
		printf 'FAIL (exit=%d)\n' "$trc"; FAIL=$((FAIL + 1))
		printf '%s\n' "$tout" | sed 's/^/      | /' | tail -30
	fi
fi
echo

echo "== summary: $PASS passed, $FAIL failed =="
if [ "$FAIL" -ne 0 ]; then
	echo "ACCEPTANCE FAILED"
	exit 1
fi
echo "ACCEPTANCE PASSED — every turn/state demo reproduced its documented invariant"
