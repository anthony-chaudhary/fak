#!/usr/bin/env bash
# race_test.sh -- the local, cgo-PREFLIGHTED way to run fak's suite under the Go
# data-race detector (issue #12 / E-001). ONE command that mirrors the CI
# `race-detector` job (.github/workflows/ci.yml) on a developer box, with the one
# guard `fak/test.sh -race` cannot give you: it refuses to run if cgo is absent.
#
#   tools/race_test.sh                         # whole module, -race -count=1 -timeout=25m
#   tools/race_test.sh ./internal/ctxmmu/      # one package, same race flags
#   tools/race_test.sh -run TestEvict ./internal/model/
#   tools/race_test.sh --check                 # only run the cgo preflight, then exit
#
# WHY THIS EXISTS (the silent-race-blind-build footgun)
# -----------------------------------------------------
# The race detector is ThreadSanitizer, which is C: it needs cgo (CGO_ENABLED=1)
# AND a working C compiler. On a toolchain without them, `go test -race` does NOT
# fail loudly -- it just builds a NON-instrumented binary and reports a cheerful
# green that never looked for a single race. The canonical Windows dev host runs
# CGO_ENABLED=0 with no gcc/clang, so a bare `fak/test.sh -race ./...` there is
# exactly that false green (see docs/testing/race-detector.md).
#
# This wrapper closes that gap: it PROVES a C compiler is present and forces
# CGO_ENABLED=1 before any `-race` build, so the only outcomes are "really ran
# instrumented" or "refused with a clear reason" -- never "silently race-blind".
#
# It delegates the actual run to fak/test.sh, reusing that script's WSL framing,
# GOTOOLCHAIN=auto toolchain fetch, and FAK_FAST=1 ext4 fast path -- this only
# adds the preflight + the CI-matching default flags on top.
#
# EXIT CODES
#   0  the instrumented suite ran (and passed)
#   2  cgo/C-compiler preflight failed -- race detector cannot run on this host
#      (the honest blocker; run it in WSL/Linux/macOS instead -- see the doc)
#   *  whatever `go test -race` itself returned (e.g. 1 on a data race / failure)
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/.." && pwd)"
FAK_TEST="$REPO_ROOT/fak/test.sh"
DOC="docs/testing/race-detector.md"

die_blocked() {
  # Honest blocker: name the missing requirement and where to run it instead,
  # rather than letting -race build a race-blind binary. Exit 2 is distinct from
  # a real test failure (1) so callers/CI can tell "couldn't run" from "found a race".
  echo "race_test.sh: BLOCKED -- $1" >&2
  echo "  The Go race detector needs cgo (CGO_ENABLED=1) + a working C compiler (gcc/clang)." >&2
  echo "  This host can't build an instrumented binary; running -race anyway would silently" >&2
  echo "  produce a NON-instrumented (race-blind) binary and a false green. Run it on a" >&2
  echo "  cgo-capable box instead (WSL/Linux/macOS) -- see $DOC." >&2
  exit 2
}

# --- Preflight 1: the Go toolchain is reachable -------------------------------
# Non-login shells (e.g. `wsl bash tools/race_test.sh`) may not have Go on PATH;
# mirror fak/test.sh's fallback so the preflight doesn't false-negative.
if ! command -v go >/dev/null 2>&1; then
  export PATH="/usr/local/go/bin:${HOME}/go/bin:${PATH}"
fi
command -v go >/dev/null 2>&1 || die_blocked "no 'go' on PATH"

# --- Preflight 2: a C compiler the race build can actually use -----------------
# Resolve the compiler Go would invoke for cgo (go env CC, default cc), then prove
# it both resolves AND runs. Fall back to the usual names so a host with gcc/clang
# but an unset CC still qualifies.
CC_BIN="$(go env CC 2>/dev/null || true)"
[ -n "$CC_BIN" ] || CC_BIN="cc"
have_cc=""
for c in "$CC_BIN" cc gcc clang; do
  if command -v "$c" >/dev/null 2>&1 && "$c" --version >/dev/null 2>&1; then
    have_cc="$c"
    break
  fi
done
[ -n "$have_cc" ] || die_blocked "no working C compiler found (tried: $CC_BIN cc gcc clang)"

echo "race_test.sh: cgo preflight OK -- C compiler '$have_cc' present; forcing CGO_ENABLED=1"

# --check: preflight only, useful in CI/scripts to gate before a long -race run.
if [ "${1:-}" = "--check" ]; then
  echo "race_test.sh: --check passed (this host can build the race detector)"
  exit 0
fi

[ -x "$FAK_TEST" ] || [ -f "$FAK_TEST" ] || die_blocked "cannot find fak/test.sh at $FAK_TEST"

# --- Build the test args: -race + CI-matching defaults + caller args -----------
# Mirror the CI job (go test -race -count=1 -timeout=25m ./...). Defaults are only
# applied when the caller didn't already supply that flag, and -count=1 forces a
# real uncached run so a green is a fresh read-back, never a stale cache hit.
args=("$@")
have_flag() { local f; for a in "${args[@]:-}"; do case "$a" in $1) return 0;; esac; done; return 1; }

final=("-race")
have_flag "-count=*" || have_flag "-count" || final+=("-count=1")
have_flag "-timeout=*" || have_flag "-timeout" || final+=("-timeout=25m")

# Append caller args; default the target to ./... when none was given.
if [ "${#args[@]}" -eq 0 ]; then
  final+=("./...")
else
  final+=("${args[@]}")
fi

export CGO_ENABLED=1
echo "race_test.sh: exec fak/test.sh ${final[*]}"
exec bash "$FAK_TEST" "${final[@]}"
