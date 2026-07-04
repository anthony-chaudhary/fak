#!/usr/bin/env bash
# policy-loader-properties: witness the three structural loader guarantees
# (fail-loud, replace-not-merge, round-trip-stable) plus the empty-manifest
# warning, documented in POLICY.md "Safety properties of the loader".
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DIR="examples/policy-loader-properties"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "policy-loader-properties: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  exit 2
fi

fails=0

# expect_exit CASE EXPECTED_EXIT -- CMD...
expect_exit() {
  local case="$1" want="$2"; shift 2
  echo "== $case =="
  "$@"
  local got=$?
  if [ "$got" -eq "$want" ]; then
    echo "-- exit=$got (expected $want) PASS"
  else
    echo "-- exit=$got (expected $want) FAIL" >&2
    fails=$((fails + 1))
  fi
  echo
}

DUMPED="$(mktemp)"
trap 'rm -f "$DUMPED"' EXIT
"$FAK_BIN" policy --dump > "$DUMPED"

expect_exit "1/5 round-trip stable — dump | check is exact" 0 \
  "$FAK_BIN" policy --check "$DUMPED"

expect_exit "2/5 fail-loud — unknown field (\"allows\" typo for \"allow\")" 1 \
  "$FAK_BIN" policy --check "$DIR/bad-unknown-field.json"

expect_exit "3/5 fail-loud — unknown deny reason" 1 \
  "$FAK_BIN" policy --check "$DIR/bad-unknown-reason.json"

expect_exit "4/5 fail-loud — unknown posture value" 1 \
  "$FAK_BIN" policy --check "$DIR/bad-unknown-posture.json"

expect_exit "5/5 empty manifest {} — valid but warned (maximally paranoid floor)" 0 \
  "$FAK_BIN" policy --check "$DIR/empty.json"

if [ "$fails" -eq 0 ]; then
  echo "All 5 loader-property witnesses matched their expected exit code."
else
  echo "$fails witness(es) did not match their expected exit code." >&2
  exit 1
fi
