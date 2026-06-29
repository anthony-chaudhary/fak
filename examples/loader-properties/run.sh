#!/bin/bash
# run.sh — witness the THREE loader safety properties from POLICY.md "Safety
# properties of the loader", plus the empty-manifest case, with one command.
#
# A misconfigured policy must fail at LOAD time, not at the first bad call in
# production. `fak policy --check` is the loader: it parses a manifest exactly the
# way `fak serve --policy` would and prints the floor it admits (or refuses, loudly).
# All five witnesses are pure parse-time adjudication — no model, no server, no
# network, no GPU.
#
#   1. round-trip exact   : --dump a manifest, --check it back  -> OK (exit 0)
#   2. fail-loud, typo    : an unknown JSON field ("allows")    -> non-zero, named
#   3. fail-loud, reason  : an unknown deny reason              -> non-zero, named
#   4. fail-loud, posture : an unknown posture value            -> non-zero, named
#   5. empty {} floor     : valid, but explicitly WARNED        -> OK + DEFAULT_DENY note
#
# The exit code gates on all five: the run FAILS if any bad manifest is accepted,
# if the round-trip is not OK, or if the empty floor is not warned.
#
#   ./run.sh                # build fak (or use $FAK_BIN), run all witnesses
#   FAK_BIN=./fak ./run.sh  # use a prebuilt binary instead of building
#
# Requires: Go (to build fak) OR a prebuilt FAK_BIN. No Python, no model, no network.
#
# NOTE ON FLAGS: the loader is `fak policy --check <FILE>` and `fak policy --dump`.
# `--check` reads a manifest FILE (it does not read stdin), so the round-trip below
# writes `--dump` to a temp file and checks that file — see README.md "A note on the
# round-trip command".
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"            # examples/loader-properties -> fak/
log(){ printf '\033[36m[loader-properties]\033[0m %s\n' "$*" >&2; }

BIN_DIR=""
TMP_DUMP=""
cleanup(){
  [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true
  [ -n "$TMP_DUMP" ] && rm -f "$TMP_DUMP" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) the kernel binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

fail=0

# check_refuses NAME FILE  — assert `policy --check FILE` exits non-zero AND prints
# a named error containing $3 (the load-fails-loud witness).
check_refuses(){
  local name="$1" file="$2" want="$3" out rc
  out="$("$BIN" policy --check "$file" 2>&1)" && rc=0 || rc=$?
  printf '  %s\n' "$out" | sed 's/^/  /'
  if [ "$rc" = 0 ]; then
    log "  ✗ $name: expected NON-ZERO exit (a silent accept is the bug this prevents), got exit 0"
    fail=1
    return
  fi
  case "$out" in
    *"$want"*) log "  ✓ $name: refused at load (exit $rc), error names \"$want\"" ;;
    *)         log "  ✗ $name: refused (exit $rc) but error did not name \"$want\""; fail=1 ;;
  esac
}

# --- property 1: ROUND-TRIP STABLE ----------------------------------------------------
echo
log "property 1 — ROUND-TRIP STABLE: the manifest the binary emits parses back exactly"
TMP_DUMP="$(mktemp)"
"$BIN" policy --dump > "$TMP_DUMP"
if OUT="$("$BIN" policy --check "$TMP_DUMP" 2>&1)"; then
  printf '  %s\n' "$OUT" | head -1 | sed 's/^/  /'
  log "  ✓ --dump | --check round-trips to a valid floor (exit 0)"
else
  printf '  %s\n' "$OUT" | sed 's/^/  /'
  log "  ✗ round-trip FAILED — --dump emitted a manifest --check rejects"
  fail=1
fi

# --- property 2: FAIL-LOUD on config errors (three witnesses) --------------------------
echo
log "property 2 — FAIL-LOUD: a malformed/unknown manifest is a fatal LOAD error, never a"
log "             silent fall-back to a more permissive default"
echo
log "  2a) unknown JSON field — \"allows\" instead of \"allow\" (the classic typo)"
check_refuses "unknown-field" "$HERE/bad-unknown-field.json" 'unknown field "allows"'
echo
log "  2b) unknown deny reason — a code outside the closed refusal vocabulary"
check_refuses "unknown-deny-reason" "$HERE/bad-deny-reason.json" 'unknown deny reason'
echo
log "  2c) unknown posture value — not fail_closed|admit_and_log"
check_refuses "unknown-posture" "$HERE/bad-posture.json" 'unknown posture'

# (bonus) a structurally malformed manifest is ALSO fail-loud, not a silent empty floor.
echo
log "  2d) malformed JSON (truncated) — invalid manifest, not an empty permissive floor"
check_refuses "malformed-json" "$HERE/bad-malformed.json" 'invalid manifest'

# --- property 3: REPLACE-NOT-MERGE + the EMPTY {} floor -------------------------------
echo
log "property 3 — REPLACE-NOT-MERGE: a loaded manifest IS the whole floor. The extreme"
log "             case is the empty manifest {} — valid, but it allows NOTHING."
if OUT="$("$BIN" policy --check "$HERE/empty.json" 2>&1)"; then
  printf '%s\n' "$OUT" | sed 's/^/  /'
  case "$OUT" in
    *"DEFAULT_DENY"*|*"nothing is affirmatively allowed"*|*"EVERY call"*)
      log "  ✓ empty {} is VALID (exit 0) but explicitly WARNED: every call -> DEFAULT_DENY" ;;
    *)
      log "  ✗ empty {} accepted but NOT warned — an adopter could ship a dead floor by accident"; fail=1 ;;
  esac
else
  rc=$?
  printf '%s\n' "$OUT" | sed 's/^/  /'
  log "  ✗ empty {} was REJECTED (exit $rc) — POLICY.md says it must be valid (the paranoid floor)"
  fail=1
fi

echo
if [ "$fail" = 0 ]; then
  log "loader properties test passed  ·  round-trip exact  ·  four bad manifests refused at load  ·  empty floor warned"
else
  log "loader properties test FAILED — see the ✗ lines above"
fi
exit "$fail"
