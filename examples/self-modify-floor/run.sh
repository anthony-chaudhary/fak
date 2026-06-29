#!/bin/bash
# run.sh — one command to witness the fak SELF_MODIFY floor (#339).
#
# A call-side (layer-1) companion to ../adjudication-demo/run.sh: that demo shows
# DEFAULT_DENY and POLICY_BLOCK; this one shows the SELF_MODIFY rung the adjudication
# demo explicitly does NOT cover. The floor is a PURE FUNCTION of (policy, call) — no
# model, no server, no network — so the witness is two `fak preflight` invocations:
#
#   1. a write-shaped call targeting policy.json   -> DENY  SELF_MODIFY (ESCALATE), glob=policy.json
#   2. the SAME allow-listed tool, harmless target -> ALLOW
#
# The exit code gates on both verdicts: the run FAILS if the protected write is not
# refused as SELF_MODIFY, or if the harmless write is not allowed.
#
#   ./run.sh                # build fak, run both witnesses, check verdicts
#   FAK_BIN=./fak ./run.sh  # use a prebuilt binary instead of building
#
# Requires: Go (to build fak) OR a prebuilt FAK_BIN. No Python, no model, no network.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"            # examples/self-modify-floor -> fak/
POLICY="$HERE/policy.json"
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

BIN_DIR=""
cleanup(){ [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# 1) the kernel binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

# preflight() folds the SAME adjudicator chain `fak serve` uses, but for ONE call, and
# prints `verdict=… reason=… by=…`. We strip the stderr "loaded floor" notice.
preflight(){ "$BIN" preflight --policy "$POLICY" "$@" 2>/dev/null; }

fail=0

# --- witness 1: a write that targets the policy file itself -> SELF_MODIFY ------------
echo
log "witness 1 — write_file targeting policy.json (a self-edit of the floor)"
OUT1="$(preflight --tool write_file --args '{"path":"policy.json","body":"{\"allow\":[\"delete_account\"]}"}')"
echo "  $OUT1"
EXP1="$(preflight --tool write_file --args '{"path":"policy.json","body":"x"}' --explain | grep -E 'disposition:|witness:' | sed 's/^/  /')"
echo "$EXP1"
case "$OUT1" in
  *"verdict=DENY"*"reason=SELF_MODIFY"*) log "  ✓ refused as SELF_MODIFY" ;;
  *) log "  ✗ expected DENY SELF_MODIFY, got: $OUT1"; fail=1 ;;
esac
case "$EXP1" in
  *"disposition: ESCALATE"*) : ;;
  *) log "  ✗ expected disposition ESCALATE"; fail=1 ;;
esac
case "$EXP1" in
  *"witness: policy.json"*) log "  ✓ bounded disclosure: witness is the ONE matched glob (policy.json)" ;;
  *) log "  ✗ expected witness=policy.json (the single offending glob)"; fail=1 ;;
esac

# --- witness 2: the SAME allow-listed tool, a harmless target -> ALLOW -----------------
echo
log "witness 2 — write_file targeting a harmless path (same tool, allowed)"
OUT2="$(preflight --tool write_file --args '{"path":"notes/2026-06-20.md","body":"hi"}')"
echo "  $OUT2"
case "$OUT2" in
  *"verdict=ALLOW"*) log "  ✓ allowed — the floor denies the PATH, not the TOOL" ;;
  *) log "  ✗ expected ALLOW, got: $OUT2"; fail=1 ;;
esac

echo
if [ "$fail" = 0 ]; then
  log "self-modify floor test passed  ·  protected write refused (SELF_MODIFY/ESCALATE, glob disclosed)  ·  harmless write allowed"
else
  log "self-modify floor test FAILED — see the ✗ lines above"
fi
exit "$fail"
