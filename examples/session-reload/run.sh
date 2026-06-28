#!/usr/bin/env bash
# run.sh — the fak session-reload (recall) operator walkthrough, end to end.
#
# The production motion this proves: a FINISHED agent session persists as a durable
# *core image* — `manifest.json` (the page table: roles + content digests + the
# quarantine state + a frozen world-version) over `cas.json` (the content-addressed
# swap device that holds the bytes). A LATER, SEPARATE process re-opens that image,
# resolves the benign pages byte-identical with zero model tokens, and — the moat —
# a page the write-time gate SEALED stays sealed across the process boundary: it is
# refused on page-in unless a witness `Clear()` ran AND the bytes pass a fresh
# content re-screen. A clearance alone never launders poison, and a tampered swap
# device fails closed at load.
#
# It runs the full reload loop and asserts each step as a witness:
#   A) process A — build a finished session (2 benign results + 1 injection + 1
#      secret), persist it to manifest.json + cas.json, and (in-process) show:
#        - a benign page resolves BYTE-IDENTICAL,
#        - the sealed injection page is REFUSED with no witness,
#        - after a witness Clear() it is STILL refused — the content re-screen
#          re-quarantines it ("clearance does not launder poison").
#   B) the two artifact files exist on disk (the whole portable image).
#   C) process B — a genuinely SEPARATE OS process re-opens the SAME image
#      (`fak debug` attaches to it as to a core dump) and proves the moat survives
#      the boundary: a benign page pages in byte-identical, the sealed page is
#      refused on page-in.
#   D) integrity — flip one byte INSIDE a stored blob (leaving its digest key
#      unchanged) and re-open: load FAILS CLOSED (digest mismatch), refusing to
#      serve any of it.
#
# Why these witnesses prove "more than a snapshot": a plain snapshot reloads
# whatever bytes it stored; recall re-adjudicates on page-in, so the seal is a
# property of the IMAGE, not of the process that wrote it (step C), and the swap
# device is self-verifying so a tamper cannot smuggle laundered bytes back in
# (step D).
#
#   ./run.sh                       # build, run the four witnesses, teardown
#
# Env knobs:
#   FAK_BIN   prebuilt fak binary to use            (default: build ./cmd/fak)
#
# Prerequisites: Go (to build fak) and a POSIX shell. No key, no model, no GPU,
# no network — every reading here is deterministic and offline.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/session-reload -> fak/
log(){ printf '\033[36m[recall]\033[0m %s\n' "$*" >&2; }
pass(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail(){ printf '  \033[31m✗\033[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

BIN_DIR=""; WORK=""; FAILS=0
cleanup(){
  [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true
  [ -n "$WORK" ] && rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 0) the fak binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

WORK="$(mktemp -d)"
IMG="$WORK/recall-image"             # the core image process A writes
IMG_BAD="$WORK/recall-image-tampered"

# --- A) process A: build + persist + the in-process reload guarantees -------------
log "A) process A: record a finished session and persist it as a core image"
A_OUT="$WORK/recall-A.txt"
"$BIN" recall --dir "$IMG" --out "$WORK/recall-report.json" >"$A_OUT" 2>&1 || {
  log "fak recall failed:"; cat "$A_OUT" >&2; exit 1
}
sed 's/^/    /' "$A_OUT" >&2
echo
# a benign page round-trips byte-identical; the sealed page is refused with no
# witness; and a clearance alone does NOT launder it (the re-screen re-quarantines).
grep -q "resolve benign account"            "$A_OUT" && grep -q "RESOLVED"        "$A_OUT" \
  && pass "benign account page resolves byte-identical" \
  || fail "expected a benign page to resolve"
grep -q "no witness Clear"                  "$A_OUT" \
  && pass "sealed injection page REFUSED with no witness" \
  || fail "expected the sealed page to be refused without a witness"
grep -q "does not launder poison"           "$A_OUT" \
  && pass "after a witness Clear() the page is STILL refused (content re-screen) — clearance does not launder poison" \
  || fail "expected the cleared injection page to be re-quarantined by the re-screen"
echo

# --- B) the artifact files on disk -----------------------------------------------
log "B) the persisted core image is two self-contained files:"
if [ -f "$IMG/manifest.json" ] && [ -f "$IMG/cas.json" ]; then
  pass "manifest.json ($(wc -c <"$IMG/manifest.json" | tr -d ' ') B, the page table) + cas.json ($(wc -c <"$IMG/cas.json" | tr -d ' ') B, the content-addressed swap)"
else
  fail "expected manifest.json + cas.json under $IMG"
fi
echo

# --- C) process B: a SEPARATE OS process re-opens the SAME image ------------------
log "C) process B: a fresh process re-opens the image and the moat survives the boundary"
B_OUT="$WORK/recall-B.txt"
# `fak debug` with no --session attaches to the existing core image at --dir, in a
# brand-new OS process that depends on nothing process A left in memory.
"$BIN" debug --dir "$IMG" --out "$WORK/cdb-report.json" >"$B_OUT" 2>&1 || {
  log "fak debug attach failed:"; cat "$B_OUT" >&2; exit 1
}
sed 's/^/    /' "$B_OUT" >&2
echo
grep -Eq "step [0-9]+ get_user_details .*RESOLVED" "$B_OUT" \
  && pass "benign page pages in byte-identical in the fresh process" \
  || fail "expected a benign page to page in across the boundary"
grep -Eq "read_refund_policy .*REFUSED"            "$B_OUT" \
  && pass "the SEALED page is refused on page-in in the fresh process (the cross-process quarantine moat)" \
  || fail "expected the sealed page to be refused on page-in across the boundary"
echo

# --- D) integrity: a tampered swap device fails closed at load -------------------
log "D) integrity: tamper one byte inside a stored blob and re-open the image"
cp -r "$IMG" "$IMG_BAD"
# cas.json is {"<sha256-hex-key>": "<base64-bytes>"}. The keys are lowercase hex, so
# the FIRST uppercase letter in the file is necessarily inside a base64 VALUE (a
# blob's bytes), never a digest key. Flip it: the blob no longer hashes to its
# address. (This is the shell analogue of the recall_test.go corrupt-CAS witness.)
awk 'BEGIN{done=0}{ if(!done && match($0,/[A-Z]/)){ c=substr($0,RSTART,1); r=(c=="Z"?"Y":"Z"); $0=substr($0,1,RSTART-1) r substr($0,RSTART+1); done=1 } print }' \
  "$IMG_BAD/cas.json" > "$IMG_BAD/cas.json.tmp" && mv "$IMG_BAD/cas.json.tmp" "$IMG_BAD/cas.json"
D_OUT="$WORK/recall-D.txt"
if "$BIN" debug --dir "$IMG_BAD" --out "$WORK/cdb-bad.json" >"$D_OUT" 2>&1; then
  fail "a tampered cas.json was loaded — integrity check did NOT fail closed"
  sed 's/^/    /' "$D_OUT" >&2
else
  if grep -Eq "corrupt CAS|digest mismatch" "$D_OUT"; then
    pass "tampered swap device REFUSED at load: $(grep -Eo 'corrupt CAS entry [0-9a-f]+ \(digest mismatch\)' "$D_OUT" | head -1)"
  else
    fail "load failed but not with a digest-mismatch reason:"
    sed 's/^/    /' "$D_OUT" >&2
  fi
fi
echo

if [ "$FAILS" -ne 0 ]; then
  log "$FAILS witness(es) FAILED"; exit 1
fi
log "all witnesses passed — a benign page round-trips byte-identical, a sealed page stays sealed across a real process boundary (clearance alone does not launder it), and a tampered swap device fails closed at load."
