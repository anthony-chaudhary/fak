#!/usr/bin/env bash
# run.sh — the fak context-debugger (cdb) operator walkthrough, end to end.
#
# `fak debug` is a CONTEXT-WINDOW DEBUGGER for a FINISHED agent session, modeled on
# `gdb`/`cdb` over a core dump. It ingests a Claude Code transcript (here the small,
# hand-crafted sample-transcript.jsonl), turns it into a CORE IMAGE — one page per tool
# result, each driven through the SAME shipped trust gate at record time — and exposes
# six read-mostly inspection surfaces. This script ingests the sample and exercises
# every one of them, asserting each as a witness:
#
#   1) info         — the core-image decomposition (pages, benign/sealed, heavy, CAS).
#   2) backtrace    — the page table (the `bt`/memory-map); sealed frames echo NO poison.
#   3) working-set  — Denning's W(query): demand-page only the pages a follow-up touches.
#   4) grep         — a read-only search over the page table that pages in NOTHING.
#   5) examine      — `x` one page through the gate: benign resolves, sealed is REFUSED.
#   6) tombstone    — suppress a page from model-visible recall; the CAS bytes SURVIVE
#                     for audit while the page leaves the working set.
#
# What cdb does NOT do: it makes the gate's decision durable and QUERYABLE, not more
# correct. The sealed pages here are real attacks; on real high-entropy data the same
# gate also yields documented FALSE POSITIVES (see README.md + CLAIMS.md §"Inherited
# detection ceiling"). cdb surfaces every seal with its reason — it does not second-guess
# the detector.
#
# The assertions are STRUCTURAL on purpose (a seal of the right kind exists, the working
# set is a strict subset, the tombstone shrinks the set while the swap device is
# byte-unchanged), so this walkthrough stays green for any well-formed sample, not just
# the one committed beside it. EXAMPLE-OUTPUT.md is a captured run of the committed sample.
#
#   ./run.sh                       # build fak, ingest the sample, run the 6 surfaces
#
# Env knobs:
#   FAK_BIN   prebuilt fak binary to use            (default: build ./cmd/fak)
#
# Prerequisites: Go (to build fak) and a POSIX shell. No key, no model, no GPU, no
# network — every reading here is deterministic and offline.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/context-debugger -> fak/
SAMPLE="$HERE/sample-transcript.jsonl"
QUERY="what refund fee did the customer account show?"
log(){ printf '\033[36m[cdb]\033[0m %s\n' "$*" >&2; }
pass(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail(){ printf '  \033[31m✗\033[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }
show(){ sed 's/^/    /' "$1" >&2; echo; }
# benign_referenced FILE -> the N in "N of M benign page(s) referenced"
benign_referenced(){ grep -oE '[0-9]+ of [0-9]+ benign' "$1" | head -1 | grep -oE '^[0-9]+'; }

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
IMG="$WORK/cdb-image"                 # the core image we ingest the sample into

# --- 1) info: ingest the sample and read the core-image decomposition -------------
log "1) info — ingest the sample transcript as a core image and read its decomposition"
INFO_OUT="$WORK/info.txt"
"$BIN" debug --session "$SAMPLE" --dir "$IMG" --cmd info >"$INFO_OUT" 2>&1 || {
  log "fak debug --session failed:"; cat "$INFO_OUT" >&2; exit 1
}
show "$INFO_OUT"
SEALED=$(grep -oE '"sealed": [0-9]+'      "$INFO_OUT" | grep -oE '[0-9]+')
HEAVY=$(grep -oE '"heavy_pages": [0-9]+'  "$INFO_OUT" | grep -oE '[0-9]+')
DEDUP=$(grep -oE '"dedup_saved": [0-9]+'  "$INFO_OUT" | grep -oE '[0-9]+')
[ "${SEALED:-0}" -ge 1 ] \
  && pass "the write-time trust gate sealed $SEALED page(s) at record time" \
  || fail "expected at least one sealed page"
[ "${HEAVY:-0}" -ge 1 ] \
  && pass "$HEAVY oversize-benign result paged OUT to the swap device (heavy)" \
  || fail "expected at least one heavy (paged-out) page"
[ "${DEDUP:-0}" -ge 1 ] \
  && pass "a duplicate read stored ONCE — content-addressed dedup saved $DEDUP B" \
  || fail "expected the duplicate read to be deduplicated"

# --- 2) backtrace: the page table — sealed frames carry metadata only -------------
log "2) backtrace (bt) — the page table; a sealed frame is a safe descriptor, never poison"
BT_OUT="$WORK/bt.txt"
"$BIN" debug --dir "$IMG" --cmd bt >"$BT_OUT" 2>&1
show "$BT_OUT"
grep -Eq 'SEAL .*(TRUST_VIOLATION|SECRET_EXFIL)' "$BT_OUT" \
  && pass "a sealed frame shows its reason code (TRUST_VIOLATION / SECRET_EXFIL), not its bytes" \
  || fail "expected at least one SEAL frame with a reason code"
if grep -q 'ignore previous instructions' "$BT_OUT" || grep -q 'AKIAIOSFODNN7EXAMPLE' "$BT_OUT"; then
  fail "the backtrace ECHOED sealed-page content (poison leak)"
else
  pass "the page table never echoes a sealed page's bytes (no poison in the map)"
fi

# --- 3) working-set: Denning's W(query) — demand-page only what the query touches --
log "3) working-set (ws) — answer a follow-up by demand-paging only the pages it references"
WS_OUT="$WORK/ws.txt"
"$BIN" debug --dir "$IMG" --cmd ws --query "$QUERY" >"$WS_OUT" 2>&1
show "$WS_OUT"
WS_BEFORE=$(benign_referenced "$WS_OUT")
WS_TOTAL=$(grep -oE '[0-9]+ of [0-9]+ benign' "$WS_OUT" | head -1 | grep -oE 'of [0-9]+' | grep -oE '[0-9]+')
if [ -n "$WS_BEFORE" ] && [ -n "$WS_TOTAL" ] && [ "$WS_BEFORE" -lt "$WS_TOTAL" ]; then
  pass "the follow-up referenced only $WS_BEFORE of $WS_TOTAL benign pages — a strict subset (the rest stayed cold)"
else
  fail "expected the working set to be a strict subset of the benign pages"
fi
grep -q 'poison in set: false' "$WS_OUT" \
  && pass "no poison in the working set (sealed pages are never candidates)" \
  || fail "expected no poison in the working set"

# --- 4) grep: a read-only search over the page table (pages in nothing) -----------
log "4) grep — search the page table for a needle; pages in NOTHING, echoes no sealed bytes"
GREP_OUT="$WORK/grep.txt"
"$BIN" debug --dir "$IMG" --cmd grep --grep "refund" >"$GREP_OUT" 2>&1
show "$GREP_OUT"
if [ -s "$GREP_OUT" ] && ! grep -q 'AKIAIOSFODNN7EXAMPLE' "$GREP_OUT"; then
  pass "grep matched benign page(s) on a content word and paged in nothing"
else
  fail "expected grep to match at least one benign page and echo no secret"
fi

# --- 5) examine: `x` one page through the gate (benign resolves, sealed refused) --
log "5) examine (x) — demand-page ONE page through the gate"
# resolve the first BENIGN step and the first SEALED step from the page table.
BENIGN_STEP=$(grep -E '^  \[ *[0-9]+\]       ' "$BT_OUT" | head -1 | grep -oE '\[ *[0-9]+\]' | grep -oE '[0-9]+')
SEALED_STEP=$(grep -E 'SEAL ' "$BT_OUT" | head -1 | grep -oE '\[ *[0-9]+\]' | grep -oE '[0-9]+')
XB_OUT="$WORK/x-benign.txt"; XS_OUT="$WORK/x-sealed.txt"
"$BIN" debug --dir "$IMG" --cmd x --step "$BENIGN_STEP" >"$XB_OUT" 2>&1
"$BIN" debug --dir "$IMG" --cmd x --step "$SEALED_STEP" >"$XS_OUT" 2>&1
show "$XB_OUT"; show "$XS_OUT"
grep -q 'RESOLVED' "$XB_OUT" \
  && pass "benign page $BENIGN_STEP RESOLVED byte-identical (the gate re-screens, then serves it)" \
  || fail "expected benign page $BENIGN_STEP to resolve"
grep -Eq 'REFUSED .*(TRUST_VIOLATION|SECRET_EXFIL)' "$XS_OUT" \
  && pass "sealed page $SEALED_STEP is REFUSED on page-in (the gate still stands on the reloaded image)" \
  || fail "expected sealed page $SEALED_STEP to be refused"

# --- 6) tombstone: suppress from recall; CAS bytes SURVIVE for audit --------------
log "6) tombstone — suppress a page in the working set from model-visible recall"
# tombstone a benign page the follow-up references. Prefer a UNIQUE page carrying PII (a
# support ticket, here); fall back to the first benign step. The target is found via grep
# (a content-word lookup that pages in NOTHING), so the demo always hits a real page. The
# witness below is structural — the page leaves the working set and the swap device is
# byte-unchanged — so it holds whichever page is chosen.
TS_STEP=$("$BIN" debug --dir "$IMG" --cmd grep --grep "PII" 2>/dev/null \
  | grep -oE '\[ *[0-9]+\]' | grep -oE '[0-9]+' | head -1)
[ -z "$TS_STEP" ] && TS_STEP="$BENIGN_STEP"
CAS_BEFORE=$(wc -c <"$IMG/cas.json" | tr -d ' ')
TS_OUT="$WORK/tombstone.txt"
"$BIN" debug --dir "$IMG" --cmd x --step "$TS_STEP" >"$WORK/x-pre-ts.txt" 2>&1   # resolves while live
"$BIN" debug --dir "$IMG" --cmd tombstone --step "$TS_STEP" \
  --reason "customer PII; suppress from recall" --requested-by "support-agent" >"$TS_OUT" 2>&1
show "$TS_OUT"
grep -q "page $TS_STEP tombstoned" "$TS_OUT" \
  && pass "page $TS_STEP tombstoned — a negative-only context-control request" \
  || fail "expected page $TS_STEP to be tombstoned"

WS2_OUT="$WORK/ws-after.txt"; X_AFTER="$WORK/x-after.txt"
"$BIN" debug --dir "$IMG" --cmd ws --query "$QUERY" >"$WS2_OUT" 2>&1
"$BIN" debug --dir "$IMG" --cmd x --step "$TS_STEP" >"$X_AFTER" 2>&1
show "$WS2_OUT"; show "$X_AFTER"
WS_AFTER=$(benign_referenced "$WS2_OUT")
if [ -n "$WS_AFTER" ] && [ "$WS_AFTER" -lt "$WS_BEFORE" ] && grep -q 'tombstoned skipped' "$WS2_OUT"; then
  pass "the tombstoned page DISAPPEARED from the working set ($WS_BEFORE of $WS_TOTAL -> $WS_AFTER of $WS_TOTAL benign)"
else
  fail "expected the working set to drop the tombstoned page"
fi
grep -Eq 'REFUSED .*tombstoned' "$X_AFTER" \
  && pass "examine of the tombstoned page is REFUSED (the model-visible path is closed)" \
  || fail "expected examine of the tombstoned page to be refused"

CAS_AFTER=$(wc -c <"$IMG/cas.json" | tr -d ' ')
if [ "$CAS_BEFORE" = "$CAS_AFTER" ]; then
  pass "the swap device is byte-for-byte unchanged ($CAS_AFTER B) — the tombstoned page's bytes SURVIVE in CAS for audit"
else
  fail "cas.json changed on tombstone ($CAS_BEFORE -> $CAS_AFTER B) — bytes should be preserved"
fi
echo

if [ "$FAILS" -ne 0 ]; then
  log "$FAILS witness(es) FAILED"; exit 1
fi
log "all witnesses passed — a finished session attached as a core image, six inspection surfaces exercised, the gate held on every page-in, and a tombstone suppressed a page from recall while keeping its bytes for audit."
