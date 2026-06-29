#!/bin/bash
# run.sh — the tool vDSO cache-hit walkthrough (#333).
#
# Replays two tiny tool-call traces through the fak kernel and prints the
# `vdso_hits=` summary line for each, side by side:
#
#   cache-friendly.json  two identical read-shaped calls  -> the second HITS  (vdso_hits=1)
#   invalidating.json    same two reads, a write between  -> the second MISSES (vdso_hits=0)
#
# The contrast is the whole point: a write-shaped completion bumps the world-version,
# so the post-write read is served by a FRESH call — never a stale cache hit. That is
# the vDSO soundness invariant ("a hit equals a fresh call") made visible.
#
#   ./run.sh                  # build fak (or honor FAK_BIN), run both traces
#   FAK_BIN=~/bin/fak ./run.sh    # use a prebuilt binary instead of building
#
# Requires: a `fak` binary — either Go on PATH (to build ./cmd/fak) or a prebuilt
# FAK_BIN. No model, no server, no network: `fak run --trace` is a pure offline replay.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"            # examples/vdso-cache-hit -> fak/
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

BIN_DIR=""
cleanup(){ [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# 1) the kernel binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak kernel -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

run_trace(){
  # $1 = trace file (repo-relative), $2 = one-line description
  local trace="$1" desc="$2"
  echo
  echo "### $trace — $desc"
  "$BIN" run --trace "$FAK_DIR/$trace"
}

run_trace "examples/vdso-cache-hit/cache-friendly.json" \
  "two identical reads — the second HITS the vDSO content cache (expect vdso_hits=1)"
run_trace "examples/vdso-cache-hit/invalidating.json" \
  "a write lands between the reads — the second MISSES (expect vdso_hits=0)"

echo
echo "Read the by= column: by=vdso is a tier-2 content-cache HIT; by=monitor is a fresh call."
echo "In invalidating.json the post-write read is by=monitor — the write bumped the"
echo "world-version, so a stale hit is impossible (a hit equals a fresh call)."
