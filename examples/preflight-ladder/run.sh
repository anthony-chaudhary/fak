#!/bin/bash
# run.sh — walk the pre-flight ladder, cheapest rung first, on real calls.
#
# Each step is one `fak preflight` witness: the kernel folds the rung ladder over a
# single proposed tool call and prints which rung produced the verdict. NOTHING here
# runs a model or executes a tool — pre-flight is the layer BEFORE either. The two
# headline rungs (#313):
#
#   rung 0  static parse   — are the args even valid JSON?      (catches malformed args)
#   rung 1  schema check    — required fields present, right types (catches schema violations)
#
# cheapest-first, escalate-on-pass: a call only reaches rung 1 if rung 0 passed, and only
# reaches the authoritative monitor if every cheap rung deferred. A rung DENY wins the fold.
#
#   ./run.sh             # the five witnesses below, one line each
#   ./run.sh --explain   # add the per-rung decision trace to every witness
#
# Requires: a `fak` binary. Set FAK_BIN to a prebuilt one, else this builds ./cmd/fak.
# No model, no network, no GPU — it completes in well under a second.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"          # examples/preflight-ladder -> fak/
EXPLAIN=""
[ "${1:-}" = "--explain" ] && EXPLAIN="--explain"
log(){ printf '\033[36m[preflight]\033[0m %s\n' "$*" >&2; }

# the kernel binary (honor a prebuilt FAK_BIN, else build it once)
BIN_DIR=""
cleanup(){ [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl, or set FAK_BIN to a prebuilt fak"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

# pf LABEL TOOL ARGS — print the witness command, then run it.
pf(){
  local label="$1" tool="$2" args="$3"
  printf '\n— %s\n' "$label"
  printf '  $ fak preflight --tool %s --args %q %s\n' "$tool" "$args" "$EXPLAIN"
  "$BIN" preflight --tool "$tool" --args "$args" $EXPLAIN
}

echo "fak pre-flight ladder — cheapest rung first, before any model turn or execution"
echo "  (default capability floor: the built-in tau2 airline tools; see 'fak policy --dump')"

# 1) well-formed call -> every cheap rung defers, the monitor ALLOWs.
pf "1. clean call         -> ALLOW (rungs 0+1 pass, monitor admits)" \
   search_flights '{"origin":"SFO","destination":"JFK"}'

# 2) malformed JSON args  -> rung 0 (static parse) DENIES before anything else looks at it.
pf "2. malformed JSON     -> DENY at RUNG-0 (by=preflight, MALFORMED)" \
   search_flights '{"origin":"SFO",}'

# 3) unquoted/garbage args -> still rung 0: the args are not valid JSON.
pf "3. unparseable args   -> DENY at RUNG-0 (by=preflight, MALFORMED)" \
   search_flights '{origin:SFO}'

# 4) wrong tool entirely  -> passes the cheap rungs (args ARE valid JSON), the monitor
#    fail-closes on an unsanctioned tool. Shows the ladder ESCALATING past rungs 0/1.
pf "4. unknown tool       -> DENY at the MONITOR (by=monitor, DEFAULT_DENY)" \
   delete_everything '{}'

# 5) a tool whose grammar the kernel does not know -> the grammar rung FAILS OPEN
#    (defers), the args parse, and the monitor decides. This is the honest fail-mode:
#    pre-flight never invents a verdict for a call it cannot reason about.
pf "5. unknown grammar    -> fail-open: grammar rung defers, monitor decides (ALLOW)" \
   search_flights '{"origin":"SFO"}'

echo
echo "every verdict above came from the rung ladder or the monitor — no model turn, no tool ran."
echo "rung-1 schema validation is REAL (internal/preflight, units 47–51) but the standalone"
echo "'fak preflight' default floor installs no per-tool schema, so rung 1 DEFERS here; see README.md."
