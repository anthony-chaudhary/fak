#!/bin/bash
# run.sh — the generic require-witness gate, end to end, with one command.
#
# It folds the SAME adjudicator chain `fak serve` uses over a single high-stakes
# tool call via `fak preflight`, and shows the gate lift a generic call to
# require-witness: the capability monitor would ALLOW `deploy`, but the witness
# rung (shipgate) overrules it to WITNESS — "held pending an independent read-back."
# That is the adjudication-time half this demo can run with no server and no model.
#
#   ./run.sh            # build/locate fak, run the three preflight verdicts
#   FAK_BIN=/path/fak ./run.sh   # use a prebuilt binary instead of building
#
# Requires: a `fak` binary (prebuilt via FAK_BIN, or Go to build ./cmd/fak).
# Pure adjudication — no network, no model, no GPU.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/witness-gate -> fak/
POLICY="$HERE/policy.json"
log(){ printf '\033[36m[witness-gate]\033[0m %s\n' "$*" >&2; }

# 1) the kernel binary: honor a prebuilt FAK_BIN, else build ./cmd/fak.
BIN="${FAK_BIN:-}"
BIN_DIR=""
cleanup(){ [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak kernel -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

run(){ "$BIN" preflight --policy "$POLICY" "$@" </dev/null; }

echo
log "1) a generic high-stakes call (deploy) — the monitor would ALLOW it, the witness rung lifts it"
run --tool deploy --args '{}'
echo

log "2) the decision trace — shipgate (rank 40) overrules the monitor's ALLOW (rank 0)"
run --tool deploy --args '{}' --explain
echo

log "3) an unsanctioned tool (confirm_transfer) — refused at the capability floor, never reaches the gate"
run --tool confirm_transfer --args '{}'
echo

log "the require-witness LIFT is shown above (verdict=WITNESS). Its RESOLUTION — WITNESS -> ALLOW"
log "when corroborated, -> DENY/UNWITNESSED when not — runs on the kernel Submit/serve path and is"
log "witnessed by:  go test ./internal/kernel -run TestRequireWitness   (see README.md)."
