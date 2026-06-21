#!/bin/bash
# run.sh — run the dynamic AgentDojo red-team battery against the stacked defense.
#
# Go-only, deterministic, in-process: NO model, NO network, NO API key. It scores
# every attack in the expanded battery against TWO defenses (detection-only and
# the shipped full-stack) and prints the per-attack verdict stream, then folds the
# outcomes into a frozen harvest LabelRow corpus. The same invocation prints
# byte-identical output on every run and platform.
#
#   ./run.sh                 # full expanded battery (seeds + generative expansion)
#   ./run.sh --seeds         # the hand-authored seed battery only
#   ./run.sh --seed 7        # fix the report ordering with a seed (still deterministic)
#   ./run.sh --json          # machine-readable outcome stream
#
# The bundled policy this battery exercises is the read-only customer-support
# policy at examples/customer-support-readonly-policy.json — an agent that may
# read but must never exfiltrate, leak a secret, or destroy state. The battery
# asks: under an ADAPTIVE attacker who rephrases past the lexical gate, does the
# defense still hold? (Answer: only the full stack does — see README.md.)
#
# Exit code IS the ASR gate: 0 iff full-stack ASR == 0 (every attack barred);
# non-zero if the shipped defense let any attack through (a regression).
#
# Requires: Go (https://go.dev/dl). Nothing else.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"   # examples/agentdojo-redteam -> fak/
POLICY="$FAK_DIR/examples/customer-support-readonly-policy.json"
log(){ printf '\033[36m[redteam]\033[0m %s\n' "$*" >&2; }

command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl"; exit 1; }
[ -f "$POLICY" ] && log "bundled policy under test: examples/customer-support-readonly-policy.json" \
                 || log "note: bundled policy not found at $POLICY (the battery still runs; the policy is the documented target shape)"

log "running the dynamic battery (deterministic, Go-only, no model)…"
echo
# The battery is a fixed matrix scored in-process; go run keeps it Go-only with no
# build artifact left behind. All flags pass straight through to the runner.
exec go run -C "$FAK_DIR" ./cmd/agentdojoredteam "$@"
