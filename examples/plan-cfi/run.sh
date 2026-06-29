#!/bin/bash
# run.sh — witness the plan-CFI (plan control-flow-integrity) adjudicator end-to-end.
#
# WHAT plan-CFI IS. It applies the classic control-flow-integrity idea (every indirect
# branch must target a known call site) to an AGENT'S PLAN. An agent's "control flow" is
# its sequence of tool calls; the approved plan is its call graph. A step that deviates
# from the approved plan shape — an "unplanned gadget" — is trapped and ESCALATED for
# human approval (a RequireApproval verdict), NOT hard-denied.
#
# WHY THIS DEMO IS A WITNESS, NOT A CLI WALKTHROUGH. plancfi is an IN-BAND Go adjudicator
# keyed by TraceID: the operator approves a Plan out-of-band (internal/plancfi.Ledger.Declare)
# and the kernel enforces it in-band on the live tool-call stream. There is no CLI verb that
# loads a plan from JSON — so the honest, runnable proof is the package's own green test
# suite, which declares the airline-booking plan and drives BOTH a conforming call (Defer)
# and a deviating call (RequireApproval) through the real Adjudicator. The bundled
# sample-allowlist.json / deviating-call.json are the human-readable rendering of those Go
# Plan{Tools, Mode} values (see README.md).
#
#   ./run.sh               # run the real plancfi witness tests (the load-bearing proof)
#   ./run.sh --check-only  # just print the witness command + the two verdicts it proves
#
# Requires: Go (to run the in-repo tests). No model, no network, no GPU.
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/plan-cfi -> fak/
log(){ printf '\033[36m[plan-cfi]\033[0m %s\n' "$*" >&2; }

# The headline witnesses: a conforming call Defers (CFI has no objection), a deviation
# Escalates (RequireApproval). TestStrictModeDenies proves escalate-vs-deny is a knob.
TESTS='TestConformingCallDefers|TestDeviationEscalates|TestNoPlanDefers|TestStrictModeDenies|TestSequenceMode|TestSessionIsolation'

cat >&2 <<'EOF'

plan-CFI witness — control-flow integrity over an agent's PLAN
  the operator declares an approved Plan per TraceID; the kernel enforces it in-band.
  legal plan shapes:        examples/plan-cfi/sample-allowlist.json
  the off-plan step:        examples/plan-cfi/deviating-call.json

  what the witness proves:
    on-plan  call  (e.g. search_flights, book_reservation) -> Defer          (CFI has no objection)
    off-plan call  (send_email — the exfil gadget)          -> RequireApproval (escalate to a human)

EOF

WITNESS="go test -run '$TESTS' -v ./internal/plancfi"
log "witness command:  $WITNESS   (run from $FAK_DIR)"

if [ "${1:-}" = "--check-only" ]; then
  log "--check-only: not executing. Run without the flag to execute the witness above."
  exit 0
fi

command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl, then re-run."; exit 1; }
log "running the plancfi witness tests…"
( cd "$FAK_DIR" && eval "$WITNESS" )
log "witness PASS — a conforming call Defers, an off-plan call returns RequireApproval."
