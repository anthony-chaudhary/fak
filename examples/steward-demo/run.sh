#!/bin/bash
# run.sh — one command to run the fak single-invariant steward demo (#324).
#
# A *steward* is a single-invariant runtime checker that fires ONLY with an
# independently-authored witness — never on the model's own claim. This demo drives a
# small frozen scenario (sample-steward.json) through a faithful, dependency-free
# re-enactment of fak's real steward package (internal/steward/steward.go) and shows:
#
#   1. a steward RAISING with an independent witness (a real alert);
#   2. the SAME apparent condition WITHOUT a witness — SUPPRESSED (the model can't
#      self-accuse): the load-bearing witness-vs-no-witness distinction; and
#   3. the META-STEWARD pruning a steward that never fires across a soak (dead-code
#      detection on the invariant layer itself).
#
#   ./run.sh                 # run the demo
#   ./run.sh --no-color      # plain output
#   ./run.sh --config PATH   # a different scenario file
#
# Requires: Python 3 (stdlib only). No model, no network, no Go toolchain — the demo is
# a deterministic re-enactment whose AUTHORITATIVE witness is the Go test suite,
# internal/steward/steward_test.go (units 87–92). To run that witness directly:
#
#   go test ./internal/steward/ -run 'TestSecretInContext|TestLeaseDisjointness|TestKPIRegression|TestVDSOSoundness|TestPrunePopulation' -v
#
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else
    echo "[demo] Python 3 not found — install it (https://www.python.org/downloads), or set PYTHON=/path/to/python3" >&2
    exit 1
  fi
fi

exec "$PY" "$HERE/demo.py" "$@"
