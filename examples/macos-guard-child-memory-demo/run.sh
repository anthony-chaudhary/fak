#!/bin/sh
# run.sh — runnable entry for the macOS default guard child-memory containment demo.
#
# Demonstrates child process-tree memory containment on macOS (Darwin):
# - Host-derived default RSS threshold: clamp(physical/4, 1GiB, 64GiB)
# - Metric typing: strictly resident set size (RSS), not commit charge
# - Fail-closed thresholding and deterministic offender selection
# - Structured receipt emission under schema fak.guard.child-resource.v1
#
# Usage:
#   ./run.sh              # full walkthrough + selfcheck
#   ./run.sh -selfcheck   # headless invariant verification
#   ./run.sh -json        # machine-readable JSON output
#
# Requires: Go 1.26+. No model, no network, no API key, no GPU.
set -e
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
exec go run "$HERE/main.go" "$@"
