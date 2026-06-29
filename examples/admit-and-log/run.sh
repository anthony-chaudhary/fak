#!/usr/bin/env bash
# admit_and_log posture demo (#337) — runs the three `fak preflight` witnesses that
# show the same read-shaped call under both postures, plus the two calls the posture
# will NOT relax. No model, no network: the kernel adjudicates a named call from a
# policy file, so every verdict is deterministic.
#
# Needs only the `fak` binary. Override with FAK=/path/to/fak if it is not on PATH.
set -euo pipefail

FAK="${FAK:-fak}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADMIT="$HERE/research-batch-policy.json"
CLOSED="$HERE/research-batch-fail-closed.json"

if ! command -v "$FAK" >/dev/null 2>&1 && [ ! -x "$FAK" ]; then
  echo "error: '$FAK' not found. Build it (go build -o fak ./cmd/fak) and put it on PATH," >&2
  echo "       or run: FAK=/path/to/fak $0" >&2
  exit 127
fi

echo "=== 1. fail_closed: read-shaped tool off the allow-list -> DEFAULT_DENY ==="
"$FAK" preflight --policy "$CLOSED" --tool read_internal_wiki --args '{}'
echo

echo "=== 2. admit_and_log: SAME call -> ADMIT with would_deny=DEFAULT_DENY metadata ==="
"$FAK" preflight --policy "$ADMIT" --tool read_internal_wiki --args '{}' --explain
echo

echo "=== 3a. admit_and_log: an explicit deny (upload_file) STILL fails closed ==="
"$FAK" preflight --policy "$ADMIT" --tool upload_file --args '{}'
echo

echo "=== 3b. admit_and_log: a write-shaped name (write_report) STILL fails closed ==="
"$FAK" preflight --policy "$ADMIT" --tool write_report --args '{}'
echo

echo "Done. The posture admits read-shaped default-denies (logging would_deny);"
echo "write-shaped calls and explicit denials are untouched."
