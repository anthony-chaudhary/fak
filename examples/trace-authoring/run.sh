#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "trace-authoring: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  exit 2
fi

echo "== minimal trace =="
"$FAK_BIN" run --trace examples/trace-authoring/minimal.json

echo
echo "== result-side quarantine trace =="
"$FAK_BIN" run --trace examples/trace-authoring/with-poison.json --engine mock
