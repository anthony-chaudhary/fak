#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "turntax: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  exit 2
fi

echo "== turn-tax workload =="
"$FAK_BIN" turntax --trace examples/turntax/sample-trace.json

echo
echo "== happy-path control =="
"$FAK_BIN" turntax --trace examples/turntax/sample-trace-happy.json
