#!/bin/bash
# run.sh - compare raw ctxmmu against the registered normgate+ctxmmu chain.
#
# Go-only and deterministic: no model, network, GPU, API key, jq, or Python.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"
CORPUS="${1:-$HERE/sample-battery.jsonl}"

log(){ printf '[normgate-evasion] %s\n' "$*"; }

command -v go >/dev/null || { log "Go not found - install from https://go.dev/dl"; exit 1; }
[ -f "$CORPUS" ] || { log "corpus not found: $CORPUS"; exit 1; }

log "corpus: ${CORPUS#$FAK_DIR/}"
echo "== registered chain with FAK_NORMGATE=off =="
FAK_NORMGATE=off go run -C "$FAK_DIR" ./cmd/ctxbench -chain -corpus "$CORPUS"

echo
echo "== registered chain with normgate enabled =="
go run -C "$FAK_DIR" ./cmd/ctxbench -chain -corpus "$CORPUS"
