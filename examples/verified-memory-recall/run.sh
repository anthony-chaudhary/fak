#!/usr/bin/env bash
# verified-memory-recall: watch `fak memory recall` turn a markdown memory store into
# a VERIFIED, intent-ranked orientation block. A note whose concrete claim still holds
# renders tagged [fresh]; a note that names a moved/deleted path is WITHHELD with the
# failing claim named (never injected wearing the authority of a fact); a prose-only
# note renders hedged [unverified]. No key, no model, no GPU, no network.
set -euo pipefail

# Run from the repo root: the default artifact verifier resolves repo-relative paths
# against the git working tree, so the "fresh" note's claim (internal/memq/exec.go)
# only checks out when the store is recalled from inside this checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "verified-memory-recall: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  echo "  (or run it straight from source:  FAK_BIN='go run ./cmd/fak' bash examples/verified-memory-recall/run.sh)" >&2
  exit 2
fi

STORE="examples/verified-memory-recall/store"
INTENT="where does the memory algebra executor live"

echo "== the store — three authored notes, one per read-time verdict =="
echo "  fresh.md  names internal/memq/exec.go   (a path that EXISTS -> verifiable, fresh)"
echo "  stale.md  names internal/gonepkg/gone.go (a path that is GONE -> refused as stale)"
echo "  prose.md  a preference, nothing checkable (-> rendered hedged, unverifiable)"
echo

echo "== the orientation block ($FAK_BIN memory recall --store $STORE) =="
"$FAK_BIN" memory recall --store "$STORE" --intent "$INTENT"
echo

echo "The stale note is not silently dropped and it is not rendered as fact — it is"
echo "listed as a refusal with the failing claim (internal/gonepkg/gone.go) named as"
echo "evidence. That is the whole point: a moved file or renamed flag can no longer"
echo "load wearing the authority of a fact. This is a backend, not a new driver —"
echo "'$FAK_BIN memory drivers' is unchanged by it."
