#!/usr/bin/env bash
# deny-in-60s: watch a default-deny floor refuse an irreversible call, then watch
# the SAME call clear under a permissive floor. No key, no model, no GPU, no network.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "deny-in-60s: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  exit 2
fi

TOOL="drop_production_database"
ARGS='{"database":"prod"}'

echo "== 1/2 DENY — the default-deny floor (policy.json) never mentions $TOOL =="
"$FAK_BIN" preflight --policy examples/deny-in-60s/policy.json --tool "$TOOL" --args "$ARGS" --explain
echo
echo "== 2/2 ALLOW — the permissive floor (policy-permissive.json) lists it =="
"$FAK_BIN" preflight --policy examples/deny-in-60s/policy-permissive.json --tool "$TOOL" --args "$ARGS" --explain
echo
echo "Same binary, same call, no model in the loop. Only the policy file changed —"
echo "the refusal is structural (DEFAULT_DENY: the floor never admitted the tool),"
echo "not a model choosing to behave."
