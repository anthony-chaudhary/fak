#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GO_BIN="${GO_BIN:-go}"
SCHEMA="examples/grammar-repair-demo/create-support-ticket.schema.json"
POLICY="examples/customer-support-readonly-policy.json"

echo "== positional call repaired =="
"$GO_BIN" run ./cmd/fak preflight \
  --policy "$POLICY" \
  --tool create_support_ticket \
  --grammar-schema "$SCHEMA" \
  --args '{"_positional":["please help me"]}' \
  --show-dispatched-args

echo
echo "== arity mismatch denied =="
"$GO_BIN" run ./cmd/fak preflight \
  --policy "$POLICY" \
  --tool create_support_ticket \
  --grammar-schema "$SCHEMA" \
  --args '{"_positional":["a","b","c"]}' \
  --show-dispatched-args
