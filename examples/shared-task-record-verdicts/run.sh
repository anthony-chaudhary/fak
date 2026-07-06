#!/usr/bin/env bash
# shared-task-record-verdicts: validate the non-acceptance verdict fixtures
# against the shared task record contract and witness the shared-item
# read-parity property from #2216. No key, no model, no GPU, no network —
# exit 0 is the witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# The internal/sharedtask Go fold was retired (faa9a66b8, #2743) as unwired; the
# live validation authority is the JSON schemas under tools/schemas/shared-*.json.
python3 examples/shared-task-record/validate_shared_items.py \
  examples/shared-task-record-verdicts
