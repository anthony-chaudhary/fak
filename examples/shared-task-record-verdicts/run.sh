#!/usr/bin/env bash
# shared-task-record-verdicts: validate the non-acceptance verdict fixtures
# against the shared task record contract. No key, no model, no GPU, no
# network — exit 0 is the witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

go test ./internal/sharedtask -run TestContractVerdictsFixtureValidates -v
