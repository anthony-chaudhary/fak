#!/usr/bin/env bash
# shared-task-record: validate the fixture sequence against the shared task
# record contract. No key, no model, no GPU, no network — exit 0 is the witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

go test ./internal/sharedtask -run TestContractSequenceFixtureValidates -v
