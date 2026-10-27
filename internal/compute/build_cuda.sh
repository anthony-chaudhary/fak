#!/usr/bin/env bash
# Compatibility entrypoint; native Go owns compilation, archives and receipts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PATH="/usr/local/go/bin:$PATH"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
exec "${GO:-go}" run ./cmd/fak-dev build-cuda legacy "$@"
