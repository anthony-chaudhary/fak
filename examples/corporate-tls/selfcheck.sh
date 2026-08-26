#!/usr/bin/env bash
# Regression witness for run.sh's strict-mode boundary. The fake fak fails the
# first command that is expected to succeed; run.sh must return that status before
# it can print any later witness or its success summary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

FAKE_FAK="$TMP/fak-fails-immediately"
OUT="$TMP/out.txt"
printf '%s\n' '#!/usr/bin/env bash' 'exit 42' >"$FAKE_FAK"
chmod +x "$FAKE_FAK"

code=0
FAK_BIN="$FAKE_FAK" bash "$ROOT/examples/corporate-tls/run.sh" >"$OUT" 2>&1 || code=$?
if [ "$code" -ne 42 ]; then
  printf 'corporate-tls strict-mode selfcheck: got exit %d, want 42\n' "$code" >&2
  cat "$OUT" >&2
  exit 1
fi
if grep -qF '== 2/7' "$OUT" || grep -qF 'corporate-tls: PASS' "$OUT"; then
  echo 'corporate-tls strict-mode selfcheck: run continued after an unexpected failure' >&2
  cat "$OUT" >&2
  exit 1
fi

echo 'corporate-tls strict-mode selfcheck: PASS (unexpected command failure stopped the run)'
