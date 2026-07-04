#!/usr/bin/env bash
# verify-release-checksums_test.sh - proves verify-release-checksums.sh
# actually catches a corrupted release asset (#1369 acceptance criterion:
# "a deliberately corrupted asset (test) makes the job fail").
#
# Run: bash .github/scripts/verify-release-checksums_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/verify-release-checksums.sh"

fails=0
pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; fails=$((fails + 1)); }

tmpd="$(mktemp -d)"
trap 'rm -rf "$tmpd"' EXIT

good="$tmpd/good"
mkdir -p "$good"
echo "archive contents" >"$good/fak_1.0.0_linux_amd64.tar.gz"
(cd "$good" && sha256sum fak_1.0.0_linux_amd64.tar.gz >fak_1.0.0_linux_amd64.tar.gz.sha256)

if bash "$SCRIPT" "$good" >/dev/null 2>&1; then
    pass "matching sha256 -> verifier succeeds"
else
    fail "matching sha256 should succeed"
fi

corrupted="$tmpd/corrupted"
mkdir -p "$corrupted"
echo "archive contents" >"$corrupted/fak_1.0.0_linux_amd64.tar.gz"
(cd "$corrupted" && sha256sum fak_1.0.0_linux_amd64.tar.gz >fak_1.0.0_linux_amd64.tar.gz.sha256)
# Corrupt the archive AFTER its sidecar was computed, mimicking a truncated
# or tampered upload landing on the release page.
echo "tampered" >>"$corrupted/fak_1.0.0_linux_amd64.tar.gz"

if bash "$SCRIPT" "$corrupted" >/dev/null 2>&1; then
    fail "corrupted archive should fail checksum verification"
else
    pass "corrupted archive -> verifier fails"
fi

missing="$tmpd/missing"
mkdir -p "$missing"
touch "$missing/fak_1.0.0_linux_amd64.tar.gz"
if bash "$SCRIPT" "$missing" >/dev/null 2>&1; then
    fail "no .sha256 sidecars should fail (nothing to verify against)"
else
    pass "no .sha256 sidecars -> verifier fails"
fi

echo
if [[ "$fails" -eq 0 ]]; then
    echo "ALL PASS"
else
    echo "$fails FAILED"
    exit 1
fi
