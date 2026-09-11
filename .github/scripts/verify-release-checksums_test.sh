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

# --- v0.55.0 regression: dist/-prefixed checksum lines ------------------------
# release-macos.yml ran shasum from dist/, so the universal sidecar's CONTENT
# (and any aggregate line folded from it) names the archive
# "dist/fak_X_darwin_universal.tar.gz". Verified from the flat assets dir that
# made `sha256sum -c` look for <assets>/dist/... -> "No such file or
# directory" -> the whole verify job red -> the release quarantined
# (v0.55.0 run 102720128853). The verifier now normalizes one leading dist/
# off the name field; these cases pin that both aggregate and sidecar forms
# verify clean, while real corruption and malformed lines still fail.
v55="$tmpd/v55"
mkdir -p "$v55"
echo "universal payload" >"$v55/fak_0.55.0_darwin_universal.tar.gz"
vh="$(cd "$v55" && sha256sum fak_0.55.0_darwin_universal.tar.gz | cut -d' ' -f1)"
printf '%s  dist/fak_0.55.0_darwin_universal.tar.gz\n' "$vh" >"$v55/SHA256SUMS"
printf '%s  dist/fak_0.55.0_darwin_universal.tar.gz\n' "$vh" >"$v55/fak_0.55.0_darwin_universal.tar.gz.sha256"
if bash "$SCRIPT" "$v55" >/dev/null 2>&1; then
    pass "dist/-prefixed aggregate + sidecar verify clean (v0.55.0 regression)"
else
    fail "dist/-prefixed checksum lines should verify clean after normalization"
fi

mixed="$tmpd/mixed"
mkdir -p "$mixed"
echo "mixed payload" >"$mixed/a.tar.gz"
mh="$(cd "$mixed" && sha256sum a.tar.gz | cut -d' ' -f1)"
printf '%s  a.tar.gz\n%s  dist/a.tar.gz\n' "$mh" "$mh" >"$mixed/SHA256SUMS"
printf '%s  a.tar.gz\n' "$mh" >"$mixed/a.tar.gz.sha256"
if bash "$SCRIPT" "$mixed" >/dev/null 2>&1; then
    pass "mixed bare + dist/-prefixed aggregate verifies clean"
else
    fail "mixed bare + dist/-prefixed aggregate should verify clean"
fi
echo
if [[ "$fails" -eq 0 ]]; then
    echo "ALL PASS"
else
    echo "$fails FAILED"
    exit 1
fi
