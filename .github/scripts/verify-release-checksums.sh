#!/usr/bin/env bash
# verify-release-checksums.sh - verify sha256 checksums for a directory of
# downloaded release assets (an aggregate SHA256SUMS file and/or per-archive
# .sha256 sidecars). Exits nonzero and prints ::error:: lines if any asset's
# sha256 does not match its checksum file, so a corrupted or tampered upload
# fails the release-artifacts verify-release job instead of shipping quietly
# (#1369).
#
# Usage: verify-release-checksums.sh <assets-dir>
set -uo pipefail

dir="${1:?usage: verify-release-checksums.sh <assets-dir>}"
cd "$dir"

fail=0

if [[ -f SHA256SUMS ]]; then
    echo "== verifying aggregate SHA256SUMS =="
    if sha256sum -c SHA256SUMS; then
        echo "ok: SHA256SUMS matches archives"
    else
        echo "::error::aggregate SHA256SUMS verification failed"
        fail=1
    fi
fi

echo "== verifying per-archive sha256 sidecars =="
shopt -s nullglob
sums=(*.sha256)
if [[ "${#sums[@]}" -eq 0 ]]; then
    echo "::error::no per-archive .sha256 assets found"
    fail=1
fi
for sum in "${sums[@]}"; do
    if sha256sum -c "$sum"; then
        echo "ok: $sum"
    else
        echo "::error::checksum mismatch for $sum"
        fail=1
    fi
done

exit "$fail"
