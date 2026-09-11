#!/usr/bin/env bash
# verify-release-checksums.sh - verify sha256 checksums for a directory of
# downloaded release assets (an aggregate SHA256SUMS file and/or per-archive
# .sha256 sidecars). Exits nonzero and prints ::error:: lines if any asset's
# sha256 does not match its checksum file, so a corrupted or tampered upload
# fails the release-artifacts verify-release job instead of shipping quietly
# (#1369).
#
# dist/ path normalization (#v0.55.0-quarantine): a checksum line's name is
# relative to the CWD the GENERATING job ran in, but `sha256sum -c` resolves
# it relative to the CWD of the VERIFYING job. release-macos.yml ran shasum
# from dist/, so the macOS universal sidecar's content reads
# `<hash>  dist/fak_X_darwin_universal.tar.gz` while this script verifies from
# the flat assets dir -- `-c` looked for `<assets>/dist/...` (No such file or
# directory) and red the whole verify job, quarantining the release
# (v0.55.0 run 102720128853; the missing universal archive in
# expected_archives also escaped the completeness check). So: strip ONE
# leading `dist/` from every name field, covering both that sidecar and any
# aggregate line still folded with the prefix. Belt and suspenders: the
# release-macos.yml fix landing in the same wave emits the sidecar BARE
# (shasum run from inside dist/), which normalizes to itself; both and either
# verify clean.
#
# Usage: verify-release-checksums.sh <assets-dir>
set -uo pipefail

dir="${1:?usage: verify-release-checksums.sh <assets-dir>}"
cd "$dir"

# Normalized checksum lists are written here (never mutating the downloaded
# asset snapshot) and -c runs against the copy.
normdir="$(mktemp -d)"
trap 'rm -rf "$normdir"' EXIT

fail=0

# normalize_sums <src> <dst>: rewrite `<hash> SP|*<name>` lines to
# `<hash>  <bare-name>`. The sha256sum format is fixed-width (64 hex, one
# space, one mode char, then the name), so the name is carved by offset --
# exact on GNU and BSD/macOS toolchains alike -- rather than pattern-guessed.
# A binary-mode line (`*`) or one CRLF terminator (Windows-authored sidecar)
# is normalized too. Unrecognized lines pass through verbatim. `--strict` below
# makes `-c` FAIL on improperly formatted lines instead of only warning them,
# so a silently-skipped malformed name still fails loud.
normalize_sums() {
    local src="$1" dst="$2" line name hash sixtysix
    : >"$dst"
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        hash="${line:0:64}"
        sixtysix="${line:64:2}"
        case "$sixtysix" in
            "  " | " *")
                name="${line:66}"
                if [[ "$name" == dist/* ]]; then
                    echo "note: normalizing dist/-prefixed checksum line -> ${name#dist/}"
                    name="${name#dist/}"
                fi
                printf '%s  %s\n' "$hash" "$name" >>"$dst"
                ;;
            *)
                printf '%s\n' "$line" >>"$dst"
                ;;
        esac
    done <"$src"
}

verify_one() {
    local label="$1" src="$2"
    echo "== verifying $label =="
    normalize_sums "$src" "$normdir/$(basename "$src")"
    if sha256sum -c --strict "$normdir/$(basename "$src")"; then
        echo "ok: $label"
    else
        echo "::error::checksum verification failed for $label"
        fail=1
    fi
}

if [[ -f SHA256SUMS ]]; then
    verify_one "aggregate SHA256SUMS" SHA256SUMS
fi

echo "== verifying per-archive sha256 sidecars =="
shopt -s nullglob
sums=(*.sha256)
if [[ "${#sums[@]}" -eq 0 ]]; then
    echo "::error::no per-archive .sha256 assets found"
    fail=1
fi
# ${sums[@]+...} keeps the empty-array loop safe under `set -u` on old bash
# (3.2 flags an unbound variable for a bare "${sums[@]}" when nothing matched).
for sum in ${sums[@]+"${sums[@]}"}; do
    verify_one "$sum" "$sum"
done

exit "$fail"
