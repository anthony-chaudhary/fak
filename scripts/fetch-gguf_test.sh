#!/usr/bin/env bash
# fetch-gguf_test.sh — offline unit tests for the fetch-gguf.sh integrity helpers.
#
# No network: every case uses local files and file:// URLs, so it runs anywhere
# curl + a sha256 tool exist. It sources fetch-gguf.sh with FETCH_GGUF_SOURCED=1
# (which suppresses main) and drives sha256_file / download_verified directly.
#
# Run: bash scripts/fetch-gguf_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCH_GGUF_SOURCED=1 source "$HERE/fetch-gguf.sh"
set +e   # we assert on exit codes explicitly; don't abort on an expected failure

fails=0
pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; fails=$((fails + 1)); }

tmpd="$(mktemp -d)"
trap 'rm -rf "$tmpd"' EXIT

# Fixture: a "good" source file and its real sha256, reachable as a file:// URL.
src="$tmpd/model.gguf"
printf 'hello-gguf-bytes' > "$src"
good_sha="$(sha256_file "$src")"
src_url="file://$src"
wrong_sha="deadbeef$(printf '0%.0s' {1..56})"   # 64 hex chars, won't match

# 1. sha256_file emits a 64-hex digest.
if [[ "${#good_sha}" -eq 64 ]]; then
    pass "sha256_file emits a 64-hex digest"
else
    fail "sha256_file digest length (${#good_sha})"
fi

# 2. Matching sha -> file placed, no .partial left behind.
dest="$tmpd/out_ok.gguf"
if download_verified "$src_url" "$dest" "$good_sha" "ok-case" >/dev/null 2>&1 \
    && [[ -f "$dest" && ! -e "$dest.partial" ]]; then
    pass "matching sha -> file placed, no .partial left"
else
    fail "matching sha case"
fi

# 3. Wrong sha -> non-zero, corrupt download deleted, no dest, no .partial.
dest="$tmpd/out_bad.gguf"
if download_verified "$src_url" "$dest" "$wrong_sha" "bad-case" >/dev/null 2>&1; then
    fail "wrong sha should have failed"
elif [[ ! -e "$dest" && ! -e "$dest.partial" ]]; then
    pass "wrong sha -> non-zero, corrupt file deleted"
else
    fail "wrong sha left an artifact on disk"
fi

# 4. curl --fail on a missing source -> non-zero, nothing written.
dest="$tmpd/out_missing.gguf"
if download_verified "file://$tmpd/does-not-exist.gguf" "$dest" "$good_sha" "missing-case" >/dev/null 2>&1; then
    fail "missing source should have failed"
elif [[ ! -e "$dest" && ! -e "$dest.partial" ]]; then
    pass "missing source -> non-zero, no artifact"
else
    fail "missing source left an artifact on disk"
fi

# 5. FAK_FETCH_SKIP_VERIFY=1 bypasses verification (even a wrong sha is accepted).
dest="$tmpd/out_skip.gguf"
if FAK_FETCH_SKIP_VERIFY=1 download_verified "$src_url" "$dest" "$wrong_sha" "skip-case" >/dev/null 2>&1 \
    && [[ -f "$dest" ]]; then
    pass "FAK_FETCH_SKIP_VERIFY=1 bypasses verification"
else
    fail "skip-verify case"
fi

# 6. Empty expected sha (custom path, no pin) -> warns but still places the file.
dest="$tmpd/out_nopin.gguf"
if download_verified "$src_url" "$dest" "" "nopin-case" >/dev/null 2>&1 \
    && [[ -f "$dest" ]]; then
    pass "no pinned sha -> warns, still places file"
else
    fail "no-pin case"
fi

# 7. Existing model + tokenizer in a non-interactive run -> no prompt, no network.
home="$tmpd/home"
mkdir -p "$home/.cache/fak-models/gguf" "$home/.cache/fak-models/tokenizers/qwen2.5"
cached_model="$home/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf"
cached_tok="$home/.cache/fak-models/tokenizers/qwen2.5/tokenizer.json"
printf 'existing-model' > "$cached_model"
printf 'existing-tokenizer' > "$cached_tok"
before="$(sha256_file "$cached_model")"
if HOME="$home" bash "$HERE/fetch-gguf.sh" < /dev/null >/dev/null 2>&1 \
    && [[ "$(sha256_file "$cached_model")" == "$before" ]]; then
    pass "non-interactive existing file -> kept without prompting"
else
    fail "non-interactive existing file case"
fi

echo
if [[ "$fails" -eq 0 ]]; then
    echo "ALL PASS"
else
    echo "$fails FAILED"
    exit 1
fi
