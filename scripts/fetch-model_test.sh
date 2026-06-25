#!/usr/bin/env bash
# fetch-model_test.sh - offline tests for fetch-model.sh integrity checks.
#
# No network or venv: the test extracts the embedded manifest-vs-weights verifier
# and runs it against tiny synthetic export directories.
#
# Run: bash scripts/fetch-model_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fails=0
pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; fails=$((fails + 1)); }

tmpd="$(mktemp -d)"
trap 'rm -rf "$tmpd"' EXIT

verify_py="$tmpd/verify_export.py"
awk "/<<'PY'/{f=1; next} /^PY$/{f=0} f" "$HERE/fetch-model.sh" > "$verify_py"
if [[ -s "$verify_py" ]]; then
    pass "extracted embedded integrity verifier"
else
    fail "extract embedded integrity verifier"
fi

mk_export() {
    local dir="$1" nbytes="$2"
    mkdir -p "$dir"
    head -c "$nbytes" /dev/zero | tr '\0' 'x' > "$dir/weights.f32"
    printf '%s' '{"a":{"dtype":"f32","shape":[2],"offset":0,"nbytes":4},"b":{"dtype":"f32","shape":[3],"offset":4,"nbytes":6}}' > "$dir/manifest.json"
}

good="$tmpd/good"
mk_export "$good" 10
if python3 "$verify_py" "$good" >/dev/null 2>&1; then
    pass "matching manifest size -> verifier succeeds"
else
    fail "matching manifest size should succeed"
fi

bad="$tmpd/bad"
mk_export "$bad" 7
if python3 "$verify_py" "$bad" >/dev/null 2>&1; then
    fail "truncated weights should fail"
elif [[ ! -e "$bad/weights.f32" ]]; then
    pass "truncated weights -> verifier fails and deletes blob"
else
    fail "truncated weights left weights.f32 behind"
fi

echo
if [[ "$fails" -eq 0 ]]; then
    echo "ALL PASS"
else
    echo "$fails FAILED"
    exit 1
fi
