#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
helper="$repo/.github/scripts/native-performance-regression.sh"
workflow="$repo/.github/workflows/native-performance-regression.yml"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/fak" <<'FAK'
#!/usr/bin/env bash
request=${3:-}
if [[ -s "$request" ]] && grep -q '"valid":true' "$request"; then
  echo '{"schema":"fak-native-performance-gate-receipt/1","outcome":"pass"}'
  exit 0
fi
if [[ -s "$request" ]] && grep -q '"regression":true' "$request"; then
  echo '{"schema":"fak-native-performance-gate-receipt/1","outcome":"regression"}'
  exit 3
fi
echo 'fak native-performance: load gate request: unavailable or malformed' >&2
exit 1
FAK
chmod +x "$tmp/fak"

run_case() {
  local name=$1 expected=$2 request=$3
  local verdict="$tmp/$name/verdict.json" summary="$tmp/$name/summary.md"
  mkdir -p "$tmp/$name"
  set +e
  bash "$helper" "$tmp/fak" "$request" "$verdict" "$summary" >/dev/null
  local status=$?
  set -e
  [[ $status -eq $expected ]]
  [[ -s "$verdict" ]]
  grep -q "Gate verdict (exit $expected)" "$summary"
}

printf '{"valid":true}\n' > "$tmp/valid.json"
printf '{"regression":true}\n' > "$tmp/regression.json"
printf '{not-json}\n' > "$tmp/malformed.json"
run_case accepted 0 "$tmp/valid.json"
run_case regression 3 "$tmp/regression.json"
run_case malformed 1 "$tmp/malformed.json"
run_case unavailable 1 "$tmp/missing.json"
grep -q '"outcome":"pass"' "$tmp/accepted/verdict.json"
grep -q '"outcome":"regression"' "$tmp/regression/verdict.json"
grep -q 'unavailable or malformed' "$tmp/malformed/verdict.json"
grep -q 'unavailable or malformed' "$tmp/unavailable/verdict.json"

helper_run='        run: bash .github/scripts/native-performance-regression.sh "$RUNNER_TEMP/fak" returned-request/gate-request.json native-performance-verdict/verdict.json "$GITHUB_STEP_SUMMARY"'
[[ $(grep -Fxc "$helper_run" "$workflow") -eq 1 ]]
! grep -q 'native-performance --gate' "$workflow"
! grep -q 'PIPESTATUS' "$workflow"

upload_block=$(sed -n '/      - name: preserve gate verdict/,$p' "$workflow")
grep -Fq '        if: always()' <<<"$upload_block"
grep -Fq '        uses: actions/upload-artifact@v4' <<<"$upload_block"
grep -Fq '          name: native-performance-verdict-${{ github.run_id }}' <<<"$upload_block"
grep -Fq '          path: native-performance-verdict/verdict.json' <<<"$upload_block"
echo 'native-performance regression workflow helper: PASS'
