#!/usr/bin/env bash
# fak_loop_run.sh -- Unix scheduler shim for `fak loop run`.
#
# Usage:
#   fak_loop_run.sh LOOP_ID SOURCE -- CMD [ARG...]
#
# Cron, launchd, and systemd should call this helper instead of invoking the
# child script directly. The OS still owns the timer; fak owns the loop ledger.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: fak_loop_run.sh LOOP_ID SOURCE -- CMD [ARG...]

Environment:
  FAK_BIN             optional fak executable path
  FAK_LOOP_LEDGER     optional ledger path (default: REPO/.fak/loops.jsonl)
  FAK_LOOP_PRINCIPAL  optional principal (default: current OS user)
EOF
}

if [ "$#" -lt 4 ] || [ "${3:-}" != "--" ]; then
  usage
  exit 2
fi

loop_id="$1"
source_name="$2"
shift 3

if [ -z "$loop_id" ] || [ -z "$source_name" ]; then
  usage
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$script_dir/.." && pwd -P)"
ledger="${FAK_LOOP_LEDGER:-$repo/.fak/loops.jsonl}"
principal="${FAK_LOOP_PRINCIPAL:-}"
if [ -z "$principal" ]; then
  principal="$(id -un 2>/dev/null || whoami 2>/dev/null || printf unknown)"
fi

fak_cmd=()
try_fak() {
  (cd "$repo" && "$@" loop status --ledger "$ledger" --json >/dev/null 2>&1)
}

if [ -n "${FAK_BIN:-}" ] && try_fak "$FAK_BIN"; then
  fak_cmd=("$FAK_BIN")
elif [ -x "$repo/fak" ] && try_fak "$repo/fak"; then
  fak_cmd=("$repo/fak")
elif [ -x "$repo/fak.exe" ] && try_fak "$repo/fak.exe"; then
  fak_cmd=("$repo/fak.exe")
elif [ -x "$repo/tools/.bin/fak" ] && try_fak "$repo/tools/.bin/fak"; then
  fak_cmd=("$repo/tools/.bin/fak")
elif command -v fak >/dev/null 2>&1 && try_fak "$(command -v fak)"; then
  fak_cmd=("$(command -v fak)")
elif command -v go >/dev/null 2>&1 && try_fak "$(command -v go)" run ./cmd/fak; then
  fak_cmd=("$(command -v go)" "run" "./cmd/fak")
fi

if [ "${#fak_cmd[@]}" -eq 0 ]; then
  echo "fak_loop_run.sh: no usable fak loop command found; set FAK_BIN or install Go" >&2
  exit 127
fi

cd "$repo"
exec "${fak_cmd[@]}" loop run \
  --ledger "$ledger" \
  --loop "$loop_id" \
  --source "$source_name" \
  --principal "$principal" \
  -- "$@"
