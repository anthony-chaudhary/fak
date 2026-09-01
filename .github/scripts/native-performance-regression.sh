#!/usr/bin/env bash
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: native-performance-regression.sh <fak> <gate-request.json> <verdict.json> <summary.md>" >&2
  exit 2
fi

fak=$1
request=$2
verdict=$3
summary=$4
mkdir -p "$(dirname "$verdict")"

set +e
"$fak" native-performance --gate "$request" 2>&1 | tee "$verdict"
status=${PIPESTATUS[0]}
set -e

{
  echo
  echo "### Gate verdict (exit $status)"
  echo '```json'
  cat "$verdict"
  echo '```'
} >> "$summary"

exit "$status"
