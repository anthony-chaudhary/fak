#!/bin/sh
set -eu
printf '%s ' "$@"
printf '\n'
if [ "${TEST_FAST_BUILD_STALL:-}" = 1 ]; then
    sleep 30 &
    child=$!
    echo "STALL_CHILD_PID=$child"
    wait "$child"
fi