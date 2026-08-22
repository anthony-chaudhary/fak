#!/bin/sh
set -eu

if [ "${1:-}" = build ]; then
    printf '%s ' "$@"
    printf '\n'
    if [ "${TEST_FAST_BUILD_STALL:-}" = 1 ]; then
        sleep 30 &
        child=$!
        echo "STALL_CHILD_PID=$child"
        wait "$child"
    fi
    exit 0
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fixture="$root/scripts/test-fast-build_test.sh"

ok=$(make --no-print-directory -C "$root" smoke-build GO="$fixture" SMOKE_BUILD_BUDGET=2s)
printf '%s\n' "$ok" | grep -F -- '-buildvcs=false ./... ' >/dev/null

timed=$(TEST_FAST_BUILD_STALL=1 make --no-print-directory -C "$root" smoke-build GO="$fixture" SMOKE_BUILD_BUDGET=0.1s SMOKE_BUILD_KILL_AFTER=0.1s 2>&1) && {
    echo "test-fast-build regression: expected timeout" >&2
    exit 1
}
printf '%s\n' "$timed" | grep -F 'test-fast build: TIMEOUT' >/dev/null
printf '%s\n' "$timed" | grep -F 'WSL /mnt/c remedy: retry from the Linux filesystem' >/dev/null
child=$(printf '%s\n' "$timed" | sed -n 's/^STALL_CHILD_PID=//p' | tail -n 1)
[ -n "$child" ] || { echo "test-fast-build regression: missing child pid" >&2; exit 1; }
if kill -0 "$child" 2>/dev/null; then
    echo "test-fast-build regression: child $child survived timeout" >&2
    exit 1
fi

echo "test-fast-build regression OK"
