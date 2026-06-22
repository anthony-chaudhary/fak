#!/usr/bin/env bash
# Launch dgx_swebench_compare.py DETACHED on the DGX so the (flaky, ~1-command-
# per-session) Slack control bridge can fire it once and then poll a host-shared
# log. EVERY shell metachar the Slack bridge escapes (>, <, &) lives in THIS file,
# never in a command typed through Slack — type only `bash <thisfile> [args]`.
#
# Poll from a FRESH bridge session (/tmp is host-shared):
#   tail -40 /tmp/swe-cmp-1/run.log ; cat /tmp/swe-cmp-1/DONE.rc
# Artifacts when DONE.rc is 0: /tmp/swe-cmp-1/compare.json, COMPARE.md
set -u
RUN="${RUN:-/tmp/swe-cmp-1}"
DRIVER="${DRIVER:-/tmp/dgx_swebench_compare.py}"
mkdir -p "$RUN"
# setsid -> new session so the bridge readback returns immediately and the session
# stays usable (a bare `nohup &` leaves the child in the bridge's process group and
# wedges it). Driver runs under cwd /srv/fleet so the prebuilt fak binary + tools
# resolve. Exit code is recorded to DONE.rc for the poller.
setsid bash -c "cd /srv/fleet && /usr/bin/python3 '$DRIVER' --run-dir '$RUN' $* ; echo \$? > '$RUN/DONE.rc'" </dev/null >"$RUN/run.log" 2>&1 &
echo "launched pid $! run=$RUN driver=$DRIVER extra_args=[$*]"
