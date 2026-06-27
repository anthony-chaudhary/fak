#!/usr/bin/env bash
# fak-serve-caffeinate.sh — assert macOS system sleep prevention, then exec fak serve.
#
# WHY this wrapper instead of `caffeinate fak serve ...` in the launchd ProgramArguments:
#   caffeinate as the top-level plist command does NOT propagate its child's stdio to its
#   own file descriptors under launchd — fak serve's logs are silently discarded and
#   StandardOutPath / StandardErrorPath end up empty.
#
# This wrapper avoids that by using exec to REPLACE itself with fak serve:
#   1. `caffeinate -is -w $$` — start caffeinate in background, holding idle-sleep and
#      system-sleep assertions, waiting on our own PID ($$).
#   2. `exec "$@"` — replace this shell process with fak serve (same PID, $$).
#      After exec, launchd's direct child IS fak serve, so:
#        - fak serve's stdio inherits the wrapper's fds → logs land in the plist paths.
#        - launchd KeepAlive tracks fak serve (the real process), not a shell wrapper.
#        - caffeinate is now waiting for fak serve's PID; it exits when fak serve exits.
#   3. When fak serve exits, caffeinate exits. launchd KeepAlive restarts this wrapper.
#      Next iteration starts caffeinate fresh and execs fak serve again. No caffeinate
#      orphans accumulate across restarts.
set -euo pipefail
caffeinate -is -w $$ &
exec "$@"
