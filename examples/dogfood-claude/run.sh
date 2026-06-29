#!/usr/bin/env bash
# run.sh — adoption-shaped wrapper for the dogfood-claude launcher.
#
# This is a THIN wrapper: it prints "what to look for" hints, then exec's the shipped
# launcher (scripts/dogfood-claude.sh) with your flags passed straight through. All the
# real work — build fak, serve a local model, front it with the kernel as a native
# Anthropic /v1/messages server, wire and launch the real Claude Code CLI, tear down on
# exit — lives in that one script. We do NOT reimplement any of it here (see README.md).
#
#   ./examples/dogfood-claude/run.sh --smoke            # curl the wire end-to-end (no model), then exit
#   ./examples/dogfood-claude/run.sh --probe "say pong"  # ONE headless live Claude Code turn, then exit
#   ./examples/dogfood-claude/run.sh                     # interactive Claude Code on the local model
#
# Every flag the launcher accepts (--smoke / --probe / --print-env / --list-accounts /
# --install / --kernel) and every FAK_DOGFOOD_* env knob passes through unchanged. The
# full reference is ../../DOGFOOD-CLAUDE.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"          # examples/dogfood-claude -> fak/
LAUNCHER="$FAK_DIR/scripts/dogfood-claude.sh"

log(){ printf '\033[36m[dogfood-example]\033[0m %s\n' "$*" >&2; }

[ -x "$LAUNCHER" ] || [ -f "$LAUNCHER" ] || {
  printf 'launcher not found: %s\n' "$LAUNCHER" >&2; exit 1; }

log "wrapping the shipped launcher: $LAUNCHER"
log "what to look for once a turn runs:"
log "  • the fak serve log (/tmp/fak-serve.log) — each /v1/messages turn + the per-call"
log "    verdict on every tool Claude proposes: ALLOW, POLICY_BLOCK / DEFAULT_DENY /"
log "    SELF_MODIFY refusals, a TRANSFORM (redacted arg), or a quarantine event."
log "  • try a deny live: ask Claude to run \`rm -rf /tmp/x\`, \`sudo ...\`, or \`git push\` —"
log "    the kernel refuses it before the shell sees it, while \`ls\`/\`cat\` run."
log "  • capability floor: examples/dogfood-claude-policy.json (see README.md)."
log "handing off to the launcher (flags passed through) ..."

exec "$LAUNCHER" "$@"
