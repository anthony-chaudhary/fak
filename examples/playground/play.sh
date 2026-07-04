#!/usr/bin/env bash
# fak playground: a guided, browser-free REPL. You propose tool calls; the fak kernel
# adjudicates each one the way a syscall interface adjudicates a syscall — and narrates
# the verdict. No API key, no model, no GPU, no network: every verdict is a pure
# function of (policy, proposed call), the offline path `fak preflight` exposes. One
# static Go binary, drop-in.
#
# Interactive (a real terminal): press Enter to fire each proposed call, then take the
# wheel in a free REPL at the end — type any tool name and watch it get judged.
# Non-interactive (piped, or in CI): the four-step guided tour runs start-to-finish on
# its own, so the same three distinct verdicts print without a human at the keyboard.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
POLICY="examples/playground/policy.json"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "playground: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  echo "  build it once (zero external deps):  go build -o fak ./cmd/fak" >&2
  exit 2
fi

INTERACTIVE=0
[ -t 0 ] && INTERACTIVE=1

hr() { printf '%s\n' "----------------------------------------------------------------------"; }

# pause "<prompt>" — in a real terminal, wait for Enter (q quits). Piped/CI: no-op.
pause() {
  [ "$INTERACTIVE" = "1" ] || return 0
  printf '\n%s ' "$1"
  local reply=""
  read -r reply || reply=""
  case "$reply" in q|Q|quit|exit) echo "bye."; exit 0;; esac
}

# propose "<title>" <tool> '<args-json>' "<lesson>" — fire one call, narrate the verdict.
step=0
propose() {
  local title="$1" tool="$2" args="$3" lesson="$4"
  step=$((step + 1))
  hr
  printf 'STEP %d — %s\n' "$step" "$title"
  printf '  proposed call:  %s %s\n\n' "$tool" "$args"
  "$FAK_BIN" preflight --policy "$POLICY" --tool "$tool" --args "$args" --explain 2>&1 | sed 's/^/    /'
  printf '\n  what happened:  %s\n' "$lesson"
}

cat <<'INTRO'
======================================================================
  fak playground — treat the tool call like a syscall
======================================================================
An agent does not get to touch the world directly; every tool call is a
request the kernel adjudicates first, exactly like a process asking the OS
to run a syscall. Below you will propose four calls against one tiny policy
(examples/playground/policy.json) and read the kernel's verdict for each.

Nothing here needs a key, a model, a GPU, or the network — the verdict is a
pure function of (policy, call), so it is deterministic and yours to poke at.
INTRO

pause "Press Enter to propose the first call — or 'q' to quit:"
propose "an allowed read — the boring, common case" \
  read_file '{"path":"README.md"}' \
  "ALLOW. read_file is on the allow-list, so an affirmative policy rung admits it. Most calls should look like this — the gate is invisible until you reach for something dangerous."

pause "Now try an irreversible shell call nobody wrote a rule about. Press Enter:"
propose "an rm -rf — watch it get denied" \
  run_shell '{"cmd":"rm -rf /"}' \
  "DENY / DEFAULT_DENY. Nobody blocklisted run_shell — it is refused because it was never ADMITTED. A default-deny floor does not have to predict the dangerous call to stop it."

pause "Next: a tool that IS named in the policy — but explicitly forbidden. Press Enter:"
propose "an explicit deny — a different kind of no" \
  send_email '{"to":"ceo@corp.example","body":"quarterly numbers attached"}' \
  "DENY / POLICY_BLOCK. send_email appears in the manifest, but under 'deny', so the reason is POLICY_BLOCK, not DEFAULT_DENY. Same verdict (DENY), a distinct, structured reason a loop can branch on."

pause "Last guided step: an ALLOWED call that happens to carry a secret. Press Enter:"
propose "a redacted secret — allowed, but sanitized first" \
  create_ticket '{"body":"printer on 3rd floor is broken","password":"hunter2"}' \
  "TRANSFORM. create_ticket is allowed, so the call runs — but 'password' is in redact_fields, so the kernel rewrites it to [REDACTED] BEFORE dispatch. A TRANSFORM sanitizes the call; a DENY stops it."

hr
cat <<'RECAP'
Three distinct verdicts in four calls: ALLOW (admitted), DENY (refused —
with two different reasons, DEFAULT_DENY vs POLICY_BLOCK), and TRANSFORM
(admitted, then sanitized). None of them was a model "choosing to behave" —
each is the policy file applied to the call. Change policy.json and every
verdict above changes with it.
RECAP

if [ "$INTERACTIVE" = "1" ]; then
  hr
  cat <<'FREE'
Your turn. Type a tool name, then JSON args, and watch the kernel judge it.
Ideas: read_file (ALLOW) · exfiltrate_secrets (DEFAULT_DENY) · send_email
(POLICY_BLOCK) · create_ticket with an "api_key" field (TRANSFORM).
Empty line or 'q' to leave.
FREE
  while true; do
    printf '\nplayground> tool name: '
    tool=""
    read -r tool || break
    [ -z "$tool" ] && break
    case "$tool" in q|Q|quit|exit) break;; esac
    printf 'playground> args JSON [default {}]: '
    uargs=""
    read -r uargs || uargs=""
    [ -z "$uargs" ] && uargs='{}'
    echo
    "$FAK_BIN" preflight --policy "$POLICY" --tool "$tool" --args "$uargs" --explain 2>&1 | sed 's/^/    /'
  done
  echo "bye."
fi
