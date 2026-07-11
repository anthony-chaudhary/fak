#!/usr/bin/env bash
# run.sh — the fak "guard on a random VM" walkthrough: the FILESYSTEM boundary.
#
# This is the filesystem twin of ../remote-vm-guard/ (the network-egress floor).
# Together they prove the same thing about two different syscall families: the fak
# capability floor rides INTO whatever sandbox/microVM you put your agent on, and
# adjudicates the syscall AT THE BOUNDARY — WITHOUT being the thing that provides
# the disk (or the network).
#
# THE SCENARIO. You move your coding agent (Claude Code, Codex, …) onto an ephemeral
# T0 — an E2B / Fly / Cloudflare / Anthropic-sandbox box — and step away. The DISK is
# the sandbox's: fak did not provision it, snapshot it, or format it. fak is the T1–T2
# reference monitor riding inside that box. A prompt-injected agent's most valuable
# filesystem moves are (1) writing into the machinery it should never touch — `.git/`
# internals, an `.ssh` private key, a `.env`, fak's own kernel source — and (2) reading
# a poisoned document straight back into its context as if it were fact.
#
# THE FLOOR. `fak guard` carries two structural filesystem rungs into the VM:
#   T1 — a write into a guarded region is refused BY SHAPE (SELF_MODIFY / ESCALATE),
#        naming ONLY the one offending glob, no model and no human in the loop;
#   T2 — a poisoned tool RESULT (a remote/local read) is QUARANTINE'd (TRUST_VIOLATION),
#        held out of the agent's context so the injection never lands.
#   …while an in-scope Read/Write of the sandbox's own disk is affirmatively ALLOWED —
#   the floor gates the PATH and the TRUST, not the disk. fak is the VFS, not the VM.
#
#   ./run.sh                       # build fak, run the FS witnesses, report PASS/FAIL
#   FAK_BIN=/path/to/fak ./run.sh  # use a prebuilt binary instead of building
#
# Needs only Go (to build fak) — NO model, key, GPU, server, or network. Every verdict
# is a live decision of the SAME kernel a guarded session runs (`fak preflight` folds the
# call-side chain; `fak demo` folds the result-side admitter), so the result is identical
# on every run.
#
# Honest fence: the shipped T1 floor refuses out-of-scope *writes* (SELF_MODIFY) into
# regions the sandbox's disk holds; the read-side *mount view* that hides a whole tree
# from the agent (#2577) and the single unified read syscall spanning local + remote
# (#2578) are the named next increments off the spine. See README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/vm-fs-guard -> fak/
POLICY="$HERE/vm-fs-floor.json"
log(){ printf '\033[36m[vm-fs-guard]\033[0m %s\n' "$*" >&2; }

BIN_DIR=""; FAILS=0
cleanup(){ [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# 1) the fak binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak ) || { log "build failed"; exit 1; }
fi

# preflight() folds the SAME call-side adjudicator chain a guarded session enforces, for
# ONE filesystem tool call, and prints `verdict=… reason=… by=…`. Every T1 verdict below
# is a pure function of (policy, call): no model, no server, no disk write actually happens.
preflight(){ "$BIN" preflight --policy "$POLICY" "$@" 2>/dev/null; }

# t1() asserts a guarded WRITE is refused SELF_MODIFY (the region the sandbox's disk holds
# that the agent must never touch), and records the ledger line.
LEDGER=""
t1_deny(){
  local desc="$1" tool="$2" args="$3"
  local out; out="$(preflight --tool "$tool" --args "$args")"
  if printf '%s' "$out" | grep -q 'verdict=DENY' && printf '%s' "$out" | grep -q 'reason=SELF_MODIFY'; then
    printf '  \033[32m✓\033[0m %-46s -> DENY  SELF_MODIFY\n' "$desc"
    LEDGER="${LEDGER}  T1  DENY        SELF_MODIFY    ${desc}\n"
  else
    printf '  \033[31m✗\033[0m %-46s -> %s (wanted DENY SELF_MODIFY)\n' "$desc" "$out"
    FAILS=$((FAILS + 1))
  fi
}
t1_allow(){
  local desc="$1" tool="$2" args="$3"
  local out; out="$(preflight --tool "$tool" --args "$args")"
  if printf '%s' "$out" | grep -q 'verdict=ALLOW'; then
    printf '  \033[32m✓\033[0m %-46s -> ALLOW\n' "$desc"
    LEDGER="${LEDGER}  --  ALLOW       (permitted)    ${desc}\n"
  else
    printf '  \033[31m✗\033[0m %-46s -> %s (wanted ALLOW)\n' "$desc" "$out"
    FAILS=$((FAILS + 1))
  fi
}

echo
log "T1 — REFUSED: a write into a region the sandbox's disk holds but the agent must not touch"
log "     (SELF_MODIFY, by shape; fak did not provide this disk — it gates the path on it):"
t1_deny  "Edit  .git/config (repo internals)"          Edit  '{"file_path":".git/config","old_string":"a","new_string":"b"}'
t1_deny  "Write ~/.ssh/id_rsa (a private key)"         Write '{"file_path":"/home/agent/.ssh/id_rsa","content":"x"}'
t1_deny  "Write /workspace/.env (a secrets file)"      Write '{"file_path":"/workspace/.env","content":"x"}'
t1_deny  "Write internal/adjudicator/decide.go (kernel)" Write '{"file_path":"internal/adjudicator/decide.go","content":"x"}'
echo
log "ALLOWED — ordinary reads/writes of the sandbox's OWN disk (the floor gates path+trust, not the disk):"
t1_allow "Read  /workspace/src/main.go (in scope)"     Read  '{"file_path":"/workspace/src/main.go"}'
t1_allow "Write /workspace/notes.md (in scope)"        Write '{"file_path":"/workspace/notes.md","content":"hi"}'
echo

# T2 — the result-side floor: a poisoned tool RESULT (a remote/local READ) is held out of
# the agent's context. `fak demo` folds the REAL ResultAdmitter chain (Kernel.AdmitResult)
# over a fetch whose body carries a prompt injection; the QUARANTINE verdict is a live
# kernel decision, never a scripted string.
log "T2 — QUARANTINE: a poisoned read result held out of the agent's context (TRUST_VIOLATION):"
DEMO="$("$BIN" demo --json 2>/dev/null || true)"
if printf '%s' "$DEMO" | grep -q '"verdict": "QUARANTINE"' && printf '%s' "$DEMO" | grep -q '"reason": "TRUST_VIOLATION"'; then
  printf '  \033[32m✓\033[0m %-46s -> QUARANTINE  TRUST_VIOLATION\n' "poisoned fetch/read result (prompt injection)"
  LEDGER="${LEDGER}  T2  QUARANTINE  TRUST_VIOLATION poisoned read held out of context\n"
else
  printf '  \033[31m✗\033[0m %-46s -> (wanted QUARANTINE TRUST_VIOLATION)\n' "poisoned fetch/read result"
  FAILS=$((FAILS + 1))
fi
echo

# (c) the exit ledger of FS decisions — the boundary's record of what it did, at a glance.
log "FS-decision ledger (the boundary's exit record):"
printf '  TIER VERDICT     REASON         CALL\n'
printf '%b' "$LEDGER"
echo

if [ "$FAILS" -ne 0 ]; then
  log "$FAILS witness(es) FAILED"; exit 1
fi
log "all witnesses passed — fak adjudicated FS syscalls INSIDE a sandbox it did not provision:"
log "  a write into guarded machinery refused (T1/SELF_MODIFY), a poisoned read quarantined"
log "  (T2/TRUST_VIOLATION), while the sandbox's own disk stayed readable/writable."
log "wrap a live agent the same way: fak guard -- claude   (the FS floor rides into the VM)."
