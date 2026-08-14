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
# THE FLOOR. `fak manage` carries two structural filesystem rungs into the VM:
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
# Honest fence — WHICH monitor refuses the read. The read-side refusals below are real and
# wired: an out-of-view call earns DENY DEFAULT_DENY from the arg-rule path view (the
# `arg_rules` block of vm-fs-floor.json), which the adjudicator evaluates on every call.
# What is NOT wired is the *other* spelling of the same idea: #2577 landed the `mount_view`
# manifest namespace + policy.MountViewRefusal (a correct, unit-tested deny-by-default
# monitor) but not its enforcement wiring, so a view declared in THAT vocabulary is inert
# on the request path — #5310 wires it. So: the capability ships and is witnessed here; the
# declarative `mount_view` syntax for it does not yet reach the request path.
#
# That gap has a PRICE, and this run measures it rather than describing it. An `arg_rules`
# view is per-(tool,arg): it gates only the tools it names, and only when the named arg is
# present. Both edges are asserted below — every one of the five FS tools this floor grants
# is aimed out of view and must be refused (an unruled tool would be an open door in a view
# claiming to be deny-by-default), and the two calls the spelling cannot express (Grep/Glob
# with no path arg, meaning "the working root") are pinned as over-refusals via t1_cost.
# When #5310 lands, those two flip to ALLOW and the cost rungs retire.
# The unified read syscall spanning local + remote (#2578) HAS landed and is witnessed by
# internal/vdso/t2_read_seam_witness_test.go. See README.md § Honest boundary.
#
# T0 is WITNESSED, never asserted (#2581). The claim this example exists to back is
# "fak decides, the sandbox provides" — so the run reads the T0 off the guest's own kernel
# before it claims to be inside one. On a bare host it still proves T1/T2, but it says
# plainly that the VM half is NOT captured instead of printing the VM sentence anyway.
#   FAK_REQUIRE_T0=1 ./run.sh     # make an unwitnessed T0 a hard failure (CI/promotion)
#
# One stream, on purpose. The narration (log()) and the per-rung verdict rows used to go to
# stderr and stdout respectively, so a terminal interleaved them nondeterministically and a
# `./run.sh > capture.txt` silently dropped every header. A witness whose captured ordering
# cannot be reproduced is not a witness, so fold stderr into stdout once, here: both fds now
# share one file description and the transcript below is byte-ordered on any box.
set -euo pipefail
exec 2>&1

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
# t1_view() asserts a PATH-VIEW refusal — the READ side of T1, and the rung that makes
# "fak is the virtual filesystem" literal: the sandbox's disk still holds the file, and
# fak simply never lets the call reach it. Deny-by-default over a path space, exactly the
# shape a mount namespace has: outside the declared view -> DEFAULT_DENY (nothing
# affirmatively permitted it), inside a read-only subtree with a write shape ->
# POLICY_BLOCK (a rule denied it). $4 is the reason code the refusal must cite, so a rung
# that starts refusing for a DIFFERENT reason fails loudly instead of passing quietly.
t1_view(){
  local desc="$1" tool="$2" args="$3" want="$4"
  local out; out="$(preflight --tool "$tool" --args "$args")"
  if printf '%s' "$out" | grep -q 'verdict=DENY' && printf '%s' "$out" | grep -q "reason=$want"; then
    printf '  \033[32m✓\033[0m %-46s -> DENY  %s\n' "$desc" "$want"
    LEDGER="${LEDGER}  T1  DENY        $(printf '%-15s' "$want")${desc}\n"
  else
    printf '  \033[31m✗\033[0m %-46s -> %s (wanted DENY %s)\n' "$desc" "$out" "$want"
    FAILS=$((FAILS + 1))
  fi
}
# t1_cost() pins an OVER-refusal: a call the view was never meant to refuse, denied anyway
# because the wired vocabulary cannot express it. It is a real verdict of the real kernel —
# so it is asserted exactly as strictly as a win — but it is recorded as a COST, not a
# capability. This is the measured price of spelling a mount view as per-tool `arg_rules`
# (see README § Honest boundary); when #5310 wires `mount_view`, these rungs should flip to
# ALLOW and this block retires. Pinning it means the price cannot drift silently.
t1_cost(){
  local desc="$1" tool="$2" args="$3" want="$4"
  local out; out="$(preflight --tool "$tool" --args "$args")"
  if printf '%s' "$out" | grep -q 'verdict=DENY' && printf '%s' "$out" | grep -q "reason=$want"; then
    printf '  \033[33m!\033[0m %-46s -> DENY  %s  (over-refusal)\n' "$desc" "$want"
    LEDGER="${LEDGER}  T1  DENY(cost)  $(printf '%-15s' "$want")${desc}\n"
  else
    printf '  \033[31m✗\033[0m %-46s -> %s (wanted DENY %s)\n' "$desc" "$out" "$want"
    FAILS=$((FAILS + 1))
  fi
}

# ---------------------------------------------------------------------------------------
# 0) T0 — the box that PROVIDES the disk, which is emphatically NOT fak.
#
# This is the rung the rest of the witness rests on: every verdict below is only
# interesting because fak computed it while riding INSIDE a T0 someone else provisioned.
# So READ the T0 off the guest's own kernel (/proc, findmnt, systemd-detect-virt) rather
# than asserting it in prose — an asserted T0 is exactly the conflation this example
# exists to refute ("fak provides the disk" vs "fak gates the disk").
T0_KIND="host"; T0_WHAT="no container or VM detected"
t0_detect(){
  local virt=""
  # a container leaves a marker file or a cgroup path; a VM leaves a hypervisor signature.
  if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    T0_KIND="container"; T0_WHAT="OCI container (runtime marker present)"
  elif [ -r /proc/1/cgroup ] && grep -qEi 'docker|containerd|lxc|kubepods|podman' /proc/1/cgroup 2>/dev/null; then
    T0_KIND="container"; T0_WHAT="OCI container (pid-1 cgroup)"
  fi
  if command -v systemd-detect-virt >/dev/null 2>&1; then
    virt="$(systemd-detect-virt 2>/dev/null || true)"
    case "$virt" in
      ""|none) : ;;
      *) [ "$T0_KIND" = host ] && { T0_KIND="vm"; T0_WHAT="hypervisor guest · $virt"; } ;;
    esac
  fi
  if [ "$T0_KIND" = host ] && [ -r /proc/version ] && grep -qi 'microsoft\|hypervisor' /proc/version 2>/dev/null; then
    T0_KIND="vm"; T0_WHAT="hypervisor guest (kernel signature)"
  fi
  [ "$T0_KIND" != host ] && [ -r /proc/version ] && T0_WHAT="$T0_WHAT · $(uname -s) $(uname -r)"
  return 0
}
# mount_of() names WHO backs a path — the source device/share and its fstype. That source
# is the sandbox's; fak appears in none of them, which is the "not the FS provider" half.
# Exactly ONE row: `findmnt -T` prints one line per matching mount entry, and a host can
# carry the same mount twice (WSL2 lists `C:\ 9p /mnt/c` twice, so a bind-mounted checkout
# under /mnt/c yields two identical rows). The callers below feed this into a single
# `printf '      workdir  %s\n'`, so a second row lands unindented and the capture stops
# matching the run. Take the head: duplicate rows describe the same mount.
mount_of(){ findmnt -no SOURCE,FSTYPE,TARGET -T "$1" 2>/dev/null | head -1 || true; }

t0_detect
echo
log "T0 — the box that PROVIDES the disk (the sandbox's job, never fak's):"
if [ "$T0_KIND" != host ]; then
  printf '  \033[32m✓\033[0m %-46s -> %s\n' "T0 witnessed from inside the guest" "${T0_KIND} · ${T0_WHAT}"
  ROOTFS="$(mount_of /)"; WORKFS="$(mount_of "$HERE")"
  [ -n "$ROOTFS" ] && printf '      rootfs   %s\n' "$ROOTFS"
  [ -n "$WORKFS" ] && printf '      workdir  %s\n' "$WORKFS"
  # fak ships no filesystem: no mount, no device, no FUSE server anywhere on this box.
  #
  # Match WHOLE FIELDS, never a substring — and consult SOURCE/FSTYPE only. Who PROVIDES a
  # filesystem is the source device and its type; the TARGET is merely where the mount was
  # hung, so a path is never evidence about its provider. That distinction is load-bearing
  # here because this repo is itself named `fak`: bind-mounting the checkout — the README's
  # own `docker run -v "$PWD:/w"` — yields `SOURCE=C:\[/work/fak] FSTYPE=9p`, and a
  # substring test read that stray "fak" as "a fak-backed mount exists" and failed this
  # rung in EVERY container, i.e. in exactly the T0 this example exists to witness. A real
  # fak-backed mount announces itself in the type (`fak` / `fuse.fak`) or as a source device
  # literally named `fak`; a bind mount of a directory called fak is 9p/overlay/ext4.
  if command -v findmnt >/dev/null 2>&1 && findmnt -rno SOURCE,FSTYPE 2>/dev/null | grep -qiE '^fak(fs)?[[:space:]]|[[:space:]](fuse\.)?fak(fs)?$'; then
    printf '  \033[31m✗\033[0m %-46s -> a fak-backed mount exists (fak would BE the FS)\n' "fak provides no filesystem here"
    FAILS=$((FAILS + 1))
  else
    printf '  \033[32m✓\033[0m %-46s -> no fak mount, device, or FUSE server\n' "fak provides no filesystem here"
  fi
else
  printf '  \033[33m!\033[0m %-46s -> host (%s)\n' "T0 NOT witnessed" "$T0_WHAT"
  log "     the T1/T2 verdicts below are real, but this run is NOT the VM-vs-boundary"
  log "     witness — it proves the boundary, not that the boundary rode into a VM."
  log "     capture the VM half by running this script INSIDE a sandbox, e.g.:"
  log "       docker run --rm -v \"\$PWD:/w\" -w /w golang:1.23 examples/vm-fs-guard/run.sh"
  if [ "${FAK_REQUIRE_T0:-0}" = 1 ]; then
    log "FAK_REQUIRE_T0=1 and no T0 was witnessed"; exit 1
  fi
fi

echo
log "T1 — REFUSED: a write into a region the sandbox's disk holds but the agent must not touch"
log "     (SELF_MODIFY, by shape; fak did not provide this disk — it gates the path on it):"
t1_deny  "Edit  .git/config (repo internals)"          Edit  '{"file_path":".git/config","old_string":"a","new_string":"b"}'
t1_deny  "Write ~/.ssh/id_rsa (a private key)"         Write '{"file_path":"/home/agent/.ssh/id_rsa","content":"x"}'
t1_deny  "Write /workspace/.env (a secrets file)"      Write '{"file_path":"/workspace/.env","content":"x"}'
t1_deny  "Write internal/adjudicator/decide.go (kernel)" Write '{"file_path":"internal/adjudicator/decide.go","content":"x"}'
echo
log "T1 — REFUSED (read side): EVERY filesystem tool this floor grants, aimed OUTSIDE the"
log "     agent's declared view, plus a write into a subtree mounted read-only. The sandbox's"
log "     disk holds all of these files; fak never lets the call reach them. A view is only"
log "     deny-by-default over the PATH SPACE if it holds for every tool that can reach a"
log "     path — one unruled tool is a door, not a view, so all five are asserted here:"
t1_view  "Read  /srv/secrets/prod.pem (out of view)"   Read  '{"file_path":"/srv/secrets/prod.pem"}' DEFAULT_DENY
t1_view  "Edit  /srv/secrets/prod.pem (out of view)"   Edit  '{"file_path":"/srv/secrets/prod.pem","old_string":"a","new_string":"b"}' DEFAULT_DENY
t1_view  "Write /srv/secrets/prod.pem (out of view)"   Write '{"file_path":"/srv/secrets/prod.pem","content":"x"}' DEFAULT_DENY
t1_view  "Grep  /srv/secrets (out of view)"            Grep  '{"path":"/srv/secrets","pattern":"KEY"}' DEFAULT_DENY
t1_view  "Glob  /srv/secrets/*.pem (out of view)"      Glob  '{"path":"/srv/secrets","pattern":"*.pem"}' DEFAULT_DENY
t1_view  "Write /workspace/vendor/lib.go (ro subtree)" Write '{"file_path":"/workspace/vendor/lib.go","content":"x"}' POLICY_BLOCK
t1_view  "Edit  /workspace/vendor/lib.go (ro subtree)" Edit  '{"file_path":"/workspace/vendor/lib.go","old_string":"a","new_string":"b"}' POLICY_BLOCK
echo
log "ALLOWED — ordinary reads/writes/searches of the sandbox's OWN disk (the floor gates"
log "     path+trust, not the disk — the same five tools, aimed INSIDE the view):"
t1_allow "Read  /workspace/src/main.go (in scope)"     Read  '{"file_path":"/workspace/src/main.go"}'
t1_allow "Write /workspace/notes.md (in scope)"        Write '{"file_path":"/workspace/notes.md","content":"hi"}'
t1_allow "Edit  /workspace/src/main.go (in scope)"     Edit  '{"file_path":"/workspace/src/main.go","old_string":"a","new_string":"b"}'
t1_allow "Grep  /workspace/src (in scope)"             Grep  '{"path":"/workspace/src","pattern":"TODO"}'
t1_allow "Glob  /workspace/src/*.go (in scope)"        Glob  '{"path":"/workspace/src","pattern":"*.go"}'
echo
log "THE PRICE of spelling this view as per-tool arg_rules (NOT a capability — a cost):"
log "     an arg rule can only gate an argument that is PRESENT, so Grep/Glob with no path"
log "     arg — an in-view call meaning \"search the working root\" — fails closed instead."
log "     Safe, but wrong: a real mount_view resolves the default root and would ALLOW these."
log "     #5310 wires that; until then these two rungs are the honest cost of the workaround:"
t1_cost  "Grep  {pattern} with no path arg (in view)"  Grep  '{"pattern":"TODO"}' DEFAULT_DENY
t1_cost  "Glob  {pattern} with no path arg (in view)"  Glob  '{"pattern":"**/*.go"}' DEFAULT_DENY
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
# The closing claim is scoped to what THIS run actually witnessed. Only a run that read a
# real T0 off the guest may say "inside a sandbox"; a host run says what it proved instead.
if [ "$T0_KIND" != host ]; then
  log "all witnesses passed — fak adjudicated FS syscalls INSIDE a ${T0_KIND} it did not provision:"
  log "  the disk came from the ${T0_KIND} (fak holds no mount there); the DECISIONS came from fak —"
  log "  a write into guarded machinery refused (T1/SELF_MODIFY), ALL FIVE granted FS tools"
  log "  refused out of view (T1/DEFAULT_DENY) and the ro subtree refused on both write shapes"
  log "  (T1/POLICY_BLOCK), a poisoned read quarantined (T2/TRUST_VIOLATION), while the"
  log "  sandbox's own disk stayed readable/writable/searchable."
else
  log "all T1/T2 witnesses passed — but T0 was NOT witnessed on this run (host, no sandbox):"
  log "  a write into guarded machinery refused (T1/SELF_MODIFY), ALL FIVE granted FS tools"
  log "  refused out of view (T1/DEFAULT_DENY), a poisoned read quarantined"
  log "  (T2/TRUST_VIOLATION), while the disk stayed readable/writable. Re-run inside a"
  log "  sandbox for the VM-vs-boundary half — see the docker line above."
fi
log "2 rungs above are marked (over-refusal): the measured price of the arg_rules spelling,"
log "  not a capability. They flip to ALLOW when #5310 wires the mount_view vocabulary."
log "wrap a live agent the same way: fak manage -- claude   (the FS floor rides into the VM)."
