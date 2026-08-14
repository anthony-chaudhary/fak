# Captured output

The T1/T2 verdicts are live decisions of the same kernel a guarded session runs
(`fak preflight` folds the call-side chain; `fak demo` folds the result-side admitter),
so **those rows are identical on any box**. The **T0 block is not** — it is read off the
guest's own kernel at run time, so it reports whatever box you actually ran on. That is
the point: an *asserted* T0 would prove nothing about the claim this example exists to
back.

**Transcript fidelity.** `run.sh` folds stderr into stdout (`exec 2>&1`), so a plain
`./run.sh > capture.txt` reproduces the ordering below exactly — before that, the narration
and the verdict rows rode different streams and interleaved differently on every run, which
would have made this file un-reproducible by construction. The only edits applied here are
mechanical and lossless: ANSI colour escapes stripped, and the `✓ · —` glyphs folded to
`PASS - -` so the block renders in any terminal, pager, and diff. No line is reordered,
added, or reworded.

The **VM block below is the complete transcript** — every line the run emitted, in order.
The **container block is abridged**, showing the T0 rung and the ledger it must agree with;
each cut is marked by a `...` line and nothing else is dropped. Both claims are checked
mechanically rather than asserted: transliterate a live run by the two rules above and the
VM block matches line-for-line, while every un-elided container segment appears in the live
container transcript as a contiguous run, in order.

## Inside a VM (the VM-vs-boundary capture)

Captured inside a hypervisor guest (WSL2, a Hyper-V lightweight utility VM standing in
for E2B/Fly/Cloudflare), run under `FAK_REQUIRE_T0=1` so an unwitnessed T0 would have
been a hard failure rather than a silent downgrade:

```text
$ FAK_BIN=$HOME/vm-fs-guard/fak FAK_REQUIRE_T0=1 ./run.sh
[vm-fs-guard] T0 - the box that PROVIDES the disk (the sandbox's job, never fak's):
  PASS T0 witnessed from inside the guest             -> vm - hypervisor guest - wsl - Linux 6.6.114.1-microsoft-standard-WSL2
      rootfs   /dev/sdf ext4   /
      workdir  C:\    9p     /mnt/c
  PASS fak provides no filesystem here                -> no fak mount, device, or FUSE server

[vm-fs-guard] T1 - REFUSED: a write into a region the sandbox's disk holds but the agent must not touch
[vm-fs-guard]      (SELF_MODIFY, by shape; fak did not provide this disk - it gates the path on it):
  PASS Edit  .git/config (repo internals)             -> DENY  SELF_MODIFY
  PASS Write ~/.ssh/id_rsa (a private key)            -> DENY  SELF_MODIFY
  PASS Write /workspace/.env (a secrets file)         -> DENY  SELF_MODIFY
  PASS Write internal/adjudicator/decide.go (kernel)  -> DENY  SELF_MODIFY

[vm-fs-guard] T1 - REFUSED (read side): EVERY filesystem tool this floor grants, aimed OUTSIDE the
[vm-fs-guard]      agent's declared view, plus a write into a subtree mounted read-only. The sandbox's
[vm-fs-guard]      disk holds all of these files; fak never lets the call reach them. A view is only
[vm-fs-guard]      deny-by-default over the PATH SPACE if it holds for every tool that can reach a
[vm-fs-guard]      path - one unruled tool is a door, not a view, so all five are asserted here:
  PASS Read  /srv/secrets/prod.pem (out of view)      -> DENY  DEFAULT_DENY
  PASS Edit  /srv/secrets/prod.pem (out of view)      -> DENY  DEFAULT_DENY
  PASS Write /srv/secrets/prod.pem (out of view)      -> DENY  DEFAULT_DENY
  PASS Grep  /srv/secrets (out of view)               -> DENY  DEFAULT_DENY
  PASS Glob  /srv/secrets/*.pem (out of view)         -> DENY  DEFAULT_DENY
  PASS Write /workspace/vendor/lib.go (ro subtree)    -> DENY  POLICY_BLOCK
  PASS Edit  /workspace/vendor/lib.go (ro subtree)    -> DENY  POLICY_BLOCK

[vm-fs-guard] ALLOWED - ordinary reads/writes/searches of the sandbox's OWN disk (the floor gates
[vm-fs-guard]      path+trust, not the disk - the same five tools, aimed INSIDE the view):
  PASS Read  /workspace/src/main.go (in scope)        -> ALLOW
  PASS Write /workspace/notes.md (in scope)           -> ALLOW
  PASS Edit  /workspace/src/main.go (in scope)        -> ALLOW
  PASS Grep  /workspace/src (in scope)                -> ALLOW
  PASS Glob  /workspace/src/*.go (in scope)           -> ALLOW

[vm-fs-guard] THE PRICE of spelling this view as per-tool arg_rules (NOT a capability - a cost):
[vm-fs-guard]      an arg rule can only gate an argument that is PRESENT, so Grep/Glob with no path
[vm-fs-guard]      arg - an in-view call meaning "search the working root" - fails closed instead.
[vm-fs-guard]      Safe, but wrong: a real mount_view resolves the default root and would ALLOW these.
[vm-fs-guard]      #5310 wires that; until then these two rungs are the honest cost of the workaround:
  ! Grep  {pattern} with no path arg (in view)     -> DENY  DEFAULT_DENY  (over-refusal)
  ! Glob  {pattern} with no path arg (in view)     -> DENY  DEFAULT_DENY  (over-refusal)

[vm-fs-guard] T2 - QUARANTINE: a poisoned read result held out of the agent's context (TRUST_VIOLATION):
  PASS poisoned fetch/read result (prompt injection)  -> QUARANTINE  TRUST_VIOLATION

[vm-fs-guard] FS-decision ledger (the boundary's exit record):
  TIER VERDICT     REASON         CALL
  T1  DENY        SELF_MODIFY    Edit  .git/config (repo internals)
  T1  DENY        SELF_MODIFY    Write ~/.ssh/id_rsa (a private key)
  T1  DENY        SELF_MODIFY    Write /workspace/.env (a secrets file)
  T1  DENY        SELF_MODIFY    Write internal/adjudicator/decide.go (kernel)
  T1  DENY        DEFAULT_DENY   Read  /srv/secrets/prod.pem (out of view)
  T1  DENY        DEFAULT_DENY   Edit  /srv/secrets/prod.pem (out of view)
  T1  DENY        DEFAULT_DENY   Write /srv/secrets/prod.pem (out of view)
  T1  DENY        DEFAULT_DENY   Grep  /srv/secrets (out of view)
  T1  DENY        DEFAULT_DENY   Glob  /srv/secrets/*.pem (out of view)
  T1  DENY        POLICY_BLOCK   Write /workspace/vendor/lib.go (ro subtree)
  T1  DENY        POLICY_BLOCK   Edit  /workspace/vendor/lib.go (ro subtree)
  --  ALLOW       (permitted)    Read  /workspace/src/main.go (in scope)
  --  ALLOW       (permitted)    Write /workspace/notes.md (in scope)
  --  ALLOW       (permitted)    Edit  /workspace/src/main.go (in scope)
  --  ALLOW       (permitted)    Grep  /workspace/src (in scope)
  --  ALLOW       (permitted)    Glob  /workspace/src/*.go (in scope)
  T1  DENY(cost)  DEFAULT_DENY   Grep  {pattern} with no path arg (in view)
  T1  DENY(cost)  DEFAULT_DENY   Glob  {pattern} with no path arg (in view)
  T2  QUARANTINE  TRUST_VIOLATION poisoned read held out of context

[vm-fs-guard] all witnesses passed - fak adjudicated FS syscalls INSIDE a vm it did not provision:
[vm-fs-guard]   the disk came from the vm (fak holds no mount there); the DECISIONS came from fak -
[vm-fs-guard]   a write into guarded machinery refused (T1/SELF_MODIFY), ALL FIVE granted FS tools
[vm-fs-guard]   refused out of view (T1/DEFAULT_DENY) and the ro subtree refused on both write shapes
[vm-fs-guard]   (T1/POLICY_BLOCK), a poisoned read quarantined (T2/TRUST_VIOLATION), while the
[vm-fs-guard]   sandbox's own disk stayed readable/writable/searchable.
[vm-fs-guard] 2 rungs above are marked (over-refusal): the measured price of the arg_rules spelling,
[vm-fs-guard]   not a capability. They flip to ALLOW when #5310 wires the mount_view vocabulary.
[vm-fs-guard] wrap a live agent the same way: fak manage -- claude   (the FS floor rides into the VM).
EXIT=0
```

All three of the issue's captures are live CLI decisions in that transcript: **(a)** the
out-of-view refusal (`DENY DEFAULT_DENY`) — on **every one of the five FS tools this floor
grants**, not just `Read`, plus the read-only-subtree write refused `POLICY_BLOCK` on both
write shapes, **(b)** the poisoned read quarantined (`QUARANTINE TRUST_VIOLATION`), and
**(c)** the exit ledger of FS decisions — nineteen rows, each one a verdict the adjudicator
computed on this run, inside a guest fak did not provision.

**Why all five tools, and not just the `Read` the issue names.** A path view is a claim
about a *path space*, not about one tool: if any granted tool can reach a path the view
excludes, the view is a door rather than a boundary. The floor this example ships grants
`Read`, `Write`, `Edit`, `Glob`, and `Grep`, so all five are aimed out of view here. That
is not theoretical tidiness — until this rung existed, the shipped manifest declared path
rules for `Read` and `Write` only, and `Edit`/`Grep`/`Glob` aimed at `/srv/secrets/prod.pem`
were **ALLOW**ed, as was a `Write` to that path (only the narrower `vendor/` deny-regex
covered `Write`). The transcript above is the first one in which the "deny-by-default over
the path space" sentence is true as written rather than true of the two rows it sampled.

**The two `(over-refusal)` rows are a cost, not a capability — and they are asserted just
as strictly.** `arg_rules` gates an argument that is *present*; `Grep {"pattern":"TODO"}`
with no `path` means "search the working root", which is *inside* the view, and the rule
cannot see it, so it fails closed. Safe but wrong. They are pinned by `t1_cost()` so the
price cannot drift silently, and they are the concrete thing #5310 buys: when `mount_view`
reaches the request path it resolves a default root, those two rows flip to `ALLOW`, and
the block retires. That is a measured promotion criterion rather than a prose one.

**What makes this the VM-vs-boundary witness, and not merely a policy test.** Three facts
in the T0 block are read off the guest, not asserted by the script:

- the kernel identifies itself as a **hypervisor guest** (`systemd-detect-virt` -> `wsl`);
- the root filesystem is **`/dev/sdf ext4`** — a block device the *hypervisor* attached.
  fak did not provision, snapshot, or format it, and could not have;
- **no fak mount, device, or FUSE server exists on the box** — the script scans every
  mount for one and fails the run if it finds one.

So the disk is unambiguously the sandbox's, while every verdict below it is fak's. That
is the epic's claim — *fak is the virtual filesystem, not the virtual machine* — as
evidence rather than as prose.

## Inside an OCI container (the E2B/Fly/Cloudflare stand-in)

The issue this example backs names its T0 as "*a container or microVM standing in for
E2B/Fly/Cloudflare*". The VM half is captured above; this is the container half, on a
stock `debian:bookworm-slim` with the checkout bind-mounted read-only and a prebuilt
binary — again under `FAK_REQUIRE_T0=1`:

```text
$ docker run --rm -v "$PWD:/w:ro" -v "$BIN_DIR:/fakbin:ro" -w /w \
    -e FAK_BIN=/fakbin/fak -e FAK_REQUIRE_T0=1 debian:bookworm-slim \
    ./examples/vm-fs-guard/run.sh
[vm-fs-guard] T0 - the box that PROVIDES the disk (the sandbox's job, never fak's):
  PASS T0 witnessed from inside the guest             -> container - OCI container (runtime marker present) - Linux 6.6.114.1-microsoft-standard-WSL2
      rootfs   overlay overlay /
      workdir  C:\[/work/fak] 9p     /w
  PASS fak provides no filesystem here                -> no fak mount, device, or FUSE server
...
[vm-fs-guard] FS-decision ledger (the boundary's exit record):
  TIER VERDICT     REASON         CALL
  T1  DENY        SELF_MODIFY    Edit  .git/config (repo internals)
  T1  DENY        SELF_MODIFY    Write ~/.ssh/id_rsa (a private key)
  T1  DENY        SELF_MODIFY    Write /workspace/.env (a secrets file)
  T1  DENY        SELF_MODIFY    Write internal/adjudicator/decide.go (kernel)
  T1  DENY        DEFAULT_DENY   Read  /srv/secrets/prod.pem (out of view)
  T1  DENY        DEFAULT_DENY   Edit  /srv/secrets/prod.pem (out of view)
  T1  DENY        DEFAULT_DENY   Write /srv/secrets/prod.pem (out of view)
  T1  DENY        DEFAULT_DENY   Grep  /srv/secrets (out of view)
  T1  DENY        DEFAULT_DENY   Glob  /srv/secrets/*.pem (out of view)
  T1  DENY        POLICY_BLOCK   Write /workspace/vendor/lib.go (ro subtree)
  T1  DENY        POLICY_BLOCK   Edit  /workspace/vendor/lib.go (ro subtree)
  --  ALLOW       (permitted)    Read  /workspace/src/main.go (in scope)
  --  ALLOW       (permitted)    Write /workspace/notes.md (in scope)
  --  ALLOW       (permitted)    Edit  /workspace/src/main.go (in scope)
  --  ALLOW       (permitted)    Grep  /workspace/src (in scope)
  --  ALLOW       (permitted)    Glob  /workspace/src/*.go (in scope)
  T1  DENY(cost)  DEFAULT_DENY   Grep  {pattern} with no path arg (in view)
  T1  DENY(cost)  DEFAULT_DENY   Glob  {pattern} with no path arg (in view)
  T2  QUARANTINE  TRUST_VIOLATION poisoned read held out of context

[vm-fs-guard] all witnesses passed - fak adjudicated FS syscalls INSIDE a container it did not provision:
...
EXIT=0
```

A **different T0 kind, detected by a different signal** (the runtime's marker file, not a
hypervisor signature), over a **different rootfs** — `overlay`, a filesystem the container
runtime composed out of image layers seconds earlier. fak provisioned none of it, and holds
no mount on it. Yet the nineteen ledger rows are **byte-identical** to the WSL2 capture
above: `diff` of the two full transcripts reports changes on **only** the T0 substrate lines
(`vm · hypervisor guest · wsl` / `/dev/sdf ext4` vs `container · OCI container` / `overlay`)
and the noun in the closing sentence. Every verdict row matches exactly — two boxes, two
filesystems, one decision.

**This capture earned its keep by failing first.** The `fak provides no filesystem here`
rung scanned mount records for the *substring* `fak` — and this repository is *named* `fak`,
so bind-mounting the checkout hands the container a mount line whose source is the checkout
path. The old pattern, run against the real mount table inside the container:

```text
$ docker run --rm -v "$PWD:/w:ro" -w /w debian:bookworm-slim bash -c \
    "findmnt -rno SOURCE,FSTYPE,TARGET | grep -iE '(^|[^a-z])fak([^a-z]|$)|fuse\.fak'"
C:\x5c[/work/fak] 9p /w
```

(`-rno` is raw mode, which escapes the backslash as `\x5c`; the same mount prints as
`C:\[/work/fak] 9p /w` in the transcript above, where `mount_of()` calls `findmnt -no`.)

That is a **9p bind mount of a directory**, not a fak-backed filesystem — but the rung read
the match as "a fak-backed mount exists (fak would BE the FS)" and failed the run. Scope of
the bug, stated exactly: it trips whenever some mount's `SOURCE`/`TARGET` contains `fak` as a
whole word, which is precisely what the `docker run -v "$PWD:/w"` recipe this example
recommends produces **from a checkout of this repo** — the container path a reader is most
likely to take. (A checkout at, say, `/src/proj` does not trip it; the archive-mounted run
used to cross-check this section did not either, because `fakhead2581` is not a whole-word
match.) The check now matches whole `SOURCE`/`FSTYPE` fields and ignores `TARGET`, because
who *provides* a filesystem is its source device and type, never the path it was hung at.

That the VM capture passed while the container recipe failed is the argument for capturing
both: the `vm` and `container` branches of the detector are different code, and only one of
them had ever been run.

## On a bare host (no sandbox)

Run the same script outside a container/VM and the T1/T2 rungs still pass, but the run
**declines to claim the VM half** instead of printing the sandbox sentence anyway:

```text
$ FAK_BIN=<a Windows-built fak.exe> ./run.sh
[vm-fs-guard] T0 - the box that PROVIDES the disk (the sandbox's job, never fak's):
  ! T0 NOT witnessed                               -> host (no container or VM detected)
[vm-fs-guard]      the T1/T2 verdicts below are real, but this run is NOT the VM-vs-boundary
[vm-fs-guard]      witness - it proves the boundary, not that the boundary rode into a VM.
[vm-fs-guard]      capture the VM half by running this script INSIDE a sandbox, e.g.:
[vm-fs-guard]        docker run --rm -v "$PWD:/w" -w /w golang:1.23 examples/vm-fs-guard/run.sh
...
[vm-fs-guard] all T1/T2 witnesses passed - but T0 was NOT witnessed on this run (host, no sandbox):
EXIT=0
```

That run is worth more than the fence it demonstrates. It was captured on a **different
OS** (a Windows host, outside any guest) using a **separately built native `fak.exe`** —
and its nineteen T1/T2 ledger rows came out byte-identical to the container capture above.
So "those rows are identical on any box" is not an assertion in this file; it is a checked
one. The ledger block never reads the T0 — it is a pure function of (policy, call) — so the
check is just to extract the 19 ledger rows from a live capture and hash them:

```text
$ ./run.sh | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^  (T1|--|T2) ' | sha256sum | head -c16
de3fc103d7ab50c4
```

Re-witnessed on 2026-08-06 on this host, that one-liner prints the same 16 hex digits for
the two boxes captured fresh — the bare Windows host (native `windows/amd64` build) and the
`debian:bookworm-slim` container (a `linux/amd64` cross-build over an `overlay` rootfs):

```text
l_host   rows=19 sha=de3fc103d7ab50c4   # bare Windows host, native windows/amd64 build
l_ctr    rows=19 sha=de3fc103d7ab50c4   # debian:bookworm-slim container, linux/amd64
```

A `diff` of the two full transcripts differs on **only** the T0 substrate lines (host vs
`container · overlay`), never on a verdict row. The WSL2 VM transcript at the top of this
file carries those same 19 ledger rows for the same reason: two operating systems, two
independently built binaries, three filesystems (`overlay` from image layers, NTFS, and
`ext4` on a hypervisor-attached device), one decision — which is what you would expect of a
verdict that is a pure function of (policy, call) and nothing else.

`FAK_REQUIRE_T0=1` turns the downgrade into a hard failure, which is the form CI and any
promotion check should run — it is what stops a bare-host run from being filed as the
VM-vs-boundary witness:

```text
$ FAK_BIN=<a Windows-built fak.exe> FAK_REQUIRE_T0=1 ./run.sh
...
[vm-fs-guard] FAK_REQUIRE_T0=1 and no T0 was witnessed
EXIT=1
```

The disclosure on a refusal is bounded to the single offending glob — `fak preflight
--policy vm-fs-floor.json --tool Edit --args '{"file_path":".git/config",...}' --explain`
reports `disposition: ESCALATE` and `witness: .git/`, never the rest of the policy.
