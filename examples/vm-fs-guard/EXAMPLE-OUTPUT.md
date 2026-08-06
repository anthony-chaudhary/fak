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
added, reworded, or omitted.

## Inside a VM (the VM-vs-boundary capture)

Captured inside a hypervisor guest (WSL2, a Hyper-V lightweight utility VM standing in
for E2B/Fly/Cloudflare), run under `FAK_REQUIRE_T0=1` so an unwitnessed T0 would have
been a hard failure rather than a silent downgrade:

```text
$ FAK_BIN=$HOME/vm-fs-guard/fak FAK_REQUIRE_T0=1 ./run.sh
[vm-fs-guard] T0 - the box that PROVIDES the disk (the sandbox's job, never fak's):
  PASS T0 witnessed from inside the guest             -> vm - hypervisor guest - wsl - Linux 6.6.114.1-microsoft-standard-WSL2
      rootfs   /dev/sdd ext4   /
      workdir  /dev/sdd ext4   /
  PASS fak provides no filesystem here                -> no fak mount, device, or FUSE server

[vm-fs-guard] T1 - REFUSED: a write into a region the sandbox's disk holds but the agent must not touch
[vm-fs-guard]      (SELF_MODIFY, by shape; fak did not provide this disk - it gates the path on it):
  PASS Edit  .git/config (repo internals)             -> DENY  SELF_MODIFY
  PASS Write ~/.ssh/id_rsa (a private key)            -> DENY  SELF_MODIFY
  PASS Write /workspace/.env (a secrets file)         -> DENY  SELF_MODIFY
  PASS Write internal/adjudicator/decide.go (kernel)  -> DENY  SELF_MODIFY

[vm-fs-guard] T1 - REFUSED (read side): a Read whose target lies OUTSIDE the agent's declared view,
[vm-fs-guard]      and a write into a subtree mounted read-only. The sandbox's disk holds both files;
[vm-fs-guard]      fak never lets the call reach them (deny-by-default over the path space):
  PASS Read  /srv/secrets/prod.pem (out of view)      -> DENY  DEFAULT_DENY
  PASS Write /workspace/vendor/lib.go (ro subtree)    -> DENY  POLICY_BLOCK

[vm-fs-guard] ALLOWED - ordinary reads/writes of the sandbox's OWN disk (the floor gates path+trust, not the disk):
  PASS Read  /workspace/src/main.go (in scope)        -> ALLOW
  PASS Write /workspace/notes.md (in scope)           -> ALLOW

[vm-fs-guard] T2 - QUARANTINE: a poisoned read result held out of the agent's context (TRUST_VIOLATION):
  PASS poisoned fetch/read result (prompt injection)  -> QUARANTINE  TRUST_VIOLATION

[vm-fs-guard] FS-decision ledger (the boundary's exit record):
  TIER VERDICT     REASON         CALL
  T1  DENY        SELF_MODIFY    Edit  .git/config (repo internals)
  T1  DENY        SELF_MODIFY    Write ~/.ssh/id_rsa (a private key)
  T1  DENY        SELF_MODIFY    Write /workspace/.env (a secrets file)
  T1  DENY        SELF_MODIFY    Write internal/adjudicator/decide.go (kernel)
  T1  DENY        DEFAULT_DENY   Read  /srv/secrets/prod.pem (out of view)
  T1  DENY        POLICY_BLOCK   Write /workspace/vendor/lib.go (ro subtree)
  --  ALLOW       (permitted)    Read  /workspace/src/main.go (in scope)
  --  ALLOW       (permitted)    Write /workspace/notes.md (in scope)
  T2  QUARANTINE  TRUST_VIOLATION poisoned read held out of context

[vm-fs-guard] all witnesses passed - fak adjudicated FS syscalls INSIDE a vm it did not provision:
[vm-fs-guard]   the disk came from the vm (fak holds no mount there); the DECISIONS came from fak -
[vm-fs-guard]   a write into guarded machinery refused (T1/SELF_MODIFY), an out-of-view read refused
[vm-fs-guard]   (T1/DEFAULT_DENY), a poisoned read quarantined (T2/TRUST_VIOLATION), while the
[vm-fs-guard]   sandbox's own disk stayed readable/writable.
[vm-fs-guard] wrap a live agent the same way: fak guard -- claude   (the FS floor rides into the VM).
EXIT=0
```

All three of the issue's captures are live CLI decisions in that transcript: **(a)** the
out-of-view `Read` refused (`DENY DEFAULT_DENY`, plus the read-only-subtree write refused
`POLICY_BLOCK`), **(b)** the poisoned read quarantined (`QUARANTINE TRUST_VIOLATION`), and
**(c)** the exit ledger of FS decisions - nine rows, each one a verdict the adjudicator
computed on this run, inside a guest fak did not provision.

**What makes this the VM-vs-boundary witness, and not merely a policy test.** Three facts
in the T0 block are read off the guest, not asserted by the script:

- the kernel identifies itself as a **hypervisor guest** (`systemd-detect-virt` -> `wsl`);
- the root filesystem is **`/dev/sdd ext4`** — a block device the *hypervisor* attached.
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
  T1  DENY        POLICY_BLOCK   Write /workspace/vendor/lib.go (ro subtree)
  --  ALLOW       (permitted)    Read  /workspace/src/main.go (in scope)
  --  ALLOW       (permitted)    Write /workspace/notes.md (in scope)
  T2  QUARANTINE  TRUST_VIOLATION poisoned read held out of context

[vm-fs-guard] all witnesses passed - fak adjudicated FS syscalls INSIDE a container it did not provision:
EXIT=0
```

A **different T0 kind, detected by a different signal** (the runtime's marker file, not a
hypervisor signature), over a **different rootfs** — `overlay`, a filesystem the container
runtime composed out of image layers seconds earlier. fak provisioned none of it, and holds
no mount on it. Yet the nine ledger rows are **byte-identical** to the WSL2 capture above
and to the Windows-host capture below: three boxes, three filesystems, two operating
systems, two independently built binaries, one decision.

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
OS** (a Windows host, outside any guest) using a **separately built `fak.exe`** — and its
nine T1/T2 ledger rows came out byte-identical to the WSL2 capture above. So "those rows
are identical on any box" is not an assertion in this file either: two operating systems
and two independently built binaries agree, which is what you would expect of a decision
that is a pure function of (policy, call) and nothing else.

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
