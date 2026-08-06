# `fak guard` on a random VM — the filesystem boundary (VM-vs-boundary witness)

Move your coding agent onto an ephemeral cloud VM (E2B, Fly, Cloudflare, Anthropic's
sandbox) and the human steps away. The **disk is the sandbox's** — fak did not
provision it, snapshot it, or format it. What fak *does* is ride inside that box as the
**T1–T2 reference monitor** and adjudicate every filesystem tool call at the boundary.
This is the filesystem twin of [`../remote-vm-guard/`](../remote-vm-guard/) (the
network-egress floor): together they show one capability floor riding into any VM, over
two syscall families.

**fak is the virtual filesystem, not the virtual machine.** ([the spine](../../docs/explainers/agent-virtual-filesystem.md))
The sandbox vendors own the box that *provides* the disk (T0); fak owns the layer that
*decides what the agent may do* on it (T1–T2). A microVM with no path floor still lets a
prompt-injected agent read every file it can reach — the floor is worth *more*, not less,
once nobody is left to click "approve."

## Run it

```bash
examples/vm-fs-guard/run.sh
```

Needs only Go (to build `fak`) — **no model, key, GPU, server, or network**. Every
verdict is a live decision of the **same kernel** a guarded session runs: `fak preflight`
folds the call-side adjudicator chain for one filesystem tool call, and `fak demo` folds
the result-side admitter (`Kernel.AdmitResult`) over a poisoned read. Those verdicts are
bit-identical on every run, and the full script normally completes in a few seconds after Go compilation. Captured output: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

**Run it inside a sandbox to capture the VM half.** The T0 rung is *witnessed, never
asserted*: the script reads the box off the guest's own kernel (`/proc`, `findmnt`,
`systemd-detect-virt`) and only claims "inside a VM it did not provision" when it
actually found one. On a bare host it still proves T1/T2 and says plainly that the VM
half is **not** captured — an asserted T0 would be exactly the conflation this example
exists to refute.

```bash
docker run --rm -v "$PWD:/w" -w /w golang:1.23 examples/vm-fs-guard/run.sh
FAK_REQUIRE_T0=1 examples/vm-fs-guard/run.sh   # unwitnessed T0 -> exit 1 (CI/promotion)

# the container capture in EXAMPLE-OUTPUT.md, reproducible on a dirty trunk: cross-build
# once, then let a stock image supply the disk and nothing else.
GOOS=linux GOARCH=amd64 go build -o "$BIN_DIR/fak" ./cmd/fak
docker run --rm -v "$PWD:/w:ro" -v "$BIN_DIR:/fakbin:ro" -w /w \
  -e FAK_BIN=/fakbin/fak -e FAK_REQUIRE_T0=1 debian:bookworm-slim \
  ./examples/vm-fs-guard/run.sh
```

Both T0 kinds the issue names are captured: a **hypervisor guest** (WSL2) and an **OCI
container**, detected by different signals over different rootfs types, agreeing row for row
on the ledger. They are separate branches of the detector, so running only one leaves the
other unexercised — the container branch shipped with a rung that failed the recipe above
until the container capture was actually taken (see EXAMPLE-OUTPUT.md § *earned its keep*).

**On a dirty shared trunk, skip the build** — point the witness at a prebuilt binary:

```bash
FAK_BIN=/path/to/fak examples/vm-fs-guard/run.sh
```

`run.sh` builds `./cmd/fak` from the *working tree*, so a peer's half-landed change
elsewhere in the module (a committed caller whose definition is still uncommitted) reds
that build and the witness never reaches its first verdict — a build failure, not an FS
verdict. The witness itself only needs `fak preflight` and `fak demo`, so any recent
binary reproduces it. This is what makes the run reproducible on a live multi-session
checkout rather than only in a clean clone.

## What it proves

The disk belongs to the sandbox; the *decisions* belong to fak. One boundary rung that
establishes *whose disk it is*, then three FS-syscall classes decided on top of it:

| # | The filesystem syscall | Verdict | Why |
|---|---|---|---|
| **T0** | *(not a syscall — the substrate)* the guest kernel's own report of the box, plus every mount on it | **witnessed** | the run reads `systemd-detect-virt` / `/proc` / `findmnt` to confirm it is inside a guest, names the device backing the rootfs (the *hypervisor's*, e.g. `/dev/sdd ext4`), and **fails if any fak-backed mount, device, or FUSE server exists** — the "fak did not provide this disk" half, as evidence |
| **T1** | `Edit .git/config`, `Write ~/.ssh/id_rsa`, `Write /workspace/.env`, `Write internal/adjudicator/decide.go` | **DENY · SELF_MODIFY** | a write into a region the sandbox's disk holds but the agent must never touch — repo internals, a private key, a secrets file, fak's own kernel source — refused *by shape*, naming only the one offending glob |
| **T1 (read side)** | `Read /srv/secrets/prod.pem` (outside the view), `Write /workspace/vendor/lib.go` (a read-only subtree) | **DENY · DEFAULT_DENY** / **DENY · POLICY_BLOCK** | the issue's capture (a): the sandbox's disk *holds* both files and fak never lets the call reach them — deny-by-default over the path space, the shape a mount namespace has. Declared as `arg_rules` in [`vm-fs-floor.json`](vm-fs-floor.json); see [Honest boundary](#honest-boundary) for why not the `mount_view` spelling |
| **allow** | `Read /workspace/src/main.go`, `Write /workspace/notes.md` | **ALLOW** | ordinary reads/writes of the sandbox's own disk — the floor gates the **path and the trust**, not the disk |
| **T2** | a poisoned fetch/read *result* carrying a prompt injection | **QUARANTINE · TRUST_VIOLATION** | an untrusted read held out of the agent's context by the result-side admitter, so the injection never lands |

The run ends with the **FS-decision ledger** — the boundary's exit record of every
verdict it issued (tier · verdict · reason · call), the filesystem counterpart of the
egress witness's blocked/allowed table.

## How it works

Both rungs are mandatory rungs of the in-process reference monitor, run ahead of any
affirmative allow, and cannot be elided by a policy — a security floor narrows, never
widens:

- **T1 (call-side)** — the self-modify floor in `internal/adjudicator/decide.go` refuses a
  write whose target matches a guarded glob (`self_modify_globs` in
  [`vm-fs-floor.json`](vm-fs-floor.json)), citing `SELF_MODIFY` with a bounded witness
  naming only the matched glob. `fak preflight --policy … --tool … --args …` folds this
  exact chain for one call.
- **T1 (call-side, read half)** — the arg-rule path view refuses a call whose *target path*
  falls outside the agent's declared view. `allow_glob` makes the view deny-by-default (a
  path matching nothing earns `DEFAULT_DENY` — nothing affirmatively permitted it), and a
  `deny_regex` marks a subtree read-only (a write into it earns `POLICY_BLOCK`). Both are
  evaluated by the adjudicator on every call, ahead of any affirmative allow.
- **T2 (result-side)** — the context-MMU's result-admission floor recognizes a
  prompt-injected tool result and returns `QUARANTINE`/`TRUST_VIOLATION`, holding the
  bytes out of context. `fak demo` folds the real `Kernel.AdmitResult` chain over it.

```text
+---------------------+     +--------------------------+     +----------------------------+
| agent FS tool call  | --> | fak guard  (T1 path floor|     | .git / .ssh / .env / kernel|
| (on the sandbox VM) |     |  + T2 read-trust floor)  | --> | -> DENY SELF_MODIFY        |
+---------------------+     +--------------------------+     | poisoned read -> QUARANTINE|
        the disk is the sandbox's            |               | in-scope read/write -> ALLOW|
                                             +-------------> +----------------------------+
```

Wrap a live agent and the FS floor rides into the VM with it:

```bash
fak guard -- claude          # every FS tool call crosses the floor; the disk stays the sandbox's
```

## Honest boundary

This witness captures the **shipped** filesystem floor: the T0 substrate read off the
guest kernel, the call-side `SELF_MODIFY` *write* refusal, and the result-side read
*quarantine* — the latter two live kernel decisions. It is deliberately **not** the full
T1 story yet:

- The issue's capture (a) — *an out-of-view `Read` refused* — **is witnessed above**, and
  it is a live adjudicator verdict, not a scripted string. What deserves naming precisely
  is **which monitor refuses it**. Two different spellings of the same idea exist in the
  manifest, and only one of them reaches the request path:

  - **`arg_rules` (wired, and what this witness uses).** An `allow_glob` over `Read`'s
    `file_path` is deny-by-default over a path space: a target matching no rule earns
    `DENY DEFAULT_DENY`, and a `deny_regex` marks a read-only subtree whose write earns
    `DENY POLICY_BLOCK`. The adjudicator evaluates these on every call, so the *capability*
    — hide a tree from the agent while the sandbox keeps providing it — ships today.
  - **`mount_view` (parses, never runs).**
    [#2577](https://github.com/anthony-chaudhary/fak/issues/2577) closed having landed the
    mount-view *kernel* — the `mount_view` namespace plus `policy.MountViewRefusal`, a
    correct and unit-tested deny-by-default reference monitor over paths — but **not its
    enforcement wiring**. Nothing on the request path calls it, so a view declared in
    *that* vocabulary is inert. Witnessed against a manifest whose view covers `src` only:

    ```text
    $ fak preflight --policy mv.json --tool Read --args '{"file_path":"secrets/id_rsa"}'
    verdict=ALLOW reason=NONE by=monitor      # out of view, yet ALLOWed
    ```

    Corroborating: `fak policy --check` prints all 17 floor dimensions and the mount view
    is in none of them. Wiring it is
    [#5310](https://github.com/anthony-chaudhary/fak/issues/5310).

  So the gap #5310 closes is one of **expressiveness, not capability**: an operator who
  wants a path view must spell it as `arg_rules` (per-tool, one rule per shape) rather than
  as one declarative `mount_view` block that applies to every FS tool at once. This witness
  deliberately shows the wired form, and says so, rather than declaring an inert field and
  letting the reader assume it did the refusing.
- The single **unified read syscall** spanning local tree query *and* remote-document
  retrieval under one trust gate ([#2578](https://github.com/anthony-chaudhary/fak/issues/2578))
  **has landed** and is witnessed directly: the *same* result-admit floor quarantines a
  poisoned **remote** document and a poisoned **local** file, and the vDSO counts a repeated
  local query as a cache-hit — pinned offline by
  [`internal/vdso/t2_read_seam_witness_test.go`](../../internal/vdso/t2_read_seam_witness_test.go).
  This `fak demo` run shows the T2 quarantine through the result-admitter; the seam test
  proves it is backend-agnostic (local *and* remote, one floor, one cache).

**Promotion evidence** (what moves this from `gen/next` toward `now`): all three of the
issue's captures are now live CLI decisions taken inside a guest fak did not provision —
the out-of-view `Read → DENY DEFAULT_DENY`, the poisoned read `→ QUARANTINE
TRUST_VIOLATION`, and the nine-row exit ledger ([`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md)),
captured under `FAK_REQUIRE_T0=1` so an unwitnessed substrate would have failed the run
instead of silently downgrading it — and now in **both** T0 kinds the issue names, a
hypervisor guest *and* an OCI container, whose ledgers agree row for row across two rootfs
types (`ext4` on a hypervisor-attached device, `overlay` composed from image layers).
Three halves promoted to get here: #2578 (the T2 read
seam, witnessed over local *and* remote reads); **T0 itself** — the run reads the substrate
off the guest kernel rather than asserting it, and found a hypervisor guest with its rootfs
on a hypervisor-attached block device and no fak mount present; and the **read-side T1
rung**, which moved from "blocked" to "witnessed" once the refusal was routed through the
monitor that is actually wired. What remains is #5310 wiring `MountViewRefusal`, which
promotes the *declaration* (one `mount_view` block covering every FS tool) rather than the
enforcement — at which point these rungs can be re-spelled in the tier's own vocabulary and
this README's two-spellings caveat retires.
**Demotion/retirement evidence**: if the epic retires the "fak is the VFS, not the VM"
claim (e.g. fak grows a T0 provider), this witness — which exists to back exactly that
claim — is retired with it. Concretely: the T0 rung fails the run the moment a fak-backed
mount appears on the box, so *this example red-lines itself* if fak ever becomes the FS
provider. That is the demotion trigger, wired rather than promised.
**Invalidating assumptions**: (1) the captured T0s are a hypervisor guest (WSL2) and an OCI
container standing in for the real E2B/Fly/Cloudflare platforms; if the boundary behaves
differently under a guest kernel that *intercepts syscalls* (gVisor, Firecracker + seccomp)
than under these substitutes, the witness must be re-run there before the strategic claim
rests on it — the detector proves *a* VM and *a* container, not *those* platforms. Both
captures here were also taken on one host, so they share a kernel (`…-microsoft-standard-WSL2`)
and are not independent of it. (2) The T0 rung witnesses the substrate, not the
enforcement path: it shows fak owns no mount, which is necessary but not sufficient for
"fak adjudicates every FS syscall" — an agent bypassing the tool interface (a raw `open()`
from spawned code) is outside what `fak preflight` decides, and this example does not claim
otherwise. (3) A *closed* tier issue does not mean an *enforced* tier — #2577 closed on the
kernel half alone, and the gap was invisible until the manifest was driven through
`fak preflight`. Read a tier's claim against a live verdict, not against its issue state.
(4) The read-side rung witnesses a path view spelled as `arg_rules`, which is **per-tool**:
the two rules here cover `Read` and `Write`, so a *third* FS tool reaching the same path
(`Edit`, `Glob`, `Grep`, a future `Stat`) is outside the view this manifest declares and
would need its own rule. A single `mount_view` block would cover them all at once — so
where the capability is equivalent today, the *completeness* of a hand-spelled view is an
operator's responsibility until #5310 lands. Treat this rung as proof the enforcement path
works, not as a template for a production view.

## Where this fits

- The FS-floor witness, standalone: `fak preflight --policy vm-fs-floor.json --tool Write --args '{"file_path":".git/config", …}'`
- The result-quarantine fold: `fak demo --json` (the `QUARANTINE` line)
- The wrapping form: [`../../cmd/fak/guard.go`](../../cmd/fak/guard.go) (`fak guard -- <agent>`)
- The network twin: [`../remote-vm-guard/`](../remote-vm-guard/) (`fak egress check`)
- The call-side self-modify witness it builds on: [`../self-modify-floor/`](../self-modify-floor/)
- The spine: [`../../docs/explainers/agent-virtual-filesystem.md`](../../docs/explainers/agent-virtual-filesystem.md) — the T0-vs-T1 tier table this is the proof of
