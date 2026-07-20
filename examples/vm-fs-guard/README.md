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
the result-side admitter (`Kernel.AdmitResult`) over a poisoned read. The result is
bit-identical on every run, and the full script normally completes in a few seconds after Go compilation. Captured output: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

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

The disk belongs to the sandbox; the *decisions* belong to fak. Three FS-syscall classes,
one boundary:

| # | The filesystem syscall | Verdict | Why |
|---|---|---|---|
| **T1** | `Edit .git/config`, `Write ~/.ssh/id_rsa`, `Write /workspace/.env`, `Write internal/adjudicator/decide.go` | **DENY · SELF_MODIFY** | a write into a region the sandbox's disk holds but the agent must never touch — repo internals, a private key, a secrets file, fak's own kernel source — refused *by shape*, naming only the one offending glob |
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

This witness captures the **shipped** filesystem floor: the call-side `SELF_MODIFY` *write*
refusal and the result-side read *quarantine*, both live kernel decisions. It is
deliberately **not** the full T1 story yet:

- The issue's ideal (a) is an *out-of-view Read refused by the **mount view***. **This is
  still not witnessed here, and the reason changed.**
  [#2577](https://github.com/anthony-chaudhary/fak/issues/2577) closed having landed the
  mount-view *kernel* — the `mount_view` manifest namespace plus `policy.MountViewRefusal`,
  a correct and unit-tested deny-by-default reference monitor over paths — but **not its
  enforcement wiring**. Nothing on the request path calls it, so a declared view is inert
  and no CLI verb can refuse an out-of-view read. Witnessed against a manifest whose view
  covers `src` only:

  ```text
  $ fak preflight --policy mv.json --tool Read --args '{"file_path":"secrets/id_rsa"}'
  verdict=ALLOW reason=NONE by=monitor      # out of view, yet ALLOWed
  ```

  Corroborating: `fak policy --check` prints all 17 floor dimensions and the mount view is
  in none of them. Wiring it is
  [#5310](https://github.com/anthony-chaudhary/fak/issues/5310). Until that lands, the
  shipped floor refuses out-of-scope **writes** (SELF_MODIFY), which is what this witness
  shows — a write-side T1 refusal standing in for a read-side one.
- The single **unified read syscall** spanning local tree query *and* remote-document
  retrieval under one trust gate ([#2578](https://github.com/anthony-chaudhary/fak/issues/2578))
  **has landed** and is witnessed directly: the *same* result-admit floor quarantines a
  poisoned **remote** document and a poisoned **local** file, and the vDSO counts a repeated
  local query as a cache-hit — pinned offline by
  [`internal/vdso/t2_read_seam_witness_test.go`](../../internal/vdso/t2_read_seam_witness_test.go).
  This `fak demo` run shows the T2 quarantine through the result-admitter; the seam test
  proves it is backend-agnostic (local *and* remote, one floor, one cache).

**Promotion evidence** (what moves this from `gen/next` toward `now`): #5310 wiring
`MountViewRefusal` into the call-side adjudicator turns witness (a) into a real read-side
out-of-view `Read → DENY DEFAULT_DENY` that `run.sh` can assert next to the existing
write-side rung — at which point all three of the issue's captures (out-of-view read,
quarantine, ledger) are live CLI decisions rather than two-of-three. #2578 has already
promoted: the T2 half is witnessed over local *and* remote reads.
**Demotion/retirement evidence**: if the epic retires the "fak is the VFS, not the VM"
claim (e.g. fak grows a T0 provider), this witness — which exists to back exactly that
claim — is retired with it.
**Invalidating assumptions**: (1) this stands in a container/microVM as a T0 substitute for
the real E2B/Fly/Cloudflare platforms; if the boundary behaves differently on a real guest
kernel (gVisor/Firecracker syscall interception) than in this substitute, the witness must
be re-run there before the strategic claim rests on it. (2) A *closed* tier issue does not
mean an *enforced* tier — #2577 closed on the kernel half alone, and the gap was invisible
until the manifest was driven through `fak preflight`. Read a tier's claim against a live
verdict, not against its issue state.

## Where this fits

- The FS-floor witness, standalone: `fak preflight --policy vm-fs-floor.json --tool Write --args '{"file_path":".git/config", …}'`
- The result-quarantine fold: `fak demo --json` (the `QUARANTINE` line)
- The wrapping form: [`../../cmd/fak/guard.go`](../../cmd/fak/guard.go) (`fak guard -- <agent>`)
- The network twin: [`../remote-vm-guard/`](../remote-vm-guard/) (`fak egress check`)
- The call-side self-modify witness it builds on: [`../self-modify-floor/`](../self-modify-floor/)
- The spine: [`../../docs/explainers/agent-virtual-filesystem.md`](../../docs/explainers/agent-virtual-filesystem.md) — the T0-vs-T1 tier table this is the proof of
