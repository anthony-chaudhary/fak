---
title: "The Agent Virtual Filesystem"
description: "The filesystem tier of fak's OS analogy, named. An agent says 'file' and 'memory' for six different things — the host block device, the file tree it is allowed to see, the read/query surface, its scratchpad, its durable memory, and the KV cache. This page separates them, draws the line each is confused with, and places fak: the reference monitor that mediates the virtual filesystem, not the box that provides it."
slug: agent-virtual-filesystem
keywords:
  - agent virtual filesystem
  - agent filesystem
  - virtual filesystem for AI agents
  - agent scratchpad
  - agent memory
  - path namespace
  - mount view
  - reference monitor
  - tool call is a syscall
  - VFS
  - retrieval
  - sandbox filesystem
date: 2026-07-03
---

# The Agent Virtual Filesystem — the FS tier of the OS analogy

fak's mental model is one borrowed idea carried at every scale: **the tool call is a
[syscall](./tool-call-is-a-syscall.md)**, the model proposes and the kernel disposes.
That analogy already has a process layer (adjudication), a memory-management unit
(`internal/ctxmmu` — paging and durability), an egress layer (`internal/egressfloor`),
and a discipline for what crosses into durable store
([memory engineering](./memory-engineering.md)). The **filesystem** layer of the same
analogy has never been named — even though every coding agent's most-used tools
(`Read`, `Write`, `Edit`, `Glob`, `Grep`) *are* file syscalls, and fak already
adjudicates them.

This page names that layer. A **virtual filesystem (VFS)** in an operating system is
the indirection between "a program asks to read a path" and "some backing store answers"
— one uniform `open/read/write/stat` interface over many different backends (a local
disk, a network mount, `/proc`, a FUSE process), with the kernel deciding, per call,
what the program is even allowed to see. The **agent virtual filesystem** is the same
indirection for an AI agent: the layer between the file-shaped tool call the model emits
and the bytes that answer it, where the kernel decides what tree exists, what may be
read, what may be written, and whether a returned byte is trusted.

## Why it needs naming: six things wear the word "file" (or "memory")

The trap is that an agent uses one vocabulary — *files*, *memory* — for six distinct
concepts with six distinct mechanisms, trust models, and lifetimes. Conflate them and
you get the failures the field keeps re-hitting: a scratchpad artifact promoted to
durable memory, a poisoned document read back as fact, two fleet agents silently
overwriting the same path, a "delete" that leaves the bytes in a cached prefix. The
disambiguation:

| Tier | What it is | Lifetime | Who owns it | fak's role |
|---|---|---|---|---|
| **T0 — host/VM store** | the real block device / sandbox FS the process runs on | machine/session | the VM or sandbox provider (E2B, Fly, Cloudflare, Anthropic's sandbox) | **not fak** — fak rides *into* it |
| **T1 — path namespace / mount view** | the *virtual* tree the agent is allowed to see and touch: deny-by-default path scope, self-modify floor | session | **fak** (the reference monitor) | mediate every path syscall |
| **T2 — read / query surface** | the unified read: local tree query (`Glob`/`Grep`) **and** remote-document retrieval (MCP resources, RAG), each trust-gated on the way back | per-call | **fak** boundary over any backend | adjudicate reads, cache the repeatable ones, quarantine poisoned results |
| **T3 — scratchpad / tmpfs** | the agent's ephemeral working FS: intermediate outputs, journaled, GC'd, checkpoint-included | turn → session | the agent, leased | expire-by-default; a lease, not durable memory |
| **T4 — durable memory** | promoted, verified, forgettable facts across sessions | bounded → durable | the [memory-engineering](./memory-engineering.md) discipline | write-time admission gate + verified recall |
| **T5 — KV-cache substrate** | the "in-memory filesystem" of the inference plane: where a context span physically lives, its stable address, addressable eviction | run | the serving layer (`internal/ctxmmu`, radix KV) | placement + bit-exact forgetting |

The one-line separations, each drawn against the sibling it blurs with:

- **T1 is not T0.** The mount view is *what the agent is allowed to see* of a store it
  does not own — a `chroot`/mount-namespace/`overlayfs` view, not the disk. fak presents
  and gates the view; the VM provides the disk.
- **T2 is not T1.** Reading is a distinct syscall from *having access* — a path can be
  in scope (T1) yet a specific read still be refused, cached, or quarantined (T2). And a
  read's backend may be a **remote** document, not a local file at all.
- **T3 is not T4.** A scratchpad artifact is `/tmp`: useful this session, gone next,
  never a fact. The default failure of every "long-term memory" feature is promoting T3
  to T4 because the model found it notable. Truth-duration, not size or location, is the
  boundary ([context is not memory](../CONTEXT-IS-NOT-MEMORY.md)).
- **T4 is not T2.** Durable memory is a *written, admitted* fact; retrieval is a *read*
  of some corpus. A vector store is a T2 mechanism a T4 engineer may select; buying one
  answers none of T4's questions (why is this remembered, is it still true, can you prove
  it is gone).
- **T5 is not T4.** The KV cache is the *substrate* of placement — one physical question
  (where do the bytes sit, what is their address) — not the *governance* of what earns
  durability. "Evict a span from the KV cache" and "forget a memory" are different planes
  ([addressable KV cache](./addressable-kv-cache.md) vs the T4 forgetting witness).

## The question everyone asks: is fak becoming a VM?

As compute moved into isolated cloud VMs and control moved onto phones, the natural read
is "fak should grow a filesystem — snapshots, forks, a sandbox." That is the wrong tier.
The isolation-primitive ladder the field settled on is a **T0** ladder:

| Level | Primitive | Isolation | Who |
|---|---|---|---|
| Strongest | hardware microVM (Firecracker / Kata / Cloud Hypervisor) | dedicated guest kernel | E2B, Fly Sprites, Vercel, Northflank |
| Strong | user-space kernel (gVisor) | syscall interception | Modal |
| Fast/narrow | V8 isolate | per-request JS sandbox | Cloudflare Workers |
| Local | OS sandbox (bubblewrap / Seatbelt) | per-process | Anthropic Claude Code |
| Weak baseline | container / devcontainer | shared kernel | DIY self-host |
| Orthogonal | git worktree | file/concurrency only, **not** security | nearly everyone |

Every row is a way to *provide* a filesystem and *isolate* the process on it. fak builds
**none** of them — that is a hardware/hypervisor play the sandbox vendors own
([landscape note](../notes/RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md)). fak
is the **T1–T2 reference monitor** that rides *inside* whichever T0 the operator chose:
the layer that decides what the agent sees and may do, independent of which box provides
the disk. The distinction is exactly Unix's own: the VFS and the block device are
different layers, and the interesting security decisions live in the VFS. A microVM with
no path-scope floor still lets a prompt-injected agent read every file it can reach; the
floor is worth *more*, not less, when the human has stepped away into an autonomous cloud
run. **fak is the virtual filesystem, not the virtual machine.** (The same fusion fak
already owns — a capability floor *and* KV/prefix reuse at one in-process hop — is why
the T1 mount view and the T5 substrate are one boundary, not two pieces of infra.)

## Prior art (fak invents none of this)

Per the project's honesty discipline
([CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md): a 0/29-novel
prior-art audit), every primitive here is decades old; the contribution is naming the
tiers at one adjudicated boundary.

- **Unix VFS** — one `open/read/write/stat` interface over many backends; the model for
  T2's "one adjudicated read over any backend."
- **Plan 9 / 9P + per-process namespaces** — every resource is a file, and each process
  gets its *own* mount view; the direct ancestor of T1 (the agent sees a private,
  composed tree, not the machine's).
- **FUSE** — a filesystem backed by a user-space process; the shape of "a read whose
  backend is a remote document or a query, not a disk block" (T2).
- **`chroot` / mount namespaces / `overlayfs`** — the mechanisms that *present* a
  restricted, layered view (T1); `overlayfs`'s upperdir is the shape of the T3 scratchpad
  over a read-only base.
- **`tmpfs` / `/tmp`** — expire-by-default working storage; the T3 lifetime model.
- **Bitemporal / cognitive-science truth-duration** — the T3↔T4 boundary long predates
  agents.
- **Agent-era T0**: Anthropic's sandbox FS + egress proxy, Fly's lazy FS, Cloudflare's
  snapshot/fork, E2B microVMs — the boundary fak rides into, not competes with.

## Where fak stands today (honest fences)

Shipped pieces that already live in these tiers — with the gaps named, not hidden:

- **T1** — `internal/vdso/pathscope.go` scopes read paths; the self-modify floor in
  `internal/adjudicator/decide.go` refuses writes into `.git/`, kernel, credential paths
  (`SELF_MODIFY`); `internal/egressfloor` refuses the SSRF/cloud-metadata class. *Gap:* a
  first-class, policy-configurable **mount view** (what tree exists to the agent at all),
  not just per-op deny rules — tracked as child work off this spine and adjacent to
  [#2358](https://github.com/anthony-chaudhary/fak/issues/2358).
- **T2** — the tool vDSO answers repeated read-only calls locally (`internal/vdso`);
  result-side admission quarantines poisoned reads (`/v1/fak/admit`). *Gap:* one
  adjudicated read syscall that spans **local tree query and remote-document retrieval**
  with the same trust gate and cache.
- **T3** — scratchpad is real but ungoverned; leasing/GC/checkpoint-inclusion is proposed
  in [#2420](https://github.com/anthony-chaudhary/fak/issues/2420),
  [#2344](https://github.com/anthony-chaudhary/fak/issues/2344),
  [#2345](https://github.com/anthony-chaudhary/fak/issues/2345).
- **T4** — the promotion ledger, verified recall over the notes backend
  (`internal/memq`, [#2346](https://github.com/anthony-chaudhary/fak/issues/2346)), and
  the write-time durability thesis
  ([context is not memory](../CONTEXT-IS-NOT-MEMORY.md)).
- **T5** — bit-exact addressable KV eviction and the placement layers
  ([four layers of agent memory](../MEMORY-LAYERS-EXPLAINER.md)).

## The one-line test

> If your agent stack cannot say, for a given "file" or "memory," **which of the six
> tiers it is** — whose box provides it, whether the agent may see it, whether the read
> is trusted, whether it survives the session, and whether it can be proven gone — you
> have file *access*, not a virtual *filesystem*.

## Related reading

- [The tool call is a syscall](./tool-call-is-a-syscall.md) — the boundary this tier sits on.
- [Memory engineering](./memory-engineering.md) — the discipline governing T4.
- [Context is not memory](../CONTEXT-IS-NOT-MEMORY.md) — the T3↔T4 truth-duration line.
- [The four layers of agent memory](../MEMORY-LAYERS-EXPLAINER.md) — the T5 placement map.
- [Addressable KV cache](./addressable-kv-cache.md) — the T5 forgetting witness.
- [Cloud / VM / remote-agent landscape](../notes/RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md) — why fak is the boundary, not the box.
