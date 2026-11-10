---
title: "Sharing work in progress"
description: "How a human engineer on the shared checkout shares, hands off, parks, and recovers uncommitted work: the fak wip checkpoint family, same-clone takeover, cross-machine handoff, and the privacy boundary of tree-wide capture."
---
# Sharing work in progress

You have uncommitted work on the shared checkout and need to share it, hand it
off, park it across an integration, or make sure it survives a crash. The
machinery is the `fak wip` family: a session's uncommitted tracked delta is
snapshotted into a gc-safe object under `refs/fak/wip/<session>` without
touching the index, the working tree, or any branch. The state is git refs, so
it survives process death and is readable by every tool in this repo; the
authoritative in-repo reference is the built-in usage text, `fak wip help`.

Two things to know before anything else:

- A checkpoint is a **tree-wide** capture: on a shared tree it includes peer
  edits. That is what makes crash recovery total, and it is why publishing one
  off-machine is an explicit decision (see [Privacy notes](#privacy-notes)).
- A checkpoint is a **durability copy**, not a second landing system. The
  default landing path is still: validate, commit explicit paths, push.

## The one-minute answer

### (a) Land your own WIP

```bash
fak wip checkpoint --session my-task                       # snapshot -> refs/fak/wip/my-task
fak wip land my-task --path internal/foo/a.go --path internal/foo/b_test.go
fak sync push
```

`checkpoint` records the whole tracked delta but stamps the `--path` arguments
as what this session **claims**; `land` commits exactly that declared file set
and refuses (exit 3, `TREE_WIDE_SNAPSHOT`) to commit an undeclared tree-wide
snapshot while another session holds a live checkpoint. `land --push` folds the
last two steps into one.

### (b) Take over a colleague's session in the same clone

```bash
fak wip reconcile --reclaim                                # ranked worklist of exact argv
fak wip reconcile adopt <session> --patch-out task.patch   # or: --into <dir-outside-repo>
# review the patch (or the isolated directory), then:
git apply task.patch
fak wip reconcile adopt <session> --land                   # optional: land through the usual refusals
```

`--reclaim` prints only the RECLAIM rows (crashed owner, delta still applies
cleanly), ranked most-decayed-first, each with the exact command to run.
Adopting takes the witnessed claim **before** the delta is re-materialized, so
two successors reading the same queue cannot recover one checkpoint twice.
`--into` must point outside this repo's working tree; `--patch-out` defaults to
`<git-dir>/fak-adopt/<session>.patch`.

### (c) Hand off across machines today

Owner side:

```bash
fak wip checkpoint --session my-task
fak wip sync --push-only                                   # publish to origin; downloads nothing
```

Peer side:

```bash
fak wip sync --fetch-only                                  # import into refs/fak/remotewip/origin/*
fak wip status --fleet                                     # who holds what: host, session, size, disposition
```

Peers can now *see* the checkpoint. Materializing it is where the current gap
lives — see [What is not possible today](#what-is-not-possible-today). Until a
dedicated verb lands, the documented-discouraged fallback is raw git against
the mirror namespace:

```bash
git fetch origin '+refs/fak/wip/*:refs/fak/remotewip/origin/*'
git diff refs/fak/remotewip/origin/<session>^1 refs/fak/remotewip/origin/<session> > task.patch && git apply task.patch
```

This works because a checkpoint commit is parented on HEAD, so
`diff <commit>^1 <commit>` is exactly the captured delta.

### (d) Suspend paths for integration

```bash
fak wip park <session> --path <p>...                       # dry-run preview (default)
fak wip park <session> --path <p>... --apply               # mint path-scoped checkpoint, clear to HEAD, integrate, reapply
fak sync reconcile --suspend-paths <p>...                  # same suspension driven through the reconcile router
```

`park` is for owner-authorized dirty paths that must survive an in-place
integration: it mints a durable path-scoped recovery checkpoint at
`refs/fak/wip/<session>/park`, clears the selected paths to HEAD, integrates
the target, and reapplies unique effects without clobbering unrelated dirty
paths. Dry-run first, always.

## Which mechanism when

| You need to... | Use | Why |
|---|---|---|
| Land a short, coherent, path-owned change | Just commit (`fak commit --path ...`) | Shared trunk remains the default for work that can be greened and landed promptly ([trunk-vs-worker rule](wip-inventory.md)). |
| Survive a crash, restart, or lost session | `fak wip checkpoint` (automatic at risky boundaries: `fak wip autocheckpoint`) | Durable copy under `refs/fak/wip/*`; protects the session, not the machine. |
| Keep dirty paths alive across an in-place integration | `fak wip park` | Path-scoped suspend + reapply; dry-run by default. |
| Run long, collision-prone, or temporarily-broken work | Detached worker worktree: `fak worktree worker prepare\|land\|reap` | Single-writer isolation that still lands through the serialized trunk gate. It is **not** a two-human collaboration mechanism: one lease, one owner PID stamp, and `reap --all-cold` can destroy a cold human's unlanded work. See [managed worker worktrees](../managed-worker-worktrees.md). |
| Share committed-but-unmerged work across clones | `fak sync push --source-commit <FULL_OID> --branch handoff/<name>` | Pushes an ancestor commit to a `handoff/` branch without moving HEAD; the only "handoff"-named surface. |
| Make uncommitted work survive machine loss or reach another host | `fak wip sync` (opt-in) | Replicates `refs/fak/wip/*` to the remote; see the privacy notes below. |

Ordinary feature branches and PRs do not solve filesystem-only WIP, stale
ownership, or landing races, so they are not the default; introduce them only
when review/cutover evidence cannot be represented by the detached-worker
receipt and serialized landing path.

## Privacy notes

- **Tree-wide capture.** A checkpoint includes every tracked change in the
  shared tree, peers' half-done work included. Reading it locally is harmless;
  publishing it is not.
- **Sync is opt-in and push-first.** Nothing runs `fak wip sync` for you. Push
  runs first and a failed push stops the sync, so a sync that errors has changed
  nothing locally.
- **A completed sync publishes every session's captures from this tree** —
  there is no "publish only my paths" option. Subject to the size gate: deltas
  beyond the default 1 MiB cap publish a metadata-only stub at the same ref;
  the session stays enumerable fleet-wide, its bytes stay on the owner host.
- **Prefer `--push-only`** unless you specifically need to import peer hosts'
  trees (`--fetch-only`). If publishing the whole tree is a problem, do not
  sync; keep the checkpoint local and move the delta by other means.

## What is not possible today

- **Cross-host restore of a peer's mirrored delta.** Peers can see your pushed
  checkpoint (`fak wip sync --fetch-only`, then `fak wip status --fleet`), but
  no verb materializes or lands a peer's mirrored delta: `restore`, `land`, and
  `reconcile adopt` read only the local `refs/fak/wip/*` namespace, and
  cross-host reclaim is explicitly deferred. The raw-git fallback in
  [section (c)](#c-hand-off-across-machines-today) is the only route today.
  Follow-up: a `fak wip handoff` verb that closes this gap.
- **Draft-PR / bundle / format-patch conventions.** None exist in this repo;
  the handoff surfaces are the ones above.

## Recovery and stale work

- `fak wip queue` — one deterministic, read-only action queue over every
  sanctioned worktree and local-only checkpoint, with the exact next command.
- `fak wip readiness` — joins the canonical WIP surfaces into one
  freshness-stamped receipt; the "what next" surface for fresh-start admission.
- [WIP checkpoint lifecycle](../notes/WIP-CHECKPOINT-LIFECYCLE-2026-08-18.md) —
  the reconcile states (SKIP/LIVE, DISCARD_WITNESSED, RECLAIM, QUARANTINE) and
  the right alternative for each.
- [Stale work](../stale-work.md) — what to do when work goes quiet.
- [WIP inventory](wip-inventory.md) — the read-only census, the age gate, and
  guarded residue collection.
- `fak wip selfcheck` — proves checkpoint → wipe → restore reproduces the delta
  byte-identical, on this machine, right now.
