---
title: "fak CLI reference — fak sync — safe shared-trunk synchronization for dirty worktrees"
description: "Operational CLI guide and architectural specification for fak sync, three-tree consistency invariants, ODB divergence hazards, and multi-node fleet rules."
---

# fak sync — safe shared-trunk synchronization for dirty worktrees

Safe sync is a command family that preserves working-tree modifications while synchronizing Git branches across multi-node agent fleets.

> **TL;DR:** Standard Git pull and merge commands can overwrite uncommitted files on shared trunks. `fak sync` provides safe fast-forward integration and push retries. It guarantees strict three-tree consistency across HEAD, Index, and Working Tree.

To inspect the current synchronization state of your repository without modifying any files, run:

```bash
fak sync check
```

## Command Overview

The `fak sync` toolset contains seven distinct subcommands:

```bash
# Assess incoming upstream changes and verify safety without mutating files
fak sync check [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]

# Apply verified fast-forward integration when all touched paths are safe
fak sync apply [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]

# Safely push current HEAD to remote with automatic race retries
fak sync push [--repo DIR] [--remote origin] [--branch B] [--retries N] [--budget 5s] [--json]

# Drain stranded commits when trunk turns green after a push blockage
fak sync drain [--repo DIR] [--remote origin] [--branch B] [--queue-file F] [--budget D] [--json]

# Reconcile diverged branches via structured non-destructive routes
fak sync reconcile [--repo DIR] [--remote origin] [--branch B] [--goal G] [--apply] [--fetch] [--json]

# Generate a deterministic JSON packet describing divergence and proposed resolution
fak sync packet [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]

# Execute a pre-built reconciliation packet with independent graph verification
fak sync execute [--packet FILE] [--repo DIR] [--remote origin] [--branch B] [--json]
```

## Architectural Foundation: The Three-Tree Invariant

Every Git checkout contains three distinct data structures:

| Tree Layer | Storage Location | State Represented |
|---|---|---|
| HEAD | `.git/objects` | The committed snapshot and root tree in the Git Object Database (ODB). |
| Index | `.git/index` | The staging area cache recording file modes, hashes, and proposed commit state. |
| Working Tree | Filesystem directories | The live files and directories where code is edited, built, and tested. |

Standard porcelain commands like `git checkout` or `git merge` update all three trees concurrently. However, automation scripts and low-level Git plumbing often operate on only one layer.

### The ODB Divergence Hazard and Phantom Deletions

When a tool performs synthetic tree manipulation directly in the Git Object Database (e.g., using `git merge-tree --write-tree` followed by `git commit-tree` and `git update-ref`):

1. `HEAD advances in ODB:` The branch reference points to the newly minted merge commit.
2. `Index remains unchanged:` The staging area retains the older pre-merge file list.
3. `Working Tree remains unchanged:` Upstream files added or modified in the new commit are not written to disk.

This produces phantom deletions. When Git runs `git status` or path-based commit checks, it compares HEAD against the Index. Because HEAD contains new upstream files that do not exist in the Index or Working Tree, Git assumes that you staged the deletion of those files.

Subsequent commits by agents or tools record those deletions permanently into the Git history. Files added upstream vanish from the repository without any explicit deletion command.

### The Stale Code Execution Hazard

ODB divergence also creates stale execution hazards. When an upstream commit modifies an existing file:

1. The new file content exists inside the HEAD commit object.
2. The old file content remains on disk in your working tree.
3. Local test commands execute against the stale disk file rather than the committed state.

Tests pass locally against obsolete code, but the pushed commit carries the upstream version. The software lands without real validation.

## Five Synchronization Invariants

To eliminate phantom deletions and tree skew across multi-node fleets, `fak sync` enforces five invariants:

1. `Tree Disjointness:` Concurrent agents must work in mutually disjoint file paths. Cross-agent path overlapping is rejected by the lane arbiter (`dos arbitrate`).
2. `Three-Tree Alignment:` HEAD, Index, and Working Tree must stay synchronized. Advancing HEAD in the ODB without updating the Index and Working Tree is strictly prohibited on active working trees.
3. `Non-Destructive Preservation:` In-flight edits belonging to other sessions must never be overwritten or deleted. `fak sync apply` enforces `--no-overwrite-ignore` and quarantines colliding untracked files.
4. `Atomic Ref Movement:` A branch reference may only advance after both Index and Working Tree updates complete successfully.
5. `Multi-Node Convergence:` Distributed fleet nodes must converge via verified fast-forwards and retry non-fast-forward push races. Divergences must be resolved via structured routes rather than forced updates.

## Multi-Node Fleet Operational Guidelines

In distributed agent environments operating across multiple compute nodes and worker worktrees:

### Pre-Flight Synchronization

Before an agent touches files or starts a task, it must run:

```bash
fak sync check --fetch
```

If the checkout is behind upstream, integrate changes safely:

```bash
fak sync apply
```

`fak sync apply` runs `git merge --ff-only --no-autostash --no-overwrite-ignore`. It succeeds only when all paths written by upstream are clean at HEAD or already byte-identical on disk.

### Publishing Finished Work

When task verification succeeds, publish through `fak sync push`:

```bash
fak sync push --retries 3 --budget 5s
```

`fak sync push` detects moving-trunk races. If a peer node lands a commit while your push is in flight, `fak sync push` checks whether your HEAD already contains the new remote commit. If so, it retries cleanly without requiring manual pull or reset cycles.

### Structured Divergence Reconciliation

When local and remote branches diverge (e.g., both contain independent commits), run:

```bash
fak sync reconcile --apply
```

`fak sync reconcile` evaluates the divergence and routes to an approved primitive:

| Route | Condition | Action Taken |
|---|---|---|
| `ROUTE_NOOP` | Local and remote tips are identical | No action needed. |
| `ROUTE_PUSH` | Local is strictly ahead of remote | Executes safe push. |
| `ROUTE_APPLY` | Local is strictly behind remote | Executes fast-forward apply. |
| `ROUTE_DISJOINT_INTEGRATE` | Local and remote touch disjoint paths | Integrates disjoint changes while updating working tree files. |
| `ROUTE_HOLD_DIRTY_COLLISION` | Uncommitted local edits collide with upstream | Refuses mutation and reports colliding paths. |
| `ROUTE_HOLD_MERGE_ACTIVE` | An in-flight merge is active (`MERGE_HEAD`) | Refuses mutation until merge completes or clears. |
| `ROUTE_DRAIN` | Trunk is red; commits must be queued | Enqueues commits in drain queue until trunk clears. |

## Diagnostic Runbook: Resolving Three-Tree Skew

If you encounter unexpected staged deletions or phantom file losses:

```bash
# 1. Inspect status for phantom deletions
git status --porcelain

# 2. Check diff against HEAD
git diff HEAD

# 3. Resynchronize Index and Working Tree to match HEAD without clobbering untracked files
git read-tree -u -m HEAD HEAD

# 4. Verify three-tree consistency
fak sync check
```

This restores missing files to disk and brings HEAD, Index, and Working Tree back into perfect alignment.
