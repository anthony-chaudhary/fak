# Three-Tree Synchronization Invariants and Multi-Node Safe Sync Rules

Safe sync is a protocol that keeps Git three trees consistent across multi-node agent fleets.

> **TL;DR:** Git requires HEAD, Index, and Working Tree to remain in lockstep. Advancing HEAD in the object database without updating Index and Working Tree creates phantom deletions. Autonomous agent fleets must follow five synchronization invariants to guarantee conflict-free landing.

To verify your workspace state before running synchronization, inspect the safe sync assessment:

```bash
fak sync check
```

## The Three Trees in Git Architecture

Git tracks state across three distinct trees:

| Tree | Storage Location | Role in Git Operations |
|---|---|---|
| HEAD | `.git/objects` | The committed commit object and root tree. |
| Index | `.git/index` | The staging area and binary cache for the next commit. |
| Working Tree | Filesystem directories | Live editable files where you compile and run tests. |

Standard Git commands like `git checkout` or `git merge` update all three trees together in one coordinated filesystem transaction. In contrast, low-level object database plumbing often modifies only one tree at a time. Take care.

## The Object Database Divergence Hazard

When automation bypasses standard working tree commands, it risks object database divergence. For example, consider synthetic tree transplantation using `git merge-tree` and `git commit-tree`:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Git Object Database (ODB)                                │
│    - New synthetic merge commit minted                      │
│    - Branch ref advanced via git update-ref to new commit   │
├─────────────────────────────────────────────────────────────┤
│ 2. Git Index (Staging Area)                                 │
│    - UNCHANGED: retains old pre-merge tree state            │
├─────────────────────────────────────────────────────────────┤
│ 3. Working Tree (Filesystem)                                │
│    - UNCHANGED: remote additions not checked out            │
└─────────────────────────────────────────────────────────────┘
```

This divergence creates phantom deletions. When upstream adds a new file `service.go`, the synthetic commit in HEAD records that file. Because the Index and Working Tree never received `service.go`, Git compares HEAD against Index. Git concludes that you deleted `service.go` in the staging area.

Subsequent commits by agents or scripts inadvertently record that deletion. Upstream code disappears from history without any explicit removal command. They vanish silently.

## The Stale Code Execution Hazard

Object database divergence also hides upstream modifications. If upstream modifies an existing file, the new blob exists in HEAD. Your working tree still contains the previous version on disk. This breaks builds.

When your test runner executes, it tests the stale local version on disk. The test suite passes locally, but your commit history records the upstream version. The resulting commit lands without real test qualification. Tests pass dishonestly.

## Five Invariants for Safe Synchronization

Autonomous agent fleets must uphold five invariants during multi-node operations:

1. `Tree Disjointness:` Concurrent workers must operate on mutually disjoint path sets. The arbiter enforces disjointness through lane leases so that concurrent workers never collide or overwrite each other when editing code across the repository. Check leases.
2. `Three-Tree Alignment:` HEAD, Index, and Working Tree must stay aligned. Advancing HEAD in the object database without updating Index and Working Tree is prohibited on active checkouts. Align always.
3. `Non-Destructive Preservation:` In-flight uncommitted edits must never be overwritten. Fast-forward integration must check `--no-overwrite-ignore` and quarantine colliding untracked files.
4. `Atomic Ref Advancement:` Branch references advance only after Index and Working Tree updates finish cleanly. Partial checkouts must trigger rollbacks.
5. `Multi-Node Convergence:` Fleet nodes must synchronize via verified fast-forwards and retry transient push races. Stalled nodes must reconcile rather than force-push.

## Multi-Node Operational Guidelines

When you coordinate multiple worker nodes across shared remotes, enforce the following workflow:

```bash
# Step 1: Pre-flight check before touching files
fak sync check

# Step 2: Safe fast-forward integration
fak sync apply

# Step 3: Publish verified changes with race retries
fak sync push
```

If local and remote branches diverge, run structured reconciliation:

```bash
fak sync reconcile --apply
```

This command evaluates the divergence. Run checks. It applies disjoint tree integration only when path sets are disjoint, and it ensures working tree files are synchronized.

## Diagnostic and Recovery Runbook

When you suspect three-tree divergence in a worker worktree, execute this diagnostic sequence:

```bash
# Check if HEAD differs from Index and Working Tree
git status --porcelain

# Check staged deletions relative to HEAD
git diff --name-only --cached --diff-filter=D

# Check uncommitted working tree differences
git diff --name-only
```

If phantom deletions appear, align the Index and Working Tree to HEAD safely:

```bash
# Refresh the index to match HEAD without destroying untracked files
git read-tree -u -m HEAD HEAD

# Verify clean status
fak sync check
```

This command restores missing files to disk and aligns the index with HEAD so that future commits never introduce phantom deletions. Act quickly.
