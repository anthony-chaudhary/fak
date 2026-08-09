# Worker-land crash recovery — 2026-08-08

`fak worktree worker land` uses an isolated index and `git commit-tree` before it
compare-and-swap updates trunk. Every candidate commit is now anchored first under:

```
refs/fak/worker-land/<worktree-name>/<candidate-sha>
```

That local named ref is the post-crash source of truth: it prevents Git GC from
collecting the candidate and remains after the worker process or terminal dies.
The pre-commit file delta also remains in the detached worker worktree until an
explicit reap; cold reap already holds trees with unlanded changes.

## Resume / inspect

```bash
fak worktree worker list
fak worktree worker recover
# inspect one RECOVERABLE candidate
git show refs/fak/worker-land/<worktree-name>/<candidate-sha>
```

`recover` emits JSON. `RECOVERABLE` means the candidate is not an ancestor of
current `HEAD`; re-run the original `worktree worker land` when its worktree and
lease are available, or use an operator-reviewed `git cherry-pick <ref>` when the
worktree is gone. `LANDED` means Git proves the candidate is already in `HEAD`.

## Guarded cleanup

```bash
# accepted only when the candidate is already reachable from HEAD
fak worktree worker recover --cleanup refs/fak/worker-land/<worktree>/<sha>

# destructive escape hatch; inspect with git show first
fak worktree worker recover --cleanup refs/fak/worker-land/<worktree>/<sha> --force
```

Cleanup rejects refs outside the recovery namespace. Without `--force`, it also
rejects every unlanded candidate. Recovery refs are intentionally retained after
a successful land so a crash immediately after trunk CAS is observable and can
be classified without trusting a session log.

These refs are machine-local. For host-loss durability of uncommitted work, use
the existing `fak wip checkpoint` remote-mirror path; worker-land recovery refs
close the narrower process/session-crash window around the off-branch commit and
trunk CAS.
