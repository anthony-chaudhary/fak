# WIP inventory

`fak wip inventory` is the read-only census for a peer-dirty checkout. It keeps source work separate from ignored/generated files, registered detached worker worktrees, stale worker residue, and `refs/fak/wip/*` checkpoints.

```powershell
fak wip inventory
fak wip inventory --json > before.json
# clear-out, crash recovery, or worktree reap
fak wip inventory --json > after.json
```

The JSON schema is `fak-wip-inventory/1`. Every count that can fail includes `known`; collection errors are explicit and the command exits nonzero rather than presenting an unknown population as zero. Samples are bounded and deterministically ordered by Git or the collector.

The command does not mutate the index, refs, worktrees, or files. The fixture test exercises tracked and untracked source work, ignored files, detached-worktree work, stale residue, and a checkpoint in one observation and compares filesystem state before and after collection.

Live witness: [`docs/_witnesses/wip-inventory-live-2026-08-17.json`](../_witnesses/wip-inventory-live-2026-08-17.json). That observation covers more than 600,000 ignored paths and records each source/recovery population independently; it is evidence for this observation, not a historical reconstruction.

## Integration path

Capture this schema immediately before and after `clear-out-wip`, worker-worktree reap, and crash recovery. Compare populations by label; never treat a drop in ignored/generated files or a full detached-worktree copy as shipped source WIP.
