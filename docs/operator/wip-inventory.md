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

## Age gate and isolation policy

Use `fak wip inventory --max-untracked-age 1h` as a recurring audit and `fak wip admit --max-untracked-age 1h ...` at task admission. Exit 3 means source work has remained filesystem-only beyond the chosen budget; checkpoint it immediately with `fak wip autocheckpoint --reason manual --session <id>`. Ignored/generated files are a separate population and never trigger this gate.

Shared trunk remains the default for a short, coherent, path-owned change that can be greened and landed promptly. Use the sanctioned detached worker path (`fak worktree worker prepare|land|reap`) when work is long-running, collision-prone, expected to leave a package temporarily broken, or cannot be landed within the untracked-age budget. The worker still lands through the serialized main-trunk gate. Ordinary feature branches and PRs do not solve filesystem-only WIP, stale ownership, or landing races, so they are not the default; introduce them only when review/cutover evidence cannot be represented by the detached-worker receipt and serialized landing path.

## Guarded residue collection

`fak worktree worker reap --all-cold` now includes `unregistered_residue` in its single JSON report. An unregistered `fak-worker-wt-*` directory is eligible only when it is beneath the canonical worker root, absent from Git's worktree registry, has no `.git` marker, has no owner or intent sidecar, and is older than the requested age floor. Dry-run is the default.

With `--apply`, an eligible non-empty directory is ZIP-archived under the dated Fleet worktree archive before removal; the archive and its `fak-worker-residue-archive/1` receipt are verified first, then source absence is verified. Empty eligible residue needs no archive. Fresh, owned, foreign-Git, unreadable, and path-escaping candidates remain with typed keep/failure reasons.
