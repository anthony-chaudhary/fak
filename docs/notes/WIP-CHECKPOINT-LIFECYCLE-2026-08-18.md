# WIP checkpoint lifecycle: when parked refs are used

A `refs/fak/wip/<session>` checkpoint is a **durability copy of uncommitted work**, not a second normal landing system. The default path remains: edit the shared tree (or a sanctioned detached worker worktree), validate, commit explicit paths, and push. A checkpoint enters the path when a session may stop, restart, lose ownership, or needs an off-machine safety copy.

## Observable states and the right alternative

| Reconcile state | Meaning | Next action |
|---|---|---|
| `SKIP` / `LIVE` | The owner is still active. The ref protects against a crash; it should not compete with that owner's normal commit. | `fak wip status`; let the owner commit normally. |
| `DISCARD_WITNESSED` / `LANDED` | Every checkpoint payload byte is already represented by HEAD. | Retire only through the witnessed lifecycle/reap path; do not hand-delete the ref. |
| `RECLAIM` / `CLOSED_DIRTY_RECOVERABLE` | The owner is gone and the entire delta still applies cleanly. | Use the exact `fak wip reconcile --reclaim --session …` command, which claims and materializes the work without deleting the source ref. |
| `QUARANTINE` / `DIVERGED` | HEAD contains one or more payload paths with different bytes. Automatic landing could revert newer work. | Run each emitted `review_commands` diff, merge intentionally, then use the normal explicit-path commit path. Never checkout a divergent path from the ref. |
| `QUARANTINE` / `CLOSED_DIRTY_RECOVERABLE` | The aggregate patch no longer applies even though no individual payload file is byte-divergent (for example, context drift or tree shape changed). | Inspect `fak wip census --json`; reconstruct deliberately rather than treating quarantine as loss. |

`fak wip reconcile` now prints `checkpoint_class`, `replication`, payload counts, `next_command`, and safe per-path `review_commands`. `QUARANTINE` means **retained for review**, not discarded and not irrecoverable.

## Durability is explicit

- `LOCAL_ONLY`: survives the owning session dying, but not loss of this clone or machine. Run `fak wip sync` when the copy must survive the workstation.
- `STALE_REMOTE`: an older checkpoint is mirrored; the current object is still local-only in practice.
- `REPLICATED`: the current checkpoint object is mirrored off-machine.
- Mirror freshness is provenance: `NEVER_SYNCED` means the clone has not looked, not that the remote has no refs.

Use `fak wip status` for replication and owner-oriented status, `fak wip census --json` for payload A/M/- evidence, `fak wip reconcile` for the action decision, and `fak sweep` only as the broad shared-tree inventory. Refs are implementation anchors; operators should use these verbs rather than `git update-ref` or `git checkout <ref>` directly.
