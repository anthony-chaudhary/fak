# Git resource ownership audit — 2026-09-01

## Decision

Adopt a small typed ownership vocabulary in `internal/gitresource`; do not infer authority from a checkout path, branch name, process label, or cleanup caller. The leaf models repository-common, per-worktree, path-set, and selected runtime resources, then admits cleanup only after an exact owner-and-epoch comparison and persisted terminal evidence.

## Evidence mapped to invariants

| Evidence | Invariant adopted |
|---|---|
| Git documents that linked worktrees share repository state except per-worktree files such as `HEAD` and `index`. [`git-worktree(1)`, DETAILS](https://git-scm.com/docs/git-worktree#_details) and [`gitglossary(7)`, worktree](https://git-scm.com/docs/gitglossary) | `HEAD`, index, and files are keyed by `WorktreeID`; refs, common config/hooks, and object maintenance are keyed by `RepositoryID`. A mutation of a common resource requires an exclusive repository-scoped lease. |
| Git exposes the effective worktree Git directory, common directory, top level, and relocated Git paths through `git rev-parse --git-dir`, `--git-common-dir`, `--show-toplevel`, and `--git-path`. [`git-rev-parse(1)`](https://git-scm.com/docs/git-rev-parse) | A workspace handle must carry canonical host, sandbox, repository, worktree, Git-dir, and common-dir identities. A root path remains diagnostic and cannot bind ownership by itself. |
| Git's repository layout distinguishes per-worktree metadata from the shared common directory; per-worktree config is opt-in via `extensions.worktreeConfig`. [`gitrepository-layout(5)`](https://git-scm.com/docs/gitrepository-layout) and [`git-worktree(1)`, CONFIGURATION FILE](https://git-scm.com/docs/git-worktree#_configuration_file) | Resource kind and scope are closed enums rather than path-prefix guesses. Configuration is common unless the resolved resource is explicitly per-worktree. |
| Codex reports describe cleanup risks around untracked no-PR worktrees, worktree locks, live processes retaining a Windows CWD, and session collisions caused by resolving the wrong worktree. [openai/codex#24703](https://github.com/openai/codex/issues/24703), [#33401](https://github.com/openai/codex/issues/33401), [#23515](https://github.com/openai/codex/issues/23515) | Cleanup preserves untracked state, live-CWD state, and foreign locks; runtime resources are explicit kinds rather than assumed to follow the Git worktree lifecycle. |
| Claude Code reports show stale cleanup missing untracked files, cleanup threatening another live session, and stale worktree reuse carrying prior uncommitted state. [anthropics/claude-code#35862](https://github.com/anthropics/claude-code/issues/35862), [#74386](https://github.com/anthropics/claude-code/issues/74386), [#51596](https://github.com/anthropics/claude-code/issues/51596) | A clean-looking path is insufficient. Reap requires exact owner plus fencing epoch, terminal evidence, and a persisted result or snapshot; resumable, dirty, untracked, unpushed, live-CWD, or foreign-lock state always preserves the worktree. |

The OSS issue reports are operational evidence, not authoritative Git specification. The Git manuals define the resource boundaries; the reports identify failure states the cleanup contract must retain.

## Adopted model

- Opaque typed IDs identify repositories, worktrees, resources, owners, leases, host/sandbox workspace bindings, and persisted results/snapshots.
- `ResourceKind` is closed over repository-common Git resources, per-worktree Git resources, explicit path sets, and selected runtime process/CWD/lock resources.
- Leases carry a closed mode, lifecycle state, and positive fencing epoch.
- `WorkspaceHandle.Validate` fails closed when any canonical identity is absent; prose and paths are never ownership proof.
- `AdmitCleanup` is compare-and-delete admission, not cleanup execution. It admits only exact terminal ownership with persisted output and no preservation signal.

## Relationship to existing fak surfaces

This leaf does not replace `internal/workerworktree`, `internal/wipref`, DOS lanes, or their recovery policy. It gives later integrations a shared type boundary. Existing worker-worktree inventory remains responsible for observing dirty/untracked/unpushed and liveness state; WIP/result surfaces remain responsible for producing persisted evidence.

## Follow-ons (not in this issue)

1. Bind `workerworktree` prepare/inventory receipts to canonical `WorkspaceHandle` values resolved through Git.
2. Translate DOS and WIP ownership receipts into `Lease` owner/epoch comparisons at cold-reap and recovery gates.
3. Project runtime CWD and lock observations into `CleanupCandidate` before any destructive worktree operation.
4. Persist cleanup decisions and their evidence as auditable receipts before wiring broader CLI admission.
