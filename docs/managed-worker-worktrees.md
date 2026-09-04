---
title: "Managed worker worktrees: lifecycle, portable defaults, and remote recovery"
description: "Operator guide and runbook for detached build-isolation worker worktrees in fak: lifecycle operations, environment configuration, and crash recovery."
---

# Managed worker worktrees

Managed worker worktrees provide filesystem and build isolation for concurrent
autonomous coding agents. A **git worktree** is an additional linked working
tree checked out from the same repository; in fak, each worker receives its own
private working directory checked out at a **detached HEAD** (a specific commit
hash rather than a branch name). This architecture allows parallel workers to edit,
build, and test code simultaneously without race conditions, while preserving the
repository's strict single-trunk discipline.

This guide provides the complete operator reference and runbook for discovering
defaults, configuring environment variables, driving lifecycle operations, and
recovering from crashes. See also the [CLI reference](cli-reference.md),
[AGENTS.md](../AGENTS.md) for shared-trunk rules, [CONTRIBUTING.md](../CONTRIBUTING.md)
for the contributor workflow, and [WIP inventory](operator/wip-inventory.md) for
tracking uncommitted work across checkouts.

## Architecture and core principles

Concurrent execution on a shared repository creates three major bottlenecks
when multiple workers execute in the same working tree (#1334 / #1333):

1. **Shared git index lock:** Simultaneous git commands collide on `.git/index.lock`.
2. **Build cache collisions:** Shared Go build caches (`GOCACHE`) cause intermediate
   compilation artifacts from one worker to turn another worker's builds red.
3. **Dirty working tree cross-talk:** In-flight uncommitted edits from one session
   leak into diffs and status checks of another session.

Managed worker worktrees eliminate these issues while adhering to the
single-source-of-truth trunk law (`OFF_TRUNK`):

- **Detached HEAD at trunk commit:** Worktrees are created at a detached commit
  pinned to trunk HEAD (`git worktree add --detach <path> <sha>`). Because the
  worktree is not attached to `main`, git allows it to exist concurrently; because
  it is not on a feature branch, it never violates the off-trunk prohibition.
- **Private build isolation:** Environment variables (`GOCACHE`, `GOTMPDIR`,
  `DISPATCH_WORKSPACE`) point inside the worker's worktree directory, isolating
  compilation caches and temporary files.
- **Single-writer landing (`land_worktree_diff`):** When a worker finishes, its
  diff-since-base is serialized and applied back onto `main` under its acquired
  lane lease. An isolated temporary index (`GIT_INDEX_FILE`) and compare-and-swap
  (CAS) ref updates prevent race conditions against concurrent trunk changes.

## Portable defaults discovery

Operators and automation scripts can inspect the active worktree configuration
without modifying any state using the `defaults` sub-command:

```bash
fak worktree worker defaults
fak worktree worker defaults --json
```

The plain-text output displays human-readable paths:

```text
schema: fak.worktree.defaults.v1
repo_root: /path/to/repo
worker_worktree_root: /path/to/worker-worktrees
root_source: environment: FLEET_WORKER_WORKTREE_ROOT
default_lease_identity_basis: lane_key_timestamp
supported_env_overrides: FLEET_WORKER_WORKTREE_ROOT
```

With `--json`, it emits machine-readable JSON adhering to the
`fak.worktree.defaults.v1` schema:

```json
{
  "schema": "fak.worktree.defaults.v1",
  "repo_root": "/path/to/repo",
  "worker_worktree_root": "/path/to/worker-worktrees",
  "root_source": "environment: FLEET_WORKER_WORKTREE_ROOT",
  "default_lease_identity_basis": "lane_key_timestamp",
  "supported_env_overrides": ["FLEET_WORKER_WORKTREE_ROOT"]
}
```

**Zero-mutation guarantee:** The `defaults` command is strictly read-only. It
performs no disk writes, git operations, or ref mutations.

## Storage roots and environment configuration

### Worker root resolution

Managed worktrees live **outside** the repository working tree so they never
appear in `git status` or interfere with uncommitted file tracking. The parent
directory for all worker worktrees is resolved in order:

1. **Environment override:** The `FLEET_WORKER_WORKTREE_ROOT` environment variable
   if set and non-empty (`root_source: "environment: FLEET_WORKER_WORKTREE_ROOT"`).
2. **Windows OS fallback:** `%LOCALAPPDATA%\Fleet\worker-worktrees` if `LOCALAPPDATA`
   is defined (`root_source: "os_fallback: LOCALAPPDATA"`).
3. **Non-Windows OS fallback:** `$TMPDIR/Fleet/worker-worktrees` or `/tmp/Fleet/worker-worktrees`
   via `os.TempDir()` (`root_source: "os_fallback: temp_dir"`).

### Directory naming convention

Each worktree directory follows the deterministic naming scheme:

```text
fak-worker-wt-<lane>-<hashed-key>
```

- `fak-worker-wt`: The constant identifying marker segment (`WorktreeMarker`).
- `<lane>`: The sanitized worker lane (for example `cmd`, `gateway`, `docs`).
- `<hashed-key>`: A 12-character SHA-1 hash of the unique worker key (issue
  number, session ID, or wave identifier).

### Build isolation environment variables

When a worker process executes inside a managed worktree, its child environment
is populated with isolated paths:

| Variable | Value | Purpose |
|---|---|---|
| `GOCACHE` | `<worktree>/.gocache` | Private Go build cache; prevents cross-worker compiler pollution. |
| `GOTMPDIR` | `<worktree>/.gotmp` | Private temporary directory for compiler operations. |
| `DISPATCH_WORKSPACE` | `<worktree>` | Repoints tools to the isolated workspace root. |
| `FLEET_WORKER_WORKTREE_DIR` | `<worktree>` | Identifies the active worktree directory to child processes. |

Disposable build directories (`.gocache` and `.gotmp`) are created by
`EnsureBuildDirs` during preparation, recreated upon reuse, and purged upon reap.

## Lifecycle operations runbook

The managed worker worktree lifecycle follows an orderly state progression:

```text
[PREPARE] -> [WORK & TEST] -> [LAND] -> [REAP]
    |                             |
    +-----> (on crash) ---------> [RECOVER]
```

### 1. `prepare` — Create or lease a detached worktree

Prepares an isolated worktree directory pinned at trunk HEAD (or an explicit
commit SHA), stamped with ownership metadata.

```bash
fak worktree worker prepare --lane <lane> --key <key> [flags]
```

#### Flags

- `--lane <name>`: **(Required)** Worker lane (for example `cmd`, `gateway`, `docs`).
- `--key <id>`: **(Required)** Worker unique key (issue number, wave ID, or session ID).
- `--base-sha <sha>`: Commit SHA to pin the detached worktree at (defaults to trunk `HEAD`).
- `--wt-root <dir>`: Parent directory override for the worktree.
- `--lease-id <id>`: Lease identity for the owner stamp (defaults to `FAK_LEASE_ID` or `resolve-<lane>`).
- `--owner-pid <pid>`: Process ID of the owning worker (defaults to current PID).
- `--capacity-reason <why>`: Advisory explanation when creating worktrees above the setpoint (50).
- `--message <msg>`: Intended signed commit message, stored in the `.intent` sidecar.
- `--path <path>`: Repeatable flag recording intended touch paths for `LAND_READY` lifecycle detection.
- `--root <dir>`: Repo root (default: discovered from working directory).

#### Behavior and output

- If a worktree with the same lane and key already exists and is clean, it is reused (`reused: true`).
- If `.worktreeinclude` exists in the repo root or worktree, declared include patterns are copied.
- Writes an owner stamp containing `pid`, `lease_id`, and `created_at`.
- Above the advisory capacity setpoint of 50 active worktrees, `capacity` returns an advisory notice.
- Emits JSON containing the worktree path, base SHA, environment map, and capacity advisory.

### 2. `list` — Inspect inventory and status evidence

Enumerates active managed worktrees and audits their lifecycle state:

```bash
fak worktree worker list
fak worktree worker list --json
```

#### Flags

- `--json`: Emits structured lifecycle inventory (`fak-worker-worktree-lifecycle/1`).
- `--capacity-reason <why>`: Records reason for retained capacity above the setpoint.
- `--remote <remote>`: Includes scrubbed cross-host snapshots for the named Git remote.
- `--fetch`: Refreshes the remote snapshot mirror before listing.
- `--root <dir>`: Repo root (default: discovered from working directory).

#### Output details

- **Standard text:** Reports count, paths, and capacity advisory to stdout/stderr.
- **`--json`:** Emits structured inventory for each worktree:
  - `path`, `base_sha`, `head_sha`.
  - `association`: `lane`, `lease_id`, state (`ASSOCIATED`, `UNASSOCIATED`).
  - `liveness`: owner PID status (`LIVE`, `DEAD`), lease status (`LIVE`, `RELEASED`).
  - `cleanliness`: clean vs dirty working tree paths.
  - `lifecycle`: lifecycle classification (`LAND_READY`, `DIRTY_UNREGISTERED`, `COLD_REAPABLE`, etc.).
  - `action`: executable `fak worktree worker land` argv when `LAND_READY`.

### 3. `land` — Apply worktree diff back to main

Applies the worktree's diff-since-base back onto the main trunk as a single
verified commit.

```bash
fak worktree worker land --worktree <dir> [flags]
```

#### Flags

- `--worktree <dir>`: **(Required)** Path of the worker worktree to land.
- `--base-sha <sha>`: Commit SHA the worktree was pinned at (diff base; default: `HEAD`).
- `--msg-file <file>`: Commit message file for `git commit -s -F` (defaults to worktree tip message).
- `--paths <path>`: Scopes the commit to specific paths; repeatable (default: entire applied diff).
- `--verify <hook>`: Pre-land verification executed inside the worktree (`off` or `go-build`).
- `--core-lock-maintenance-witness <claim>`: Witness claim required when modifying core-locked paths.
- `--recovery-remote <remote>`: Git remote receiving candidate recovery ref before trunk CAS.
- `--require-remote-recovery`: Refuses trunk CAS if remote candidate read-back fails.
- `--disambiguation-timeout-ms <ms>`: Disambiguation deadline (1..900000 ms; default 120000 ms).
- `--unsafe-skip-symptom-witness`: Bypasses mandatory fail-to-pass test witness for `fix(*)` commits.
- `--root <dir>`: Repo root (default: discovered from working directory).

#### Landing mechanics and safety guarantees

1. **Local recovery ref anchor:** Before touching the trunk ref, the candidate commit
   is anchored locally at `refs/fak/worker-land/<worktree-name>/<candidate-sha>`. If the
   process terminates during landing, the commit is not lost.
2. **Isolated index:** Staging and commit construction execute in a throwaway
   index (`GIT_INDEX_FILE`), avoiding contention on the shared `.git/index`.
3. **Compare-and-swap (CAS):** Trunk `HEAD` is updated using an atomic CAS ref update.
   If a peer landed in the gap, CAS fails and retries (up to 5 attempts).
4. **Readback verification:** After committing, `LandReadbackVerify` confirms trunk
   `HEAD` contains the worker's intended paths (`LAND_READBACK_MISMATCH` refusal if missing).
5. **Symptom witness:** Fix commits (`fix(...)`) must include a test that reproduces
   the failure on the parent commit and passes on the fix.

### 4. `reap` — Clean removal and bulk sweeps

Releases worktrees that are no longer needed. Supports single-worktree targeted
mode and bulk cold sweeps:

```bash
fak worktree worker reap --worktree <dir> [--superseded-by <sha>] [--max-wait <duration>]
fak worktree worker reap --all-cold [--apply] [--age-floor-min <min>] [--even-if-unlanded]
```

#### Flags

- `--worktree <dir>`: Target worktree directory (single-worktree mode).
- `--superseded-by <sha>`: Authorizes dirty worktree cleanup only if `<sha>` is on trunk and matches bytes.
- `--max-wait <duration>`: Timeout deadline for inspection and removal (default `10s`).
- `--all-cold`: Bulk mode: enumerates all worktrees and selects cold candidates.
- `--apply`: Actually delete selected worktrees (**dry-run by default**; reports plan without deleting).
  Can also be enabled via `FAK_WORKTREE_COLD_COLLECT=apply`.
- `--age-floor-min <min>`: Minimum age in minutes for a dead-lease worktree to be eligible (default `30`).
- `--even-if-unlanded`: In bulk mode, also deletes worktrees kept only because they contain uncommitted diffs.
  **Destructive:** destroys uncommitted work; use only for abandoned sessions.
- `--root <dir>`: Repo root (default: discovered from working directory).

#### Bulk sweep safety invariants

- **Dry-run by default:** Without `--apply`, reports candidate worktrees, bytes, and
  reasons without removing any files.
- **Lease liveness gate:** A worktree whose lane lease is still active is **never** reaped.
- **Age grace floor:** Keeps worktrees younger than `--age-floor-min` even if the lease
  appears released, preventing races with recently finished workers.
- **Work preservation:** Worktrees with uncommitted changes are preserved and reported
  as `held_by_work` unless `--even-if-unlanded` is explicitly specified.
- **Unregistered residue:** Stale directories under the worker root without valid Git
  metadata are identified as `unregistered_residue` and archived to zip before deletion.

### 5. `gc` — Owner-stamped leak garbage collection

Performs leak garbage collection targeting worktrees abandoned by crashed processes:

```bash
fak worktree worker gc [--max-age <duration>]
fak worktree worker gc --apply [--max-age <duration>]
```

#### Flags

- `--max-age <duration>`: Minimum owner-stamp age before eligibility (default `30m`).
- `--dry-run`: Reports candidates without deleting (**default behavior**).
- `--apply`: Removes eligible worktrees and runs `git worktree prune`.
  `--dry-run` and `--apply` are mutually exclusive.
- `--root <dir>`: Repo root (default: discovered from working directory).

#### Dual qualification rule

A worktree is eligible for `gc` removal only when **both** conditions are proven:

1. **Owner process is dead:** The PID recorded in the owner stamp no longer exists.
2. **Lane lease is released:** The lease oracle confirms the stamped lease is inactive.

### 6. `publish` and `recover` — Remote publication and crash recovery

Provides durability against local host loss and tools for resuming interrupted lands.

#### Remote publication

Publishes a scrubbed snapshot of local worktree states to a remote Git ref:

```bash
fak worktree worker publish --remote origin --dry-run
fak worktree worker publish --remote origin --apply
```

#### Crash recovery inventory

When a worker crashes after `commit-tree` but before or during trunk CAS, the candidate
commit remains anchored under `refs/fak/worker-land/<worktree>/<candidate-sha>`.
To inspect and recover:

```bash
fak worktree worker recover
fak worktree worker recover --remote origin --fetch
```

#### Recovery candidate states

| State | Meaning | Recommended action |
|---|---|---|
| `LOCAL_ONLY` | Candidate exists locally; protected against process crash. | Inspect with `git show <ref>`; re-run `land` or cherry-pick. |
| `REPLICATED` | Candidate exists locally and in verified remote mirror. | Safe against host loss; re-run `land` or cherry-pick. |
| `REMOTE_ONLY` | Found on remote mirror but missing locally (fresh clone). | Restore local ref via printed command, then inspect/land. |
| `LANDED` | Git history proves the candidate is already in trunk `HEAD`. | Safe to clean up. |

#### Guarded recovery cleanup

Local recovery refs can be pruned once landed:

```bash
fak worktree worker recover --cleanup refs/fak/worker-land/<worktree>/<sha>
fak worktree worker recover --cleanup refs/fak/worker-land/<worktree>/<sha> --force
```

Remote recovery refs require ancestry verification:

```bash
fak worktree worker recover --remote origin \
  --cleanup-remote refs/fak/worker-land/<worktree>/<sha> \
  --worktree-name <worktree>

fak worktree worker recover --remote origin \
  --cleanup-remote refs/fak/worker-land/<worktree>/<sha> \
  --worktree-name <worktree> --apply
```

Cleaning a peer's remote recovery ref additionally requires `--allow-peer`.

## Sub-commands summary

| Sub-command | Purpose | Default mode | Primary receipt / schema |
|---|---|---|---|
| `defaults` | Discover resolved roots and supported env overrides | Read-only | `fak.worktree.defaults.v1` |
| `prepare` | Create/lease detached worktree with isolated env | Mutating | `worktreePrepareOut` |
| `list` | Enumerate active worktrees and lifecycle states | Read-only | `fak-worker-worktree-lifecycle/1` |
| `land` | Apply worktree diff onto trunk as a verified commit | Mutating | `workerworktree.Result` |
| `reap` | Release single worktree or perform bulk cold sweep | Dry-run (`--all-cold`) | `worktreeColdReapOut` |
| `gc` | Collect dead-owner, released-lease worktrees | Dry-run | `workerworktree.GCReport` |
| `publish` | Publish scrubbed host lifecycle snapshot to remote | Dry-run | `SnapshotPublishResult` |
| `recover` | Enumerate recovery candidates and clean up landed refs | Read-only | `worktreeWorkerRecoverOut` |
