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
2. **Build-state ownership:** Disposable caches and compiler temporary files need
   a clear cleanup owner so one worker's cleanup does not remove another's state.
3. **Dirty working tree cross-talk:** In-flight uncommitted edits from one session
   leak into diffs and status checks of another session.

Managed worker worktrees eliminate these issues while adhering to the
single-source-of-truth trunk law (`OFF_TRUNK`):

- **Detached HEAD at trunk commit:** Worktrees are created at a detached commit
  pinned to trunk HEAD (`git worktree add --detach <path> <sha>`). Because the
  worktree is not attached to `main`, git allows it to exist concurrently; because
  it is not on a feature branch, it never violates the off-trunk prohibition.
- **Private build isolation by default:** `GOCACHE`, `GOTMPDIR`, and
  `DISPATCH_WORKSPACE` point inside the worker's directory. An explicit shared
  cache opt-in changes only `GOCACHE`; temporary files and source remain isolated.
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
uses these paths:

| Variable | Value | Purpose |
|---|---|---|
| `GOCACHE` | `<worktree>/.gocache` by default, or the absolute `FAK_SHARED_GOCACHE` path | Private cache by default; explicit shared-cache reuse when selected. |
| `GOTMPDIR` | `<worktree>/.gotmp` | Private temporary directory for compiler operations. |
| `DISPATCH_WORKSPACE` | `<worktree>` | Repoints tools to the isolated workspace root. |
| `FLEET_WORKER_WORKTREE_DIR` | `<worktree>` | Identifies the active worktree directory to child processes. |

`EnsureBuildDirs` creates or recreates the selected cache and private temporary
directory when invoked before Go. It reports directory-creation errors and never
recreates a missing worktree. Reaping removes directories inside the worktree;
an external shared cache is outside that cleanup boundary.

### Opt into an operator-owned shared Go cache

Set `FAK_SHARED_GOCACHE` to an absolute directory outside all disposable
worktrees. `WorktreeEnv` uses an explicitly supplied caller-map value first,
otherwise the process environment. Whitespace is trimmed. Unset, empty, `off`,
relative, or NUL-containing values retain `<worktree>/.gocache`; an explicit
empty or invalid caller value also suppresses an ambient opt-in. An ambient
`GOCACHE` alone does not enable this option. Inspect the returned `GOCACHE` in
the preparation receipt to see the effective selection.

Validation applies to the selected `GOCACHE` path. Other caller-map values,
including the original `FAK_SHARED_GOCACHE` entry, are forwarded unchanged;
constructing a valid process environment remains the caller's responsibility.

Go's content-addressed build cache supports concurrent Go commands. Share it
only among trusted writers, keep `GOTMPDIR` private, and manage the external
directory's retention separately. This option adds no cache collector and does
not give the worktree reaper ownership of shared storage. Unset
`FAK_SHARED_GOCACHE` to restore the private-cache default; existing isolated-cache
behavior is otherwise unchanged.

Go does not detect changes to external C libraries imported through cgo. After
changing those libraries, force the affected rebuild (for example, `go build -a`)
or coordinate cache invalidation with the cache owner. Never clear a shared cache
while other builds are using it. See `go help cache` for Go's cache contract.

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
- `--owner-pid <pid>`: Process ID of the owning worker. Defaults to the `prepare` process itself, which exits as soon as it prints its receipt; the tree is then protected from the prepare-time dead sweep only for the dead-owner grace window (15 minutes from prepare, or while dirty). Pass the long-lived worker's PID.
- `--capacity-reason <why>`: Advisory explanation when creating worktrees above the setpoint (50).
- `--message <msg>`: Intended signed commit message, stored in the `.intent` sidecar.
- `--path <path>`: Repeatable flag recording intended touch paths for `LAND_READY` lifecycle detection.
- `--root <dir>`: Repo root (default: discovered from working directory).

#### Behavior and output

- If a worktree with the same lane and key already exists and is clean, it is reused (`reused: true`). Reuse requires a live git registration (listed, not prunable, `.git` link present) and is re-read after stamping; a gutted target is refused (`ORPHAN_TARGET_REFUSED`) and a target that vanishes mid-prepare reports `PREPARE_REUSE_LOST`.
- Attempts for the same lane and key are serialized by a per-target lock under `.fak-worker-prepare-locks`; a timed-out attempt removes its partial tree before releasing it, and a concurrent attempt waits (up to its budget, or `FAK_WORKER_PREPARE_LOCK_WAIT` when unbounded) or reports `PREPARE_BUSY` without touching the target.
- `--message` and `--path` are validated before anything is materialized.
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
- `--verify <hook>`: Pre-land verification executed inside the worktree (`off` or `go-build`). Use `off` only for a pre-existing red: the same packages fail with the same first error on the landing base, and every condition and receipt in `.claude/skills/commit-clean/SKILL.md` "Pre-existing red" holds.
- `--core-lock-maintenance-witness <claim>`: Witness claim required when modifying core-locked paths.
- `--recovery-remote <remote>`: Git remote receiving candidate recovery ref before trunk CAS.
- `--require-remote-recovery`: Refuses trunk CAS if remote candidate read-back fails.
- `--disambiguation-timeout-ms <ms>`: Disambiguation deadline (1..900000 ms; default 120000 ms).
- `--unsafe-skip-symptom-witness`: Bypasses mandatory fail-to-pass test witness for `fix(*)` commits.
- `--symptom-timeout <duration>`: Wall-clock budget for the `fix(*)` symptom witness (two scratch worktrees plus a candidate and a parent `go test` compile; default `10m`). A run the budget kills abstains as `SYMPTOM_TIMEOUT`, never `SYMPTOM_NO_MATCH`; raise it for large packages or a loaded host.
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

#### Land-verify candidates (`--candidates`)

`land` verifies a workspace whose `go.work` escapes the repository in a detached
checkout named `.fak-cand-validate-*` beside the repository (in the system temp
directory otherwise), and removes it when verification ends. A land that is killed
first (for example by a supervisor's deadline) cannot remove it. `gc --candidates`
collects those leaked checkouts:

```bash
fak worktree worker gc --candidates [--legacy-max-age <duration>]
fak worktree worker gc --candidates --apply [--legacy-max-age <duration>]
```

A candidate named `.fak-cand-validate-<pid>-<start>-<n>` is eligible when that pid
and process start time no longer identify a running process. A name without an
owner identity is eligible once untouched past `--legacy-max-age` (default `2h`). Each
directory is removed before `git worktree prune` runs against the repository it was
registered in. A land killed inside `git worktree add` also leaves git's
`initializing` lock on a registration whose directory is gone, which a plain prune
never clears; the sweep removes that lock under the same owner/age rule so the
prune collects it. Candidates never hold unique work: each is a copy of an existing
commit plus a diff the worker worktree still holds.

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

## Preservation-only preparation (explicit opt-in)

`fak worktree worker prepare --preserve-existing` creates only a new uniquely
keyed detached Git worker. Default prepare behavior is unchanged. This mode
uses native Git creation directly on every platform: it deliberately avoids
Darwin block-clone and its pool/fallback lifecycle. It never calls the sweep,
pool enumeration/acquisition, same-key reuse, reset, clean, force-reap, partial
cleanup or Git prune. Existing checkouts, registrations and worker sidecars
are retained. Existing or unreadable targets, dangling symlinks, registrations
(including prunable ones), owner/pool/legacy `.idle`/intent/message sidecars and their `.tmp` siblings are conflicts.
Choose a genuinely new admitted key; this mode does not automatically recover
or discard a previous partial attempt.

In addition to `--lane`, `--key` and the full `--base-sha`, callers must supply
`--owner-pid` for the actual long-lived owner, `--lease-id`, `--lease-holder`,
`--lease-generation`, one or more concrete `--admitted-path` source paths, and
`--reserve-bytes` from the current resource policy. Obtain these lease values
through normal ownership admission. Do not fabricate a lease, holder, generation
or PID. An explicitly supplied empty `--lease-id` is refused before ambient or generated defaults. Preservation preparation only reads the lease and rechecks the native
fence and path ownership; it does not acquire, renew, release or publish it.
An unfenced legacy lease must migrate through the normal admission workflow.
The owner must keep the admitted lease alive through prepare and later work.

Admission, process liveness and disk headroom are rechecked around creation and
before the ready receipt. Concrete paths must remain nonempty after normalization; `.` and wildcard/path traversal forms are refused. The exact full commit, clean checkout, index-lock
absence, new registration and stamped owner/worker lease are verified. Unknown
inventory, stale fencing, ownership mismatch, unreadable resources or
insufficient space refuse without contraction. Disk admission estimates twice
the pinned tracked blob bytes plus the explicit caller reserve. This estimate
is not a disk quota or protection from unrelated concurrent writers. Fresh
external resource admission (including memory pressure, build slots, applicable
host policy and source ownership) remains mandatory. Worker count and
`--capacity-reason` remain advisory telemetry; no new count limit is introduced.

The mode skips `.worktreeinclude`, shared Git exclude edits and pool stamping;
`--sandbox-compatible` is refused before preparation. It does not materialize
source overlays. Necessary new-worker writes are the target lock, detached Git
registration/checkout, owner sidecar and `lease.json`, plus optional new intent
metadata when the existing `--message`/`--path` pair is supplied. All metadata is
created exclusively at its final path with no replacement, temp-rename fallback
or failure deletion. Pinned tracked `lease.json`/`lease.json.tmp` is refused.
Intent paths must be a subset of concrete admitted paths; immutable intent and
message publication happen inside the target lock, with no expansion or CLI
postprocessing after lock release. `lease.json`
may appear in raw Git status; native cleanup-status helpers already filter it.
The success receipt has `preserve_existing: true` and the ordinary isolated Go
environment. Build-directory creation and later validation remain separate.

Failures leave surviving new directories, registrations and metadata untouched
without emitting a ready receipt or environment. Git may remove its own failed
addition; no promise is made that every partial directory/registration survives.
The `preserved` failure field observes whether the target still exists, failing
toward preservation when inspection is inconclusive. No automatic cleanup, index
retry, no-checkout fallback or destructive recovery runs. Git itself can roll
back its own failed add; qualification must prove that the supported Git
version does not alter pre-existing registrations, including missing/prunable
ones. Checkout-policy qualification supports only Git 2.45.0. Explicit false
fsmonitor values are disabled; `submodule.active` is dormant only when the full
pinned tree has no gitlinks or `.gitmodules`. Filter definitions are dormant only
when every pinned path has a complete `filter: unspecified` attribute result.
No filter or hook is executed during proof. The creation hooks
`reference-transaction`, `post-index-change` and `post-checkout` must be absent
in the source and new checkout contexts; existing, unreadable or indirect hook
paths refuse. The pinned tree must not be able to materialize a hook or its
ancestor in the new checkout, including case-folded matches. With
`core.hooksPath` set, its value and native-reported hook locations must be
ASCII; an absolute location also requires an ASCII prospective target identity.
Relative ASCII hooks may use Unicode source and target roots, but Unicode
normalization aliases for configured hook identities are not qualified. Sparse
checkout, conditional includes, active or unknown attributes, unavailable
sources and query warnings are refused. Context-dependent includes are refused
before add instead of guessing the future administrative path.

With `extensions.worktreeConfig` enabled, the qualified context may have no
`config.worktree`, or a directly resolved regular readable file containing
exactly one `core.bare` record with the literal parsed value `false` and one
nonempty ASCII `core.hooksPath` value without NUL, CR or LF. Extra entries,
duplicate keys, includes and indirect sources refuse. Directly parsed records
must match the records reported at that exact Git configuration origin. When
present, Git must create a byte-identical, independent regular-file copy in the
actual new administrative directory; hardlink aliases refuse. Only that proven
child origin may map back to the admitted source origin. Every other origin,
value and source byte remains bound.

The source proof is rechecked before add. Both the original source and actual
new checkout are rechecked before status and before publication; configuration
bytes and complete tree/attribute results must still agree with admission.
Existing peers and their fsmonitor safeguards are never queried or altered.
No configuration, hook or credential rewrite is used to qualify this shape.
Safeguards are not silently disabled. Repeated proof detects observed drift;
it does not lock configuration or hook paths against unrelated writers between
checks and Git execution. An approved quiescent admission remains required.
The add explicitly sets `gc.worktreePruneExpire=never`, which requires
platform/Git-version integration qualification rather than an assumption.
Concurrent external deletion is outside the per-target lock contract; final
readback refuses a lost registration and preserves whatever evidence remains.

This capability requires independent review, scoped ownership and native
qualification before deployment. A committed launcher does not authorize
operational execution of an uncommitted patch. Bootstrap and publication must
use the repository's approved committed-tool rollout; this proposal grants no
permission to provision a worker, bypass admission, or execute a baseline.

Preservation preparation canonicalizes repository and worker roots to one
physical absolute identity before path inspection, locking, admission and Git
creation. An explicit relative `--wt-root` resolves against the repository root;
symlink aliases resolve to the same target and lock. Missing worker-root suffixes
are retained only after their existing ancestor has been resolved; unreadable or
dangling existing ancestors refuse. This closes the mismatch between process
working-directory checks and Git's repository-relative worktree target.

Preservation capacity telemetry uses only the registration census and the pure
advisory classifier. It never invokes lifecycle/cleanliness probes or produces
contraction recommendations, including above50 workers and after refusals.
Default preparation retains its existing advisory behavior.

After final registration readback, the primitive rechecks owner/lease/resource
authority before success inside the target lock. The CLI also rechecks authority
immediately before exporting a ready environment and emits a refusal if fencing
has been lost. The resource gate ends with a native lease fence after disk/source
queries. These are read-side checks, not an atomic lease reservation against
arbitrary external writers; an approved quiescent admission and normal renewal
remain required. No lifecycle query runs after the primitive publication fence.

Preservation fixtures create isolated Git repositories directly without ordinary
Prepare or mutable sweep/backend setup, and exclude system/global Git config.
Full CLI fixtures cover actual missing-lease and insufficient-disk refusals,
success above the advisory setpoint, peer-status isolation, aliases/relative
roots and publication revocation. These fixtures must be executed through the
approved bounded native qualification workflow before operational use; merely
including them in this proposal is not a passing validation receipt.

Admitted and intent paths must equal the existing fence's normalized pathname
exactly. Leading/trailing whitespace (including Unicode whitespace), path
aliases and every other fence-normalization change refuse; an internal space in
an otherwise unchanged concrete filename remains valid. Thus ownership checks
and immutable intent storage name the same source path.

The peer-status sentinel fixture includes a positive control: after the CLI
absence check, a real status probe must trigger the sentinel or the fixture
fails. A separate full CLI publication fixture uses a native Go helper-process Git proxy solely
inside the test process to CAS the actual isolated lease ref to a new holder and
generation during CLI publication's resource query. It requires the real lease
fence to refuse publication with no ready environment and verifies that the
fixture ref changed. This is real-store integration-fixture coverage when run;
mocked-gate unit fixtures do not establish that authority behavior. All of these
fixtures remain unexecuted in the task-local proposal.

The fsmonitor helper and Git proxy are named aliases of the already built Go
test executable, dispatched before testing flag parsing by fixture-only init
logic. They are hard-linked when possible, with an exclusive native binary-copy
fallback, and never invoke a shell, `go run` or a nested compiler. Helper children
have explicit deadlines. The fsmonitor writes the positive-control sentinel
natively; the proxy delegates to the original absolute Git executable and CAS
updates only the isolated fixture lease during publication's resource query.
No executable shell helpers or policy exceptions are required.

Preservation fixture subprocesses use an explicit environment allowlist rather
than inherited Git, SSH, credential, proxy, trace or loader settings. Git is
resolved to an absolute approved executable; each fixture supplies an empty
template directory, private HOME/XDG roots, disabled global/system config and
file-only transport. Only the deliberately configured native fsmonitor/proxy
configuration and isolated peer index are added explicitly. Hostile-environment
regressions cover config-write redirection and template-hook import using fresh
temporary fixtures; they remain unexecuted until source qualification.

Qualification must unset `FAK_WORKSPACE_ROOT` before invoking the launcher.
The documentation helpers then resolve the extracted candidate from their test
working directory, rather than an ambient shared checkout. Allocate an exclusive
fresh TMPDIR namespace before the launcher, with TMP/TEMP bound to the same
namespace, because committed-tree extraction may reap stale trees in os.TempDir.
These environment changes apply to fixture/qualification isolation; production
Git policy and the existing peer fsmonitor safeguard remain unchanged.
