---
name: git-subagent-sync
description: Synchronize subagent changes to git on shared trunk across all lanes and all untracked WIP by default (or scoped via --lane/--paths). Enforces single-source-of-truth invariants, disjoint file-tree fencing via `dos arbitrate`, detached build-isolation worktrees, and atomic coordinator landing via `fak sweep --apply` or `fak commit --path`. Prevents index corruption, off-trunk drift, peer clobbering, and unverified worker self-reports.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--lane <lane>] [--paths <p>...] [--worker-dir <dir>] [--push] (defaults to all lanes and all untracked WIP)"
metadata:
  opencode: agent-permission
---

# /git-subagent-sync — Synchronize Subagent Changes to Git on Shared Trunk

Coordinated, conflict-free synchronization protocol for parallel subagents and isolated workers operating on the shared trunk (`main`). By default, discovers, arbitrates, validates, and synchronizes **all lanes and all untracked WIP** across the repository (or narrows to an explicit `--lane` or `--paths` when provided). Prevents index lock collisions, uncommitted file cross-talk, branch drift, and peer clobbering by strictly enforcing coordinator-mediated git mutations, disjoint filesystem leases via `dos arbitrate`, detached build-isolation worktrees, and non-forgeable witness gates.

## Core Invariants

1. **`main-is-single-source`**: All committed work lands directly on trunk (`main`). Feature branches, side branches, and divergent branch worktrees are prohibited. The repository trunk guard rejects off-trunk transactions with `OFF_TRUNK`.
2. **All lanes & untracked WIP by default**: In the absence of an explicit `--lane` or `--paths` constraint, synchronization automatically inventories the complete working tree, discovers all untracked WIP files (`??`), maps untracked files to their canonical lanes, arbitrates leases per lane, and drives each lane through isolated validation, package tests, atomic landing, and DOS witness gates.
3. **Untracked WIP preservation**: Untracked source files (`??`) are first-class candidate WIP. They are mapped to their respective lane via `dos.toml` and committed as part of their package; they are never wiped, abandoned, or hidden behind wildcard `.gitignore` rules.
4. **Explicit paths only (No blanket staging)**: Git staging and landing operations require explicit pathspecs. "All lanes" does NOT mean `git add -A` or a blanket commit; it means iterating over all dirty lanes and landing each lane group with explicit pathspecs and its bindable `(fak <lane>)` stamp.
5. **Detached worktrees only**: Concurrent filesystem isolation utilizes detached HEAD worktrees pinned at an explicit trunk SHA (`git worktree add --detach <path> <sha>` managed via `fak worktree worker prepare`). Never attach branch worktrees.
6. **Coordinator owns git**: Workers and subagents never execute mutating git commands (`commit`, `push`, `merge`, `rebase`, `checkout`, `add`, `reset`). The coordinator process exclusively manages index operations, CAS trunk landing, and push interactions.
7. **Package-scoped verification only**: Verification executes within isolated scopes (`fak validate --mine <p>...` and targeted package tests under WSL/Linux) without running unisolated global sweeps that contaminate clean builds with peer WIP.

## Architecture & Roles

```
[Upstream Trunk: origin/main]
              │
      Coordinator Process
    ┌─────────┴─────────┐
    │  - Pre-flight checks & git fetch
    │  - Full working-tree inventory (modified + all untracked WIP)
    │  - Lane classification via `fak sweep --json` & `dos.toml`
    │  - Multi-lane arbitration loop: `dos arbitrate` per lane
    │  - Allocate detached worktree: `fak worktree worker prepare`
    │  - Ingest receipts & verify independently (`fak validate`, `go test`)
    │  - Atomic lane landing: `fak sweep --apply` / `fak commit --path`
    │  - Commit witness per lane: `dos commit-audit`, `dos verify`
    │  - Handle no-lane residuals (goals, root configs)
    │  - Final review audit: `dos review origin/main..HEAD`
    │  - Cleanup: `fak worktree worker reap` & lease release
    └─────────┬─────────┘
              │ (Fenced Worktree / Write Boundary)
      Subagent / Worker
    ┌─────────┴─────────┐
    │  - Bounded implementation in allocated file tree
    │  - Private compilation cache: GOCACHE, GOTMPDIR
    │  - Local package unit tests
    │  - ZERO git mutations (read-only git inspection only)
    │  - Emit structured completion receipt to coordinator
    └───────────────────┘
```

### Coordinator Responsibilities
- Discover layout and active repository state via `dos doctor`, `git status --porcelain`, and `fak sweep --json`.
- By default, inventory **all lanes and all untracked WIP**, grouping untracked files into their canonical lanes.
- Acquire exclusive lane leases from `dos arbitrate` before initiating subagent dispatch or landing.
- Provision detached worktree isolation via `fak worktree worker prepare` with dedicated build cache roots (`GOCACHE`, `GOTMPDIR`, `DISPATCH_WORKSPACE`).
- Ingest worker completion receipts without trusting narrative assertions ("tests pass", "complete").
- Independently verify candidate deltas per lane using isolated compilation checks (`fak validate --mine`) and targeted package test suites.
- Execute atomic landing on trunk per lane using `fak sweep --apply --lane <lane>`, `fak worktree worker land`, or locked explicit-path staging (`fak commit --path`).
- Witness each landed commit via `dos commit-audit HEAD`, `dos verify fak <leaf>`, and `dos review`.
- Reap detached worktrees and release lane leases upon completion.

### Worker / Subagent Responsibilities
- Execute modifications strictly inside the declared file tree or assigned detached worktree directory.
- Use isolated compiler directories defined in the environment.
- Run local tests inside the private worktree or target package.
- Adhere strictly to the git prohibition: NEVER execute `git commit`, `git push`, `git checkout -b`, `git add`, `git reset`, or `git merge`.
- Return a structured completion receipt specifying modified files, newly created files, test outputs, and execution status.

## Step-by-Step Execution Protocol

### Phase 1: Pre-Flight, Full-Tree Census & Lane Discovery (All Lanes by Default)

1. **Refresh upstream tracking without churn**:
   ```bash
   git fetch origin main
   ```
   Inspect for mid-flight merges:
   ```bash
   test ! -f .git/MERGE_HEAD || exit 4 # MERGE_IN_PROGRESS
   ```
   If a merge is active, pause and allow the owning process to resolve it.

2. **Verify index lock availability**:
   Verify `.git/index.lock` is absent. If present, inspect lock staleness rather than forcing removal.

3. **Full Working-Tree Inventory (Tracked + All Untracked WIP)**:
   Capture the full dirty census across all lanes:
   ```bash
   git status --porcelain
   fak sweep --json
   ```
   Parse all modified, deleted, and untracked (`??`) files:
   - **Package source & tests**: map each untracked file to its canonical lane according to `dos.toml [lanes]` (e.g. `cmd/fak/arms.go` -> `cmd`, `internal/compute/numa_topology.go` -> `compute`, `internal/gateway/arms.go` -> `gateway`).
   - **Goal specifications**: map `goals/**` to the `goal` lane or `internal/goalregistry`.
   - **Documentation & notes**: map `docs/**` to the `docs` lane.
   - **Tooling & test scripts**: map `tools/**` to the `tools` lane.
   - **Root residuals**: group root-level files (`Makefile`, `campaign-baseline.json`) for explicit leaf stamping.

4. **Determine Scope (All Lanes vs Explicit Lane)**:
   - If `--lane <lane>` or `--paths <p>...` is explicitly passed, narrow execution to that specific lane or pathset.
   - **Default**: Select **all discovered lanes** with dirty tracked or untracked WIP files for sequential synchronization.

### Phase 2: Multi-Lane Arbitration & Ordering

1. **Order Discovered Lanes**:
   Sort discovered lanes to land dependencies before dependents (e.g. core libraries `internal/model`, `internal/compute`, `internal/gateway` before entrypoints `cmd/fak`, and docs/tooling after code).

2. **Acquire Exclusive Lane Lease**:
   For each target lane in the queue, query the lane arbiter before modifying or committing:
   ```bash
   dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive --tree <lane_paths>...
   ```
   Ensure outcome is `acquire`. If outcome is `refuse`, proceed to the next disjoint lane or back off if blocked by transient contention (`LOCK_BUSY`).

3. **Provision Detached Worktree (When Dispatching Isolated Subagents)**:
   When coordinating isolated workers, allocate a detached worktree directory pinned to trunk `HEAD`:
   ```bash
   fak worktree worker prepare --lane <lane> --key <task-or-issue-id>
   ```
   Capture `worker_worktree` and `base_sha` from the JSON payload.

### Phase 3: Per-Lane Receipt Ingestion & Independent Verification

For each lane being synchronized:

1. **Audit touched files against arbitration fence**:
   Extract modified and untracked paths for this lane. Assert every path belongs to the arbitrated file-tree lease. Reject unmapped paths outside the lease (`COLLISION_RISK`).

2. **Validate prospective delta in isolation**:
   Run isolated compilation, formatting, and vetting across the lane's modified AND untracked paths:
   ```bash
   fak validate --mine <lane_paths>...
   ```
   Confirm build, vet, and formatting pass while masking peer WIP.

3. **Execute targeted test witness on-device**:
   Run package tests for the target lane (WSL/Linux on Windows):
   ```bash
   go test -v ./internal/<lane>/...
   go vet ./internal/<lane>/...
   ```
   Proceed only when execution yields `CLAIM_TEST_GREEN`. Never accept worker-narrated test success without coordinator verification.

### Phase 4: Per-Lane Atomic Trunk Landing

For each verified lane:

1. **Prepare Conventional Commit subject**:
   Format subject line with DCO sign-off (`-s`) and bindable lane trailer `(fak <leaf>)`:
   ```
   <type>(<scope>): <concise summary> (fak <leaf>)
   ```

2. **Lint commit message and paths**:
   ```bash
   fak commit --preview -m "<type>(<scope>): <summary> (fak <leaf>)" --path <lane_paths>...
   ```

3. **Execute atomic trunk landing**:
   **Option A — In-Tree Lane Sweep Landing (`fak sweep --apply`)**:
   ```bash
   fak sweep --apply --lane <lane> -m "<type>(<scope>): <summary> (fak <leaf>)" [--push]
   ```
   **Option B — In-Tree Locked Commit (`fak commit --path`)**:
   ```bash
   fak commit \
     --path <p1> --path <p2> ... \
     -m "<type>(<scope>): <summary> (fak <leaf>)" \
     [--push]
   ```
   **Option C — Detached Worktree Landing (`fak worktree worker land`)**:
   ```bash
   fak worktree worker land \
     --worktree "<worker_worktree>" \
     --paths <lane_paths>... \
     --verify go-build \
     --msg-file "<prepared-msg-file>"
   ```

4. **Verify commit claim shape & leaf stamp immediately**:
   ```bash
   dos commit-audit HEAD
   dos verify fak <leaf>
   ```
   Confirm verdict is `OK` (`diff-witnessed`) and `shipped: true`.

5. **Reap worker worktree if used**:
   ```bash
   fak worktree worker reap --worktree "<worker_worktree>" --superseded-by HEAD
   ```

### Phase 5: No-Lane Residuals, Push & Multi-Lane Summary Receipt

1. **Synchronize No-Lane Residuals**:
   For any remaining untracked files with no auto-inferred lane:
   - `goals/**`: commit with stamp `(fak goal)` or record in goal lifecycle.
   - Root configs / tooling (`Makefile`, `campaign-baseline.json`): commit with explicit paths and matching leaf (e.g. `(fak build)`, `(fak devindex)`).
   - Proven disposable junk: clean with `fak sweep --clean-junk` or `fak tree-doctor --sweep-scratch`.

2. **Confirm zero residual review debt**:
   ```bash
   dos review origin/main..HEAD
   ```
   Confirm `has_residual` is `false`.

3. **Push to trunk**:
   If `--push` was requested or trunk is green, push landed commits:
   ```bash
   git push origin main
   ```

4. **Emit structured sync receipt**:
   ```json
   {
     "schema": "fak-subagent-sync/1",
     "mode": "all-lanes",
     "lanes_synced": ["agent", "compute", "gateway", "cmd"],
     "untracked_wip_cleared": 43,
     "commits": [
       {
         "lane": "agent",
         "commit_sha": "<sha1>",
         "paths": ["internal/agent/blackboard_bench_test.go", "internal/agent/subagent_synthesis.go"],
         "witness": { "test": "CLAIM_TEST_GREEN", "audit": "diff-witnessed", "stamp": "shipped" }
       },
       {
         "lane": "compute",
         "commit_sha": "<sha2>",
         "paths": ["internal/compute/decode_numapool.go", "internal/compute/numa_topology.go"],
         "witness": { "test": "CLAIM_TEST_GREEN", "audit": "diff-witnessed", "stamp": "shipped" }
       }
     ],
     "status": "ALL_LANES_COMMITTED"
   }
   ```

## Refusal & Recovery Matrix

| Refusal Token | Exit Code | Trigger Condition | Exact Recovery Action |
|---|---|---|---|
| `LOCK_BUSY` | 3 | Another process holds `.git/index.lock` or the advisory commit lock. | Wait with backoff (e.g. 500ms, 1s, 2s). In multi-lane mode, proceed to next disjoint lane or wait for lock release; do not remove active locks manually. |
| `WRITER_LEASE_HELD` | 3 | Worktree writer lease held by an active sync-apply window (#4240). | Transient contention. Back off and poll until writer lease clears, then retry landing. |
| `MERGE_IN_PROGRESS` | 4 | A git merge is active (`.git/MERGE_HEAD` exists). | Do not attempt path-scoped staging. Wait for merge completion or unstage local paths and halt. |
| `PATHSPEC_RACE` | 1 | Staged commit contained files outside the requested explicit pathspec. | Commit was held locally unpushed. Inspect `git show --stat HEAD`, verify unrequested paths, unstage them, and never force-push. |
| `PUSH_REJECTED` | 1 | Remote rejected push (non-fast-forward conflict on origin). | Run `git fetch origin main`, reconcile in place via trunk pull without `--autostash`, re-verify, and push without force. |
| `STALE_BASE_DELETION` | 4 | Working copy predates upstream modifications and would silently overwrite peer lines. | Refresh modified paths from `origin/main`, reapply the subagent delta onto the fresh baseline, re-run tests, and re-commit. |
| `STALE_UNTRACKED` | 4 | Path is untracked locally but already exists on origin/main. | Local HEAD is behind remote. Run `git fetch origin main` and compare via `git show origin/main:<path>` before landing. |
| `COLLISION_RISK` | 4 | Requested write tree overlaps with a concurrent active lease holder. | Run `dos arbitrate --workspace .` to inspect conflicting holders. Wait for lease expiration or select an alternate disjoint lane from `free_clusters`. |
| `OFF_TRUNK` | 4 | Current working tree is detached or checked out on a branch other than `main`. | Return to trunk: `git checkout main`. Sanctioned worker worktrees (`fak worktree worker`) are exempt when landing via CAS. |
| `CORE_SELF_MODIFY` | 4 | Attempted modification of frozen/core-locked files (e.g. `internal/abi/**`, policy anchors). | Abort unauthorized modification. If permitted by maintenance mandate, provide explicit `--core-lock-maintenance-witness <claim>`. |

## Verification and Witness Gate

Every execution of `git-subagent-sync` must pass the following checkable verification commands before declaring synchronization complete:

1. **Prospective Build & Vet Isolation**:
   ```bash
   fak validate --mine <paths>...
   ```
   *Exit criteria*: Exit code 0; zero compile errors, gofmt clean, vet clean.

2. **Targeted Package Test Execution**:
   ```bash
   go test -v ./internal/<lane>/...
   go vet ./internal/<lane>/...
   ```
   *Exit criteria*: Exit code 0; status is `CLAIM_TEST_GREEN`.

3. **Commit Subject & Leaf Lint**:
   ```bash
   fak commit --preview -m "<type>(<scope>): <summary> (fak <leaf>)" --path <paths>...
   ```
   *Exit criteria*: Exit code 0; Conventional Commits syntax valid, DCO present, trailer matches active lane.

4. **Diff Witness Verification**:
   ```bash
   dos commit-audit HEAD
   ```
   *Exit criteria*: Verdict `OK`, witness category `diff-witnessed`.

5. **Stamp Recognition**:
   ```bash
   dos verify fak <leaf>
   ```
   *Exit criteria*: `shipped: true` from recognized repository stamp grammar.

6. **Residual Review Audit**:
   ```bash
   dos review origin/main..HEAD
   ```
   *Exit criteria*: `has_residual: false`.

7. **Clean Working Tree Audit**:
   ```bash
   git status --porcelain
   fak sweep --json
   ```
   *Exit criteria*: All intended lanes and untracked WIP files committed cleanly; zero unowned residual debris.


## Refusal & Recovery Matrix

| Refusal Token | Exit Code | Trigger Condition | Exact Recovery Action |
|---|---|---|---|
| `LOCK_BUSY` | 3 | Another process holds `.git/index.lock` or the advisory commit lock. | Wait with backoff (e.g. 500ms, 1s, 2s). If lock persists, verify holder PID; do not remove active locks manually. |
| `WRITER_LEASE_HELD` | 3 | Worktree writer lease held by an active sync-apply window (#4240). | Transient contention. Back off and poll until writer lease clears, then retry landing. |
| `MERGE_IN_PROGRESS` | 4 | A git merge is active (`.git/MERGE_HEAD` exists). | Do not attempt path-scoped staging. Wait for merge completion or unstage local paths and halt. |
| `PATHSPEC_RACE` | 1 | Staged commit contained files outside the requested explicit pathspec. | Commit was held locally unpushed. Inspect `git show --stat HEAD`, verify unrequested paths, unstage them, and never force-push. |
| `PUSH_REJECTED` | 1 | Remote rejected push (non-fast-forward conflict on origin). | Run `git fetch origin main`, reconcile in place via trunk pull without `--autostash`, re-verify, and push without force. |
| `STALE_BASE_DELETION` | 4 | Working copy predates upstream modifications and would silently overwrite peer lines. | Refresh modified paths from `origin/main`, reapply the subagent delta onto the fresh baseline, re-run tests, and re-commit. |
| `COLLISION_RISK` | 4 | Requested write tree overlaps with a concurrent active lease holder. | Run `dos arbitrate --workspace .` to inspect conflicting holders. Wait for lease expiration or select an alternate disjoint lane from `free_clusters`. |
| `OFF_TRUNK` | 4 | Current working tree is detached or checked out on a branch other than `main`. | Return to trunk: `git checkout main`. Sanctioned worker worktrees (`fak worktree worker`) are exempt when landing via CAS. |
| `CORE_SELF_MODIFY` | 4 | Attempted modification of frozen/core-locked files (e.g. `internal/abi/**`, policy anchors). | Abort unauthorized modification. If permitted by maintenance mandate, provide explicit `--core-lock-maintenance-witness <claim>`. |

## Verification and Witness Gate

Every execution of `git-subagent-sync` must pass the following checkable verification commands before declaring synchronization complete:

1. **Prospective Build & Vet Isolation**:
   ```bash
   fak validate --mine <paths>...
   ```
   *Exit criteria*: Exit code 0; zero compile errors, gofmt clean, vet clean.

2. **Targeted Package Test Execution**:
   ```bash
   go test -v ./internal/<lane>/...
   go vet ./internal/<lane>/...
   ```
   *Exit criteria*: Exit code 0; status is `CLAIM_TEST_GREEN`.

3. **Commit Subject & Leaf Lint**:
   ```bash
   fak commit --preview -m "<type>(<scope>): <summary> (fak <leaf>)" --path <paths>...
   ```
   *Exit criteria*: Exit code 0; Conventional Commits syntax valid, DCO present, trailer matches active lane.

4. **Diff Witness Verification**:
   ```bash
   dos commit-audit HEAD
   ```
   *Exit criteria*: Verdict `OK`, witness category `diff-witnessed`.

5. **Stamp Recognition**:
   ```bash
   dos verify fak <leaf>
   ```
   *Exit criteria*: `shipped: true` from recognized repository stamp grammar.

6. **Residual Review Audit**:
   ```bash
   dos review origin/main..HEAD
   ```
   *Exit criteria*: `has_residual: false`.

7. **Worktree Cleanup Verification**:
   ```bash
   fak worktree worker list --json
   ```
   *Exit criteria*: Allocated worktree is removed or cleanly marked `CLEANUP_READY`.
