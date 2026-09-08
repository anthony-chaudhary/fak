---
name: git-subagent-sync
description: Synchronize subagent changes to git on shared trunk across all lanes and all untracked WIP by default (or scoped via --lane/--paths). Enforces single-source invariants, safe merge convergence via `fak sync check`, `fak sync reconcile --apply`, and `fak sync push`, dual-repo synchronization (fak & fak-private), disjoint file-tree fencing via `dos arbitrate`, detached build-isolation worktrees, and atomic coordinator landing via `fak sweep --apply` or `fak commit --path`. Prevents index corruption, off-trunk drift, peer clobbering, and unverified worker self-reports. Use when synchronizing subagent changes across lanes on shared trunk.
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
6. **Safe Landing within Worker Boundaries**: Autonomous safe git sync landing (`fak sync check`, `fak sync reconcile --apply`, `fak commit --path`, `fak sync push`) is active by default working within each worker process upon test verification. Raw uncoordinated git commands (`git add -A`, raw unverified git commits/merges) remain strictly prohibited.
7. **Package-scoped verification only**: Verification executes within isolated scopes (`fak validate --mine <p>...` and targeted package tests under WSL/Linux) without running unisolated global sweeps that contaminate clean builds with peer WIP.
8. **Autonomous landing upon task completion by default**: When an agent, worker, or subagent completes any assigned task, it must follow the safe landing process itself by default working within each worker process rather than leaving uncommitted edits or waiting for interactive prompts. If coordinator context preservation or build isolation is required, the coordinator dispatches a dedicated landing/worker subagent to execute the verification and landing steps autonomously.
9. **Safe Merge & Structured Convergence**: Strictly forbid unverified raw merges, `--autostash`, or force-pushes. Check for `.git/MERGE_HEAD` (`MERGE_IN_PROGRESS`); if active, halt and wait. Integrate upstream via `fak sync check` and `fak sync reconcile --apply` (supporting `ROUTE_DISJOINT_INTEGRATE`, `ROUTE_SUPERSET_MERGE`, and `ROUTE_HOLD_DIRTY_COLLISION` with `fak wip park`).
10. **Dual-Repo Synchronization (`fak` & `fak-private`)**: When cross-repo dependencies, shared interfaces (`pkg/*`), or companion platform modules are touched, synchronize both repositories. Check dirty states, fetch remotes for both, fast-forward both (`fak sync reconcile --apply` in `fak`, `git pull --ff-only` or `fak-sync repo` in `fak-private`), run `go work sync`, and verify boundary leak safety (`python tools/scrub_public_copy.py --audit-staged`).

## Architecture & Roles

```
[Upstream Trunk: origin/main]
              │
      Coordinator Process
    ┌─────────┴─────────┐
    │  - Pre-flight checks & fak sync check
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
    │  - Autonomous safe git landing: `fak sync check`, `fak sync reconcile --apply`, `fak commit --path`, `fak sync push`
    │  - Emit structured completion receipt with commit SHA
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
- Adhere strictly to safe trunk landing: NEVER execute raw uncoordinated git commands (`git commit`, `git push`, `git checkout -b`, `git add -A`, `git reset`, or `git merge`). Instead, execute landing through the guarded fak verbs: `fak sync check`, `fak sync reconcile --apply`, `fak commit --path`, and `fak sync push` unprompted by default upon test verification.
- Return a structured completion receipt specifying modified files, newly created files, test outputs, execution status, and the landed origin commit SHA.

## Step-by-Step Execution Protocol

### Phase 1: Pre-Flight, Full-Tree Census & Lane Discovery (All Lanes by Default)

1. **Refresh upstream tracking and verify safe merge state**:
   ```bash
   fak sync check --fetch
   fak sync reconcile --apply
   ```
   Inspect for mid-flight merges:
   ```bash
   test ! -f .git/MERGE_HEAD || exit 4 # MERGE_IN_PROGRESS
   ```
   If a merge is active, pause and allow the owning process to resolve it. Never abort, reset, or finish a peer's merge.

   **Dual-Repo Pre-flight (when cross-repo dependencies, shared interfaces `pkg/*`, or companion modules are involved)**:
   When coordinating changes across both `fak` and `fak-private`:
   - Inspect dirty state in both checkouts (`git status --porcelain`).
   - Fetch remotes and lock refs for both repositories (`fak sync check --fetch` in `fak`, `git fetch origin` / `fak-sync repo` in `fak-private`).
   - Inspect `.git/MERGE_HEAD` in both checkouts; if either repository is in a merge state, unstage local paths and wait.
   - Run `go work sync` to verify multi-module workspace graph alignment across both trees.

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

3. **Safe divergence handling and merge routing**:
   Before executing landing, check if upstream advanced during arbitration or verification:
   - Strictly avoid unverified raw git merges, `--autostash`, or force-pushes.
    - Route divergence through structured reconciliation:
      ```bash
      fak sync check
      fak sync reconcile --apply
      ```
      Execute the evaluated safe route:
      - `ROUTE_APPLY`: clean fast-forward convergence via `fak sync reconcile --apply` (or `fak sync apply`).
     - `ROUTE_DISJOINT_INTEGRATE`: disjoint commit file-trees merged cleanly.
     - `ROUTE_SUPERSET_MERGE`: textless `-s ours` verified convergence.
     - `ROUTE_HOLD_DIRTY_COLLISION`: if uncommitted local paths collide, park them safely via `fak wip park` (or `--suspend-paths`), converge, and reapply.
   - Re-verify prospective package tests with `fak validate --mine` after convergence before landing the commit.

4. **Execute atomic trunk landing**:
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

5. **Verify commit claim shape & leaf stamp immediately**:
   ```bash
   dos commit-audit HEAD
   dos verify fak <leaf>
   ```
   Confirm verdict is `OK` (`diff-witnessed`) and `shipped: true`.

6. **Reap worker worktree if used**:
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

3. **Push to trunk & dual-repo coordination**:
   If `--push` was requested or trunk is green, push landed commits safely.

   **Single-repo push (`fak` only)**:
   ```bash
   fak sync push
   ```
   `fak sync push` automatically retries non-fast-forward races when a peer lands between fetch and push while HEAD already contains origin, and halts safely on genuine divergence.

   **Dual-repo push (`fak` & `fak-private`)**:
   When cross-repo dependencies, shared interfaces (`pkg/*`), or companion platform modules were modified:
   - Audit the staged/committed public delta for private leak needles before push:
     ```bash
     python tools/scrub_public_copy.py --audit-staged
     ```
     Ensure zero leak needles (exit code 0).
   - Push companion `fak-private` changes first or synchronously using its sanctioned workflow (`fak-sync push`).
   - Push `fak` commits via `fak sync push`.
   - Confirm both repositories have reached their intended synchronized tip.

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
| `MERGE_IN_PROGRESS` | 4 | A git merge is active (`.git/MERGE_HEAD` exists). | Strictly forbid unverified raw merges, `--autostash`, or force-pushes. If the merge is owned by a peer, do not abort or finish it; unstage local paths and wait. If resolving owned convergence, integrate via `fak sync reconcile --apply` (or `fak sync apply` `--ff-only`) or `fak sync reconcile` (`ROUTE_DISJOINT_INTEGRATE` / `ROUTE_SUPERSET_MERGE` / `ROUTE_HOLD_DIRTY_COLLISION` with `fak wip park`). |
| `PATHSPEC_RACE` | 1 | Staged commit contained files outside the requested explicit pathspec. | Commit was held locally unpushed. Inspect `git show --stat HEAD`, verify unrequested paths, unstage them, and never force-push. |
| `PUSH_REJECTED` | 1 | Remote rejected push (non-fast-forward conflict on origin). | Never force-push or rebase with `--autostash`. Run `fak sync check`, then `fak sync reconcile --apply` (or `fak sync apply`), re-verify affected packages, and push with `fak sync push`. If cross-repo work was landed, synchronize and push both `fak` and `fak-private`. |
| `STALE_BASE_DELETION` | 4 | Working copy predates upstream modifications and would silently overwrite peer lines. | Refresh modified paths from `origin/main`, reapply the subagent delta onto the fresh baseline, re-run tests, and re-commit. |
| `STALE_UNTRACKED` | 4 | Path is untracked locally but already exists on origin/main. | Local HEAD is behind remote. Run `fak sync check --fetch` and compare via `git show origin/main:<path>` before landing. |
| `COLLISION_RISK` | 4 | Requested write tree overlaps with a concurrent active lease holder. | Run `dos arbitrate --workspace .` to inspect conflicting holders. Wait for lease expiration or select an alternate disjoint lane from `free_clusters`. If dirty paths collide during convergence, park conflicting WIP via `fak wip park` (`ROUTE_HOLD_DIRTY_COLLISION`). |
| `DUAL_REPO_SKEW` | 4 | Interface mismatch or un-synchronized commits between `fak` and companion `fak-private` across shared packages (`pkg/*`). | Synchronize remotes for both repos, run `go work sync`, re-verify interface compatibility across workspaces, and align commits. |
| `PUBLIC_LEAK_DETECTED` | 1 | Staged or committed public copy contains private needles or credentials. | Run `python tools/scrub_public_copy.py --audit-staged`, scrub offending private artifacts, and never push until leak check exits 0. |
| `OFF_TRUNK` | 4 | Current working tree is detached or checked out on a branch other than `main`. | Return to trunk via `fak sync apply` (or `git checkout main`). Sanctioned worker worktrees (`fak worktree worker`) are exempt when landing via CAS. |
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

6. **Boundary Leak Safety Audit (Dual-Repo)**:
   ```bash
   python tools/scrub_public_copy.py --audit-staged
   ```
   *Exit criteria*: Exit code 0; zero private leak needles detected.

7. **Residual Review Audit**:
   ```bash
   dos review origin/main..HEAD
   ```
   *Exit criteria*: `has_residual: false`.

8. **Clean Working Tree Audit**:
   ```bash
   git status --porcelain
   fak sweep --json
   ```
   *Exit criteria*: All intended lanes and untracked WIP files committed cleanly; zero unowned residual debris.

9. **Worktree Cleanup Verification**:
   ```bash
   fak worktree worker list --json
   ```
   *Exit criteria*: Allocated worktree is removed or cleanly marked `CLEANUP_READY`.
