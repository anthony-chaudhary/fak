# Dual-Repo Workspace Topology and Safe Synchronization Mechanics

**Date:** 2026-09-03  
**Status:** Architectural Exploration & Operational Guidance  
**Scope:** Multi-repository development, workspace directory layouts, and safe multi-remote synchronization across the public engine and companion repositories.  
**Related Documents:** `docs/dev-process-private-boundary.md`, `ARCHITECTURE.md`, `CLAUDE.md`.

---

## 1. Context & The Need for Dual-Repo Synchronization

As `fak` establishes its boundary model (separating the public open-core engine runtime from private companion extensions, autonomous factory tools, and proprietary serving layers), developers and autonomous agents frequently work across both repositories concurrently.

In this topology:
- **The Public Core (`fak`)** owns the open-source kernel runtime (`cmd/fak`, `internal/engine`, `internal/model`, `internal/ctxmmu`), the frozen ABI, and exported SDK packages (`pkg/abi`, `pkg/scorecard`, `pkg/fakclient`).
- **The Companion Repository (`fak-private`)** owns deployment orchestration, private autonomous queue dispatchers, scorecards, and lab hardware bridges, resolving the public engine via Go workspace tooling (`go.work`).

Because development occurs rapidly across both domains, an active workspace quickly suffers from **skew** if repositories are pulled independently:
1. **Public Skew:** Pulling upstream changes in public `fak` without pulling `fak-private` may leave private factory and dispatch tools out-of-sync with updated ticket queues or platform scripts.
2. **Private Skew:** Pulling changes in `fak-private` without pulling `fak` causes Go workspace compilation breaks whenever private packages consume newly updated or modified `pkg/*` interfaces from the public engine.
3. **Lease/Ref Stagnation:** Autonomous worker coordination depends on git ref namespaces (`refs/fak/locks/*`). Coordinated sweeps require fetching updated refs across active trees.

Hence, operational sessions require a disciplined, safe process to **pull and synchronize both repositories simultaneously**.

---

## 2. Directory Layout Analysis: Sibling Folders vs. Container Folder (`fak-all`)

A recurring architectural question is whether the two repositories should continue living as sibling folders or be grouped into a single container folder (e.g., `fak-all/`):

### Topology A: Sibling Directories (Current Baseline)
```
work/
├── fak/             (git checkout: public core)
└── fak-private/     (git checkout: companion repo)
```
- **Strengths:**
  - Zero disruption to existing tooling, paths, and guard configurations.
  - Native integration with `repo_guard.py` (`private_companion_roots(ws)` automatically identifies `fak-private` as `<ws>-private`).
  - Shallow path depth on Windows environments where path length and filesystem watcher limits matter.
- **Limitations:**
  - Sits at the same hierarchy level as other unrelated directories in `work/`.
  - Manual coordination required when opening workspaces in editors or scripting multi-repo operations.

### Topology B: Dedicated Container Directory (`fak-all/`)
```
work/
└── fak-all/         (non-versioned workspace container directory)
    ├── fak/         (git checkout: public core)
    └── fak-private/ (git checkout: companion repo)
```
- **Strengths:**
  - Clean namespace encapsulation: encapsulates the entire `fak` ecosystem into a single top-level folder.
  - Sibling relative paths remain intact: relative references between the two (`../fak` from `fak-private`, or `../fak-private` from `fak`) continue to work transparently.
  - Enables container-level meta-configurations (such as a multi-root IDE workspace configuration `fak-all.code-workspace` or an outer `.editorconfig`) without checking them into either repository.
- **Risks & Hazards (Safety Guardrails Required):**
  1. **NEVER Initialize a Git Repository at the Container Level:**
     - The container directory `fak-all/` must **NEVER** be a single git repository.
     - Creating a git repository at the `fak-all/` root would blur the public/private boundary, risking catastrophic leakage of private code, credentials, and notes into public remotes.
     - Both `fak` and `fak-private` must remain completely independent git checkouts with separate `.git` trees and remotes.
  2. **Do NOT Use Git Submodules:**
     - Public `fak` cannot have `fak-private` as a git submodule because public users cannot clone or authenticate to private repositories, and submodule pointers leak private URLs.
     - Submodules create detached-HEAD friction and ref synchronization overhead in fast-moving autonomous agent loops.
  3. **Context MMU & Tool Admission Boundaries:**
     - If an agent or harness is launched with `workspace_root` set to `fak-all/`, tool permissions and guard layers (`repo_guard.py`) lose their unambiguous anchor.
     - In `fak`, `private_companion_roots(ws)` strips `-private` and verifies that writes only target the authorized companion. If the workspace root is set to `fak-all/`, guard checks fail to match the repository name unless explicitly adapted.
     - Agents operating in public `fak` must maintain their execution root in `fak/` to prevent cross-contamination of unscrubbed private content into public patches.
  4. **Workspace Memory Keys:**
     - Agent harnesses that key project memory off the absolute workspace path (e.g., hash keys like `C--work-fak`) would resolve a new project key (e.g., `C--work-fak-all-fak`), requiring explicit migration of local memory stores.

---

## 3. Protocol for Safe Dual-Repo Synchronization

To safely pull and maintain synchronization across both trees (whether arranged as sibling directories or within a container folder), any synchronization workflow or automated command must adhere to the following invariant sequence:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SAFE DUAL-REPO SYNC SEQUENCE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. PRE-CHECK DIRTY STATE                                                    │
│    - Inspect `git status --porcelain` on both `fak` and `fak-private`.      │
│    - If unstaged modifications exist, refuse or explicitly auto-stash:      │
│      `git stash push -m "dual-sync-autostash-<timestamp>"`                  │
│                                                                             │
│ 2. FETCH ALL REMOTES & REFS                                                 │
│    - Run `git fetch --all --tags --prune` in `fak`.                         │
│    - Run `git fetch --all --tags --prune` in `fak-private`.                 │
│    - Synchronize coordination refs (`refs/fak/locks/*`).                    │
│                                                                             │
│ 3. FAST-FORWARD-ONLY INTEGRATION                                            │
│    - Execute `git pull --ff-only origin <branch>` in `fak`.                 │
│    - Execute `git pull --ff-only origin <branch>` in `fak-private`.         │
│    - Fast-forward guarantees that local diverges never trigger accidental   │
│      merge commits or conflict markers across trees.                        │
│                                                                             │
│ 4. GO WORKSPACE RECONCILIATION                                              │
│    - Run `go work sync` to align module graphs across both modules.         │
│    - Verify build readiness: `go test -run=^$ ./...` or `fak arch-check`.   │
│                                                                             │
│ 5. BOUNDARY & LEAK AUDIT                                                    │
│    - Run staged scrub audit (`tools/scrub_public_copy.py --audit-staged`).  │
│    - Ensure zero private needles or out-of-tree artifacts entered `fak/`.   │
│                                                                             │
│ 6. RESTORE WORKING STATE                                                    │
│    - If changes were auto-stashed in Step 1, cleanly pop/re-apply:          │
│      `git stash pop` (verifying zero conflict artifacts).                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Goal Artifact Synchronization (`fak goal sync`, `fak-sync goal`)

In addition to git tree synchronization, autonomous agent sessions maintain execution continuity across restarts and feed factory dispatch by synchronizing goal artifacts (`goals/GOAL-<slug>.md`, `goals/subagents/*.md`, `.fak/goal-registry.json`, and `.fak/goal-park/`) to the companion store (`fak-private/goals/fak/`):

- **Push (`fak goal sync push` / `fak goal sync push --commit`)**: Copies active and achieved goal specifications, child sub-goals, canonical registry entries, and park records from the working tree into `fak-private/goals/fak/`. Passing `--commit` commits the synced artifacts in the target repository.
- **Pull (`fak goal sync pull`)**: Restores goal artifacts from `fak-private/goals/fak/` into the working tree, skipping newer local files unless `--force` is specified.
- **Status (`fak goal sync status`)**: Evaluates hash-level synchronization status across all goal artifacts.
- **Lifecycle Integration (`fak-sync goal`)**: Functions as the unified goal sync verb within the safe dual-repo sync lifecycle, ensuring uncommitted loop state and intent ledgers survive host crashes and agent handoffs.

---

## 4. Cross-Repository Context & Agent Memory Synchronization

When autonomous agents or developers work alternatively between `fak` (the public engine) and `fak-private` (the private factory/companion), managing **agent context and memory** presents a distinct synchronization challenge:
- How does an agent working on private dispatch or cluster tooling know what changed in public `fak`?
- How do architectural findings or bug fixes discovered in one tree get communicated to the other?
- How do we prevent high-taint private context (credentials, customer data, proprietary serving algorithms) from polluting public agent prompts or commits?

### 4.1 The Asymmetric Information Flow Model (Lattice Flow)
Context cannot flow symmetrically between the two trees. It follows a strict asymmetric flow:

```
┌─────────────────────────────────────────────────────────────┐
│                    PUBLIC REPO (fak)                        │
│                 "Open Engine & Public Memory"               │
│  - Public ABI changes, bug fixes, engine benchmarks         │
│  - Public agent memory: agent-memory/fak/                   │
│  - Public goal artifacts: goals/GOAL-*.md, goal registry    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ 1. DOWNLINK: Unrestricted Ingestion
                               │    (Private agents freely read & inherit
                               │     public engine context)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 PRIVATE REPO (fak-private)                  │
│                "Factory & Proprietary Context"              │
│  - Autonomous dispatch queues, scorecards, hardware bridges │
│  - Durable goal store: goals/fak/                           │
│  - Private agent memory: agent-memory/fak-private/          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ 2. UPLINK: Scrubbed Distillation ONLY
                               │    (Strict fail-closed redaction gate:
                               │     issue_scrub.py / scrub_public_copy.py)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    PUBLIC REPO (fak)                        │
│  - Scrubbed issue descriptions & sanitized task contracts   │
└─────────────────────────────────────────────────────────────┘
```

1. **Public $\rightarrow$ Private (Downlink / Ingestion): Free & Automatic**
   - Public code, public documentation, and public agent memories are open.
   - Agents operating in `fak-private` can freely read, index, and query public `fak/pkg/*`, `fak/docs/`, and public memory files.
2. **Private $\rightarrow$ Public (Uplink / Distillation): Redacted & Fail-Closed**
   - Private context possesses high confidentiality taint.
   - Private context **must never** be directly synced, copied, or injected into public agent sessions.
   - It may only cross into public `fak` through sanitized distillations (such as scrubbed issue tickets, public pull requests, or sanitized architecture RFCs) that have passed automated scrub gates (`tools/scrub_public_copy.py`).

### 4.2 Synchronization Across the Context Tiers

| Context Tier | Storage / Mechanism | Sync Strategy Across Repositories |
|---|---|---|
| **Durable Goal Artifacts** | `goals/` & `.fak/goal-registry.json` $\rightarrow$ `fak-private/goals/fak/` | Synced via `fak goal sync [push\|pull\|status]` or `fak-sync goal`. Pushes active goals, child sub-goals, and registry transitions to the private companion to survive local crashes and feed factory dispatch. |
| **Durable Agent Memory** | `$CLAUDE_CONFIG_DIR/projects/<key>/memory/` | Separate project keys (`C--work-fak` vs `C--work-fak-private`). Private agents ingest public memories as an overlay; public agents never load private memory. |
| **Runtime Lease Coordination** | Git references (`refs/fak/locks/*`) | Synced via `git fetch`. Both public and private workers observe lock states, contract leases, and liveness heartbeats in real time. |
| **Change Feeds & Revocations** | In-kernel `fak_changes` feed | Propagates typed mutations and invalidations so peer agents evict stale caches when shared data changes. |
| **Durable Session Recall** | Context MMU core images (`fak recall`) | Persisted core dumps in CAS storage can be mounted read-only across workspace boundaries under cryptographic digest verification. |

---

## 5. Architectural Summary & Recommendation

1. **Keep Git Repositories Decoupled:** Regardless of directory nesting, `fak` and `fak-private` must remain distinct git repositories with separate remotes, independent commit histories, and isolated commit gates.
2. **Container Directory (`fak-all/`) is Structurally Feasible:** If an operator prefers organizing both trees under `fak-all/`, this is fully compatible with existing tooling provided that:
   - `fak-all/` is a plain filesystem directory, never a git repository.
   - The relative sibling relationship (`fak` and `fak-private` at the same relative depth) is preserved.
   - Agent execution sessions anchor their working directory in the target repository (`fak/` or `fak-private/`), rather than the outer container.
3. **Mechanize the Dual-Pull & Goal Sync:** Provide dedicated helpers (e.g., via `fak dev sync`, `fak goal sync`, or `fak-sync goal`) that automate the 6-step safe pull protocol and goal artifact mirroring, preventing manual skew between the engine, companion layers, and factory queues.
4. **Enforce Asymmetric Context Synchronization:** Treat context flow as a one-way lattice: private agents inherit public facts and memories freely; public agents only receive sanitized, scrubbed task distillations.
