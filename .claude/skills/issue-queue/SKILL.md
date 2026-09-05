---
name: issue-queue
description: One repeatable, evidence-backed pass that manages, prioritizes, and resolves a bounded batch of backlog issues worst-first using specialized subagents. Inspects the issue queue via `fak issue-orchestrator` and `fak dispatch order`, partitions into ready-leaves, triage, and subdivide cohorts, arbitrates lane leases via `dos arbitrate`, dispatches parallel isolated worker subagents, independently witnesses reproduction tests with cross-validators, re-measures queue burndown with `--compare`, and commits with `(fak <leaf>)`. Use when managing the issue queue, picking the next bounded batch of issues to resolve, or advancing issue burndown.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--lane <name>] [--top N] [--subdivide] [--triage] [--from-issues <path>]  (no args = baseline + resolve top 1-3 ready issues)"
---

# /issue-queue — manage, queue, and resolve backlog issues in bounded batches

> **What this does.** Every open issue in the repository represents a pending deliverable, bug,
> or feature. When left unmanaged, the issue backlog rots into sprawling, un-prioritized,
> collision-prone work.
>
> While `/issue-orchestrator` coordinates large-scale multi-wave campaigns across many workers,
> `/issue-queue` executes the **atomic, bounded issue management and resolution pass**:
> baseline and inspect the issue queue, partition issues into actionable cohorts (ready dispatchable
> leaves, epics needing decomposition, and scope triage), pick a bounded batch of 1–3 high-priority
> units, arbitrate lane leases (`dos arbitrate`), delegate implementation to a coordinated team of
> specialized subagents (`researcher`, `worker`, `cross-validator`, `issue-auditor`), independently
> witness the reproduction test and fix, prove burndown with `--compare`, and commit by explicit path.

The shape: **baseline issue queue (`fak issue-queue --json` / `fak dispatch order`) → partition into ready/subdivide/triage cohorts → research & explore candidates (parallel `researcher` / `explore` subagents) → arbitrate leases (`dos arbitrate`) → parallel implementation wave (`worker` subagents via `task`) → adversarial verification & gap audit (`cross-validator` & `issue-auditor` subagents) → prove queue burndown (`--compare`) → commit cleanly by explicit path with `(#N)` and `(fak <lane>)` → release lease.**

---

## The Rule of Bounded Batches

Never attempt a repository-wide sweep in one pass. Enforce strict scoping:

1. **Max 1–3 issues per batch**: Focus on 1–3 closely related leaves or a single critical `P0`/`P1` issue.
2. **Shift-left proof by default**: Every bug fix must ship with a reproduction unit test that fails before the fix and passes after. Features must ship with concrete unit/contract tests. Never accept mocks or self-report claims.
3. **Queue taxonomy discipline**:
   - **Ready Leaves**: Atomic S0/S1 tasks with a single witness and bounded package tree (`internal/<lane>/`).
   - **Subdivide Queue**: Epics or multi-subsystem issues (>15 expected steps or touching 3+ packages) routed to decomposition before dispatch.
   - **Triage Queue**: Issues with missing acceptance criteria, ambiguous scope, or missing likely files routed to repair before dispatch.
   - **Cooling / Held Queue**: Issues under backoff cooldown or held by active lane leases in `.dos/lane-journal.jsonl`.
4. **Collision safety**: Pre-dispatch lane arbitration via `dos arbitrate` is mandatory before touching files.
5. **Coordinator stays clean via specialized subagents**: Substantive investigation, code edits, and verification are delegated to isolated subagents via the `task` tool (Gemini 3.8 Flash high `variant: high`). The coordinator verifies, witnesses, and commits.
6. **Non-forgeable witness**: Issues are closed only by git ancestry on trunk (`Fixes #N` with `(fak <lane>)`), verified by `dos commit-audit` and `dos verify`.

---

## Step 1 — Measure Baseline & Inventory Queue

Capture the open backlog baseline and inspect queue cohorts using allocated scratch storage (`fak tree-doctor --scratch-dir issue-queue`):

```bash
# Allocate scratch directory for queue artifacts:
fak tree-doctor --scratch-dir issue-queue

# Snapshot current issue queue state:
fak issue-queue --json > _scratch/issue-queue/baseline.json

# View top prioritized issues across lanes:
fak issue-queue --top 5
```

If querying live GitHub issues:
```bash
gh issue list --state open --json number,title,body,labels > _scratch/issue-queue/live-issues.json
fak issue-queue --from-issues _scratch/issue-queue/live-issues.json --json > _scratch/issue-queue/baseline.json
fak issue-queue --from-issues _scratch/issue-queue/live-issues.json --top 5
```

Note the headline queue metrics:
- **Total Issues Evaluated**: total backlog size.
- **Dispatchable Ready Leaves**: atomic S0/S1 issues ready for immediate dispatch.
- **Subdivide Epics**: oversized issues requiring decomposition.
- **Triage Issues**: issues requiring scope or acceptance criteria repair.
- **Held Excluded**: issues currently held by active leases in `.dos/lane-journal.jsonl`.

---

## Step 2 — Parallel Scoping & Pre-Dispatch Arbitration

### 1. Select Bounded Target Batch (1–3 Issues)
Choose 1–3 target issues based on leverage:
- **Option A: High Priority / High Centrality Hotspots (`P0`/`P1`, `Core` centrality)**:
  Issues in core leaves (`gateway`, `engine`, `model`, `compute`) with high inbound impact.
- **Option B: Quick Leaf Wins (Highest Velocity)**:
  Self-contained bug fixes or feature additions sitting at 2–4 expected steps with bounded write surfaces.

### 2. Pre-Dispatch Research & Path Mapping (Parallel Subagents)
Launch parallel specialized discovery subagents in a single turn to research candidate context and verify disjoint boundaries without polluting coordinator context:

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "researcher",
      "description": "Research #1024 gateway timeout context",
      "prompt": "Investigate issue #1024: Fix gateway streaming timeout.\nAnalyze relevant files under internal/gateway/, recent commits, and existing test patterns.\nReturn a compact receipt: (1) root cause hypothesis, (2) exact files to edit, (3) proposed reproduction test strategy."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "explore",
      "description": "Explore model KV cache paths",
      "prompt": "Inspect internal/model/ for KV cache allocation and recycling paths for issue #1035.\nVerify that its write tree is strictly confined to internal/model/ and shares no files with internal/gateway/."
    }
  }
]
```

*(If an issue involves deep architectural or concurrency invariants, launch a `deep-reason` subagent for lock ordering and invariant analysis).*

### 3. Verify Smallness & Contract
```bash
# Verify smallness / atomicity (single deliverable, witness == 1):
fak dispatch issue-smallness-lint --issue <number> --json

# Verify issue contract and structure:
fak-dev issue contract --from-issues _scratch/issue-queue/live-issues.json --json
```

### 4. Arbitrate Lane Leases
Verify lane lease availability before dispatch:
```bash
dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive
```
*(Or invoke the `dos_arbitrate` MCP tool).*

- **Outcome `acquire`**: Admitted. Proceed to dispatch.
- **Outcome `refuse`**: Lane is held by a peer. Pick another candidate from the queue or wait out the lease.

---

## Step 3 — Parallel Implementation Worker Wave

Launch all selected issues' implementation workers in parallel in a single turn using multiple `task` tool calls with `subagent_type="worker"` (using Gemini 3.8 Flash high `variant: high`). Each worker operates under strict package fences:

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Implement fix for #1024",
      "prompt": "You are an isolated worker resolving issue #1024: Fix gateway streaming timeout.\nLane: gateway\nBoundaries: Edit ONLY files inside internal/gateway/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nDeliverable:\n  1. Write an atomic reproduction test in internal/gateway/stream_timeout_test.go demonstrating the defect.\n  2. Implement the minimal fix in internal/gateway/.\n  3. Verify package tests pass.\nVerification: Execute ONLY package-scoped tests: `go test -v ./internal/gateway` and `go vet ./internal/gateway`. NEVER run `go test ./...`.\nReturn Contract: Return a compact 3-line receipt: (1) files created/modified, (2) package test results, (3) confirmation of fix."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Implement feature for #1035",
      "prompt": "You are an isolated worker resolving issue #1035: Add model KV cache recycling.\nLane: model\nBoundaries: Edit ONLY files inside internal/model/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nDeliverable:\n  1. Write an atomic reproduction test in internal/model/kv_recycle_test.go.\n  2. Implement the KV cache recycling path in internal/model/.\n  3. Verify package tests pass.\nVerification: Execute ONLY package-scoped tests: `go test -v ./internal/model` and `go vet ./internal/model`. NEVER run `go test ./...`.\nReturn Contract: Return a compact 3-line receipt: (1) files created/modified, (2) package test results, (3) confirmation of feature."
    }
  }
]
```

Each worker executes the concrete implementation:
1. **Reproduction Unit Test**: Focused test reproducing the defect or specifying the contract.
2. **Minimal Package Implementation**: Code changes strictly confined to `internal/<lane>/`.
3. **Package Verification**: Scoped checks (`go vet ./internal/<lane>`, `go test -v ./internal/<lane>`).

---

## Step 4 — Parallel Adversarial Cross-Validation & Gap Audit

The coordinator does NOT trust worker narration (`dos-witness-claim`). Instead, launch parallel verification subagents to independently audit the landed work before commit:

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "cross-validator",
      "description": "Cross-validate #1024 and #1035",
      "prompt": "You are an adversarial cross-validator.\nTasks to verify: #1024 (internal/gateway) and #1035 (internal/model).\nResponsibilities:\n  1. Inspect `git diff internal/gateway` and `git diff internal/model` to verify that edits did not leak outside declared package boundaries.\n  2. Execute on-device tests: `go test -v ./internal/gateway` and `go test -v ./internal/model`.\n  3. Verify that reproduction tests genuinely assert the expected behavior and do not test trivial mocks.\nReturn a structured DOS-style proof verdict: VERIFIED or REJECTED with concrete command witnesses."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "issue-auditor",
      "description": "Audit QA gaps and follow-ons",
      "prompt": "You are an issue auditor inspecting the changes in internal/gateway and internal/model.\nUncover unhandled edge cases, boundary conditions, or failure modes in the newly added code.\nIf real follow-ons or gaps exist, draft structured GitHub issue tickets with clear reproduction steps and acceptance criteria."
    }
  }
]
```

Coordinator independently verifies:
1. **Toolchain check**: `go vet ./internal/<lane>` and `go test -v ./internal/<lane>`.
2. **Diff boundary check**: `git diff --stat internal/<lane>`.
3. **Audit review**: Incorporate cross-validator proof verdicts and any auditor follow-on tickets.

---

## Step 5 — Re-Measure and Prove Queue Burndown

Re-evaluate the issue queue against the saved baseline to prove burndown:

```bash
# Re-snapshot issue state:
gh issue list --state open --json number,title,body,labels > _scratch/issue-queue/updated-issues.json
fak issue-queue --from-issues _scratch/issue-queue/updated-issues.json --compare _scratch/issue-queue/baseline.json
```

Confirm:
- Target issues are resolved and retired from the dispatchable count.
- Closed count delta reflects the completed batch.
- No new unclassified drift was introduced.

---

## Step 6 — Commit Cleanly by Explicit Path

Commit each finished leaf independently on the trunk. Include a Conventional-Commits subject, signed-off DCO (`-s`), issue citation `(#N)`, and required `(fak <leaf>)` trailer:

```bash
fak commit --path internal/<laneA> -m "fix(<laneA>): resolve <summary> (#1024) (fak <laneA>)"
fak commit --path internal/<laneB> -m "feat(<laneB>): add <summary> (#1035) (fak <laneB>)"
```
*(Fallback: `git commit -s -m "..." -- internal/<lane>` without `git add -A`).*

Verify `git status` shows your lanes are committed cleanly and no peer WIP was staged.

---

## Step 7 — Release Lane Leases & Clean Scratch

Release all acquired lane leases:

```bash
dos lease-lane release --lane <laneA>
dos lease-lane release --lane <laneB>
```

Reap the campaign scratch directory:
```bash
fak tree-doctor --reap-scratch issue-queue --json
```

Report the one-line completion receipt to the operator with the checkable commit SHAs, issue numbers, and test witnesses.
