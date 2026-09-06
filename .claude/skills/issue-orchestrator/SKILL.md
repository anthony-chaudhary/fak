---
name: issue-orchestrator
description: Plan, partition, and coordinate multi-wave parallel subagent campaigns to resolve and close GitHub issues across the repository. Uses `fak issue-orchestrator --plan-waves` (or `fak issue-lanes`) to partition the active issue backlog into pairwise tree-disjoint, collision-free cohorts, arbitrates lane and tree leases via `dos arbitrate`, dispatches parallel isolated worker subagents per wave via the `task` tool, independently witnesses landed bug fixes and features, commits with explicit paths and issue citations, and drives campaign burndowns. Use when running bulk issue resolution, feature wave burndowns, or multi-agent issue campaigns.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--top 10] [--target-issues 5] [--wave-size 4] [--max-waves 2] [--from-issues issues.json]  (no args = default focused wave: top 10 candidates, max 2 waves)"
---

# /issue-orchestrator — campaign-scale multi-wave issue resolution

> **The Campaign Coordinator for General Issue Work.** While atomic issue resolution targets 1–3
> issues in isolation via [`/issue-queue`](../issue-queue/SKILL.md) (the issue counterpart of [`/debt-clean`](../debt-clean/SKILL.md)),
> `/issue-orchestrator` coordinates **large volumes of general issue work**
> across multiple concurrent subagent waves. It partitions the active issue backlog into
> **pairwise tree-disjoint, concurrent-safe cohorts**, verifies or acquires lane leases via
> `dos arbitrate`, dispatches parallel worker subagents via the `task` tool, independently
> witnesses each worker's effects, commits explicit paths with issue citations `(#N)`, and
> tracks campaign burndown velocity against a baseline.

The shape: **baseline campaign (`fak issue-orchestrator --json`) → automated safe wave planning (`fak issue-orchestrator --plan-waves`) → pre-dispatch arbitration (`dos arbitrate`) → parallel subagent wave dispatch (`task`) → harvest & independently witness → commit leaf-by-leaf with `(#N)` → compare burndown → loop.**

---

## Defaults and Operational Boundaries

When invoked without explicit arguments or under underspecified requests, apply these reasonable defaults:

- **Bounded Evaluation (`--top 10`)**: ALWAYS bound backlog evaluation to the top 10–15 candidate issues (e.g. `fak issue-orchestrator --top 10`). Never run unbounded sweeps across hundreds of backlog tickets, which wastes tokens evaluating irrelevant triage queues.
- **Action-First Velocity**: Prioritize immediate wave execution over prolonged baseline analysis. Plan Wave 1, arbitrate, dispatch workers, and land code.
- **Target Issues**: Default to resolving a focused cohort of **3–5 issues** per session.
- **Wave Concurrency (`--wave-size`)**: Default to **4 concurrent workers** per wave (or 3 on constrained platforms). Never exceed 5 concurrent workers.
- **Max Waves per Run (`--max-waves`)**: Default to **1–2 waves** per run. Avoid unbounded multi-wave loops without checkpoints.
- **Scratch & State Hygiene**: Baseline snapshots belong in allocated scratch or temporary JSON files (`fak tree-doctor --scratch-path issue-orchestrator/baseline.json`), never untracked root dumps. Clean them up on completion.
- **Worker Isolation**: Each subagent gets exactly one issue and one package lane (`internal/<lane>`), touches only declared files, runs only package-scoped tests (`go test -v ./internal/<lane>`, `go vet ./internal/<lane>`), and returns a 3-line receipt.
- **Witness & Commit Cadence**: Coordinator independently verifies (`go vet`, `go test`) and commits each successful worker leaf individually with DCO sign-off, Conventional Commits, issue number citation `(#N)`, and lane ship-stamp `(fak <lane>)` before proceeding to the next wave.

---

## Architectural Invariants (The Orchestration Laws)

1. **Automated Tree-Disjoint Concurrency**:
   Never allow concurrent subagents to touch overlapping directories or packages. Two workers
   editing the same Go package break compilation for each other. Use `fak issue-orchestrator --plan-waves --wave-size N`
   to partition candidate issues into verified pairwise tree-disjoint waves with zero cross-package
   import contention.
2. **Live Lease Arbitration & Exclusion**:
   Before dispatching any subagent, the coordinator must verify or acquire the lane lease via
   `dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive` (or the `dos_arbitrate` MCP tool).
   `fak issue-orchestrator --plan-waves` automatically discovers and excludes any active leases held in
   `.dos/lane-journal.jsonl`. If an issue's lane is held by another agent, the orchestrator skips it and advances
   the next candidate from the wave plan without colliding.
3. **Multi-Orchestrator Concurrency**:
   Multiple orchestrator loops or independent agents can operate concurrently on the repository:
   each agent respects the shared lane taxonomy and lease journal. When two agents seek work simultaneously,
   `dos arbitrate` grants mutually disjoint lanes to each, preventing collision by construction.
4. **Serial Core Gates**:
   High-blast-radius core leaves (`internal/abi`, `internal/kernel`, `internal/adjudicator`,
   `internal/policy`, `internal/gateway`, `internal/vdso`, `internal/shipgate`, `internal/architest`) touch root types imported across the repository.
   They must NEVER run concurrently with peer workers—the planner automatically isolates them into
   dedicated, single-worker serial waves (`serial_singleton`).
5. **Strict Subagent Fences & Scoped Verification**:
   Subagents must edit ONLY within their assigned package directory. They must NEVER touch root
   files, `go.mod`, `go.sum`, `dos.toml`, or sibling packages. Subagents must run ONLY package-scoped
   tests (`go test -v ./internal/<lane>`, `go vet ./internal/<lane>`), NEVER broad `go test ./...` which
   could fail on unrelated in-flight peer edits.
6. **Subdivide Epics, Triage Unclear Scope**:
   Oversized issues (>15 expected steps or touching 3+ subsystems) are routed to the `Subdivide` queue
   to be broken into atomic leaves before dispatch. Issues with unclear scope or missing acceptance
   criteria are routed to `Triage`. The orchestrator dispatches only concrete leaf units.
7. **Coordinator Stays Clean**:
   The coordinator dispatches, adjudicates conflicts, witnesses proofs, and tracks the campaign burndown.
   Substantive implementation and test runs happen inside isolated subagent tasks (`subagent_type="general"`
   using Gemini 3.8 Flash high `variant: high`).
8. **Witness Before Close**:
   An issue is resolved only when its reproduction or contract test passes green on disk. The commit message
   must cite the issue number `(#N)` and carry the `(fak <lane>)` trailer.

---

## Phase 1 — Quick Bounded Planning & Target Selection

Plan the immediate actionable wave bounded to top candidate issues:

```bash
# Fast path: plan immediate actionable wave bounded to top candidate issues:
fak issue-orchestrator --top 10 --max-waves 1

# If tracking a formal burndown comparison, capture baseline bounded to cohort:
fak issue-orchestrator --top 15 --max-waves 2 --json > campaign-baseline.json
```

From the output, determine:
- **Starting Scope**: e.g., `10 total issue(s) evaluated · 2 dispatchable · 0 subdivide · 8 triage`.
- **Campaign Target**:
  - *Default Focused Target*: Resolve **2–4 issues** in Wave 1 immediately.
  - *Alternative Point Target*: Retire a fixed step budget (e.g. `--target-points 15`).
- **Wave Capacity**: Standard reasonable wave size is **2–4 parallel subagents** (`--wave-size 4`, max 5).
- **Campaign Horizon**: Standard execution batch is **1–2 waves** (`--max-waves 1` or `2`).

---

## Phase 2 — Automated Concurrent-Safe Wave Planning

Generate provably collision-free, concurrent-safe waves using `fak issue-orchestrator`:

```bash
# Plan campaign waves bounded to top candidate issues (wave size 4, max 2 waves):
fak issue-orchestrator --top 15 --wave-size 4 --max-waves 2

# Or plan from a specific issue source (e.g. local snapshot or gh export):
fak issue-orchestrator --from-issues issues.json --wave-size 4 --max-waves 2

# Or emit machine-readable wave plan JSON:
fak issue-orchestrator --top 15 --wave-size 4 --max-waves 2 --json > wave-plan.json
```

The planner automatically:
1. **Discovers and Excludes Held Leases**: Scans `.dos/lane-journal.jsonl` and excludes all active held leases in the workspace.
2. **Enforces Pairwise Tree-Disjointness**: Proves that no two issues in a wave share directory paths or parent/child containment.
3. **Decouples Package Import Contention**: Checks the Go internal import graph to ensure workers in the same wave do not contend on shared internal package APIs.
4. **Isolates Serial Singletons**: Identifies critical core leaves and schedules them into dedicated `serial_singleton` single-worker waves.
5. **Extracts Subdivide & Triage Queues**: Pulls oversized epics into `Subdivide` with a child-issue budget and unready tickets into `Triage`.

---

## Phase 3 — Pre-Dispatch Arbitration & Parallel Subagent Dispatch

For each wave in the plan:

### 1. Pre-Dispatch Lane Arbitration

Before launching workers, verify each lane in the wave is free to acquire:

```bash
dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive
```
*(Or invoke the `dos_arbitrate` tool).*

- **Admitted (`outcome: acquire`)**: Proceed to dispatch worker on `<lane>`.
- **Refused (`outcome: refuse`)**: Another peer agent took the lane. Log refusal, skip this lane, and advance the next candidate issue from the wave plan.

### 2. Launch Parallel Subagents

Launch all admitted workers in parallel in a single turn using multiple `task` tool calls with `subagent_type="worker"` under strict fences (using Gemini 3.8 Flash high `variant: high` for hard work):

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Resolve #1024 gateway streaming timeout",
      "prompt": "You are an isolated worker for issue #1024: Fix gateway streaming timeout.\nLane: gateway\nBoundaries: Edit ONLY files under internal/gateway/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/gateway` and `go vet ./internal/gateway`. NEVER run `go test ./...`.\nDeliverable: Fix the timeout issue and add a reproduction unit test.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) confirmation of defect fix."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Resolve #1035 model kv cache recycling",
      "prompt": "You are an isolated worker for issue #1035: Add model KV cache recycling.\nLane: model\nBoundaries: Edit ONLY files under internal/model/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/model` and `go vet ./internal/model`. NEVER run `go test ./...`.\nDeliverable: Implement the KV cache recycling path with reproduction unit tests.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) confirmation of defect fix."
    }
  }
]
```

---

## Phase 4 — Wave Harvest, Cross-Validation & Independent Witnessing

When the parallel subagents finish:

1. **Audit Receipts**: Verify each subagent reported green tests strictly within its declared package.
2. **Parallel Adversarial Verification & Gap Audit**:
   Launch parallel `cross-validator` and `issue-auditor` subagents to independently evaluate the changes:
   - `cross-validator`: Independently verifies git diff boundaries, executes on-device package tests, and produces DOS-style proof verdicts.
   - `issue-auditor`: Inspects diffs for unhandled edge cases, failure modes, and QA gaps, filing or drafting structured follow-on tickets.
3. **Independent Verification**: Coordinator independently runs package-scoped checks:
   ```bash
   go vet ./internal/<laneA> ./internal/<laneB>
   go test -v ./internal/<laneA> ./internal/<laneB>
   ```
4. **Commit by Explicit Path**:
   Commit each finished leaf independently on the trunk with the issue citation and ship-stamp trailer:
   ```bash
   fak commit --path internal/<laneA> -m "fix(<laneA>): resolve gateway streaming timeout (#1024) (fak <laneA>)"
   fak commit --path internal/<laneB> -m "feat(<laneB>): add model KV cache recycling (#1035) (fak <laneB>)"
   ```
   *(Fallback: `git commit -s -m "..." -- internal/<laneA>` without `git add -A`).*
5. **Release Lane Leases**:
   Release the acquired lane leases so peer agents can work on the lanes:
   ```bash
   dos lease-lane release --lane <laneA>
   dos lease-lane release --lane <laneB>
   ```

---

## Phase 5 — Burndown Check & Loop Progression

Compare against the campaign baseline to measure velocity and progress:

```bash
fak issue-orchestrator --compare campaign-baseline.json
```

Check the delta:
- How many issues closed in this wave?
- What is the total campaign progress percentage?
- How many waves remain in the current horizon?

**Decision Gate**:
- If the campaign target is achieved: emit the final campaign summary report.
- If more progress is needed: generate Wave N+1 from `fak issue-orchestrator --plan-waves` and continue the loop.
