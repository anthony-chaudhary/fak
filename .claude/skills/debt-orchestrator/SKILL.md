---
name: debt-orchestrator
description: Plan, partition, and coordinate sustained multi-wave parallel subagent campaigns to retire large volumes of maturity debt across the repository. Uses `fak debt-lanes --plan-waves` to partition WIP into pairwise tree-disjoint, collision-free cohorts (4–8 concurrent workers per wave), arbitrates lane leases via `dos arbitrate`, dispatches specialized subagents (`worker`, `cross-validator`, `issue-auditor`, `deep-reason`), independently witnesses landed improvements, ratchets the production-grade denominator, and drives long-running campaigns across 5–10+ waves to target grade milestones (e.g. Grade B 80%, Grade A 90%). Use when running bulk debt reduction, campaign-scale maturity upgrades, or multi-agent wave burndowns.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--target-grade 80%|85%|90%] [--points 1000] [--wave-size 6] [--max-waves 8]  (no args = continuous multi-wave campaign to next letter grade)"
---

# /debt-orchestrator — campaign-scale multi-wave maturity debt retirement

> **The Campaign Coordinator.** While `/debt-clean` is the atomic unit that retires 1–3
> hotspots in a single batch, `/debt-orchestrator` coordinates **large volumes of work**
> across multiple high-concurrency subagent waves over **sustained, long-running campaigns**.
> It partitions the active WIP backlog into **pairwise tree-disjoint, concurrent-safe cohorts**
> (4–8 parallel workers per wave), acquires lane leases via `dos arbitrate`, dispatches specialized
> worker subagents via the `task` tool, cross-validates landed improvements with adversarial subagents,
> commits leaf-by-leaf, and ratchets the production grade denominator across **5–10+ sequential waves**
> to reach ambitious milestone targets (e.g. Grade C 76.4% → Grade B 80.0%+ → Grade A 90.0%).

The shape: **baseline campaign → automated safe wave planning (`fak debt-lanes --plan-waves --wave-size 6 --max-waves 8`) → pre-dispatch arbitration (`dos arbitrate`) → parallel specialized subagent dispatch (4–8 workers) → wave harvest & parallel cross-validation / code-debt audit → commit leaf-by-leaf & release leases → compare burndown & O(1) context carryover → loop until milestone achieved.**

---

## Defaults and Operational Boundaries

When invoked without explicit arguments or under underspecified requests, apply these operational defaults:

- **Target Sizing**: Default to climbing a full letter grade milestone (e.g. Grade C 76.4% → Grade B 80.0%+, or Grade B → Grade A 90.0%) or retiring a major point block (e.g. retire 500–2,000+ debt points).
- **Wave Concurrency (`--wave-size`)**: Default to **6 concurrent workers** per wave (scaled to 4–8 based on repository capacity and candidate density). Never throttle to 1–2 when tree-disjoint candidate leaves are available.
- **Campaign Horizon (`--max-waves`) & Working Longer**: Default to running **5–10 sequential waves** per campaign session (or an unbounded campaign loop with `--max-waves 0` until the target grade or point goal is met). Do NOT stop after a single wave or prompt the operator for permission between waves; sustain execution autonomously across the planned wave queue.
- **Specialized Subagent Allocation**:
  - `worker`: Substantive implementation worker (using Gemini 3.8 Flash high variant for hard work) for package leaves. Implements core types, functions, writes unit tests, and adds benchmarks.
  - `cross-validator`: Parallel adversarial verification on-device during wave harvest.
  - `issue-auditor`: Edge case analysis and QA gap audit during wave harvest.
  - `deep-reason`: High-difficulty reasoning for serial singleton core leaves (`internal/kernel`, `internal/adjudicator`, `internal/abi`) with complex concurrency invariants.
- **Context Window Budgeting & O(1) State Carryover**:
  - Subagents return compact 3-line receipts: (1) files created/edited, (2) test result, (3) new maturity facts. Heavy command outputs stay strictly inside the subagent boundary.
  - At wave boundaries, discard raw compiler output and verbose command logs. Maintain an O(1) Campaign Progress Ledger in coordinator context:
    `Wave N | Lanes [L1, L2, L3, L4, L5, L6] | Commits [sha1, ...] | Points: +X.X | Grade: Y% -> Z%`
  - Re-arm the next wave with fresh context, enabling 5–10+ wave runs without context degradation or token budget exhaustion.
- **Scratch & State Hygiene**: Baseline snapshots belong in allocated scratch (`_scratch/debt-orchestrator/baseline.json`), reaped with `fak tree-doctor --reap-scratch debt-orchestrator --json` on campaign completion.
- **Autonomous Progress & Recovery**: Do not pause between waves. If a lane is held by a peer or contentious, skip/defer it, continue with admitted lanes, and advance immediately to the next wave.

---

## Architectural Invariants (The Orchestration Laws)

1. **Automated Tree-Disjoint Concurrency**:
   Never allow concurrent subagents to touch overlapping directories or packages. Two workers
   editing the same Go package break compilation for each other. Use `fak debt-lanes --plan-waves --wave-size N`
   to partition candidate lanes into verified pairwise tree-disjoint waves with zero cross-package
   import contention.
2. **Live Lease Arbitration & Exclusion**:
   Before dispatching any subagent, the coordinator must verify or acquire the lane lease via
   `dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive` (or the `dos_arbitrate` MCP tool).
   `fak debt-lanes --plan-waves` automatically discovers and excludes any active leases held in
   `.dos/lane-journal.jsonl`. If a lane is held by another agent, the orchestrator skips it and advances
   the next candidate from the wave plan without colliding.
3. **Multi-Orchestrator Concurrency**:
   Multiple orchestrator loops or independent agents can operate concurrently on the repository:
   each agent respects the shared lane taxonomy and lease journal. When two agents seek work simultaneously,
   `dos arbitrate` grants mutually disjoint lanes to each, preventing collision by construction.
4. **Serial Core Gates & Deep-Reasoning Allocation**:
   High-blast-radius core leaves (`internal/abi`, `internal/kernel`, `internal/adjudicator`,
   `internal/policy`, `internal/gateway`, `internal/vdso`, `internal/shipgate`, `internal/architest`) touch root
   types imported across the repository. They must NEVER run concurrently with peer workers—the planner automatically
   isolates them into dedicated, single-worker serial waves (`serial_singleton`). Deploy `subagent_type="deep-reason"`
   for these leaves to ensure lock invariants, concurrency safety, and frozen ABI contracts are strictly preserved.
5. **Strict Subagent Fences & Scoped Verification**:
   Subagents must edit ONLY within their assigned `internal/<lane>/` package directory. They must NEVER
   touch root files, `go.mod`, `go.sum`, `dos.toml`, or sibling packages. Subagents must run ONLY package-scoped
   tests (`go test -v ./internal/<lane>`, `go vet ./internal/<lane>`), NEVER broad `go test ./...` which
   could fail on unrelated in-flight peer edits.
6. **Coordinator Stays Clean & O(1) Carryover**:
   The coordinator dispatches, adjudicates conflicts, witnesses proofs, and tracks the campaign burndown.
   Substantive implementation and test runs happen inside isolated subagents. Discard heavy logs at wave
   boundaries and carry forward only compact progress ledger rows to sustain long multi-wave sessions.
7. **Specialized Subagent Allocation**:
   Deploy specialized subagents matched to the task: `worker` for package implementation, `cross-validator`
   for adversarial test verification, `issue-auditor` for QA gap audits, and `deep-reason` for complex
   concurrency/architecture leaves.
8. **Sustained Multi-Wave Campaign Endurance**:
   Do not stop after a single wave. Execute the campaign continuously across 5–10+ sequential waves until
   the milestone target grade is achieved or the queue is fully drained. Persist through transient locks
   or skipped lanes without halting the campaign loop.
9. **Denominator Level-Setting**:
   Retiring debt means increasing `RealizedPoints` by advancing the maturity curve (0.0 → 10.0).
   The orchestrator never deletes work to artificially shrink the denominator.

---

## Phase 1 — Campaign Sizing & Baseline

Capture the campaign baseline and calculate the distance to the target milestone using allocated scratch storage (`fak tree-doctor --scratch-dir debt-orchestrator`):

```bash
# Allocate scratch directory for campaign artifacts:
fak tree-doctor --scratch-dir debt-orchestrator

# Capture baseline snapshot in allocated scratch:
fak debt-lanes --json > _scratch/debt-orchestrator/baseline.json

# View top debt hotspots:
fak debt-lanes --top 10
```

From the output, determine:
- **Starting Grade**: e.g. `Grade C (76.4%)`, Denominator `13,303.5` pts, Realized `10,161.3` pts.
- **Campaign Target**:
  - *Grade B Target (80.0%)*: requires `13,303.5 * 0.80 = 10,642.8` pts (need `+481.5` realized pts).
  - *Grade A Target (90.0%)*: requires `13,303.5 * 0.90 = 11,973.2` pts (need `+1,811.9` realized pts).
  - *Point Target*: e.g. retire `500–2,000` total debt points.
- **Wave Capacity**: High-throughput wave size is **4–8 parallel subagents** (default `--wave-size 6`).
- **Campaign Horizon**: Long-running horizon of **5–10 waves** (`--max-waves 8` or `10`, or `0` = all necessary).

---

## Phase 2 — Automated Concurrent-Safe Wave Planning

Generate provably collision-free, concurrent-safe waves using `fak debt-lanes --plan-waves`:

```bash
# Plan campaign waves with high-concurrency wave size 6 and 8-wave campaign horizon:
fak debt-lanes --plan-waves --wave-size 6 --max-waves 8

# Or plan to a specific target grade milestone:
fak debt-lanes --plan-waves --wave-size 6 --max-waves 8 --target-grade 80%

# Or emit machine-readable wave plan JSON to scratch:
fak debt-lanes --plan-waves --wave-size 6 --max-waves 8 --json > _scratch/debt-orchestrator/wave-plan.json
```

The planner automatically:
1. **Discovers and Excludes Held Leases**: Scans `.dos/lane-journal.jsonl` and excludes all active held leases in the workspace.
2. **Enforces Pairwise Tree-Disjointness**: Proves that no two lanes in a wave share directory paths or parent/child containment.
3. **Decouples Package Import Contention**: Checks the Go internal import graph to ensure workers in the same wave do not contend on shared internal package APIs.
4. **Isolates Serial Singletons**: Identifies critical core leaves and schedules them into dedicated `serial_singleton` single-worker waves.
5. **Projects Realized Lift & Grade Climb**: Calculates the exact debt retired and production grade climb per wave.

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
- **Refused (`outcome: refuse`)**: Another peer agent took the lane. Log refusal, skip this lane, and advance the next candidate lane from the wave plan.

### 2. Partitioning by Action Type

Partition each lane by its scheduled maturity action type:
- **Package-Isolated Actions** (`implement`, `test`, `benchmark`, `clean comments`): Run directly in parallel fenced subagents bounded strictly to `internal/<lane>/`.
- **Integration Actions** (`integrate` into `cmd/`, `dogfood` in `internal/maturity/runtime-proofs.json`): Require cross-package coordinator wiring. Do not assign cross-package modifications to package-fenced subagents; wire integration actions from the coordinator.

### 3. Select Specialized Subagent Type

- Standard package leaves (`implement`, `test`, `benchmark`) → `subagent_type="worker"`.
- Complex serial singletons (`internal/kernel`, `internal/adjudicator`) → `subagent_type="deep-reason"`.

### 4. Launch High-Concurrency Parallel Subagents (Single Turn)

Launch all admitted workers in parallel in a single turn using multiple `task` tool calls with `subagent_type="worker"` under strict fences (using Gemini 3.8 Flash high `variant: high` for hard work). Inject the lane's actual `<next_action>` into the worker prompt:

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Mature internal/harnesshint",
      "prompt": "You are a bounded worker for internal/harnesshint.\nGoal: Fulfill next action: integrate harnesshint: connect into production commands or registration graph\nBoundaries: Edit ONLY files under internal/harnesshint/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/harnesshint` and `go vet ./internal/harnesshint`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Mature internal/managedocs",
      "prompt": "You are a bounded worker for internal/managedocs.\nGoal: Fulfill next action: integrate managedocs: connect into production commands or registration graph\nBoundaries: Edit ONLY files under internal/managedocs/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/managedocs` and `go vet ./internal/managedocs`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Mature internal/observer",
      "prompt": "You are a bounded worker for internal/observer.\nGoal: Fulfill next action: integrate observer: connect into production commands or registration graph\nBoundaries: Edit ONLY files under internal/observer/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/observer` and `go vet ./internal/observer`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Mature internal/roofline",
      "prompt": "You are a bounded worker for internal/roofline.\nGoal: Fulfill next action: integrate roofline: connect into production commands or registration graph\nBoundaries: Edit ONLY files under internal/roofline/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/roofline` and `go vet ./internal/roofline`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Mature internal/sensecheck",
      "prompt": "You are a bounded worker for internal/sensecheck.\nGoal: Fulfill next action: integrate sensecheck: connect into production commands or registration graph\nBoundaries: Edit ONLY files under internal/sensecheck/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/sensecheck` and `go vet ./internal/sensecheck`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "worker",
      "description": "Mature internal/sessionsearch",
      "prompt": "You are a bounded worker for internal/sessionsearch.\nGoal: Fulfill next action: integrate sessionsearch: connect into production commands or registration graph\nBoundaries: Edit ONLY files under internal/sessionsearch/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/sessionsearch` and `go vet ./internal/sessionsearch`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
    }
  }
]
```

---

## Phase 4 — Wave Harvest, Cross-Validation, Verification, Commit & Lease Release

When the parallel subagents finish:

1. **Audit Receipts**: Verify each subagent reported green tests strictly within its declared package.
2. **Parallel Adversarial Verification & Gap Audit**:
   Launch parallel `cross-validator` and `issue-auditor` subagents to independently evaluate the changes:
   - `cross-validator`: Independently verifies git diff boundaries, executes on-device package tests, and produces DOS-style proof verdicts.
   - `issue-auditor`: Inspects diffs for unhandled edge cases, failure modes, and QA gaps, preventing newly introduced debt.
3. **Independent Verification**: Coordinator independently runs package-scoped checks across all wave leaves:
   ```bash
   go vet ./internal/<laneA> ./internal/<laneB> ./internal/<laneC> ./internal/<laneD> ./internal/<laneE> ./internal/<laneF>
   go test -v ./internal/<laneA> ./internal/<laneB> ./internal/<laneC> ./internal/<laneD> ./internal/<laneE> ./internal/<laneF>
   ```
4. **Maturity Verification**:
   ```bash
   fak debt-lanes --lane <laneA>
   fak debt-lanes --lane <laneB>
   ```
   Confirm maturity score climbed, gap decreased, and debt principal dropped.
5. **Code-Debt Hygiene Check**:
   Run pre-commit code-debt hygiene check to ensure no god-functions (>1500 lines) or cyclomatic complexity traps were introduced:
   ```bash
   fak code-debt --path internal/<laneA>
   fak code-debt --path internal/<laneB>
   ```
6. **Commit by Explicit Path**:
   Commit each finished leaf independently on the trunk with signed-off DCO and the required `(fak <lane>)` trailer:
   ```bash
   fak commit --path internal/<laneA> -m "fix(<laneA>): advance maturity curve and retire debt (fak <laneA>)"
   fak commit --path internal/<laneB> -m "fix(<laneB>): advance maturity curve and retire debt (fak <laneB>)"
   fak commit --path internal/<laneC> -m "fix(<laneC>): advance maturity curve and retire debt (fak <laneC>)"
   fak commit --path internal/<laneD> -m "fix(<laneD>): advance maturity curve and retire debt (fak <laneD>)"
   fak commit --path internal/<laneE> -m "fix(<laneE>): advance maturity curve and retire debt (fak <laneE>)"
   fak commit --path internal/<laneF> -m "fix(<laneF>): advance maturity curve and retire debt (fak <laneF>)"
   ```
   *(Fallback: `git commit -s -m "..." -- internal/<laneA>` without `git add -A`).*
7. **Release Lane Leases**:
   Release each lane lease after commit so leases do not leak into `Held Lanes Excluded` in subsequent waves or peer runs:
   ```bash
   dos lease-lane release --lane <laneA>
   dos lease-lane release --lane <laneB>
   dos lease-lane release --lane <laneC>
   dos lease-lane release --lane <laneD>
   dos lease-lane release --lane <laneE>
   dos lease-lane release --lane <laneF>
   ```

---

## Phase 5 — Burndown Check, O(1) Context Carryover & Autonomous Multi-Wave Loop

Compare against the scratch baseline to measure campaign velocity:

```bash
fak debt-lanes --compare _scratch/debt-orchestrator/baseline.json
```

Check the delta:
- Did `Production Grade` increase toward the campaign goal?
- Did `Total Debt` drop?
- Did `WIP Dilution` decrease?

### Record O(1) Progress Entry

Record a compact 1-line entry in the coordinator's O(1) Campaign Progress Ledger:
```
Wave N Complete: 6 lanes retired (+36.0 pts) · Grade C 76.4% → C 78.2% · Total Debt -38.9 pts
```

### Autonomous Multi-Wave Loop Progression (Working Longer)

- **Campaign Target Achieved**:
  1. Clean up allocated campaign scratch:
     ```bash
     fak tree-doctor --reap-scratch debt-orchestrator --json
     ```
  2. Emit the final campaign summary report.
- **Campaign Target Not Yet Met (Sustained Execution)**:
  1. **DO NOT STOP.** Do not pause or prompt the operator for confirmation between waves.
  2. Discard verbose test logs and compiler outputs from Wave N to preserve clean coordinator context.
  3. Immediately advance to Wave N+1 from the planned wave queue.
  4. If the wave plan has drained but the milestone target is still unmet, re-run wave planning:
     ```bash
     fak debt-lanes --plan-waves --wave-size 6 --max-waves 8
     ```
     to generate the next wave cohort and continue the loop.
  5. If a lane encounters transient lock contention or peer lease hold, skip it, advance the next disjoint candidate lane from the wave plan, and sustain campaign momentum.
