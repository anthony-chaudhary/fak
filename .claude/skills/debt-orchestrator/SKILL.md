---
name: debt-orchestrator
description: Plan, partition, and coordinate multi-wave parallel subagent campaigns to retire large volumes of maturity debt across the repository. Uses `fak debt-lanes --plan-waves` to partition WIP into pairwise tree-disjoint, collision-free cohorts, arbitrates lane leases via `dos arbitrate`, dispatches parallel isolated worker subagents per wave, independently witnesses landed improvements, ratchets the production-grade denominator, and drives the system to target grade milestones (e.g. 80%, 85%, 90%). Use when running bulk debt reduction, campaign-scale maturity upgrades, or multi-agent wave burndowns.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--target-grade 80%] [--points 500] [--wave-size 4]  (no args = plan campaign to reach next letter grade)"
---

# /debt-orchestrator — campaign-scale multi-wave maturity debt retirement

> **The Campaign Coordinator.** While `/debt-clean` is the atomic unit that retires 1–3
> hotspots in a single batch, `/debt-orchestrator` coordinates **large volumes of work**
> across multiple concurrent subagent waves. It partitions the active WIP backlog into
> **pairwise tree-disjoint, concurrent-safe cohorts**, verifies or acquires lane leases via
> `dos arbitrate`, dispatches parallel worker subagents via the `task` tool, independently
> witnesses each worker's effects, and ratchets the production grade denominator toward the
> campaign goal (e.g. Grade C 76.4% → Grade B 80.0%+).

The shape: **baseline campaign → automated safe wave planning (`fak debt-lanes --plan-waves`) → pre-dispatch arbitration (`dos arbitrate`) → parallel subagent wave dispatch → harvest & independently witness → commit leaf-by-leaf → compare burndown → loop.**

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
4. **Serial Core Gates**:
   High-blast-radius core leaves (`internal/abi`, `internal/kernel`, `internal/adjudicator`,
   `internal/policy`, `internal/gateway`, `internal/vdso`) touch root types imported across the repository.
   They must NEVER run concurrently with peer workers—the planner automatically isolates them into
   dedicated, single-worker serial waves (`serial_singleton`).
5. **Strict Subagent Fences & Scoped Verification**:
   Subagents must edit ONLY within their assigned `internal/<lane>/` package directory. They must NEVER
   touch root files, `go.mod`, `go.sum`, `dos.toml`, or sibling packages. Subagents must run ONLY package-scoped
   tests (`go test -v ./internal/<lane>`, `go vet ./internal/<lane>`), NEVER broad `go test ./...` which
   could fail on unrelated in-flight edits.
6. **Coordinator Stays Clean**:
   The coordinator dispatches, adjudicates conflicts, witnesses proofs, and tracks the campaign burndown.
   Substantive implementation and test runs happen inside isolated subagent tasks (`subagent_type="general"`).
7. **Denominator Level-Setting**:
   Retiring debt means increasing `RealizedPoints` by advancing the maturity curve (0.0 → 10.0).
   The orchestrator never deletes work to artificially shrink the denominator.

---

## Phase 1 — Campaign Sizing & Baseline

Capture the campaign baseline and calculate the distance to the target milestone:

```bash
fak debt-lanes --json > campaign-baseline.json
fak debt-lanes --top 10
```

From the output, determine:
- **Starting Grade**: e.g. `Grade C (76.4%)`, Denominator `13,303.5` pts, Realized `10,161.3` pts.
- **Campaign Target**:
  - *Grade B Target (80.0%)*: requires `13,303.5 * 0.80 = 10,642.8` pts (need `+481.5` realized pts).
  - *Grade A Target (90.0%)*: requires `13,303.5 * 0.90 = 11,973.2` pts (need `+1,811.9` realized pts).
  - *Point Target*: e.g. retire `300` total debt points.
- **Wave Capacity**: Standard wave size is **3–5 parallel subagents**.

---

## Phase 2 — Automated Concurrent-Safe Wave Planning

Generate provably collision-free, concurrent-safe waves using `fak debt-lanes --plan-waves`:

```bash
# Plan campaign waves with default wave size 4:
fak debt-lanes --plan-waves --wave-size 4 --max-waves 5

# Or emit machine-readable wave plan JSON:
fak debt-lanes --plan-waves --wave-size 4 --max-waves 5 --json > wave-plan.json
```

The planner automatically:
1. **Discovers and Excludes Held Leases**: Scans `.dos/lane-journal.jsonl` and excludes all active held leases in the workspace.
2. **Enforces Pairwise Tree-Disjointness**: Proves that no two lanes in a wave share directory paths or parent/child containment.
3. **Decouples Package Import Contention**: Checks the Go internal import graph to ensure workers in the same wave do not contend on shared internal package APIs.
4. **Isolates Serial Singletons**: Identifies critical core leaves and schedules them into dedicated `serial_singleton` single-worker waves.
5. **Projects Realized Lift**: Calculates the exact debt retired and production grade climb per wave.

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

### 2. Launch Parallel Subagents

Launch all admitted workers in parallel in a single turn using multiple `task` tool calls with strict fences:

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "general",
      "description": "Mature internal/faultlab",
      "prompt": "You are a bounded worker for internal/faultlab.\nGoal: Advance maturity curve: add initial core implementation and unit tests.\nBoundaries: Edit ONLY files under internal/faultlab/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/faultlab` and `go vet ./internal/faultlab`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "general",
      "description": "Mature internal/gcpgpu",
      "prompt": "You are a bounded worker for internal/gcpgpu.\nGoal: Advance maturity curve: add initial core implementation and unit tests.\nBoundaries: Edit ONLY files under internal/gcpgpu/. NEVER touch go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/gcpgpu` and `go vet ./internal/gcpgpu`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) test result, (3) new maturity facts."
    }
  }
]
```

---

## Phase 4 — Wave Harvest & Independent Verification

When the parallel subagents finish:

1. **Audit Receipts**: Verify each subagent reported green tests strictly within its declared package.
2. **Independent Verification**: Coordinator independently runs package-scoped checks:
   ```bash
   go vet ./internal/<laneA> ./internal/<laneB>
   go test -v ./internal/<laneA> ./internal/<laneB>
   ```
3. **Maturity Verification**:
   ```bash
   fak debt-lanes --lane <laneA>
   fak debt-lanes --lane <laneB>
   ```
   Confirm maturity score climbed and debt principal dropped.
4. **Commit by Explicit Path**:
   Commit each finished leaf independently on the trunk:
   ```bash
   fak commit --path internal/<laneA> -m "fix(<laneA>): advance maturity curve and retire debt (fak <laneA>)"
   fak commit --path internal/<laneB> -m "fix(<laneB>): advance maturity curve and retire debt (fak <laneB>)"
   ```
   *(Fallback: `git commit -s -m "..." -- internal/<laneA>` without `git add -A`).*

---

## Phase 5 — Burndown Check & Loop Progression

Compare against the campaign baseline to measure campaign velocity:

```bash
fak debt-lanes --compare campaign-baseline.json
```

Check the delta:
- Did `Production Grade` increase toward the campaign goal?
- Did `Total Debt` drop?
- Did `WIP Dilution` decrease?

**Decision Gate**:
- If the campaign goal is achieved: emit the final campaign summary report.
- If more progress is needed: generate Wave N+1 from `fak debt-lanes --plan-waves` and continue the loop.
