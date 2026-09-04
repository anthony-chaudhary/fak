---
name: debt-orchestrator
description: Plan, partition, and coordinate multi-wave parallel subagent campaigns to retire large volumes of maturity debt across the repository. Queries `fak debt-lanes`, partitions the WIP backlog into pairwise tree-disjoint cohorts, dispatches parallel isolated worker subagents per wave, independently witnesses landed improvements, ratchets the production-grade denominator, and drives the system to target grade milestones (e.g. 80%, 85%, 90%). Use when running bulk debt reduction, campaign-scale maturity upgrades, or multi-agent wave burndowns.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--target-grade 80%] [--points 500] [--wave-size 4]  (no args = plan campaign to reach next letter grade)"
---

# /debt-orchestrator — campaign-scale multi-wave maturity debt retirement

> **The Campaign Coordinator.** While `/debt-clean` is the atomic unit that retires 1–3
> hotspots in a single batch, `/debt-orchestrator` coordinates **large volumes of work**
> across multiple concurrent subagent waves. It partitions the 780+ active WIP backlog into
> **pairwise tree-disjoint cohorts**, dispatches parallel worker subagents via the `task` tool,
> independently witnesses each worker's effects, and ratchets the production grade denominator
> toward the campaign goal (e.g. Grade C 75.4% → Grade B 80.0%+).

The shape: **baseline campaign → partition into disjoint cohorts → dispatch parallel subagent
wave → harvest & independently witness → commit leaf-by-leaf → compare burndown → loop.**

---

## Architectural Invariants (The Orchestration Laws)

1. **Tree-Disjoint Concurrency**:
   Never allow concurrent subagents to touch overlapping directories or packages. Two workers
   editing the same Go package break compilation for each other. Every concurrent wave must
   consist of pairwise disjoint leaf paths (`internal/<laneA>/**` vs `internal/<laneB>/**`).
2. **Serial Core Gates**:
   High-blast-radius core leaves (`internal/abi`, `internal/kernel`, `internal/adjudicator`)
   touch root types imported across the repository. They must NEVER run concurrently with
   peer workers—dispatch them as dedicated, single-worker serial waves.
3. **Coordinator Stays Clean**:
   The coordinator dispatches, adjudicates conflicts, witnesses proofs, and tracks the
   campaign burndown. Substantive implementation and test runs happen inside isolated subagent
   tasks (`subagent_type="general"`).
4. **Denominator Level-Setting**:
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
- **Starting Grade**: e.g. `Grade C (75.4%)`, Denominator `13,255.5` pts, Realized `10,000.3` pts.
- **Campaign Target**:
  - *Grade B Target (80.0%)*: requires `13,255.5 * 0.80 = 10,604.4` pts (need `+604.1` realized pts).
  - *Grade A Target (90.0%)*: requires `13,255.5 * 0.90 = 11,930.0` pts (need `+1,929.7` realized pts).
  - *Point Target*: e.g. retire `300` total debt points.
- **Wave Capacity**: Standard wave size is **3–5 parallel subagents**.

---

## Phase 2 — Cohort Partitioning (Collision-Free Waves)

Group candidate debt lanes into ordered, collision-free cohorts:

### Cohort Archetypes

- **Cohort 1: Disjoint Enabling Stubs (High Volume / Rapid Lift)**:
  Lanes at `0.0` or `1.0/8.0` (`proposed` or `stub`) that have disjoint directories.
  *Example Wave (Size 4)*:
  - Worker 1: `internal/blobcommon/`
  - Worker 2: `internal/breathgate/`
  - Worker 3: `internal/childproc/`
  - Worker 4: `internal/ciyaml/`
  *Advancing each from 0.0 to 4.0 yields 8.0 realized points each = +32.0 pts in one wave!*

- **Cohort 2: Stewardship & Hardening (Low Risk / High Stability)**:
  Lanes that already have code and tests, but lack exported documentation or contract comments.
  *Example Wave (Size 4)*:
  - Worker 1: `internal/treedoctor/`
  - Worker 2: `internal/pathlint/`
  - Worker 3: `internal/boundarylint/`
  - Worker 4: `internal/codelint/`

- **Cohort 3: Core De-risking (Serial Singleton)**:
  Lanes with `interest.band == "critical"` (`internal/abi`, `internal/adjudicator`).
  *Must run as a single worker with no concurrent siblings.*

---

## Phase 3 — Concurrent Subagent Wave Dispatch

Launch the wave's subagents in parallel in a single turn using multiple `task` tool calls:

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "general",
      "description": "Mature internal/blobcommon",
      "prompt": "You are a bounded worker for internal/blobcommon.\nGoal: Fulfill next action: add initial core implementation and unit tests.\nBoundaries: Edit ONLY files under internal/blobcommon/.\nVerification: Run `go test -v ./internal/blobcommon` and `go vet ./internal/blobcommon`.\nReturn: Return a compact 3-line receipt: (1) files created, (2) test result, (3) new maturity facts."
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "general",
      "description": "Mature internal/breathgate",
      "prompt": "You are a bounded worker for internal/breathgate.\nGoal: Fulfill next action: add initial core implementation and unit tests.\nBoundaries: Edit ONLY files under internal/breathgate/.\nVerification: Run `go test -v ./internal/breathgate` and `go vet ./internal/breathgate`.\nReturn: Return a compact 3-line receipt: (1) files created, (2) test result, (3) new maturity facts."
    }
  }
]
```

---

## Phase 4 — Wave Harvest & Independent Verification

When the subagents finish:

1. **Audit Receipts**: Verify each subagent reported green tests within its declared package.
2. **Independent Proof**: Coordinator independently runs:
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
   git add internal/<laneA>
   git commit -s -m "fix(<laneA>): advance maturity curve and retire debt (fak <laneA>)" -- internal/<laneA>

   git add internal/<laneB>
   git commit -s -m "fix(<laneB>): advance maturity curve and retire debt (fak <laneB>)" -- internal/<laneB>
   ```

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
- If more progress is needed: select the next disjoint cohort and trigger Wave N+1.
