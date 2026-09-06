---
name: debt-orchestrator
description: Plan, partition, and coordinate sustained multi-wave parallel subagent campaigns to retire large volumes of maturity debt across both the public fak engine and private fak-private platform repositories. Defaults to `--target-repo both` when fak-private is present. Uses `fak debt-orchestrator` (or `fak debt-lanes --plan-waves`) to partition candidate lanes into pairwise tree-disjoint, collision-free cohorts (4–8 concurrent workers per wave), arbitrates lane leases via `dos arbitrate` and git contract locks, dispatches specialized subagents (`worker`, `cross-validator`, `issue-auditor`, `deep-reason`), independently witnesses tests per repo (`go test` vs `go -C ..\fak-private test`), audits public commits for leaks via `tools/scrub_public_copy.py`, commits leaf-by-leaf with distinct trailers (`(fak <leaf>)` vs `(fak-private <leaf>)`), ratchets the production-grade denominator, supports dynamic validation envelopes (10/50/100 units), and drives campaigns across 5–10+ waves to milestone targets (e.g. Grade B 80%, Grade A 90%).
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--target-repo both|fak|fak-private] [--target-grade 80%|85%|90%] [--points 1000] [--wave-size 6] [--max-waves 8]  (no args = continuous multi-wave campaign to next letter grade)"
---

# /debt-orchestrator — campaign-scale multi-wave maturity debt retirement

> **The Dual-Repo Campaign Coordinator.** While `/debt-clean` is the atomic unit that retires 1–3
> hotspots in a single batch, `/debt-orchestrator` coordinates **large volumes of work**
> across multiple high-concurrency subagent waves over **sustained, long-running campaigns** across
> both repositories: public engine (`fak`) and private platform/serving (`fak-private`).
> It defaults to `--target-repo both` (`both|fak|fak-private`) when `fak-private` is present,
> partitions candidate lanes into **pairwise tree-disjoint, concurrent-safe cohorts**
> (4–8 parallel workers per wave), acquires lane leases via `dos arbitrate` and git contract locks,
> dispatches specialized worker subagents via the `task` tool with repo-scoped fences,
> cross-validates landed improvements with adversarial subagents, audits public commits with `tools/scrub_public_copy.py`,
> commits leaf-by-leaf with repo-appropriate trailers (`(fak <leaf>)` vs `(fak-private <leaf>)`),
> and ratchets the production grade denominator across **5–10+ sequential waves**
> to reach ambitious milestone targets (e.g. Grade C 76.4% → Grade B 80.0%+ → Grade A 90.0%).

The shape: **baseline campaign (`fak debt-orchestrator --target-repo both --json`) → automated safe wave planning (`fak debt-orchestrator --plan-waves --wave-size 6 --max-waves 8`) → pre-dispatch arbitration (`dos arbitrate` & contract locks) → parallel specialized subagent dispatch (4–8 workers per wave) → wave harvest, independent verification & leak check → commit leaf-by-leaf & release leases → compare burndown & O(1) context carryover → loop until milestone achieved.**

---

## Defaults and Operational Boundaries

When invoked without explicit arguments or under underspecified requests, apply these operational defaults:

- **Target Repository (`--target-repo`)**: Default to `both` when `fak-private` is available as a sibling repository (`../fak-private` or `C:\work\fak-private`), otherwise `fak`. The orchestrator balances waves across public and private lanes to prevent platform interface skew.
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

## Architectural Invariants (The Dual-Repo Orchestration Laws)

1. **Default Dual-Repo Target (`--target-repo both`)**:
   Unless explicitly restricted via `--target-repo fak` or `--target-repo fak-private`, the orchestrator plans and balances waves across both repositories. Dual-repo synchronization is mandatory because `fak-private` imports `fak/pkg/*` via `go.work`.
2. **Cross-Repo Tree-Disjoint Partitioning**:
   Never allow concurrent subagents to touch overlapping directories or packages. Lanes are partitioned into pairwise tree-disjoint cohorts:
   - **`fak` public lanes**: `internal/<lane>`, `pkg/<lane>`, `cmd/<lane>`.
   - **`fak-private` private lanes**: `platform/<lane>`, `cmd/<lane>`, `tools/<lane>`.
   Concurrent subagents within a wave must NEVER share directory paths, package imports, or repo-level contention.
3. **Live Lease Arbitration & Git Contract Locks**:
   Before dispatching any subagent, the coordinator must verify or acquire the lane lease:
   - For `fak` lanes: `dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive`
   - For `fak-private` lanes: `dos arbitrate --workspace ..\fak-private --lane <lane> --kind keyword --mode exclusive`
   - Check git contract locks (`refs/fak/locks/contract-*` and `refs/fak/locks/<lane>`).
   `fak debt-orchestrator --plan-waves` automatically discovers and excludes any active leases held in `.dos/lane-journal.jsonl`. If a lane is held, skip it and advance the next candidate from the wave plan.
4. **Serial Core Gates & Deep-Reasoning Allocation**:
   High-blast-radius core leaves (`internal/abi`, `internal/kernel`, `internal/adjudicator`,
   `internal/policy`, `internal/gateway`, `internal/vdso`, `internal/shipgate`, `internal/architest`, `platform/gateway`, `platform/ctxmmu`) touch root types imported across the repository. They must NEVER run concurrently with peer workers—the planner automatically isolates them into dedicated, single-worker serial waves (`serial_singleton`). Deploy `subagent_type="deep-reason"` for these leaves to ensure lock invariants, concurrency safety, and frozen ABI contracts are strictly preserved.
5. **Strict Subagent Fences & Scoped Verification per Repo**:
   Subagents run with absolute filesystem boundaries:
   - Subagents assigned to `fak` work strictly inside `internal/<lane>/` or `pkg/<lane>/`. They NEVER touch `fak-private`, root files, `go.mod`, `go.sum`, or sibling packages.
   - Subagents assigned to `fak-private` work strictly inside `platform/<lane>/` (or designated `cmd/`, `tools/`). They NEVER touch public `fak`.
   - Subagents run package-scoped tests only: `go test -v ./internal/<lane>` (for `fak`) or `go -C ..\fak-private test -v ./platform/<lane>` (for `fak-private`). NEVER run broad `go test ./...` which could fail on unrelated in-flight peer edits.
6. **Coordinator Stays Clean & O(1) Carryover**:
   The coordinator dispatches, adjudicates conflicts, witnesses proofs, and tracks the campaign burndown. Substantive implementation and test runs happen inside isolated subagents. Discard heavy logs at wave boundaries and carry forward only compact progress ledger rows to sustain long multi-wave sessions.
7. **Mandatory Leak Check Before Public Commits**:
   Before ANY commit is made to public `fak`, the coordinator MUST execute the leak scrubber audit:
   ```bash
   python tools/scrub_public_copy.py --audit-staged --root .
   ```
   If ANY needle or forbidden pattern is detected, the commit is BLOCKED immediately. If drafting external issues or PR notes alongside staged code, also run `python tools/issue_scrub.py --check < notes.md`.
8. **Distinct Commit Trailers**:
   - Public `fak` commits MUST end with `(fak <leaf>)`.
   - Private `fak-private` commits MUST end with `(fak-private <leaf>)`.
   Commits stage explicit paths only; `git add -A` and `git commit -a` are strictly forbidden.
9. **Dynamic Validation Envelopes (10/50/100 Units)**:
   For high-leverage infrastructure leaves (`session`, `recall`, `vcache`, `ctxmmu`, `dispatch`, `cmd/fak-server`), advance beyond 1-step unit tests using dynamic validation envelopes (`NOTES-maturity-curve-ops-and-dynamic-dogfooding-scale-2026-09-04.md`):
   - **10 Units (Micro)**: Boundary fuzzing, parameter permutations, CLI replay against edge scenarios.
   - **50 Units (Matrix)**: Concurrency stress matrix, multi-lane contention, token-pressure pacing with `YIELDED_IO`.
   - **100 Units (Macro)**: Dark fleet shadow execution, macro-wave migration, live LLM evaluation suites.
10. **Denominator Level-Setting**:
    Retiring debt means increasing `RealizedPoints` by advancing the maturity curve (0.0 → 10.0). The orchestrator never deletes work to artificially shrink the denominator.

---

## Phase 1 — Campaign Sizing & Baseline

Capture the campaign baseline and calculate the distance to the target milestone using allocated scratch storage (`fak tree-doctor --scratch-dir debt-orchestrator`):

```bash
# Allocate scratch directory for campaign artifacts:
fak tree-doctor --scratch-dir debt-orchestrator

# Capture baseline snapshot in allocated scratch (supports --target-repo both|fak|fak-private):
fak debt-orchestrator --target-repo both --json > _scratch/debt-orchestrator/baseline.json

# View top debt hotspots across target repositories:
fak debt-orchestrator --target-repo both --top 10
```

From the output, determine:
- **Starting Grade**: e.g. `Grade C (76.4%)`, Denominator `13,303.5` pts, Realized `10,161.3` pts.
- **Target Repository Split**: Balance of candidate lanes across public `fak` and private `fak-private`.
- **Campaign Target**:
  - *Grade B Target (80.0%)*: requires `13,303.5 * 0.80 = 10,642.8` pts (need `+481.5` realized pts).
  - *Grade A Target (90.0%)*: requires `13,303.5 * 0.90 = 11,973.2` pts (need `+1,811.9` realized pts).
  - *Point Target*: e.g. retire `500–2,000` total debt points.
- **Wave Capacity**: High-throughput wave size is **4–8 parallel subagents** (default `--wave-size 6`).
- **Campaign Horizon**: Long-running horizon of **5–10 waves** (`--max-waves 8` or `10`, or `0` = all necessary).

---

## Phase 2 — Automated Concurrent-Safe Wave Planning

Generate provably collision-free, concurrent-safe waves across both repositories using `fak debt-orchestrator`:

```bash
# Plan campaign waves across both repositories with wave size 6 and 8-wave horizon:
fak debt-orchestrator --target-repo both --wave-size 6 --max-waves 8

# Or plan to a specific target grade milestone:
fak debt-orchestrator --target-repo both --wave-size 6 --max-waves 8 --target-grade 80%

# Or emit machine-readable wave plan JSON to scratch:
fak debt-orchestrator --target-repo both --wave-size 6 --max-waves 8 --json > _scratch/debt-orchestrator/wave-plan.json
```

The planner automatically:
1. **Discovers and Excludes Held Leases**: Scans `.dos/lane-journal.jsonl` in both workspaces and excludes all active held leases.
2. **Enforces Pairwise Tree-Disjointness**: Proves that no two lanes in a wave share directory paths or parent/child containment.
3. **Decouples Package Import Contention**: Checks the Go import graph to ensure workers in the same wave do not contend on shared package APIs.
4. **Isolates Serial Singletons**: Identifies critical core leaves and schedules them into dedicated `serial_singleton` single-worker waves.
5. **Projects Realized Lift & Grade Climb**: Calculates the exact debt retired and production grade climb per wave across both repositories.

---

## Phase 3 — Pre-Dispatch Arbitration & Parallel Subagent Dispatch

For each wave in the plan:

### 1. Pre-Dispatch Lane Arbitration & Contract Locks

Before launching workers, verify each lane in the wave is free to acquire in its respective workspace:

```bash
# For a fak public lane:
dos arbitrate --workspace . --lane <lane> --kind keyword --mode exclusive

# For a fak-private private lane:
dos arbitrate --workspace ..\fak-private --lane <lane> --kind keyword --mode exclusive
```
*(Or invoke the `dos_arbitrate` tool).*

Check git contract locks (`refs/fak/locks/contract-*` and `refs/fak/locks/<lane>`).
- **Admitted (`outcome: acquire`)**: Proceed to dispatch worker on `<lane>`.
- **Refused (`outcome: refuse`)**: Another peer agent took the lane. Log refusal, skip this lane, and advance the next candidate lane from the wave plan.

### 2. Partitioning by Action Type & Dynamic Envelopes

Partition each lane by its scheduled maturity action type:
- **Package-Isolated Actions** (`implement`, `test`, `benchmark`, `clean comments`): Run directly in parallel fenced subagents bounded strictly to their package directory (`internal/<lane>/`, `pkg/<lane>/`, or `platform/<lane>/`).
- **Integration Actions** (`integrate` into `cmd/`, `dogfood` in `runtime-proofs.json`): Require cross-package coordinator wiring. Do not assign cross-package modifications to package-fenced subagents; wire integration actions from the coordinator.
- **High-Leverage Infrastructure (10/50/100 Dynamic Envelopes)**: For critical subsystems (`session`, `recall`, `vcache`, `ctxmmu`, `dispatch`), assign dynamic verification envelopes to ensure concurrency stress, fuzzing, and empirical dogfooding receipts.

### 3. Select Specialized Subagent Type

- Standard package leaves (`implement`, `test`, `benchmark`) → `subagent_type="worker"`.
- Complex serial singletons (`internal/kernel`, `internal/adjudicator`, `internal/abi`, `platform/gateway`) → `subagent_type="deep-reason"`.

### 4. Launch High-Concurrency Parallel Subagents (Single Turn)

Launch all admitted workers in parallel in a single turn using multiple `task` tool calls with `subagent_type="worker"` under strict repo fences (using Gemini 3.8 Flash high `variant: high` for hard work). Inject the lane's actual `<next_action>` into the worker prompt:

#### For a `fak` Public Lane:
```json
{
  "tool": "task",
  "parameters": {
    "subagent_type": "worker",
    "description": "Mature fak public lane: internal/<lane>",
    "prompt": "You are an isolated worker for public repo fak at `internal/<lane>`.\nGoal: Fulfill next action: <next_action>.\nTarget Repo: fak (path: .)\nBoundaries: Edit ONLY files under internal/<lane>/. NEVER touch fak-private, go.mod, go.sum, dos.toml, or root files.\nVerification: Run ONLY package-scoped tests: `go test -v ./internal/<lane>` and `go vet ./internal/<lane>`. NEVER run `go test ./...`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
  }
}
```

#### For a `fak-private` Private Lane:
```json
{
  "tool": "task",
  "parameters": {
    "subagent_type": "worker",
    "description": "Mature fak-private lane: platform/<lane>",
    "prompt": "You are an isolated worker for private repo fak-private at `platform/<lane>`.\nGoal: Fulfill next action: <next_action>.\nTarget Repo: fak-private (path: ..\\fak-private)\nBoundaries: Edit ONLY files inside `..\\fak-private\\platform\\<lane>/`. NEVER touch public fak, go.mod, go.sum, or root files.\nVerification: Run ONLY package tests: `go -C ..\\fak-private test -v ./platform/<lane>` and `go -C ..\\fak-private vet ./platform/<lane>`.\nReturn: Return a compact 3-line receipt: (1) files created/edited, (2) package test result, (3) new maturity facts."
  }
}
```

---

## Phase 4 — Wave Harvest, Verification, Leak Audit, Commit & Lease Release

When the parallel subagents finish:

1. **Audit Receipts**: Verify each subagent reported green tests strictly within its declared package.
2. **Parallel Adversarial Verification & Gap Audit**:
   Launch parallel `cross-validator` and `issue-auditor` subagents to independently evaluate the changes:
   - `cross-validator`: Independently verifies git diff boundaries, executes on-device package tests, and produces DOS-style proof verdicts.
   - `issue-auditor`: Inspects diffs for unhandled edge cases, failure modes, and QA gaps, preventing newly introduced debt.
3. **Independent Verification**: Coordinator independently runs package-scoped checks:
   - For `fak` lanes:
     ```bash
     go vet ./internal/<lane>
     go test -v ./internal/<lane>
     ```
   - For `fak-private` lanes:
     ```bash
     go -C ..\fak-private vet ./platform/<lane>
     go -C ..\fak-private test -v ./platform/<lane>
     ```
4. **Maturity Verification**:
   ```bash
   fak debt-orchestrator --target-repo both --lane <lane>
   ```
   Confirm maturity score climbed, gap decreased, and debt principal dropped.
5. **Code-Debt Hygiene Check**:
   Run pre-commit code-debt hygiene check to ensure no god-functions (>1500 lines) or complexity traps were introduced:
   ```bash
   fak code-debt --path internal/<lane>
   ```
6. **Mandatory Leak Audit (For `fak` Public Commits)**:
   Before staging or committing any public change, execute the leak scrubber audit:
   ```bash
   git add internal/<lane>
   python tools/scrub_public_copy.py --audit-staged --root .
   ```
   *MUST pass with 0 exit code before proceeding.*
7. **Commit by Explicit Path with Distinct Repo Trailers**:
   - For public `fak` commits:
     ```bash
     fak commit --path internal/<lane> -m "fix(<lane>): advance maturity curve and retire debt (fak <lane>)"
     ```
     *(Fallback: `git commit -s -m "fix(<lane>): advance maturity curve and retire debt (fak <lane>)" -- internal/<lane>` without `git add -A`).*
   - For private `fak-private` commits:
     ```bash
     git -C ..\fak-private commit -s -m "fix(<lane>): advance maturity curve and retire debt (fak-private <lane>)" -- platform/<lane>
     ```
8. **Release Lane Leases**:
   Release each lane lease after commit so leases do not leak into `Held Lanes Excluded` in subsequent waves or peer runs:
   ```bash
   # For fak lanes:
   dos lease-lane release --workspace . --lane <lane>

   # For fak-private lanes:
   dos lease-lane release --workspace ..\fak-private --lane <lane>
   ```

---

## Phase 5 — Burndown Check, O(1) Context Carryover & Autonomous Multi-Wave Loop

Compare against the scratch baseline to measure campaign velocity:

```bash
fak debt-orchestrator --target-repo both --compare _scratch/debt-orchestrator/baseline.json
```

Check the delta:
- Did `Production Grade` increase toward the campaign goal?
- Did `Total Debt` drop across both repositories?
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
     fak debt-orchestrator --target-repo both --wave-size 6 --max-waves 8
     ```
     to generate the next wave cohort and continue the loop.
  5. If a lane encounters transient lock contention or peer lease hold, skip it, advance the next disjoint candidate lane from the wave plan, and sustain campaign momentum.
