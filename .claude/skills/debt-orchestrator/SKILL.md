---
name: debt-orchestrator
description: Coordinate bounded, evidence-backed maturity debt work in the current repository, with isolated workers and independent verification. Use for debt burndowns or explicitly requested sustained campaigns.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[repository or lane] [--wave-size N] [--perf-focus] [--opencode-commands] [--spawn-opencode] (no args = one bounded pass)"
metadata:
  opencode: agent-permission
---

# /debt-orchestrator — Multi-Agent Maturity Debt Burndown & Performance Acceleration

Retire concrete maturity debt gaps across the codebase using verified lane leases, tree-disjoint multi-agent wave dispatch (via in-harness subagents or OpenCode leaf workers), and rigorous 3x harsher standards on factors driving long-term continual performance gains.

The shape: **baseline debt lanes (`fak debt-lanes --json`) → plan concurrent-safe waves (`fak debt-orchestrator --plan-waves --perf-focus`) → arbitrate leases (`dos arbitrate`) → parallel wave dispatch (in-harness `task` or OpenCode leaf workers via `--opencode-commands`) → harvest & independently witness → autonomous safe git landing (`fak sync`, `fak commit --path`, `fak sync push`) → compare burndown (`--compare`) → loop.**

---

## Default 9-Surface Class Discovery

Discovery scans all **9 standard surface classes** by default across the repository, ensuring complete workspace inventory without hiding peripheral, tool, or stewardship debt:

1. **`internal`**: Core runtime, mediation, security, and compute packages.
2. **`pkg`**: Public shared packages and client SDKs.
3. **`platform`**: Hardware and OS platform-specific implementations.
4. **`cmd`**: Command-line interface binaries and verbs.
5. **`tools`**: Internal developer tooling and submodules.
6. **`skills`**: Agent skill packs (`.claude/skills`, `.agents/skills`).
7. **`workflows`**: CI/CD workflows (`.github/workflows`).
8. **`examples`**: Example code and runnable demos.
9. **`docs`**: Technical architecture and reference documentation.

To restrict scanning to package roots only (`internal`, `pkg`, `platform`), pass `--no-expanded-surfaces` (or `--expanded-surfaces=false`).

---

## The Debt Generation Lifecycle

Maturity debt is an active, deterministic accounting system driven by three core lifecycle mechanisms:

1. **1/10 Stub Admission**:
   - Every new package or unit of work added to the codebase is immediately admitted to the production denominator at **1/10th (10%) target maturity** admission.
   - Stubs cannot hide unmeasured: creating a skeleton directory or placeholder package instantly claims its production ceiling in the denominator while contributing minimal realized points.
   - This ensures immediate structural visibility for new code and blocks phantom growth.

2. **Dilution from Immature WIP**:
   - Any incomplete, unhardened, or untested work-in-progress expands the production denominator (`DenominatorPoints`) while contributing fractionally to realized points (`RealizedPoints`).
   - This creates measurable **dilution from WIP** (`DilutionFromWIP`), mathematically lowering the system-wide production grade (`GradePercent`) and grade letter until the unit reaches target maturity.
   - Dilution forces teams to finish and harden existing WIP before opening new speculative surfaces.

3. **Dynamic Detector Debt**:
   - 15 independent, typed detector dimensions continuously inspect all scanned surfaces on disk.
   - Detectors inspect unit code, test files, benchmarks, and configuration for structural defects:
     - **Thin tests**: test files with zero assertions or mock-only passes.
     - **Stub markers**: `TODO`, `FIXME`, `XXX`, or `panic("not implemented")` calls (safely scanned across lines >64KB).
     - **Modularity deficits**: god-files (>1500 lines) and god-functions (>200 lines).
     - **Formulaic comment gaming**: "Contract:"/"Invariant:" stuffing or excessive comment bloat (>35%).
     - **Performance hazards**: unbenchmarked core/enabling paths or missing runtime proofs.
     - **Hygiene & safety**: unsafe pointer arithmetic, subprocess execution, undocumented exports, race/fuzz gaps.
   - Findings dynamically attach carrying cost interest surcharges (elevating the rate up to the critical compounding band >25%) and degrade health verdicts (`healthy` → `degraded` → `critical`).

---

## The 3x Harsher Performance Discipline

To ensure that debt retirement translates directly into long-term continual performance gains (rather than synthetic metric-chasing or comment padding), the debt orchestrator enforces five uncompromising performance gates:

1. **Substantive Benchmarks Mandatory (Core & Enabling)**:
   - Any Core or Enabling lane lacking substantive benchmarks (`!Benchmarked`) incurs an automatic interest carrying cost surcharge (`unbenchmarked_core_perf_hazard (+0.06)` / `unbenchmarked_enabling_perf_hazard (+0.04)`) and a 3x harsher health deduction (-0.15).
   - Core runtime paths cannot achieve `healthy` status without passing benchmarks.
   - Benchmark functions MUST use `b.Run` or loop over `b.N` in `<unit_of_work>/*_test.go`; empty loops or mocked timers are rejected by structure.
   - Baseline results must be registered in `BENCHMARK-AUTHORITY.md`.

2. **Real Runtime Proofs & Dogfooding (Mocks Prohibited)**:
   - "Mocks hide integration bugs" (Hermes' rule). Core and Enabling paths without real loopback execution or dogfooding (`!Dogfooded`) incur carrying cost surcharges (`unproven_runtime_core_perf_hazard (+0.05)`) and 3x harsher health deductions (-0.15).
   - Real runtime proof must be wired into `runtime-proofs.json` or verified through on-device integration tests (`*integration*test.go`).
   - Purely synthetic mock-only tests do not count toward debt retirement.

3. **Zero Tolerance for Modularity Deficits & God-Constructs**:
   - God-files (>1500 lines) and god-functions (>200 lines), model hardcoding, and tight coupling carry severe compounding interest surcharges (+0.08 to +0.10) and block advancing to `hardened` or `production_grade` rungs.
   - These monolithic blocks prevent compiler vectorization, SIMD/Vulkan/Metal acceleration, and clean parallelization.
   - Workers must decompose oversized files into single-responsibility units and remove model coupling.

4. **Zero Tolerance for Formulaic Comment Gaming**:
   - Comments do NOT award maturity points. Adding formulaic "Contract:", "Invariant:", or "Fail-closed:" headers is flagged as comment gaming and penalized with interest rate surcharges.
   - Comment bloat (>35% comment ratio) is classified as bad debt and prevents promotion to production grade.

5. **Performance-Focused Wave Prioritization (`--perf-focus`)**:
   - When `--perf-focus` (or `--harsher-perf`) is passed to `fak debt-orchestrator`, wave planning sorts Core and Enabling lanes with performance debt (unbenchmarked, unproven runtime, modularity deficits) ahead of peripheral debt, regardless of raw principal totals.

---

## Scope and Stop Conditions

- **Default pass**: 1 bounded pass with at most 1 admitted wave (4–8 small outcomes across verified disjoint lanes).
- **Scale to hardware**: Do not exceed physical CPU cores or create thrashing.
- **Sustained campaigns**: Continue through verified waves until the target grade or target points are achieved, budget is exhausted, or a checkable blocker is encountered.
- **Inspect active ownership**: Skip held work (`.dos` leases) and leave a resumable receipt. Never launch duplicate workers on held lanes.

---

## Step 1 — Measure Baseline & Plan Safe Waves

Capture baseline state and plan concurrent-safe waves:

```bash
# Capture full baseline:
fak debt-lanes --workspace <repo-root> --json > baseline.json
fak debt-lanes --top 5

# Plan waves with 3x harsher performance focus:
fak debt-orchestrator --workspace <repo-root> --wave-size 4 --max-waves 2 --perf-focus

# Plan waves with ready-to-run OpenCode leaf worker commands:
fak debt-orchestrator --workspace <repo-root> --wave-size 4 --max-waves 2 --perf-focus --opencode-commands

# Emit machine-readable wave plan JSON:
fak debt-orchestrator --workspace <repo-root> --wave-size 4 --max-waves 2 --perf-focus --opencode-commands --json > wave-plan.json
```

Use filtering flags to target specific debt areas:
- `--no-expanded-surfaces`: disable expanded surfaces; scan package roots only (`internal`, `pkg`, `platform`).
- `--surface <class>`: filter by surface class (`internal`, `pkg`, `platform`, `cmd`, `tools`, `skills`, `workflows`, `examples`, `docs`).
- `--perf-focus` / `--harsher-perf`: prioritize unbenchmarked, unproven, and modularity debt.
- `--health degraded,critical`: target only unhealthy or compounding lanes.
- `--criticality core,enabling`: focus on performance-critical infrastructure.
- `--query <text>`: filter across lane name, unit of work, companion, or health issue tokens (`unbenchmarked`, `unproven_runtime`, `missing_tests`, `modularity_deficit`).
- `--cross-index`: display dual-repo companions, inbound blast radius, and DOS trees.

---

## Step 2 — Pre-Dispatch Lease Arbitration

Before launching workers, verify tree-disjointness and acquire leases using DOS:

```bash
# Adjudicate lane admission:
dos arbitrate --lane <lane> --tree <unit_of_work>/**

# Acquire durable lease:
dos lease-lane acquire --lane <lane> --owner debt-orchestrator --tree <unit_of_work>/**
```

Never dispatch concurrent workers touching overlapping package directories. Shared Go package edits cause immediate compile breakages.

---

## Step 3 — Wave Dispatch (Subagents or OpenCode Leaf Workers)

Choose between in-harness subagents (Option A) or OpenCode leaf workers (Option B):

### Option A: In-Harness Subagents (`task` Tool)

Dispatch concurrent subagents across pairwise tree-disjoint lanes in a single coordinator response:

```text
task subagent_type="worker" prompt="Maturity Debt Lane: gateway (internal/gateway)..."
task subagent_type="worker" prompt="Maturity Debt Lane: compute (internal/compute)..."
task subagent_type="worker" prompt="Maturity Debt Lane: engine (internal/engine)..."
```

Instruct each worker to:
1. Confine edits strictly to `internal/<lane>/...` (1–3 files).
2. Fulfill missing performance deliverables (benchmarks, runtime proofs, modularity cleanup).
3. Verify package tests and benchmarks (`go test -v ./internal/<lane>/...`, `go test -bench=. ./internal/<lane>/...`).
4. Execute autonomous safe git landing (`fak sync check`, `fak commit --path`, `fak sync push`).

### Option B: OpenCode Leaf Workers (`--opencode-commands`)

Use generated OpenCode CLI commands to launch detached, user-visible child workers:

```bash
# Inspect generated commands from wave plan:
fak debt-orchestrator --wave-size 4 --max-waves 1 --perf-focus --opencode-commands

# Example generated invocation:
opencode run --variant high --auto --title "Debt: gateway (internal/gateway)" "Maturity Debt Lane: gateway..."
```

#### Five Invariants of OpenCode Leaf Workers:
1. **High Reasoning Effort (`--variant high`)**: Child workers run with Gemini 3.8 Flash high reasoning mode for rigorous verification.
2. **Direct Leaf Execution (No Nested Tasks)**: Leaf workers execute deliverables directly within package boundaries. Calling the `task` tool or attempting nested delegation is strictly prohibited (prevents subagent depth limit recursion failure, #12028).
3. **Automated Approvals (`--auto`)**: Workers run unattended with permissions granted for local file operations.
4. **Mandatory 4-Phase Delivery & Landing Pipeline**:
   - Phase 1 [Implement]: Author tests, benchmarks, runtime wiring, or modularity refactor.
   - Phase 2 [Verify]: Run package verification commands (`go test`, `go vet`, benchmarks).
   - Phase 3 [Autonomous Safe Git Landing]: Commit by explicit path and push unprompted (`fak sync check`, `fak commit --path`, `fak sync push`).
   - Phase 4 [Receipt]: Output compact 3-line receipt upon completion.
5. **Milestone Progress Reporting**: Report progress via structured comment tags (`<!-- fak:progress milestone="..." delta="..." tests="..." -->`).

---

## Step 4 — Witness and Harvest

Workers return compact receipts. The coordinator independently verifies delivered evidence:

1. **Verify Git Commits**: Confirm commits landed on trunk with valid trailers (`(fak <lane>)`):
   ```bash
   git log -n 5 --oneline
   ```
2. **Verify Performance Witnesses**: Run benchmarks and tests on the touched packages:
   ```bash
   go test -v ./internal/<lane>/...
   go test -bench=. -benchmem ./internal/<lane>/...
   go vet ./internal/<lane>/...
   ```
3. **Check Denominator Honesty**: Confirm the production grade increased without shrinking the denominator:
   ```bash
   fak debt-lanes --workspace <repo-root> --compare baseline.json
   ```

---

## Step 5 — Close the Pass & Release Leases

1. Release finished leases:
   ```bash
   dos lease-lane release --lane <lane> --owner debt-orchestrator
   ```
2. Clean up allocated scratch directories:
   ```bash
   fak tree-doctor --reap-scratch debt-orchestrator --json
   ```
3. Emit compact verdict receipt (<3 lines):
   - **Line 1**: Summary of retired debt, grade delta, and waves completed.
   - **Line 2**: List of matured lanes, touched paths, and landed commit SHAs.
   - **Line 3**: Checkable verification command.

---

## Verification and Witness

Verify this skill definition using the project gates:

```bash
# 1. Structural admission and anti-slop verification:
python tools/skill_slop_scorecard.py .claude/skills/debt-orchestrator/SKILL.md --corpus .claude/skills

# 2. Frontmatter portability check:
python tools/skill_frontmatter_lint.py --check

# 3. Synchronize cross-harness adapter for OpenCode (.agents/skills/debt-orchestrator/SKILL.md):
go run ./cmd/fak-project-assets sync --json

# 4. Verify zero unexplained parity gaps across harnesses:
go run ./cmd/fak-project-assets parity --json
```
