# The Multi-Dimensional Maturity Curve: Beyond Basic Tests to True Verification

**Date:** 2026-09-03  
**Status:** Canonical operational & architectural doctrine  
**Scope:** Agent fleet dispatch, issue fan-out, capability lifecycles, and verification gates  

---

## 1. The Core Tension: Scope Limiting vs. Shipping "Raw"

To prevent autonomous coding agents and smaller models from hallucinating, context-exhausting, or creating sprawling, unreviewable diffs, repository doctrine enforces **strict scope discipline**:
- **Atomic units (S0/S1 leaves)**: Restrict write surfaces to 1–3 files within a single package or lane.
- **Spine first**: Ship the smallest working end-to-end path in the initial session.

While this discipline is essential, its historical operational side-effect was that **capabilities shipped "raw" and were promptly abandoned**:
1. A worker landed a minimal prototype or raw spine with a single basic unit test.
2. The commit was stamped `(fak <leaf>)`, `dos verify` marked the phase shipped, and the tracking issue closed.
3. The capability was left permanently on lower lifecycle rungs: in the repository's maturity scorecard, **575 out of 583 capabilities were bottlenecked at `tested`**, with only 5 having registered runtime proofs and 0 having default surfaces.
4. Because no automated trigger pushed the capability further, the raw code was never integrated, profiled, or tested under real agent workloads.

**The Invariant:** Limiting scope to an initial raw spine is the **first step** of a capability's life, never its terminal state. When a scoped leaf lands, the system must automatically generate structured successor issues across a multi-dimensional maturity curve and dispatch agents to advance it.

---

## 2. The 6-Dimensional Maturity Curve

Moving beyond the binary `"a *_test.go exists"` check requires evaluating capabilities across six concrete dimensions:

```
                  ┌───────────────────────────────────────────────────────────┐
                  │              The 6-Dimensional Maturity Curve             │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
       ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
       ▼                  ▼                     ▼                     ▼                  ▼
┌──────────────┐   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   ┌──────────────┐
│  DIMENSION 1 │   │  DIMENSION 2 │      │  DIMENSION 3 │      │  DIMENSION 4 │   │  DIMENSION 5 │
│Composability │   │  Shift-Left  │      │Multi-Faceted │      │ Performance  │   │  Sim / Gym / │
│& Integration │   │& Debt Hygiene│      │  Dogfooding  │      │ & Benchmarks │   │ Agent Evals  │
└──────────────┘   └──────────────┘      └──────────────┘      └──────────────┘   └──────────────┘
```

### Dimension 0: Unit & Boundary Verification (The Test Witness)
- **Defect Repro Witness**: Bug fixes and behavior changes must include a test that fails on pre-change trunk and passes on the patch.
- **Refusal & Error Coverage**: Assert every error return and recovery message (no silent `nil` drops).
- **Determinism & Race Cleanliness**: Byte-identical execution across runs and `-race` clean.

### Dimension 1: Composability & Cross-Subsystem Integration
- **Loopback Integration**: Real multi-package composition without mocks (e.g. `leaseref` ➔ `blastlease` ➔ `knownbad`). Hermes' rule: *"Mocks hide integration bugs."*
- **Cross-Layer Contracts**: Proving that downstream packages consume upstream receipts without assumptions or type drift.
- **Auto-Fanned Issue**: `test(<leaf>): add loopback composability test with <upstream-caller>`.

### Dimension 2: Shift-Left Invariants & Debt Hygiene
- **Architectural Tier Cleanliness**: Strict compliance with `internal/architest` tier definitions (no upward imports).
- **Swallowed Error Elimination**: Zero unhandled errors; all refusals must resolve to tokens in `dos.toml [reasons]`.
- **Ephemeral Resource Containment**: Zero unmanaged background daemons (no LaunchAgents with `KeepAlive=true`, no persistent systemd units). Ephemeral child lifecycles bound to `trap cleanup EXIT INT TERM` and zero resident idle memory.
- **Auto-Fanned Issue**: `shift-left(<leaf>): audit architest tier, error floor, and ephemeral process containment`.

### Dimension 3: Multi-Faceted Operational Dogfooding
- **Tier 1 (CLI Seam)**: Reachable through a production `cmd/fak` verb.
- **Tier 2 (Runtime Proof)**: Passing deterministic command registered in `internal/maturity/runtime-proofs.json`.
- **Tier 3 (Self-Hosting)**: `fak` executes the capability on its own source code, git commits, or agent dispatch sessions.
- **Tier 4 (Telemetry Ledger)**: Durable JSONL ledger tracking invocations, hit rates, and error distributions on real workloads.
- **Auto-Fanned Issue**: `dogfood(<leaf>): exercise live runtime execution and record in runtime-proofs.json`.

### Dimension 4: Performance, Allocation & Net-True Benchmarks
- **Micro-Benchmarks**: `func Benchmark*` asserting strict `ns/op` and `B/op` budgets under `testing.AllocsPerRun`.
- **Zero-Allocation Hot Paths**: Bounded allocations per record (e.g. sublinear scaling $\mathcal{O}(N)$ on parsers).
- **Net-True Accounting**: Matched-envelope speedups (setup + inference + recovery overhead) beating baseline.
- **Authority Promotion**: Promoting measured gains to `BENCHMARK-AUTHORITY.md` and tagging `CLAIMS.md`.
- **Auto-Fanned Issue**: `bench(<leaf>): add allocation micro-benchmarks and latency budget profile`.

### Dimension 5: Gyms, Simulations & Agent Evals
- **Deterministic Fault Injection**: Chaos simulation testing mid-line stream truncation, malformed UTF-8, binary corruption, and network drops.
- **Agent Evaluation Gyms**: Testing behavior in end-to-end task environments (Terminal-Bench 4 / `tb4bench`, AgentDojo red-team simulations, SWE-bench fixtures).
- **Replay Stability**: Byte-identical determinism when replaying recorded session transcripts.
- **Auto-Fanned Issue**: `eval-gym(<leaf>): add adversarial fault-injection and TB4 evaluation fixture`.

---

## 3. Keeping Commits Light: The Fan-Out Hand-Off

Individual contributors and subagents must **not** be burdened with satisfying all dimensions in a single commit:

1. **Worker lands S0/S1 Spine**:
   - Scope: 1–3 files in `internal/<leaf>/`.
   - Witness: 1 reproduction unit test proving the core through-line works.
2. **Landing Trigger Fires (`fak maturity route --live` / `issue fanout`)**:
   - Decomposes the capability's next lifecycle steps into dedicated GitHub issues tagged with stable marker keys (`<!-- fak-maturity-work-key: ... -->`).
3. **Dispatch Wave Consumes Follow-Ons**:
   - Autonomous workers pick up the specialized follow-ons in parallel waves (one worker on composability, another on allocation benchmarking, another on fault simulation).
4. **Maturity Scorecard Re-Derives Truth**:
   - As witnesses land on disk, `fak maturity` automatically promotes the capability and schedules the next rung.

---

## 4. Empirical Proof Witnessed on `internal/blastlease` (2026-09-03)

This multi-dimensional maturity walk was verified end-to-end on `internal/blastlease`:

1. **Initial State**: Sitting raw at Rung 1 (`prototyped`) with zero tests.
2. **Issue Creation**: `fak maturity route --live` created Issue **#10862** (`maturity(blastlease): add tests for the capability`).
3. **Worker 1 (True Verification)**: Created `internal/blastlease/blastlease_test.go` (`TestRead`, `TestLive` with real git-backed `leaseref`). Promoted to Rung 2 (`tested`).
4. **Follow-On Creation**: Generated targeted issues for the advanced dimensions:
   - Issue **#10872**: Composability integration test with `leaseref` and `knownbad`.
   - Issue **#10874**: Allocation micro-benchmarks and latency budgets.
   - Issue **#10875**: Adversarial fault injection and JSONL truncation recovery.
5. **Parallel Agent Execution**:
   - **Composability Witness**: `composability_test.go` (`TestComposability`) verified `leaseref` ➔ `blastlease.Live` ➔ `knownbad.TreesIntersect` without mocks (0.09s).
   - **Fault Simulation Witness**: `fault_simulation_test.go` (`TestFaultSimulation`) verified fail-closed recovery on mid-line truncations, corrupt types, and oversized 512KB lines.
   - **Performance Witness**: `blastlease_bench_test.go` verified `BenchmarkRead` (166 ns/record, 6.01M records/s) and `TestAllocationBudget` (28 allocs for 10 records, well within `<= 40` budget).
6. **Maturity Re-Derivation**: `fak maturity --json` re-read ground truth from disk and promoted `blastlease.benchmarked = true`.
7. **Closure**: All issues (#10862, #10872, #10874, #10875) closed with non-forgeable test and benchmark outputs.
