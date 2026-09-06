---
name: debt-clean
description: One repeatable, evidence-backed pass that retires a bounded batch of maturity debt worst-first across the system's dedicated debt lanes. Features rich queryability (--query, --health), cross-indexes related items (dual-repo companions, tests, runtime-proofs, benchmarks, inbound blast radius), targets high-carrying-cost or degraded hotspots, advances maturity with tests, integration, and benchmarks, re-measures with --compare to prove the denominator was level-set, and commits by explicit path with (fak <leaf>). Use when cleaning or retiring maturity debt across units of work.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--lane <name>] [--query <text>] [--health healthy|degraded|critical] [--cross-index] [--target-repo both|fak|fak-private] [--top N]  (no args = baseline + clean top 1-3 debt hotspots)"
---

# /debt-clean — retire maturity debt worst-first in bounded batches

> **What this does.** Every unit of work (feature, module, leaf) in the system carries a
> dedicated debt lane scored on a continuous 0.0–10.0 maturity curve. When new work is added
> at a partial maturity (e.g. 1/10 stub), it immediately enters the production grade
> denominator as debt with an accrued carrying cost.
>
> This skill executes a **bounded, disciplined cleaning pass**: query and baseline the debt lanes,
> cross-index related things (dual-repo companions, dependents, proofs, benchmarks), run pre-flight
> health diagnostics, pick 1–3 high-leverage hotspots (worst carrying cost, critical compounding interest,
> or degraded health), fulfill the exact next action that advances the maturity curve, prove the debt dropped
> and health upgraded with `--compare`, and commit by explicit path.

The shape: **query & baseline debt lanes → cross-index related things & pre-flight health → pick bounded batch (1–3 units) → execute next action
(tests/integration/benchmarks/clean bloat) → verify package → prove debt drop & health upgrade with `--compare` → commit by
explicit path.**

---

## The Rule of Bounded Batches

Never attempt a repository-wide sweep in one pass (hundreds of WIP lanes cannot and should
not be rushed in a single turn). Enforce strict scoping:

1. **Max 1–3 units of work per batch**: Focus on 1–3 closely related leaves or a single
   critical hotspot.
2. **Shift-left proof**: Every maturity climb requires genuine disk evidence (passing
   `*_test.go`, benchmarks, real runtime wiring), never synthetic markers or mock-only claims.
3. **No comment gaming; excess comments is BAD debt**: Comments are NOT tracked as positive
   debt retirement. Do not add formulaic "Contract:", "Invariant:", or "Fail-closed:" comment headers
   or restate syntax. In fact, excess comments and comment bloat (>35% comment ratio or formulaic
   keyword clutter) are penalized as BAD debt with interest rate penalties.
4. **Denominator honesty**: Advancing maturity increases realized points without shrinking
   the denominator, genuinely raising the production-grade percentage.
5. **Multi-dimensional health upgrade**: Debt retirement must elevate the lane's health status
   from `degraded` or `critical` toward `healthy`, resolving flagged issue tokens.

---

## Step 1 — Measure Baseline & Query Health

Run `fak debt-lanes` to capture the current state, query candidate work, and audit system health:

```bash
# Capture full baseline:
fak debt-lanes --json > baseline.json
fak debt-lanes --top 5

# Query lanes by keyword (matches lane name, unit of work, companion, drivers, or health issues):
fak debt-lanes --query <keyword> --top 5

# Filter lanes by multi-dimensional health verdict (healthy, degraded, critical):
fak debt-lanes --health degraded,critical --top 5

# Cross-index related things, dual-repo companions, and blast radius:
fak debt-lanes --cross-index --top 10
```

### Multi-Dimensional Lane Health Dimensions
- **`healthy`**: Score >= 0.85, test passing, clean comment hygiene, integrated into production graph, documented/benchmarked if core, low carrying interest.
- **`degraded`**: Untested stubs (`missing_tests`), excess comment bloat (`comment_ratio > 35%` or formulaic keyword noise), disconnected wiring (`integrated=false`), unbenchmarked, or unproven in runtime.
- **`critical`**: Compounding carrying cost (>25% interest rate) or untested core paths (`criticality=core && has_tests=false`) with high inbound dependents.

Note the headline figures:
- **Production Grade**: current letter and percentage (e.g. `Grade C, 75.4%`).
- **Fleet Health**: healthy, degraded, and critical counts with average health score.
- **Denominator Points**: total production baseline points.
- **Total Debt**: total principal + carrying cost.
- **Top Hotspots**: lanes carrying the highest carrying cost or highest maturity gap.

---

## Step 2 — Cross-Index Related Things & Select Target (1–3 Units)

Inspect the specific target's status, evidence, health findings, and cross-indexed artifacts:

```bash
fak debt-lanes --lane <lane-name> --json
```

### Cross-Indexing Checklist for Target Lane
1. **Dual-Repo Companion**:
   - If working in `fak` (`internal/<lane>`), check `Related.CompanionUnitOfWork` for counterpart in `fak-private` (`platform/<lane>`).
   - If working in `fak-private`, check counterpart in `fak` (`internal/<lane>` or `pkg/<lane>`).
   - Run `go run ./cmd/fak-boundary explain <unit_of_work>` before making cross-repo edits to ensure 5-gate compliance.
2. **Blast Radius & Inbound Dependents**:
   - Inspect `Related.Dependents` (packages importing this lane). Core leaves with >10 dependents require non-breaking public APIs.
   - Inspect `Related.Dependencies` (packages imported by this lane).
3. **Registered DOS Trees**:
   - Check `Related.DosTrees` declared in `dos.toml` to verify directory boundaries for subagent task fencing.
4. **Witness Proofs & Benchmarks**:
   - Check `Related.ProofWitnesses` against `internal/maturity/runtime-proofs.json`.
   - Check `Related.BenchmarkWitnesses` against `BENCHMARK-AUTHORITY.md`.

Choose targets based on leverage:
- **Option A: High Carrying Risk (Critical Compounding Interest)**:
  Lanes where `interest.band == "critical"` (>25% rate) or `health.status == "critical"`.
- **Option B: Untested / Skeleton Stubs (Highest Maturity Gap)**:
  Lanes sitting at `0.0` or `1.0/8.0` (`proposed` or `stub`). Adding core types and tests jumps maturity from `1.0` to `4.0+`.
- **Option C: Comment Bloat / Hygiene Cleanup**:
  Lanes flagged with `excess_comments` where pruning formulaic comments restores clean structure and cuts bad debt interest.

Read the lane's `next_action` field—this is the exact action required to climb the curve.

---

## Step 3 — Delegate to Subagent (Keep Coordinator Context Clean)

Per `AGENTS.md`, the primary agent coordinates; substantive investigation, code editing, and test runs are delegated to isolated subagents via the `task` tool so coordinator context stays clean and focused.

Launch a targeted subagent:
```json
{
  "subagent_type": "general",
  "description": "Clean debt in <lane>",
  "prompt": "You are an isolated worker advancing the maturity curve for <unit_of_work> (internal/<lane> or pkg/<lane>).\nGoal: fulfill the exact next action: <next_action>.\nTarget Directory: <unit_of_work>\nBoundaries: Edit ONLY files inside <unit_of_work>/. Do not touch sibling packages.\nVerification: Execute `go test -v ./<unit_of_work>` and `go vet ./<unit_of_work>`.\nReturn Contract: Return a compact receipt with: (1) files modified/created, (2) test result receipt, (3) brief 1-line rationale. Do not return full command logs."
}
```

The subagent executes the concrete work needed:
- **If missing tests (`HasTests == false`)**:
  Writes real unit tests (`<name>_test.go`) in `<unit_of_work>/`. Covers positive and
  negative branches, edge cases, and ensures `go test ./<unit_of_work>/...` passes cleanly.

- **If missing implementation (`HasCode == false` or stub)**:
  Implements core functions, types, and logic in `<unit_of_work>/`. Adheres to existing
  patterns and keeps the package focused.

- **If unintegrated (`Integrated == false`)**:
  Connects the package into production commands or registration graph.

- **If undogfooded (`Dogfooded == false`)**:
  Executes real runtime path and captures passing proof in `internal/maturity/runtime-proofs.json`.

- **If unbenchmarked (`Benchmarked == false`)**:
  Adds substantive `Benchmark*` functions with `b.N` loops measuring production operations.

- **If carrying excess comments (`ExcessComments == true`)**:
  Prunes formulaic comment bloat, redundant syntax explanations, and keyword clutter. Code
  must be clean and self-documenting; invariants belong in tests and code assertions. NEVER
  add formulaic "Contract:", "Invariant:", "Fail-closed:" comments.

---

## Step 4 — Coordinator Independent Witnessing

The coordinator does NOT trust worker narration—it independently witnesses the result (the `dos-witness-claim` invariant):

1. **Verify toolchain execution**:
   - For public `fak` units (`internal/<target>` or `pkg/<target>`):
     ```bash
     go vet ./<unit_of_work>
     go test -v ./<unit_of_work>
     ```
   - For private `fak-private` units (`platform/<target>`):
     ```bash
     go -C ..\fak-private vet ./platform/<target>
     go -C ..\fak-private test -v ./platform/<target>
     ```

2. **Verify maturity curve ascent & health upgrade**:
   ```bash
   fak debt-lanes --lane <target> --json
   ```

Verify that:
- `Maturity` increased (e.g. from `1.0` or `2.0` to `4.0` or `7.5+`).
- `MaturityGap` decreased.
- `DebtPrincipal` and `CarryingCost` dropped.
- `RealizedContribution` increased.
- `Health.Status` improved (e.g. from `critical` to `degraded`, or `degraded` to `healthy`).
- `Health.Issues` resolved target issue tokens (e.g. `missing_tests`, `excess_comments`, `disconnected_wiring`).

---

## Step 5 — Re-Measure and Prove Debt Drop & Health Upgrade

Run the scorecard against the saved baseline to prove the drop:

```bash
fak debt-lanes --compare baseline.json
fak debt-lanes --lane <target>
```

Confirm:
- `Total Debt` shows a negative delta (`-X.X pts`).
- `Health Status` shows the lane upgraded with no regression in companion lanes.
- `Production Grade` percentage improved or held steady.
- `Denominator` remained stable (no work was deleted to cheat the score).
- `WIP Dilution` decreased.

---

## Step 6 — Commit Cleanly by Explicit Path

Follow trunk discipline: stage only the explicit paths modified for this single unit of
work. Include a Conventional-Commits subject, signed-off DCO (`-s`), and the required repo trailer:

- **For public `fak` commits**:
  ```bash
  fak sync check
  python tools/scrub_public_copy.py --audit-staged --root .
  fak commit --path <unit_of_work> -m "fix(<target>): advance maturity curve and retire debt (fak <target>)" [--push]
  fak sync push
  ```
- **For private `fak-private` commits**:
  ```bash
  git -C ..\fak-private add platform/<target>
  git -C ..\fak-private commit -s -m "fix(<target>): advance maturity curve and retire debt (fak-private <target>)" -- platform/<target>
  ```

Verify `git status` shows your lane is committed and no peer WIP was touched.
