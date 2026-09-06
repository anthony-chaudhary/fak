---
name: debt-clean
description: One repeatable, evidence-backed pass that retires a bounded batch of maturity debt worst-first across the system's dedicated debt lanes. Inspects `fak debt-lanes`, targets the highest-carrying-cost hotspots (or compounding-interest core lanes), advances maturity with tests, integration, and benchmarks, re-measures with `--compare` to prove the denominator was level-set and total debt dropped, and commits by explicit path with `(fak <leaf>)`. Use when cleaning or retiring maturity debt across units of work.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--lane <name>] [--top N] [--critical-only]  (no args = baseline + clean top 1-3 debt hotspots)"
---

# /debt-clean — retire maturity debt worst-first in bounded batches

> **What this does.** Every unit of work (feature, module, leaf) in the system carries a
> dedicated debt lane scored on a continuous 0.0–10.0 maturity curve. When new work is added
> at a partial maturity (e.g. 1/10 stub), it immediately enters the production grade
> denominator as debt with an accrued carrying cost.
>
> This skill executes a **bounded, disciplined cleaning pass**: baseline the debt lanes,
> pick 1–3 high-leverage hotspots (worst carrying cost or critical compounding interest),
> fulfill the exact next action that advances the maturity curve, prove the debt dropped with
> `--compare`, and commit by explicit path.

The shape: **baseline debt lanes → pick bounded batch (1–3 units) → execute next action
(tests/integration/benchmarks/clean bloat) → verify package → prove debt drop with `--compare` → commit by
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

---

## Step 1 — Measure Baseline

Run `fak debt-lanes` to capture the current state and identify top hotspots:

```bash
fak debt-lanes --json > baseline.json
fak debt-lanes --top 5
```

Note the headline figures:
- **Production Grade**: current letter and percentage (e.g. `Grade C, 75.4%`).
- **Denominator Points**: total production baseline points.
- **Total Debt**: total principal + carrying cost.
- **Top Hotspots**: lanes carrying the highest carrying cost or highest maturity gap.

---

## Step 2 — Select Bounded Target (1–3 Units)

Choose targets based on leverage:

- **Option A: High Carrying Risk (Critical Compounding Interest)**:
  Lanes where `interest.band == "critical"` (>25% rate) because they are core runtime
  modules with high inbound dependents (e.g. `internal/abi`, `internal/adjudicator`).
  Retiring even 1.0–2.0 points of gap here yields massive carrying-cost relief.

- **Option B: Untested / Skeleton Stubs (Highest Maturity Gap)**:
  Lanes sitting at `0.0` or `1.0/8.0` (`proposed` or `stub`). Writing their initial core
  declarations and unit tests jumps their maturity from `1.0` to `4.0+`, cutting their gap
  in half.

Inspect the specific target's status and evidence:

```bash
fak debt-lanes --lane <lane-name> --json
```

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

2. **Verify maturity curve ascent**:
   ```bash
   fak debt-lanes --lane <target>
   ```

Verify that:
- `Maturity` increased (e.g. from `1.0` or `2.0` to `4.0` or `7.5+`).
- `MaturityGap` decreased.
- `DebtPrincipal` and `CarryingCost` dropped.
- `RealizedContribution` increased.

---

## Step 5 — Re-Measure and Prove Debt Drop

Run the scorecard against the saved baseline to prove the drop:

```bash
fak debt-lanes --compare baseline.json
```

Confirm:
- `Total Debt` shows a negative delta (`-X.X pts`).
- `Production Grade` percentage improved or held steady.
- `Denominator` remained stable (no work was deleted to cheat the score).
- `WIP Dilution` decreased.

---

## Step 6 — Commit Cleanly by Explicit Path

Follow trunk discipline: stage only the explicit paths modified for this single unit of
work. Include a Conventional-Commits subject, signed-off DCO (`-s`), and the required repo trailer:

- **For public `fak` commits**:
  ```bash
  git add <unit_of_work>
  python tools/scrub_public_copy.py --audit-staged --root .
  git commit -s -m "fix(<target>): advance maturity curve and retire debt (fak <target>)" -- <unit_of_work>
  ```
- **For private `fak-private` commits**:
  ```bash
  git -C ..\fak-private add platform/<target>
  git -C ..\fak-private commit -s -m "fix(<target>): advance maturity curve and retire debt (fak-private <target>)" -- platform/<target>
  ```

Verify `git status` shows your lane is committed and no peer WIP was touched.
