---
title: "conceptbench - FAK-concept fidelity benchmark (epic #2721)"
description: "The plan for conceptbench: which model best handles fak's own concepts, measured per concept, and what fak must adapt to raise concept fidelity."
---

# conceptbench — FAK-concept fidelity benchmark (epic #2721)

> Local plan/index for the fan-out filed under **epic #2721**. This file also carries the
> epic's children-checklist (the `gh issue comment` post was blocked by the guard's
> preview-confirm gate this session — captured here instead per operator instruction).
> Filed 2026-07-05. Home: `cmd/conceptbench`; leaf `conceptbench`; ship trailer `(fak conceptbench)`.

## The question this benchmark answers

Which model handles **fak's own concepts** best — per concept — and what must fak *adapt* so
weaker models stay productive? The user's hypothesis: Opus / Fable handle **fak injection
tool-call items** (the adjudicated verdict/disposition stream) better than smaller models right
now. This benchmark makes that checkable, and feeds the model-switching machinery already in the
tree (`cmd/fak/dispatch_model_policy.go` resolver + downgrade chain; `internal/dispatchtick`
`ModelSwitchableReason`, 36185e28).

It is the **third benchmark axis**, complementary to what already exists:
- **Compute** — `cmd/modelbench` (pure-Go forward-pass tok/s, RSS). *How fast* fak runs a model.
- **Raw-vs-fak agentic** — epic #868 (SWE-bench / Terminal-Bench / τ-bench / AgentDojo). *Does fak help.*
- **Concept fidelity (this)** — *which model handles fak's concepts best, per concept*, dos-refereed.

## Why it's trustworthy: deterministic referee, not an LLM judge

Every concept is graded by the **dos kernel** reading git/artifacts — not by a judge model. The
unifying doctrine is **tool-call-is-a-syscall**: the model *proposes* a call, the kernel *disposes*
with a verdict ∈ {ALLOW, DENY, TRANSFORM, QUARANTINE, REQUIRE_WITNESS, DEFER}; a DENY carries a
disposition ∈ {RETRYABLE, WAIT, ESCALATE, TERMINAL}. Handling that injected stream correctly is the
headline of "fak injection tool-call items."

## The concept axis (each row = a benchmark category with a checkable witness)

| # | Concept | What the model must do | dos / fak referee |
|---|---|---|---|
| 1 | Commit-stamp + trunk fidelity | Conventional-Commits + `(fak <leaf>)` trailer + `-s` DCO, by explicit path, on `main` | `dos_verify` + `dos_commit_audit` `diff-witnessed` + stamp grammar + `OFF_TRUNK`=0 |
| 2 | Lane / lease correctness | Pick a disjoint lane before touching a tree | `dos_arbitrate` admits; disjoint-tree rule |
| 3 | Structured refusal | Refuse with a token from the closed vocab, not prose | `dos_check_reason` `known=true` (prose ⇒ `UNCLASSIFIED`) |
| 4 | **Verdict/disposition + call repair** *(headline)* | Right tool + valid args; honor DENY/QUARANTINE; accept a TRANSFORM repaired-arg; recover per disposition; no guard bypass; no hallucinated tool | verdict honored; `toolDescriptors()` resolves; disposition-correct recovery |
| 5 | Guard hook-protocol | Valid `fak.task-handoff.v1` JSON on Stop; survive deny-all continue | handoff-JSON schema validity |
| 6 | Witness/verify honesty | Report `not yet` w/ evidence vs claim-shipped-when-not | `dos_commit_audit` `CLAIM_UNWITNESSED` rate |

Injection/referee surfaces are concrete: tool schemas in `internal/gateway/mcp.go` `toolDescriptors()`
(`fak_adjudicate`/`fak_syscall`/`fak_read`/`fak_admit`/`fak_changes`/…); guard injection in
`cmd/fak/guard_child.go` + `guard_mcp.go` + `guard_toolproc_hooks.go` + `guard.go:132-141`; closed
refusal vocab + stamp grammar in `dos.toml`.

## What we measure

pass@1 fidelity per (model×concept); plus fak-native signals — guard-refusal rate, no-commit
reason-class mix (`internal/dispatchtick/witness.go`), tokens/turn tax, wall-clock. **Honesty gate:**
no leaderboard *claim* until real (non-replay) runs exist — mirrors #868's `result_claim_allowed`.

## What we adapt (the payoff loop)

Measure → find where a model falls off a concept → change fak (simpler injection format / refusal-vocab
hint / `lane_models` pin routing a failed concept to a frontier arm / a new `ModelSwitchableReason`
class) → re-measure. Owned by #2741.

## Filed issues — children checklist (epic #2721)

**Spine (ship first):**
- [ ] #2729 — spine: one concept × two models × dos grade × one report row

**Harness infra:**
- [ ] #2730 — task corpus + fixture schema (`fak.conceptbench.task.v1`)
- [ ] #2731 — model-driver registry (Anthropic gateway + in-kernel serve + raw arm)
- [ ] #2732 — dos-refereed grader adapter (concept → kernel referee)

**Concept scenarios (one per concept, each scenario + grader):**
- [ ] #2733 — commit-stamp + trunk fidelity
- [ ] #2734 — lane / lease correctness (`dos_arbitrate`)
- [ ] #2735 — structured refusal vocabulary (`dos_check_reason`)
- [ ] #2736 — **verdict/disposition handling + tool-call repair (headline)**
- [ ] #2737 — guard hook-protocol compliance (`fak.task-handoff.v1`)
- [ ] #2738 — witness/verify honesty (`dos_commit_audit`)

**Roll-up:**
- [ ] #2739 — report + per-(model×concept) leaderboard + `result_claim_allowed` honesty gate
- [x] #2740 — `cmd/conceptbench` runner + contract mode + catalog workload-kind —
  **shipped**: `cmd/conceptbench` (replay runner + `--contract` honesty gate + core-budget
  knobs + `fak.conceptbench.report.v1`) with a replay-backed smoke test; `concept-benchmark`
  workload-kind wired into `tools/bench_plan.py` (`KIND_META`/`KINDS`/`RECHECK_DAYS`) and
  declared in `experiments/benchmark/catalog.json` `workload_kinds`. Witness: `go test
  ./cmd/conceptbench` green, `python tools/bench_plan_test.py` green (27 tests incl.
  `ConceptBenchmarkKindTest`), and a `bench_plan.py` run surfacing the `concept-benchmark`
  coverage cell on every bench-node.
- [ ] #2741 — analysis: what we learned + which fak affordances to adapt per model
- [ ] #2742 — research: SOTA methodology alignment (τ-bench / BFCL / Terminal-Bench / MCP-Bench)

**Suggested order:** ship #2729 → infra #2730–#2732 → scenarios #2733–#2738 (concurrent, disjoint
files) → roll-up #2739–#2742. Milestone 5 — *Win the benchmarks we enter*.

## SOTA positioning (research grounding, owned by #2742)

- **τ-bench / τ²-bench** — policy adherence (retail/airline/telecom), LLM-judge or DB-state grader,
  single end-to-end number. Borrow the framing; swap the judge for the deterministic dos kernel and
  make *fak's own doctrine* the policy.
- **BFCL v4** — function-selection + argument accuracy, agentic/multi-turn weighted. Informs the
  verdict/repair concept (right tool, valid args, state across turns).
- **Terminal-Bench 2** — long-horizon real-shell tasks. Real fixtures; ours are fak-concept scoped.
- **MCP-Bench** — tool-use across many MCP servers; informs the injected-tool surface.
- Field critique (single-number benchmarks don't localize *where* a failure originates): our
  per-concept decomposition is exactly that localization.
