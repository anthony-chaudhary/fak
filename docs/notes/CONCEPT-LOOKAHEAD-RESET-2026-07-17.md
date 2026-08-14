---
title: "The look-ahead reset: speculative rollout, witnessed distillation, reset-with-lesson"
description: >
  A concrete intervention under epic #5202 (intervention as a state operator). Because an LLM
  agent's state IS its copyable context prefix, you can fork at a clean checkpoint, speculatively
  roll the agent forward to see what WOULD have happened, witness the outcome at rung W2/W3,
  distill it into a compact lesson, then reset to the checkpoint and re-render the continuation
  with the lesson folded in. The rendered trajectory is poison-free yet carries the foresight —
  it strictly dominates both rollback-alone (which loses the information) and nudge-alone (which
  keeps the poison). The paired fork also turns the fundamental problem of causal inference into
  an EXACT instance-level counterfactual. This note carries the theory + dominance proof and a
  cheapest-first dogfood plan grounded in real fak seams.
date: 2026-07-17
family_key: fak-lookahead-reset-key
prior_art: ["#5202", "#2533", "#2540", "#4992", "#2760", "#3619", "#3547", "#760"]
---

# The look-ahead reset

Status: concept/research note under epic **#5202** (intervention as a typed state operator).
Nothing here changes runtime. It develops the single intervention the operator flagged as
highest-value — *reset the trajectory but fold in the information of what would have happened* —
into a formal operator with a dominance proof, an exact-counterfactual estimand, and a
cheapest-first dogfood plan mapped to real fak packages.

Provenance: two parallel deep-dives (theory/proof; applied dogfooding over the fak tree), on top
of [the intervention-as-state-operator synthesis](CONCEPT-INTERVENTION-AS-STATE-OPERATOR-2026-07-17.md).
Citations are from model training memory (verify where marked). Applied seam claims were read
from the tree but some transport glue is explicitly unverified — see the fences.

## The one-line idea

> Fork at a clean checkpoint → let the agent **speculatively run the path forward** (the
> look-ahead) → **witness** where it actually ends → **distill** that outcome into a verified
> *lesson* → **reset** to the checkpoint and re-render the continuation with the lesson folded
> in. The agent "wakes" at the branch point carrying the *foresight* of what would have
> happened, but none of the *poison* of the confused path it took to learn it.

This is **Model-Predictive Control where the world-model is the agent run forward on itself** —
exact (bias ≈ 0) instead of a learned approximation — made free and lossless by the copyable
belief state. It keeps the lesson and drops the poison, which is exactly what neither rollback
nor nudging can do alone.

## The operator

State at turn `t` is `x_t = (C_t, S_t)`: the context prefix `C_t` (which *is* the belief state —
the policy is a frozen function of it) and the workspace snapshot `S_t` (git SHA + tree + leases).
Both are losslessly copyable; that is the entire license. The operator composes five typed
sub-operators:

1. **Fork** `Φ: x_t ↦ {x_t^(i)}` — bit-identical, hermetic copies at a *clean* checkpoint
   (pre-poison), each sandboxed: own worktree, disjoint lease, irreversible external effects
   masked by a guard U-edit.
2. **Rollout** `R_k: (x_t^(i), π, seed s_i, u_i) ↦ τ_i` — run `k` turns under continuation
   directive `u_i`. `u_0 = ∅` reproduces the incumbent (drifting) path; `u_1..` are alternatives.
   Each rollout runs under its *own* trajctl regime gate and detour budget.
3. **Witness** `W: τ_i ↦ (y_i, r_i)` — outcome `y_i` at rung `r_i ∈ {W2, W3}` computed from
   artifacts (test verdict, build state, curve slope, goal event), **never** from the fork-agent's
   narration. Truncation with no discriminating event is `ABSTAIN`, not failure.
4. **Distill** `D: (τ_i, y_i, r_i) ↦ ℓ_i` — a compact lesson, produced by a *fresh* distiller
   context (not the fork-agent summarizing itself), then audited so every claim is entailed by a
   cited artifact/span hash (commit-audit discipline applied to lessons).
5. **Reset + render** `A: (x_t, {ℓ_i}) ↦ x_t^+` — discard all forks, restore `x_t`, and render the
   selected lessons **on an observation channel** (a verifier tool-result carrying its rung and
   evidence hash), not as assistant self-talk. From the continuing agent's view it ran a cheap
   oracle query at turn `t`.

### The lesson — the sharp part

The right minimality notion is not statistical sufficiency (reconstructing the transcript) but
**Blackwell decision-sufficiency**: `ℓ` is the cheapest *garbling* of the rollout that preserves
the value of the branch-point decision. Equivalently an information-bottleneck with a hard
novelty constraint: maximize `I(ℓ; consequence-relevant future)` s.t. `|ℓ|` small **and
`I(ℓ; poison) = 0`**.

```
Lesson {
  branch_action X : the continuation, at the granularity of its CAUSE class
  outcome y       : witnessed event + polarity ("fails F at t+j" / "reaches goal G")
  rung r          : W2 | W3 — the authority this lesson may assert at
  cause C         : checkable proposition, carrying ITS OWN rung
                    (W3 cause may assert; W1 cause renders fail-closed as "hypothesis")
  scope           : validity qualifier — seeds tried, horizon k ("failed within k" ≠ "cannot succeed")
  evidence_ptr    : content-hash of the stored rollout artifacts — re-witnessable downstream
  salvage         : witnessed gains from the discarded segment worth carrying (commits, verified facts)
}
```

The **cause field is load-bearing**: a bare "path X failed" prunes only the literal action; the
agent takes a surface-variant `X′` sharing the mechanism and re-enters the same basin. The cause
defines the *equivalence class* the lesson prunes. Under-distill → no generalization; over-distill
→ re-import the poison and the attention drag the reset exists to remove. Rule of thumb: **distill
everything witnessed, drop everything narrated**; the `salvage` field is the compensator for
genuinely useful witnessed work in the discarded segment.

## Dominance proof

Information sets: rollback-alone acts on `{x_t}`; nudge-alone on `{x_t, poison, prose}`;
look-ahead reset on `{x_t, ℓ}`.

**Assumptions.** A1 witness soundness (the W2/W3 verdict is correct); A2 distillation fidelity
(ℓ preserves consequence-relevant content, asserts causes only at their earned rung); A3
evidence-monotone policy (the model updates on trusted-channel, witnessed, cited statements —
ICL-as-implicit-Bayes, Xie et al., from memory); A4 relevance/stationarity (hermetic fork, no
leakage, no environment drift); A5 clean checkpoint (`x_t` predates the poison); A6 strictness
(`P(ℓ changes the argmax) > 0` — holds exactly when the gate fires, since the rolled-out incumbent
is the modal continuation and was witnessed to fail).

**vs. rollback-alone.** `{x_t, ℓ}` strictly refines `{x_t}` by a truthful signal about
consequences. **Blackwell's theorem** (an experiment weakly dominates any garbling of it, for
every decision problem) and **Howard's non-negativity of the value of information** (1966): adding
a truthful signal cannot lower, and under A6 strictly raises, the value of a rational decision.
Control-theoretic twin: **Bertsekas' rollout-improvement property** — a policy acting on rollout
evaluations of a base policy is no worse than the base policy. Rollback-alone *is* the base policy
restarted from `x_t`; look-ahead reset is one rollout-improvement step carried as `ℓ`. Without `ℓ`
the reset agent re-derives its continuation from the same prior that drifted — identically (same
seed) or i.i.d. from the same distribution over mistakes (fresh seed).

**vs. nudge-alone.** The nudge carries two liabilities. *State:* the poison stays — posterior
corruption `Δ ≈ n·δ` (linear in poisoned turns, the many-shot scaling regime; Anil et al. 2024,
from memory) plus provenance-blind self-conditioning (Zhang et al., snowballing) plus positional
attention decay. *Content:* a restated objective has likelihood-ratio ≈ 1 between the wrong latent
`z*` and `z_true`, moving the posterior `O(m·ε)` against `O(n·δ)`. The look-ahead reset zeroes the
first (state is the clean `x_t`) and inverts the second: a witnessed consequence — "X executed,
failed test T at t+j, artifact H" — is maximally discriminating *exactly on the corrupted
coordinates*, so the posterior lands KL-closer to truth. Control view: inside an attractor basin
the input-to-terminal-state gain is exponentially damped (that is what stability means); the same
tokens spent at the pre-capture branch point have `O(1)` gain. The reset crosses the separatrix;
the lesson prevents re-entry. Neither half alone does both.

**Where dominance fails (honestly).** If A1/A2 break, `ℓ` is **high-precision poison** on the
most-trusted channel — strictly worse than rollback. Dominance is therefore *conditional*; the
guardrails below are what restore it in expectation. This is the symmetric doctrine from #5202:
the kernel does not believe the supervisor either.

## The counterfactual becomes exact

**Holland's fundamental problem** (1986): you cannot observe `Y(a)` and `Y(a′)` on the same unit.
Here the unit is the branch point `x_t`, and because its state is losslessly copyable and the
environment forkable, the unit-level effect is *observed, not estimated*:

- Potential outcome `Y(x_t, u_i, s) = g(W(R_k(x_t; π, s, u_i)))`.
- Unit-level contrast `Δ_{ij}(x_t, s) = Y(x_t, u_i, s) − Y(x_t, u_j, s)` — both terms on the same
  unit, same seed. Shared seeds across arms = **common random numbers**, so `Var(Δ)` is small.
- Arm value `v_i = E_s[Y(x_t, u_i, s)]` by Monte-Carlo; the only error is sampling error, not model
  bias — the "world model" is the real system speculatively executed.

**Selection rule:** `i* = argmax_i v̂_i` by witnessed outcome, ties broken by cost; render `ℓ_{i*}`
plus the *negative* lessons of pruned arms (a pruned arm's lesson is what prevents re-entry).
Disciplines: guard the **winner's curse** (validate `v̂_{i*}` on held-out seeds, or assert only the
*ordering* at W2); and note this is **decision-time planning, not policy learning** — no weights
update, no reward loop; the rollout scores gate one harness action and fold into context as
evidence (fak doctrine intact: scores gate, never train).

**Structural bonus — the probe subsumes do-no-harm.** #5202's residual harm channel is regime
*misclassification* (a healthy run labeled DRIFT). Rolling out `u_0` (the incumbent path) first
converts that classifier call into a witnessed one: if the fork *recovers*, do nothing — the run
was healthy and the fork proved it, at token cost instead of trajectory cost. Intervention becomes
conditioned on witnessed counterfactual failure, not a signal threshold.

## Literature situation and the novelty delta

MPC / receding-horizon with the agent as its own *exact* world-model (contrast learned world
models — Dreamer, MuZero — cheap but biased; here bias ≈ 0, cost is tokens/horizon/seed-variance;
trajctl's folded curve supplies the MPC terminal value). MCTS backup is *numeric*; here the backup
is *linguistic and causal* (generalizes over the cause class). Closest prior and the exact delta:
**LATS** keeps the live tree and stitches reflections into the *uncleaned* context, no evidence
rungs, no admission gate; **Reflexion** is the degenerate 1-rollout with *self-narrated*
distillation (untrusted rung), reset only at episode boundary; **Tree of Thoughts** scores by
*self-evaluation* (W0/W1 — the banned rung); **RAP** uses the LLM as an *approximate* world model
(unneeded when state is copyable); **ExIt/HER** distill search into *weights*. The neuroscience
analogue is tight: vicarious trial-and-error / hippocampal preplay, and **Mattar–Daw prioritized
replay** (`gain × need`) is literally the value-of-information rule for which branch deserves a
rollout — the animal simulates the shock arm and stores "that path leads to shock" without paying
the shock. Huang et al. (ICLR 2024, "LLMs cannot self-correct reasoning yet", from memory) is the
negative result that *motivates* witnessed distillation: intrinsic self-correction fails without
external ground truth.

**Novelty delta, exactly:** (1) the world model is *free and exact* from the copyable prefix — no
learned model, no model bias; (2) the reset keeps **only the distilled lesson** — not the tree
(LATS), not the transcript (Reflexion), not weight updates (ExIt/HER) — a clean prefix that still
carries the future; (3) distillation is **witnessed at a typed evidence rung**, fail-closed, vs.
self-evaluation everywhere in LLM-search; (4) **admission control** — a regime gate + a VoI test
decide *whether* to fork at all (search is an intervention with a harm budget, not a default); (5)
the paired fork doubles as an **exact instance-level evaluator of interventions themselves**.

## When to fire it — value of information vs. cost

Fire when `EVOI > c·(N·k·tokens)`, where `EVOI ≈ E[max_i v_i] − max_i E[v_i]` plus the
avoided-misclassification term. This concentrates on: the regime gate is non-HEALTHY (or the next
action is high-stakes/**irreversible** — VoI scales with irreversibility, so reversible steps are
their own cheap experiments and need no fork); genuine uncertainty over continuations; expensive-
to-unwind decisions. **Branching** `N` stays 1–3 (marginal value of the N-th arm decays like the
expected-max of N draws; Mattar–Daw `gain × need` prioritizes candidate arms). **Horizon** `k` =
the minimum reaching a *discriminating witness event*, capped by the detour budget and extended
past the plausible self-correction window (so a recoverable path is not pruned). **Termination** is
guaranteed: bounded `N × k × seeds`, each rollout under its own gate, killed on STALL, booked
against the `DETOUR_OVERRUN` budget — and rollout tokens are *spent*, never credited to the
progress curve (the ban on length-monotone metrics applies to the fork's accounting too).

## Applied: the fak seam map

| Step | Reusable today | Missing glue |
|---|---|---|
| Fork (context) | `internal/sessionimage` `ForkDir` = `SnapshotDir` (#2760) + `BranchDir` (CoW, lineage); `fak session fork`; ACRFence keep-bit (`witness.json`) | captures fak's *logical* session, **not** the Claude Code harness transcript — the flagship `fak manage -- claude` prefix lives in the transcript JSONL; fork transport (`--fork-session` or JSONL-copy-under-new-uuid) is **unverified glue** |
| Fork (workspace) | `internal/workerworktree` `Prepare/Reap` (detached worktree pinned to trunk SHA; isolated Land CAS #3619 / readback #3547 for the eventual winner); `internal/shadowgit` step→write attribution | replicate the live session's *uncommitted* diff into the fork (`git diff` → `git apply`); put the shadow dir **outside** the worktree so evidence survives Reap |
| Rollout | headless resume spawn `rwResumeArgv`/`rwSpawnResume` + launch broker; `internal/loopmgr` `AdmitSpeculation` (EV = P(correct)×saved > cost, closed reasons) as the pay-for-rollout gate | turn cap (`--max-turns`, verify CLI); cwd pinned to fork; a **rollout policy floor** under `fak manage` (deny push/gh/steer). NB `cmd/fak/headless.go` is headless-*lint*, not a run path; `sessionreplay` is single-decision replay, not a rollout engine |
| Witness | `internal/trajctl` rungs W0–W3 + curve/Signal; `internal/trajctlhook` `CheapScorers` (CommitProgress W3 + ActivityDivergence W2) + `GitEvidenceResolver` | point `GitEvidenceResolver` at the fork worktree; **persist evidence pre-Reap** (exit codes, diffstat, shadowgit SHAs) or W3 lessons decay to dangling refs |
| Distill | `internal/sessionreset/model_distill.go` (opt-in `SummarizeFunc`, cost gate, graceful decline, pinned instruction) is the template | the witness gate (below) + the `Lesson` leaf |
| Reset + render | `internal/sessionctl/next.go` Move×Render; `trajctl.GatewaySteer` POST `/v1/fak/session/{id}/steer` (#760, one-nudge-per-episode); doomloop outbox + `drain --deliver`; SessionStart `source=compact` additionalContext (`guard_sessionstart.go`); full reset via `internal/sessionreset` seed | `SteerDecision{Lesson, Rung}` extension; a `KindLookahead` ledger row |

## Three dogfood protocols, cheapest-first

**P0 — PreCompact mini-lookahead (async, nearly free).** In `runGuardPreCompact`, on an *allow*
decision, after the existing boundary-row append: gate via `AdmitSpeculation`, then spawn detached
(transcript-fork + `workerworttree.Prepare` + diff patch + `claude --resume <fork> -p "Continue the
current plan" --max-turns 3` under a deny-push floor) and **exit 0 immediately — never delay the
hook**. The runner witnesses at exit (build/affected-tests → W3; ActivityDivergence → W2), distills
a `Lesson`, writes it to a per-session lesson file. `runGuardSessionStart` gains a `source=compact`
branch: if a fresh lesson (same base SHA) exists, inject it as additionalContext next to the
compacted summary. Fail-open. **Done:** one long real session with ≥3 auto-compactions shows ≥1
rollout that produced a W2+ lesson and was injected at SessionStart(compact) — verified from ledger
rows + transcript, not self-report.

**P1 — doomloop OBSERVE → fork-confirm-before-NUDGE.** In `cmd/fak/doomloop.go` `tick/sample`, when
`Correction == CorrectObserve` and `BurningFlatStreak == TripWindows-1`: fork + roll `k` turns, feed
the *fork's* effort/progress counters back through the pure `doomloop.Classify` (classifier
untouched; all wiring in the shell). Fork verdict DOOM → enqueue the nudge now, enriched with the
lesson; fork verdict HEALTHY (W3 progress in the fork) → ledger a suppressed-nudge row with a new
closed reason (`SPEC_RECOVERED`, declared in dos.toml). Extend `calibrate.go` to score
heuristic-vs-simulation disagreement. **Done:** ledger has ≥K OBSERVE-stage confirm rollouts with
fork verdict + rung, and calibrate reports the measured false-positive rate of the K-window
heuristic.

**P2 — full drift-triggered look-ahead reset.** At Stop-hook turn end where `Signal == DRIFT|STALL`
and steer would compose the generic re-anchor: (1) paired checkpoint (new seam); (2) roll k=5–8
headless in the fork; (3) witness (CheapScorers + build/test + shadowgit); (4) witness-gated
distill; (5) render, at two strengths — **mild** (deliver lesson via `GatewaySteer` /
`RenderSystemDirective`, drifted history stays) or **strong** (the true reset: relaunch
`claude --resume <pre-drift-fork-id> -p "<lesson splice> Continue."`, dropping the drifted suffix).
Every event ledgers as `KindLookahead`. **Done:** one real long-horizon session where a DRIFT fired,
the full cycle ran, and the post-reset curve reaches its next W3 landing (witnessed commit,
`dos_verify`-confirmable) in fewer turns than the pre-drift drift span.

## Measurement: paired-fork A/B + SUTVA fence

Because state is a copyable prefix, arms are **serial re-runs from the identical paired
checkpoint** — no concurrency, so no lease contention. At a real branch point, run 4 arms:
(1) reset+lesson, (2) rollback-alone, (3) let-drift (as a fork, never the live shared session),
(4) nudge-alone. Each: fresh transcript fork + own detached worktree at the same base SHA +
identical diff patch, k-turn budget, deny-push floor, **Reap at end — no arm lands**. Witnessed
scores: turns-to-first-W3 (GitEvidenceResolver on the arm worktree, evidence captured pre-Reap),
curve slope, binary goal-attained. Ledger on the existing `FoldWorktreeAB` / `fak dispatch
worktree-ab` pattern, extended to a 4-arm `lookahead-ab` fold.

**SUTVA fence:** (a) detached worktrees make arm writes physically disjoint even on the same
logical paths; (b) no arm lands — the only trunk mutation is the operator-chosen winner afterward,
under a normal lane lease (`dos_arbitrate` at land time only); (c) the arm guard floor denies
steer/gh/push and the launch broker caps one live resume at a time; (d) scoring is worktree-local;
(e) base SHA + peer commits during the experiment are recorded as covariates, and an arm whose base
no longer matches trunk parent is flagged, not pooled.

## Witnessed-distillation contract

A new pure leaf `internal/lookahead` (tier 1–2, stdlib + trajctl):

```go
type RolloutEvidence struct {
    ForkSessionID string; BaseSHA string; Turns int
    Rung      trajctl.WitnessRung   // max rung across Witnesses
    Witnesses []trajctl.EvidenceRef // build/test exits, shadowgit snapshot SHAs, curve rows
}
type LessonKind string // FACT | RISK
type Lesson struct {
    Claim string; Kind LessonKind
    Rung  trajctl.WitnessRung; Evidence RolloutEvidence
    ExpiresSHA string // stale once trunk moves past base
}
```

`DistillLesson` reuses the `sessionreset.modelDistill` decline pattern (nil seam / short transcript
/ model error → ok=false, never poisons the seed), then the **gate**: W3 evidence may emit
`Kind=FACT`; W2 may emit only `Kind=RISK`; a W2 lesson asserting a FACT is refused with a closed
reason `LESSON_OVERCLAIMS` (declared in dos.toml); W1/W0 rollout evidence is refused outright —
judge opinion and self-report never masquerade as foresight. Consumers render the rung visibly
("Witnessed (W3): …" vs "Risk flag (W2): …") at three plug points: the steer packet, SessionStart
additionalContext, and a sessionreset contributor. Staleness at inject time = `ExpiresSHA` ancestry
check (same spirit as `dos_recall` / `GitEvidenceResolver`).

## Buildable-this-week vs needs-new-seam

**Reusable as-is:** sessionimage checkpoint/fork + keep-bit; workerworttree Prepare/Reap (+isolated
Land for the winner); shadowgit; trajctl rungs/curve/steer + regime gate; trajctlhook
RunTurnEnd/RunCompaction; doomloop classifier + outbox + drain + calibrate; PreCompact/SessionStart/
Stop hook shells; headless resume spawn + launch broker; `AdmitSpeculation`; sessionreset seam;
`FoldWorktreeAB`; `dos verify` / commit-audit.

**Needs new seam (only these):** (1) **paired atomic checkpoint** — bind a transcript fork to
(base SHA + uncommitted-diff patch) as one addressable object (a sessionimage `workspace.json`
sibling part), plus the transcript-fork transport itself; (2) **rollout-in-fork runner** —
`fak lookahead roll`: fork + worktree + guarded turn-capped headless spawn + witness collection +
reap; (3) the small `Lesson` leaf + `SteerDecision`/ledger-row extensions. P0 and P1 are buildable
on seams 2+3 (P0 needs no paired checkpoint — the compaction boundary is already quiescent); P2 and
the 4-arm A/B need seam 1.

## Failure modes and guardrails

1. **Rollout drifts / is uninformative.** Fork runs under its own regime gate + detour budget,
   killed on STALL. Asymmetry: a `u_0` rollout that reproduces the parent's drift *is* the desired
   witness; *alternative* arms need a perturbed seed/directive to explore rather than re-derive the
   same mistake. No discriminating event within `k` → `ABSTAIN`, **no lesson minted**.
2. **Distillation infidelity.** Fresh distiller context; every claim cites artifact/span hashes and
   passes an entailment audit; the cause carries its own rung; a W1 cause renders fail-closed.
3. **A wrong lesson is high-precision poison** (worst case, rides the trusted channel). Authority
   stamping (assert only at the earned rung); journaled pre-image (COMPENSABLE); an acceptance
   predicate `φ` — if the continuation *witnesses* a contradiction of the lesson, the lesson is
   tombstoned, not argued with; `evidence_ptr` keeps it permanently re-witnessable.
4. **Over-pruning a recoverable path** (the "don't kill self-correcting trajectories" concern).
   `k` extends past the plausible self-correction window; score on curve-trend + terminal witness,
   not first error; mandatory scope qualifier ("failed within k under seeds S" ≠ "cannot succeed");
   censored arms scored `ABSTAIN`, and selection prefers *unknown* over *negative* for them.
5. **Side-effect leakage from the fork.** Hermeticity is a precondition: detached worktree, lease
   disjointness via `dos_arbitrate`, irreversible external actions masked by a guard U-edit. Honest
   consequence: outcomes that *require* an irreversible action cannot be witnessed speculatively —
   those decision points stay operator-gated; say so rather than stub-and-pretend.
6. **Seed non-transfer / selection bias.** Multi-seed where budget allows, scope-qualify where not;
   hold-out-seed validation against the winner's curse. Standing rule: rollout scores gate harness
   actions only — never a reward signal, never fine-tuning data (keeps this decision-time planning,
   not a Goodhart-exposed learning loop).

## Fences (verify before building)

- sessionimage forks fak's *logical* session, **not** the Claude Code transcript — the transcript
  is the real copyable prefix for flagship `fak manage -- claude` sessions, and its fork transport
  (`--fork-session` / JSONL-copy) is **unverified**; grep found no repo reference to `--fork-session`.
- `--max-turns` CLI support for the turn cap is unverified.
- Fork-worktree SHAs dangle after Reap — durable witness must be recorded pre-Reap.
- `cmd/fak/headless.go` is headless-*lint*, not a run path; the real headless primitive is the
  resume-watchdog spawn. `internal/sessionreplay` is single-decision replay, not a rollout engine.
- Nothing here is shipped; per-instance *benefit* remains unprovable — the honest per-episode claim
  is "took (φ witnessed)"; population license is #5202's e-process/SPIBB machinery, with the paired
  fork now serving as both actuator and its own exact evaluator.
