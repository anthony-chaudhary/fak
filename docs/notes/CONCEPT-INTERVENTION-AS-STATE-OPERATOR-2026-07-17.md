---
title: "Intervention as a state operator: the physics-and-proof layer beneath next()"
description: >
  A structural theory of what an agent intervention IS. Today every intervention (the
  trajctl re-anchor nudge, every sessionctl Next move, the doomloop NUDGE) is a token
  disturbance appended to an unchanged, poisoned context, escalated on elapsed time, topping
  out at hand-to-human. Four independent lenses — control theory, Bayesian belief-state, the
  mechanistic actuator surface, and causal verification — converge on the same replacement:
  an intervention is a TYPED OPERATION on agent state / action-space / dynamics, routed by a
  diagnosis of which latent corrupted, stamped with the evidence-rung it may assert at,
  carrying its own witnessed acceptance predicate, escalating in STRENGTH not time, gated to
  do no harm, and evaluated on an EXACT forked counterfactual. This is the physics-and-proof
  layer under the next() actuation seam (#4992), the trajctl steering ladder (#2533), and the
  out-of-band control vocabulary (#2753).
date: 2026-07-17
family_key: fak-intervention-operator-key
prior_art: ["#2533", "#2540", "#2753", "#4992", "#4993", "#4994", "#5170"]
---

# Intervention as a state operator

Status: concept/research note. Nothing here changes runtime. It states the theory that the
existing actuation seams *actuate*, isolates the residual none of them cover — what a move
mechanically does to agent state, and how to prove it moved the trajectory to a *better* one
rather than merely a *different* one — and parks a spine + children.

Provenance: synthesized from four parallel deep-dives (control theory, Bayesian belief-state,
the mechanistic actuator surface, causal verification), each grounded in
`internal/trajctl/steer.go`, `internal/sessionctl/next.go`, `internal/doomloop/doomloop.go`,
and [the out-of-band control note](OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05.md).

## The one-sentence reframe

> Today an intervention is a **token disturbance** `u(t)` added to an otherwise-unchanged
> system, applied at one point in state, escalated on elapsed *time*, topping out at
> "hand to a human." It should be a **typed operation on agent state, action-space, or
> dynamics** — routed by a diagnosis of what actually corrupted, stamped with the evidence
> rung it is entitled to assert at, carrying its own acceptance test, and escalating in
> *strength* (not elapsed time) until the witnessed curve measurably turns — gated to do no
> harm to a healthy run.

Every shipped lever is the weak member of the weak class. The `trajctl` re-anchor nudge
(`ComposeReAnchor`) *appends* `{objective, plan, curve}` to a context that still contains
whatever drove the drift. Every `sessionctl` Next move renders as a `user-splice` /
`system-directive` prose splice. The `doomloop` ladder climbs `OBSERVE → NUDGE → ESCALATE`,
where `ESCALATE` fires because the burning-flat *streak got longer* — not because the nudge
was *witnessed to have failed* — and its top rung is "give up to an operator," never "pull a
stronger lever."

## Why prose fails — one mechanism, four derivations

The four lenses each derive the failure of "re-read your objective" from their own first
principles, and it is the same mechanism every time.

- **Control theory.** A bad trajectory is a *stable fixed point* — an attractor basin. A
  bounded input added to the input of an unchanged closed loop is exactly the perturbation
  the basin is stable against. That is what "basin" means. You cannot nudge out of a basin;
  you must cross the separatrix — which requires changing the *state* discontinuously
  (surgery, rollback) or the *dynamics map* itself (tool-permission, constrained decoding),
  not adding tokens to a poisoned context.
- **Bayesian belief-state.** By intervention time the posterior `P(z | C_t)` over the latent
  `z = (goal, world-model, plan)` has *concentrated* on a wrong `z*`, and the corruption grows
  ~linearly in the number of poisoned turns (`Δ₀ ≈ n·δ`, following the many-shot / Bayesian-
  scaling results). A restated objective is content already ~equally probable under `z*` and
  `z_true` — likelihood-ratio ≈ 1 — an `O(m·ε)` nudge against `O(n·δ)` corruption. And the
  model is **provenance-blind**: it conditions on its *own* prior assertions as if they were
  exchangeable evidence, so snowballing is a positive-feedback filter (models separately
  recognize the majority of their own snowballed falsehoods in a fresh session — the knowledge
  is intact; the *conditioning set* is poisoned). The failure is **not a missing instruction**;
  instructions are likelihood terms, and the failure lives in the accumulated prior.
- **Actuator / mechanistic.** The appended nudge competes for softmax attention against the
  whole poisoned prefix, with gain that decays by position (lost-in-the-middle) and by length
  (context rot). A guard rule or a grammar mask is *not a competitor in that lottery* — it acts
  after the distribution is computed, and the bad action has measure zero. Persuasion vs.
  physics. And intrinsic self-correction — exactly what a nudge requests — reliably fails
  *without external ground truth*.
- **Causal verification.** Even when it helps, you cannot *prove* it helped on the instance
  that matters (the fundamental problem of causal inference), and the current ladder escalates
  on elapsed time, never checking whether the last intervention took.

## The new object: a typed intervention

Replace the string with a record. This is where the four lenses fuse.

```
Intervention {
  operand      : STATE(x) | ACTION_SET(U) | POLICY(π) | TRANSITION(F) | DISTURBANCE | HALT
  diagnosis    : which latent corrupted — G(goal) | W(world-model) | P(plan/progress)
  authority    : the witness rung (W0..W3) of the evidence this op may assert
  reads        : trajctl signal+rung, culprit span/turn ids, guard policy, workspace SHA, lease
  mutates      : exactly ONE operand
  accept φ     : a W2/W3 predicate on the next k turns — behavior/evidence, never "ok I'll re-anchor"
  reversibility: REVERSIBLE | COMPENSABLE (journaled pre-image) | IRREVERSIBLE
  witness      : {pre-state hash, op, post-state hash, curve-delta @ +k} — the op itself gets a W-rung
}
```

Four load-bearing moves:

### (a) Operate on state / action-space / dynamics, not on the input

The four-operand taxonomy ranks the whole actuator surface, and the ranking **inverts** today's
default. Strongest and most reliable are **U-edits** (revoke the tool / grammar-mask the bad
move so the looping action is *unrepresentable*) and **x-edits** (context *surgery* deleting the
poisoning span plus its inferential descendants; **rollback** to a pre-corruption checkpoint;
**handoff** to a fresh context composed only from *witnessed* artifacts). Weakest — today's floor
— is the token append. Ranked, with fak availability:

| Actuator | Operand class | Mechanism | fak today |
|---|---|---|---|
| Tool-permission / affordance change | U-edit | forbidden action structurally cannot execute; structured refusal returns | **yes** (guard) |
| Handoff / fresh brief from witnessed state | x-reset | discards poisoned context; new KV, fresh attention | partial (`sessionctl` reopen) |
| Constrained decoding / grammar | U-edit (token) | bad move masked to −∞ logit, mass renormalizes to legal moves | **yes** (grammar seam, self-hosted engine) |
| Context surgery (excise poison + descendants; pin verified facts high) | x-edit | removes tokens attention keeps conditioning on | **yes** mechanically (compaction / `fak_context_spans` tombstone-restore — today aimed at token budget, not epistemic hygiene) |
| Checkpoint rollback (context + workspace, atomic) | x-edit | restores a pre-corruption snapshot | partial (git side yes; paired context snapshot no) |
| Observation shaping / tool-result mediation | F-edit | rewrites what the environment returns; models weight tool observations as ground truth | **yes** (browser/tool mediation) |
| Sampler / logit-bias change | π-edit (output) | escape a degenerate repetition loop | partial (own engine) |
| Activation steering / representation editing | π-edit (internal) | edits residual-stream activations | no (needs engine hook; open-weights only) |
| **Context-append (today's nudge)** | disturbance | adds tokens; poison stays | **yes** (shipped floor) |
| Halt / escalate | termination | delegate to operator | **yes** |

The structural claim: **actuators that change the action space or the dynamics map ≫ a token
disturbance to an unchanged system.** The strong actuators fak can field are already behind
shipped enforcement seams (guard, grammar, mediation, compaction); the gap is plumbing — they
are not yet wired as `trajctl`/`next()` actuators.

### (b) Route by diagnosis (the missing router)

`trajctl`'s signals are too coarse because they do not say *which latent broke*.

- **G (goal drift)** — objective forgotten, world-model intact → prose re-anchor is correct
  *here and only here* (high λ against the goal latent, ~0 against the world latent). Today's
  nudge is a G-edit misapplied to W-corruption.
- **W (world-model corruption)** — a wrong premise established and snowballed → **witness
  forcing** (redirect the agent to *run the experiment* — `git cat-file -t <sha>`; condition on
  the tool result), or **excision + a retraction certificate** (excise the culprit span and its
  descendant closure; assert the corrected belief with its W3 witness).
- **P (plan overrun / detour)** — `DETOUR_OVERRUN` → plan-state re-projection.

### (c) Authority = witness rung — the kernel does not believe the *supervisor* either

fak's founding doctrine is "the kernel does not believe the agents." The symmetric, previously
missing half: **a supervisor injecting an unverified "fact" on a high-trust channel is the most
dangerous poisoning source in the system** — high-gain wrong evidence. So an intervention may
assert only at the rung it carries: `W3` → may retract a belief / assert a fact; `W1–W2` → may
only *instruct verification* (which mints its own W3); `W0` → may only *ask a question*. A "you
seem off track" (W0) rightly does almost nothing — and under sycophancy it produces surface
compliance without belief change, which is worse than nothing.

### (d) Self-verifying, and escalate strength not time

Each intervention carries an acceptance predicate `φ` — a W2/W3 check on the next `k` turns (a
commit touching the re-anchored lane; the failing test actually executed; the curve slope
flipping). The asymmetry is the design: **`φ` *failing* is directly observed — a witnessed
no-op — so escalation is licensed deterministically, per instance, with no counterfactual
needed. `φ` *passing* proves only uptake, never benefit.** This replaces the time-driven
`doomloop` `ESCALATE`: a failed soft nudge does not wait out a longer streak and quit to a
human — it climbs the *strength* ladder (append → observation-shaping → sampler →
tool-policy/grammar → surgery → rollback → handoff → halt), weakest-sufficient rung that
actually crosses the separatrix. A failed rung is *evidence*, ledgered, and gates the next.

## The unification that dissolves the hardest problem

The causal ceiling is that you cannot observe `Y(intervene)` and `Y(¬intervene)` on the same
trajectory. The belief lens removes it for *offline* evaluation:

> **For an LLM agent the belief state is copyable — the context prefix IS the state.** So the
> counterfactual twin is exact, not estimated: fork the trajectory at intervention time, apply
> the op in one branch, no-op (or an alternative op) in the other, and directly measure the
> effect.

This is the deepest convergence: the actuator lens's **rollback/handoff**, the belief lens's
**fork-and-probe evaluation**, and the causal lens's **randomized twin arm** are *the same
primitive* — a checkpoint/fork — wearing three hats. Building it once buys the strongest
actuator, the exact evaluator, and the causal identification strategy simultaneously.

That yields a two-tier proof that keeps the honest epistemics:

- **Per-instance (provable, online).** Did `φ` fire? A W3-checkable bit. Report "intervention
  *took* (witnessed)" — never "worked."
- **Population (provable, offline).** `τ = E[B⁺ − B⁻]` over paired forks, where `B` is a
  strictly-proper score of the agent's credences on load-bearing sub-claims against **W3
  verifiers** — "witnessed, not self-reported" applied to *epistemics*. Evaluate under an
  **e-process / confidence sequence** (anytime-valid, because the supervisor peeks forever;
  fixed-`n` p-values are invalid under optional stopping). Formalize the regime gate as **safe
  policy improvement**: maximize lift on the bad regime subject to a bounded harm ceiling on
  the healthy regime (SPIBB shape, with do-nothing as baseline and deviation licensed only in
  drift/stall cells). Anti-Goodhart: **W3 outcome is the only confirmatory endpoint** (an op
  lifting the W2 score but not the W3 goal is logged `SURROGATE_ONLY`), plus a **placebo
  (content-free) nudge arm** to prove it is the content not the interruption, plus a McCrary
  density test for agents learning to dodge the threshold, plus a permanent ~5% hold-out so
  "enabled" stays reversible on evidence after a model upgrade.

**Do-no-harm, sharpened:** with a hard regime gate, healthy-regime harm is zero *by
construction* — the residual harm channel is signal *misclassification* (a truly-healthy run
labeled DRIFT). So the constraint discharges into: no fired sub-stratum shows a confidence-
sequence upper bound below `−δ`. Minimal license to enable an actuator by default: (1) `τ > 0`
e-process crossed on the W3 outcome; (2) no harm pocket; (3) `φ`-uptake above a floor; (4)
effect survives the W2→W3 outcome swap; (5) the hold-out arm stays open.

## SUTVA is a real violation here

fak's shared-tree fleet breaks the no-interference assumption: one worker's intervention
mutates state peers observe (locks, leases, the tree). Outcomes must be **lane-local** or the
randomization **clustered by lease-epoch**, or the population numbers mean nothing. This is not
academic — it is the first thing the calibration child must get right.

## Positioning — the residual under the actuation family

This note is the *physics-and-proof* layer; it does not duplicate the actuation-seam family, it
supplies the theory the seams actuate.

- **`next()` (#4992, children #4993 seam / #4994 lower-stophook)** — the one witnessed move all
  drivers lower to. This note types *what a move does to state* and *how to prove it helped*;
  the typed `Intervention` record is the payload `next()` should carry.
- **trajectory-control (#2533, re-anchor nudge #2540)** — supplies the regime gate, the witness
  rungs, the curve. This note supplies the *diagnosis router*, the *strength ladder*, and the
  *fork-paired τ* the steering ladder currently lacks.
- **out-of-band control (#2753)** — supplies the closed control *vocabulary* (redirect /
  add-constraint / edit-state / checkpoint / fork). This note ranks those ops by operand class
  and gives each an authority stamp, an acceptance predicate, and a reversibility class.
- **policy-amendment classes (#5170)** — the discipline for *who may amend the guard floor*; a
  `TOOL_POLICY` intervention is a guard-floor amendment and must obey those classes.

## The spine and the fan-out

**Spine (S):** this note + a typed `Intervention` record generalizing `SteerDecision` /
`NextRecord` to carry `operand`, `diagnosis`, `authority`, `φ`, `reversibility`, and the
`pre-hash / op / post-hash / curve-delta@+k` witness. Turns "intervention" from a string into a
verifiable state operation.

**Children (worst-first):**

1. The G/W/P **diagnosis router** — refine the `trajctl` signal into a latent-type so the
   controller *selects an actuator* instead of always nudging.
2. **Authority stamping + fail-closed assert rule** — `authority = min witness rung`; refuse a
   `W3` assertion the supervisor cannot witness. Reuse the `dos_check_reason` closed vocabulary.
3. **Excision-for-hygiene** — point compaction at the culprit span + descendant closure (an
   epistemic-hygiene primitive distinct from budget-driven compaction), reusing the
   `fak_context_spans` / tombstone-restore seam.
4. **Tool-policy / grammar U-edit actuators** through the guard, behind the regime gate, with
   the structured-refusal `fix` field; a `TOOL_POLICY` op obeys the #5170 amendment classes.
5. **`φ` acceptance predicates + strength-escalation ladder** — replace the time-driven
   `doomloop` `ESCALATE` with witnessed-`φ`-driven strength escalation.
6. **Paired checkpoint / fork** (context + workspace SHA + lease, atomic) — the one new seam
   that unlocks the strongest actuator *and* the exact evaluator *and* the causal twin at once.
7. **Calibration child as an e-process** — emit `{e-value, confidence sequence}` per
   `(actuator × regime × outcome-version)`; the CS lower bound is what "enable-by-default" reads.
   Lane-local / lease-epoch-clustered outcomes per the SUTVA fence.

## Honest fences

- **Nothing here is shipped.** The referee (witness rungs), the detector (`trajctl`/`doomloop`),
  and — behind the guard / compaction / grammar seams — the strong actuators exist; the typed
  contract, the router, authority stamping, the paired checkpoint, and the e-process calibrator
  do not.
- **Per-instance *benefit* is unprovable forever.** The system must refuse to phrase it: the
  maximal honest claim is "took (`φ` witnessed); policy effect CS = [ℓ, u]."
- **The Bayesian abstraction is falsified in its strict form** (martingale violations, order
  effects, context rot). Use log-odds / precision as a *design calculus* that predicts the sign
  and rough magnitude of effects — not as a theorem.
- **Excision is not clean conditioning-set surgery** — the model may have re-encoded the poison
  into its own later tokens; descendant closure is only approximable via span provenance.
- **Activation steering is uniquely reachable** (fak ships its own GGUF engine) but research-
  grade; API-served models permit neither it, logit-bias, nor token-grammar. The reliable wins
  are the U-edits and x-edits, which work regardless.
- **Citation hygiene.** The two in-tree anchors the regime gate and re-anchor nudge cite
  (`arXiv:2602.03338`, `arXiv:2505.02709`) did not resolve on search during this synthesis and
  should get a `dos_citation_resolve` pass — the regime gate's load-bearing "do no harm" claim
  rests on the first.
