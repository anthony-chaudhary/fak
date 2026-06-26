---
title: "Credit assignment over spans: PRMs, RL credit, and agent-memory usefulness"
description: "Prior-art survey 4/5 for the reward-over-spans epic (#860): how the RL and agent-training literature assigns credit to intermediate steps/context — process reward models, temporal-difference / hindsight credit, and retrieval-usefulness signals — and the span-credit-assignment design those patterns produce. The deliverable is one reward equation, reward(s) = success-gated × recency-discounted × confound-normalized attention mass, with each term justified by a named prior-art pattern and mapped onto fak's already-shipped ctxplan.OutcomeFromAttention / Forecast.Learn / Weights.Learn / rsiloop seams. Honest spine: attention-as-importance is CONTESTED, so the producer is a swap point — if the survey's sibling (#861/#866) refutes attention, the same Outcome interface takes the leave-one-out signal (#863) instead."
slug: credit-assignment-over-spans
keywords:
  - credit assignment
  - process reward models
  - PRM
  - Math-Shepherd
  - temporal-difference learning
  - eligibility traces
  - hindsight credit assignment
  - generalized advantage estimation
  - self-RAG
  - retrieval reranking
  - mixture-of-experts gating
  - attention as importance
  - span reward
  - context credit assignment
date: 2026-06-25
---

# Credit assignment over spans: PRMs, RL credit, and agent-memory usefulness

This is survey 4/5 under the reward-over-spans epic
([#860](https://github.com/anthony-chaudhary/fak/issues/860)): H2O computed accumulated
per-token attention, used it once for a greedy KV-eviction decision, and threw it away. The
epic's move is to **promote that witnessed attention from transient eviction-state to a
persisted, span-attributed REWARD** — credit assignment over the agent's context units (tool
results, messages, retrieved docs, sub-agent outputs). A span that the useful part of a turn
attended heavily *earned its keep*; a span that never draws attention is noise an RSI loop
should learn to stop carrying, stop retrieving, stop generating.

This memo is the deliverable for
[#864](https://github.com/anthony-chaudhary/fak/issues/864).

That is, restated, a **credit-assignment problem**: given an outcome, attribute it back to the
intermediate things that produced it. The RL and agent-training literature has spent a decade
on exactly this question for *steps* and *retrieved items*; this note surveys what they did, and
extracts the patterns that transfer to "which context spans earned their keep". The product is a
single design — one reward equation with three terms, each justified by a prior-art pattern,
and each mapped onto a fak seam that is **already in the tree** (the substrate landed under
[#851](https://github.com/anthony-chaudhary/fak/issues/851) / [#858]; this note specifies the
reward that rides it, it does not invent new substrate).

The honest boundary is carried from the epic and is the whole risk: **attention-as-importance is
contested** ("Attention is not Explanation", Jain & Wallace 2019, vs "Attention is not not
Explanation", Wiegreffe & Pinter 2019). So the design below is a *spec for an experiment*
([#866](https://github.com/anthony-chaudhary/fak/issues/866) /
[#867](https://github.com/anthony-chaudhary/fak/issues/867)), not a claim that attention is the
right reward. The producer is deliberately a swap point: the reward equation is written against an
`Outcome` interface fak already ships, and if the validity survey ([#861]) or the LOO correlation
([#866]) refutes attention, the *same* interface takes the leave-one-out / gradient signal
([#863]) instead. The design succeeds by finding the right signal, not by forcing attention to be
it.

## 1. The four prior-art families, and the one transferable idea in each

### 1.1 Process Reward Models — reward the STEP, gated by the OUTCOME

A process reward model (PRM) scores each intermediate reasoning step, not just the final
answer. "Let's Verify Step by Step" (Lightman et al., 2023; the PRM800K dataset) showed a
human-labeled per-step verifier beats an outcome-only verifier on MATH — process supervision
both performs better and is more aligned, because it rewards *where* the reasoning was sound
rather than only *whether* the conclusion was right. The cost is the human step-labels.

Math-Shepherd (Wang et al., 2024) removed the human from the loop: it estimates a step's
correctness **automatically**, as the empirical probability that continuing from that step
reaches the correct final answer (Monte-Carlo completion rollouts). This is the load-bearing
transfer for us. It says: *a step's credit is gated by the outcome it leads to.* A step that sits
on paths that mostly fail earns little credit even if it looks locally fluent; a step on paths
that mostly succeed earns a lot.

> **Transferred pattern A — the outcome gate.** A span-attention reward is the *context-side*
> analogue of a PRM: reward each context UNIT by its contribution, not just the turn by its final
> answer. But — exactly as Math-Shepherd gates step-credit by downstream success — **attention
> should be credited only on turns that produced a verified-good output.** Attention on a turn
> that was confidently wrong is anti-signal: it tells you which spans *misled* the model, and
> crediting them teaches the planner to keep the very context that caused the error.

### 1.2 RL credit assignment — temporal-difference, advantage, and hindsight

Classical RL is the theory of assigning a delayed scalar reward back to the decisions that
earned it. Three pieces transfer:

- **Temporal-difference learning and eligibility traces** (Sutton & Barto, *Reinforcement
  Learning*, 2nd ed.; TD(λ)). An eligibility trace makes a state/action *eligible* for credit
  when reward arrives, with eligibility decaying geometrically — a thing touched τ steps ago is
  credited by `λ^τ`. This is precisely "a span attended N turns ago should get less credit than
  one attended now".
- **Generalized Advantage Estimation** (Schulman et al., 2016). GAE's λ knob trades bias for
  variance in how far back an advantage is propagated. The lesson for us is not the exact
  estimator but the *dial*: how aggressively to discount older attention is a tunable, and the
  right setting is empirical, not assumed.
- **Hindsight Credit Assignment** (Harutyunyan et al., NeurIPS 2019). HCA reassigns credit by
  asking, *given the outcome that actually occurred, how responsible was this action* — a
  likelihood-ratio reweighting that flows credit from a known outcome backward to the decisions
  that made it more likely. The framing matters here: credit is computed **conditioned on the
  realized outcome**, which is the same conditioning the outcome gate (§1.1) imposes.

> **Transferred pattern B — recency discount (the eligibility trace).** Accumulate a span's
> reward over the turns it was resident as a *discounted* sum, `reward(s) = Σ_t γ^(T−t) a_s(t)`,
> not a flat sum. A span attended this turn is fresher evidence of usefulness than one attended
> ten turns ago and never since. γ is the eligibility-trace decay; γ=1 recovers the un-discounted
> cumulative baseline.

### 1.3 Agent memory / retrieval reward — which retrieved item was USEFUL

RAG and agent-memory systems face the span-credit problem under a different name: of the items I
retrieved, which actually helped, so I can rerank or stop retrieving the rest?

- **Self-RAG** (Asai et al., 2023) trains the model to emit *reflection tokens* that critique its
  own retrieval — `IsRel` (is this passage relevant?), `IsSup` (is the output supported by it?),
  `IsUse` (was it useful?). The signal that a retrieved unit earned its keep is produced by the
  consuming model itself, at the point of use.
- **Retrieval reranking from downstream success.** A recurring pattern (e.g. distilling a reader's
  attention or its answer-correctness back into the retriever, as in attention-distillation
  retrievers): the reranker is trained on whether the retrieved item *led to a good answer*, i.e.
  the retrieval reward is again **outcome-gated downstream success**, not a standalone relevance
  score.

> **Transferred pattern C — usefulness is judged at the point of consumption, by downstream
> success.** This reinforces the outcome gate (A) and adds a scope: the *same* per-span reward
> generalizes past KV-pruning to credit a **retrieval** (did the retrieved doc draw attention?),
> a **tool call** (did its result get attended, or was it wasted budget?), and a **sub-agent
> output** (did the synthesizer attend to what it returned?). One reward, fleet-wide credit.

### 1.4 Attention as a learned value/router signal — where it works, and the contest

There are regimes where attention *is* used as a value signal and it works: mixture-of-experts
gating (Shazeer et al., 2017, the sparsely-gated MoE layer) is a learned router whose gate
weights decide which experts are worth running; retrieval-attention and attention-rollout
(Abnar & Zuidema, 2020) treat attention as a propagated importance. So attention-as-value is not
a priori illegitimate.

But the explanation literature is explicitly **divided** on whether attention weights are valid
importance: "Attention is not Explanation" (Jain & Wallace, 2019) shows attention can be permuted
without changing the prediction; "Attention is not not Explanation" (Wiegreffe & Pinter, 2019)
shows that under the right test attention *does* carry explanatory signal. And attention has
documented **confounds** — attention sinks / the BOS dump (StreamingLLM, Xiao et al., 2023),
positional bias, induction-head artifacts — that inflate mass on tokens that are not actually
"useful context". H2O itself only ever used accumulated attention for a *budget* decision, never
claimed it was a value.

> **Transferred pattern D — confound normalization, and the contest is the gate.** Before
> attention mass can be a reward it must have its known confounds **subtracted** (the sink
> baseline, positional bias) — this is the domain of sibling survey
> [#862](https://github.com/anthony-chaudhary/fak/issues/862), and it enters the reward as a
> normalization term, not an afterthought. And because validity is contested
> ([#861](https://github.com/anthony-chaudhary/fak/issues/861)), the whole reward is *gated on an
> empirical validation* against an attribution that is not attention (exact-eviction leave-one-out,
> [#863]/[#866]) — if they do not correlate, attention is not the signal and the same seam takes
> the LOO reward instead.

## 2. The design — one reward equation, three justified terms

Fold the four transferred patterns into a single per-span reward, computed for each candidate
context span `s` at turn `T`:

```
reward(s, T) = success_gate(T) · Σ_{t ≤ T, s resident at t}  γ^(T−t) · m_s(t)
```

where

| Term | What it is | Prior-art justification | Default / off-ramp |
|---|---|---|---|
| `success_gate(T)` | a scalar in [0,1] from the turn's own verdict — only credit attention on a turn that produced a verified-good output | PRM outcome supervision (Lightman 2023); Math-Shepherd automatic step-credit (Wang 2024); Hindsight CA conditioning on the realized outcome (Harutyunyan 2019) | `1.0` (un-gated) reproduces today's behavior — the gate is a strict *tightening*, never a loosening |
| `γ^(T−t)` | the recency discount on attention witnessed `T−t` turns ago | TD(λ) eligibility traces (Sutton & Barto); the GAE λ dial (Schulman 2016) | `γ=1` recovers the un-discounted cumulative accumulator (already shipped, #855) |
| `m_s(t)` | **confound-normalized** attention mass on `s` at turn `t` (raw mass minus the sink/positional baseline), in [0,1] | attention sinks / positional confounds (StreamingLLM, Xiao 2023); the validity contest (Jain & Wallace 2019 / Wiegreffe & Pinter 2019) → #862 subtracts the confounds, #861/#866 validates the residual | raw mass `a_s(t)` is the `m = a` degenerate case — and the whole signal is swappable for LOO if it fails validation |

Each term is **degenerate-to-current**: set `success_gate=1`, `γ=1`, `m=a` and the equation is
exactly the flat cumulative attention the substrate already accumulates. That is deliberate — the
design adds three *named, separately-ablatable* knobs onto the shipped reward, so the experiment
([#866]) can turn each on and measure its contribution rather than shipping a monolith.

The equation is also **signal-agnostic in `m`**: it is the attention *instantiation* of a general
span-credit reward. If [#861]/[#866] refute attention, replace `m_s(t)` with the per-span
leave-one-out delta ([#863]) and every other term — the outcome gate, the recency discount, the
fleet-wide scope — still holds. The design's contribution is the *credit-assignment structure*,
not the attention measurement.

## 3. Mapping onto fak — the seams already exist

The reward does not need new substrate. It is a tightening of three functions that landed under
the attention-witness epic. (Symbols verified against HEAD; cited by symbol + test, not line.)

### 3.1 `ctxplan.Outcome` is the credit interface; `OutcomeFromAttention` is the producer

`ctxplan.Outcome{Hits, Faults, Wasted}` (in `internal/ctxplan/learn.go`) is the witnessed
feedback the planner closes its loop over. `OutcomeFromAttention` (in
`internal/ctxplan/outcome_attention.go`, #858) is the production producer: a resident span
witnessed at attention mass ≥ `DefaultHitThreshold` (0.01) is a `Hit`, below it is `Wasted`, an
elided span demand-paged back is a `Fault`; a pin is a `Hit` by construction; an unwitnessed
resident span teaches nothing (fail-closed). This is the **un-gated, un-discounted, raw-mass**
version of §2's equation — `success_gate=1`, `γ=1`, `m=a`. The three design terms attach here:

- **Outcome gate (term 1).** `OutcomeFromAttention` / `LearnFromAttention` today credit a `Hit`
  regardless of whether the turn was good. The gate is a `success_gate(T)` argument: when the turn
  failed its verdict, fold an *empty* (or down-weighted) Outcome, so a wrong turn's attention does
  not promote the spans that misled it into the next plan's intents. The verdict source is the
  honest hard part — the fak-shaped answer is the same evidence the RSI loop already trusts (a
  real `TruthClean`/`SuiteGreen` verdict, a downstream tool-success, a benchmark pass), never the
  model's self-report. `Hit`/`Wasted`/`Fault` already separate "drew on this" from "idled"; the
  gate decides *whether this turn's separation counts*.
- **Recency discount (term 2) is partly shipped.** The rolling per-span accumulator (#855,
  `5dc7e7e`) already keeps **EMA + cumulative** variants — and an EMA *is* a geometric recency
  discount (`γ ≈ 1 − α`). The cumulative variant is the `γ=1` baseline. So term 2 is not new code;
  the design's claim is to *name the EMA's decay as the eligibility-trace γ* and let the
  experiment pick it, rather than treat EMA-vs-cumulative as an arbitrary smoothing choice.
- **Confound normalization (term 3) is #862's output.** Feed `OutcomeFromAttention` the
  sink/positional-normalized attribution, not raw mass — i.e. the `Attribution` it consumes is
  already `m`, produced by #862. `SignalNoiseFromAttention` (in
  `internal/ctxplan/signalnoise.go`) is the sibling consumer of the same attribution and gets the
  same normalized input for free.

### 3.2 `Forecast.Learn` / `Weights.Learn` are the consumers — already wired for a graded reward

`Forecast.Learn` promotes the content-tokens of faulted spans into the predicted `Intents` (the
planner "learns what to predict" from where it was wrong). `Weights.Learn` runs one online
logistic-gradient step that labels every needed span (`Hit` or `Fault`) `y=1` and every `Wasted`
span `y=0`, tuning the cost knobs (`Relevance/Utility/Durability/Recency/Primacy`) so the score
better separates needed from wasted. Two things matter for this design:

1. **The learner is already gradient-graded, so a graded reward fits without a rewrite.** Today
   the label is binary (1 for Hit/Fault, 0 for Wasted). The outcome gate and recency discount
   turn that into a *weighted* label — a `Hit` on a successful, recent turn carries more gradient
   than one on an old or failed turn. That is a per-row sample weight on the existing gradient,
   not a new learner.
2. **There is already a span-level Q-value slot to persist the reward into.** `Weights.Utility`
   reads `recall.Page.Utility` off `Cell.Attrs["utility"]` — a witness-gated, `[0, UtilityMax]`
   (4.0) clamped utility the planner already scores with. The accumulated `reward(s,T)` is exactly
   the production witness that slot was waiting for: the rolling, success-gated, discounted
   attention reward *is* the learned outcome-utility, fed back so a span that has earned attention
   over a session scores higher next turn. This closes the loop the `Utility` signal was designed
   for but had no production producer.

### 3.3 `rsiloop` is the outer keep-or-revert — a new fitness axis

`internal/rsiloop` (`Harness`, `Measurement{Metric, SuiteGreen, TruthClean}`, the non-forgeable
`shipgate.Evaluate` keep-bit) is where a *candidate planner configuration* is kept only if it
shows a real, suite-green, truth-clean gain over latest `main`. The reward gives it a new
`Metric`: **witnessed attention signal-to-noise over a turn sequence** (the fraction of resident
attention mass that landed on spans the success-gated reward credits). A candidate forecast/weight
configuration that *raises* witnessed S/N turn-over-turn is KEPT; one that does not is REVERTED.
Because the metric is derived from the model's own witnessed attention (not the loop author's
number), it inherits rsiloop's anti-forgery property — the reward cannot be faked into a keep.

## 4. The honest boundary (carried verbatim from the epic)

1. **Attention-as-importance is CONTESTED.** This design does not assume attention = value. Term 3
   exists *because* the raw signal has confounds, and the whole reward is gated on the empirical
   validation in [#866] (correlate the attention reward against exact-eviction leave-one-out — the
   fak edge, since fak owns the forward pass and can compute a true LOO). If they do not correlate,
   the deliverable is the *alternative* ([#863]: gradient × attention, leave-one-out, influence
   functions) wired to the same `Outcome` interface, not a forced attention reward.
2. **The outcome gate needs a TRUSTWORTHY verdict.** The whole value of term 1 collapses if
   `success_gate(T)` is the model's self-assessment. It must be a witnessed verdict (the same
   discipline rsiloop's `TruthClean`/`SuiteGreen` keep-bit already enforces). Where a turn has no
   such verdict, the conservative default is to *not credit* (gate→0), not to credit ungated —
   fail-closed, exactly as `OutcomeFromAttention` already skips unwitnessed spans.
3. **Shadow-first.** The shipped producer is pure and off the planning path: producing the
   Outcome and even folding it through the learners changes no live plan until a driver adopts the
   revised Forecast/Weights (#858's two-posture pattern). This design keeps that posture — record
   the reward, validate it against LOO, and only *then* let it steer a plan.

## 5. Deliverable summary (the spec the experiment implements)

The span-credit-assignment reward is `reward(s,T) = success_gate(T) · Σ γ^(T−t) · m_s(t)`, with
each term justified by a named prior-art pattern (PRM/Math-Shepherd outcome gate; TD(λ)/GAE
recency discount; attention-sink/explanation-contest confound normalization) and each mapped onto
a fak seam that already ships (`ctxplan.OutcomeFromAttention` produces it, `Forecast.Learn` /
`Weights.Learn` consume it into the `recall.Page.Utility` Q-slot, `rsiloop` keeps-or-reverts on
witnessed S/N). Every term is degenerate-to-current (set the three knobs to identity and the
equation is the shipped flat cumulative attention), so [#866] can ablate each term independently.
The producer is a swap point: if attention fails validation, the same interface takes the LOO
reward ([#863]). This note is survey 4/5; the defensible-novelty boundary + one-sentence claim
that gates the experiment is sibling [#865].

### References

- Lightman et al., ["Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050), 2023 (PRM800K, process supervision).
- Wang et al., ["Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"](https://arxiv.org/abs/2312.08935), 2024 (automatic process reward).
- Sutton & Barto, [*Reinforcement Learning: An Introduction*, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html) (TD(λ), eligibility traces).
- Schulman et al., ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438), 2016 (the λ bias/variance dial).
- Harutyunyan et al., ["Hindsight Credit Assignment"](https://arxiv.org/abs/1912.02503), NeurIPS 2019.
- Asai et al., ["Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"](https://arxiv.org/abs/2310.11511), 2023.
- Shazeer et al., ["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538), 2017.
- Abnar & Zuidema, ["Quantifying Attention Flow in Transformers"](https://arxiv.org/abs/2005.00928) (attention rollout), 2020.
- Xiao et al., ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453) (StreamingLLM), 2023.
- Zhang et al., ["H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"](https://arxiv.org/abs/2306.14048), NeurIPS 2023.
- Jain & Wallace, ["Attention is not Explanation"](https://arxiv.org/abs/1902.10186), NAACL 2019; Wiegreffe & Pinter, ["Attention is not not Explanation"](https://arxiv.org/abs/1908.04626), EMNLP 2019.
- Cohen-Wang et al., ["ContextCite: Attributing Model Generation to Context"](https://arxiv.org/abs/2409.00729), 2024.
