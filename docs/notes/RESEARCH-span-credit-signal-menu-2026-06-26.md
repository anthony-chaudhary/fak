---
title: "Span-credit signal menu: attention, gradients, leave-one-out, influence, and the fak fallback"
description: "Prior-art survey 3/5 for the reward-over-spans epic (#860): ranked alternatives to attention for assigning credit to context spans, and the recommended two-stage design for fak: attention proposes candidates, exact-eviction leave-one-out verifies them when the cache witness supports it."
slug: span-credit-signal-menu
keywords:
  - span reward
  - credit assignment
  - attention as importance
  - leave-one-out attribution
  - exact eviction
  - gradients
  - integrated gradients
  - influence functions
  - TracIn
  - ContextCite
  - RSI
date: 2026-06-26
---

# Span-credit signal menu: attention, gradients, leave-one-out, influence, and the fak fallback

This is the deliverable for
[#863](https://github.com/anthony-chaudhary/fak/issues/863), the alternatives survey under
the reward-over-spans epic
([#860](https://github.com/anthony-chaudhary/fak/issues/860)). Its job is to answer the
fallback question: if attention is too noisy to be a reward, what signal should fak use
instead?

The answer is a ranked menu, not a single magic signal. The recommended default is:

> attention proposes, exact eviction disposes.

Use attention-derived span reward only as a cheap prior for ranking candidates. Then run
exact-eviction leave-one-out on the top and bottom candidates and promote the attention
proxy only if it correlates with that ablation witness. If it does not correlate, keep the
same `Outcome`/reward seam but feed it leave-one-out deltas, gradients, or another named
producer instead of attention.

## Ranked menu

| Rank | Signal | Faithfulness | Cost | fak witnessability | Best use | Failure mode |
|---|---|---|---|---|---|---|
| 1 | Exact-eviction leave-one-out for selected spans | Highest practical counterfactual signal when the backend can remove the resident span exactly and re-decode the probe. | `O(K * probe_decode)` for `K` checked spans; `O(S * probe_decode)` if every span is checked. | High for supported in-kernel cache states: `internal/model/span_reward_shadow.go` records `LeaveOneOutDelta`, `EvictRepositionMaxDiff`, and `EvictVsRecomputeMaxDiff`. | Gold witness for top/bottom candidates; labels for later proxy training; final fallback when attention is refuted. | Still not free; exact only at the witnessed cache boundary. Middle-context eviction may differ from a full no-span recompute, so `EvictVsRecomputeMaxDiff` must be reported. |
| 2 | Top/bottom-K exact-eviction leave-one-out | Almost rank 1 for the decision the planner needs: prove that the best-looking and worst-looking candidates are real. | `O(min(S, 2K) * probe_decode)` plus the already-paid forward telemetry. | High: `SpanRewardOptions.TopBottomK` already checks only reward extremes and the tests assert that behavior. | The default verifier for #866: cheap enough to run as a shadow gate, strong enough to refute a bad prior. | Can miss mid-ranked spans. Quality depends on the prior that selected the extremes. |
| 3 | Gradient attribution: Integrated Gradients, Grad x Input, Attention x Gradient, gradients w.r.t. span KV | Strong local sensitivity signal: it asks how the output/loss changes under differentiable perturbation of a span representation. | Usually cheaper than full leave-one-out over many spans if a backward path is available; more expensive and more invasive than attention telemetry. | Medium. It requires a gradient-capable model path, retention of the right activations/KV tensors, and a witness format fak does not currently ship for live inference. | Fallback when exact eviction is unavailable; training/fine-tuning loops; a second opinion when attention and LOO disagree. | Not causal truth by itself. Saturation, baseline choice, gradient noise, non-linear interactions, and missing backward support can mislead. |
| 4 | Context attribution / ContextCite-style ablation or surrogate attribution | Directly addresses the same question: which context chunks supported the generated answer? | Medium to high, depending on sampling, surrogate fit, and how many chunks are ablated. | Medium. Good offline benchmark or external witness, but not automatically kernel-owned unless bound to the fak run artifact. | Cross-check fak's LOO labels; evaluate pruning/poisoning decisions; compare against prior-art context-attribution methods. | Surrogate mismatch and sampling error; does not by itself prove a fak cache-state counterfactual. |
| 5 | Influence functions and TracIn | Faithful for training-data or training-trajectory influence under their assumptions; weaker fit for live prompt-span credit. | High: needs gradients, checkpoints, Hessian approximations or checkpoint trajectories, and often training data access. | Low to medium. Useful when fak owns or records the training/update path; poor first choice for a live prompt cache. | Training provenance, dataset/debug influence, long-horizon RSI analysis. | Answers "which training example/update mattered?", not necessarily "which resident context span mattered in this turn?" |
| 6 | Attention-derived normalized reward | Cheapest signal: already produced during the forward pass and easy to persist per span. | Near zero incremental cost after attention telemetry is wired. | High: fak can witness the model-run telemetry, normalize confounds, and store a replayable report. | Candidate generator, sort key, cheap prior, regression telemetry. | The attention-as-explanation debate applies. Treat as reward only after correlation against LOO or another ablation witness. |
| 7 | Learned surrogate reward over spans | Potentially cheap after enough labels exist. | Training plus drift monitoring; cheap at inference once trained. | Medium if trained from fak-witnessed labels and versioned with the run; low if imported as an opaque scorer. | Amortize exact-eviction labels once #866 accumulates enough data. | Learns the biases of its labels and can silently drift out of domain. |

The ordering is by faithfulness times cost for fak's current substrate, not by general
academic prestige. Full leave-one-out is the strongest measurement, but top/bottom-K LOO is
the deployable version because a planner usually needs to know whether its kept and dropped
candidates are sensible, not compute a perfect attribution vector for every span on every
turn.

## Recommended design

Use a three-stage producer/verifier pipeline:

1. **Outcome gate first.** A span reward is only positive on a witnessed-good turn or task.
   Failed-turn attention is diagnostic or anti-signal, never positive training evidence.
2. **Cheap prior.** Compute attention-derived reward for all spans:
   `reward(s,T) = success_gate(T) * recency_discount(s,T) * normalized_attention_mass(s,T)`.
   This is the existing #864/#865 shape, but it remains telemetry until validated.
3. **Exact verifier.** Run exact-eviction leave-one-out on the top and bottom candidates from
   that prior. `internal/model/span_reward_shadow.go` already produces a closed verdict:
   `CORRELATE`, `REFUTE`, or `INSUFFICIENT`.

The promotion rule should be mechanical:

- If normalized attention reward beats raw attention and reaches the pre-registered Spearman
  threshold against checked leave-one-out deltas, the attention producer may be used as a
  cheap proxy for that model/task class.
- If the report is `REFUTE`, attention is not the reward for that slice. Use the checked LOO
  deltas as labels or direct reward values and keep the `Outcome` interface unchanged.
- If the report is `INSUFFICIENT`, do not flip live planning or RSI. Gather more checked
  spans, more turns, or a different producer.

This preserves the novelty boundary from
[`RESEARCH-span-reward-novelty-boundary-2026-06-26.md`](RESEARCH-span-reward-novelty-boundary-2026-06-26.md):
fak is not claiming that attention proves causality. fak is claiming that an agent kernel can
measure a cheap proxy from its own forward pass, falsify it with a stronger cache-state
counterfactual, and gate any self-improvement keep on an external outcome witness.

## Why exact eviction changes the economics

Generic leave-one-out context attribution usually means rebuilding or re-querying the model
for many perturbed prompts. That is expensive and often lives outside the serving runtime.
fak has a narrower but stronger path when the context span is resident in a cache it controls:
clone the cache, evict the span, re-decode the probe, and measure the final-logit delta.

That does not make LOO free. It makes selected-span LOO practical enough to use as a shadow
gate:

- Attention cost is `O(forward telemetry already paid)`.
- Full LOO cost is `O(S * probe_decode)` for `S` candidate spans.
- Top/bottom-K LOO cost is `O(min(S, 2K) * probe_decode)`.
- Gradient attribution cost is roughly one or more backward passes plus activation/KV
  retention, if the backend exposes them.
- Influence/TracIn cost depends on stored training checkpoints or gradients and is usually
  outside the live prompt-cache loop.

The useful fak angle is therefore not "LOO is free." It is "the kernel can cheaply verify the
most decision-relevant candidates under a cache-state witness it owns."

## Prior-art basis

**Gradient attribution.** Integrated Gradients supplies axioms and a path integral from a
baseline to the input; Grad x Input and saliency-map methods use local gradients as
sensitivity. These are better-motivated than plain attention when the question is "what would
change the output?", but they still depend on baseline/perturbation choices and on a
gradient-capable path.

**Occlusion / leave-one-out.** Occlusion sensitivity asks the direct counterfactual question:
hide or remove part of the input and observe the output change. For span reward this is the
cleanest interpretation: remove span `s`, re-run or re-decode, measure delta. fak's exact
cache eviction is a system-level implementation of that counterfactual for supported resident
spans.

**Influence methods.** Influence functions and TracIn are useful when the object of credit is
a training example, checkpoint update, or learning trajectory. They should be cited as
adjacent credit-assignment tools, not as the first prompt-span reward. Their evidence is most
useful if fak later wires RSI/training updates whose provenance must be audited.

**Context attribution.** ContextCite is the closest external framing: attribute generated text
to context chunks and use that for verification, pruning, and poisoning analysis. The honest
fak delta is operational, not conceptual: a kernel-owned cache witness can validate a subset
of span attributions with exact eviction and persist the run artifact.

**Attention debate.** The "attention is not explanation" exchange is exactly why attention is
ranked below ablation and gradients. Attention is still valuable because it is already paid for
and covers all spans. It is a proposal distribution, not the adjudicator.

## Mapping to current code

- `internal/model/span_reward_shadow.go` implements the recommended verifier: attention-based
  normalized reward plus exact KV-eviction LOO, with `TopBottomK`, `CheckedRows`,
  `SpearmanRewardDelta`, `ConfoundNormalizationImproved`, and closed verdicts.
- `internal/model/span_reward_shadow_test.go` already proves the top/bottom-K sampling shape
  and records the exactness boundary through `EvictVsRecomputeMaxDiff`.
- `internal/ctxplan/snfitness.go` keeps the RSI fitness path shadow-only until #866 proves
  correlation against exact-eviction LOO. That is the right hold: no live planner flip just
  because attention telemetry exists.
- `internal/ctxplan/outcome_attention.go` and the forecast/weight learners are the consumer
  seam. If #866 refutes attention, the same seam can consume LOO deltas or another
  `Outcome` producer.

## Downstream rule

Any future doc or benchmark that says "span reward" must name the producer and verifier:

- **producer:** attention, exact LOO, gradient, ContextCite-style attribution, influence, or
  a trained surrogate;
- **verifier:** exact eviction, full recompute, gradient audit, external attribution report,
  or no verifier yet;
- **promotion state:** proxy accepted, proxy refuted, insufficient data, or direct LOO label.

If those fields are absent, call the number telemetry, not reward.

## Sources

Primary sources:

- Integrated Gradients / Axiomatic Attribution for Deep Networks: <https://arxiv.org/abs/1703.01365>
- Deep Inside Convolutional Networks / saliency maps: <https://arxiv.org/abs/1312.6034>
- Visualizing and Understanding Convolutional Networks / occlusion sensitivity: <https://arxiv.org/abs/1311.2901>
- Understanding Black-box Predictions via Influence Functions: <https://arxiv.org/abs/1703.04730>
- TracIn / Estimating Training Data Influence by Tracing Gradient Descent: <https://arxiv.org/abs/2002.08484>
- ContextCite: <https://arxiv.org/abs/2409.00729>
- Attention is not Explanation: <https://arxiv.org/abs/1902.10186>
- Attention is not not Explanation: <https://arxiv.org/abs/1908.04626>

Local source files:

- `internal/model/span_reward_shadow.go`
- `internal/model/span_reward_shadow_test.go`
- `internal/ctxplan/snfitness.go`
- `internal/ctxplan/outcome_attention.go`
- `docs/notes/RESEARCH-credit-assignment-over-spans-2026-06-25.md`
- `docs/notes/RESEARCH-span-reward-novelty-boundary-2026-06-26.md`
