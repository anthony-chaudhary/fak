---
title: "Is witnessed span-attention a valid reward? Verdict, aggregation, and failure modes"
description: "Prior-art survey 1/5 for the reward-over-spans epic (#860): the attention-as-explanation debate, what it permits fak to claim, and the exact aggregation rule for using witnessed span attention as a reward proxy only after ablation validation."
slug: witnessed-span-attention-validity
keywords:
  - span reward
  - attention as explanation
  - attention attribution
  - attention rollout
  - attention flow
  - leave-one-out attribution
  - exact eviction
  - RSI
date: 2026-06-26
---

# Is witnessed span-attention a valid reward? Verdict, aggregation, and failure modes

This is the deliverable for
[#861](https://github.com/anthony-chaudhary/fak/issues/861), the first prior-art survey
under the reward-over-spans epic
([#860](https://github.com/anthony-chaudhary/fak/issues/860)). The question is narrow:
before fak uses attention mass as a reward over context spans, is attention a valid
importance signal at all?

Verdict:

> Witnessed span-attention is defensible as a cheap, relative, span-level proposal signal.
> It is not defensible as a standalone causal explanation or as an accepted reward until it
> correlates with an ablation witness such as exact-eviction leave-one-out.

That verdict is deliberately weaker than "attention is explanation." It is also stronger
than "attention is useless." The literature says raw attention can be noisy and manipulable,
but it also retains signal under better tests, especially when used as a relative ranking and
checked against perturbation, gradient, or ablation evidence.

## What the debate permits

| Source | What it says | Rule for fak |
|---|---|---|
| Jain and Wallace, "Attention is not Explanation" | Attention weights can be poorly correlated with gradient importance, and different attention distributions can preserve similar predictions. Source: <https://arxiv.org/abs/1902.10186>. | Do not call raw attention a causal explanation. Do not reward spans solely because a heatmap is high. |
| Wiegreffe and Pinter, "Attention is not not Explanation" | Whether attention explains depends on the definition of explanation and the test design; adversarial attention alone does not settle usefulness. Source: <https://arxiv.org/abs/1908.04626>. | Keep attention as a candidate signal, but define the test that promotes or refutes it. |
| Serrano and Smith, "Is Attention Interpretable?" | Attention noisily predicts importance but is not a fail-safe indicator; gradient rankings can better predict effects in some cases. Source: <https://arxiv.org/abs/1906.03731>. | Use attention as a prior, not the final label; keep gradient or LOO as fallback producers. |
| Abnar and Zuidema, "Quantifying Attention Flow in Transformers" | Raw attention is less reliable because information mixes across layers; rollout and flow better approximate token relevance against ablation/gradient scores. Source: <https://arxiv.org/abs/2005.00928>. | Avoid single-layer or last-layer heatmaps. Aggregate across layers/heads/queries, and treat rollout/flow as an offline cross-check when the goal is explanation rather than direct KV-read accounting. |

The common denominator is a testable posture: attention may be useful when the claim is
relative and operational, but it must be validated against a stronger witness for the model,
task, and span type in question.

## The aggregation fak should use

For the first reward producer, use the signal fak already witnesses:

1. Emit post-softmax attention rows from the model seam
   (`internal/model/attn_observer.go`).
2. Attribute each row's key-position weights to the owning semantic span through the
   `From`/`Len` ledger (`internal/kvmmu/attention.go`).
3. Sum across heads, layers, and consumer query positions into per-span mass.
4. Normalize by total emitted mass and by the expected attention mass for that span's
   position and length.
5. Use the result as a relative ranking, then validate the top/bottom candidates with
   exact-eviction leave-one-out (`internal/model/span_reward_shadow.go`).

The default aggregation is therefore:

```text
raw_mass(s) = sum over consumer rows r, heads h, layers l, positions p in span(s)
              attention(l, h, r, p)

proxy_reward(s) = success_gate * recency_discount *
                  max(0, raw_mass(s) - expected_mass(position(s), len(s)))
```

Important choices:

- **Consumer rows only.** For reward, score how future/probe tokens read resident context.
  Do not reward a span for attention emitted while constructing itself. The shadow scorer
  already uses this posture by measuring probe rows after `ContextIDs`.
- **Span-level mass, not token heatmaps.** Aggregate over all token positions in the span,
  then normalize for span length. This reduces per-token noise and matches the unit fak can
  evict, quarantine, page, and account for.
- **All observed heads/layers by default.** A last-layer heatmap is too brittle. fak's
  first shipped witness should use the sum of direct reads across emitted rows because the
  cache question is operational: which resident K/V positions did the model read?
- **Rollout/flow as cross-check, not default producer yet.** Attention rollout and flow are
  better suited to "which input influenced the final representation through the residual
  stack?" They require assumptions and machinery beyond the shipped cache-read witness. Use
  them offline if direct attention and LOO disagree, or after fak persists the needed
  per-layer matrix stream cheaply.
- **Relative ranking only.** The proxy's natural use is ordering spans for verification,
  eviction, or review. Absolute values are not portable across models, prompts, or context
  layouts without calibration.

## Promotion rule

Attention is accepted only after the #866-style shadow report proves it for a slice:

- `CORRELATE`: normalized attention reward reaches the pre-registered Spearman threshold
  against checked exact-eviction LOO deltas and beats raw attention. Attention may be used as
  the cheap proxy for the matching model/task/span class.
- `REFUTE`: attention is not the reward for that slice. Keep the `Outcome` consumer seam but
  switch the producer to leave-one-out deltas, gradients, or another attribution signal.
- `INSUFFICIENT`: no live planner or RSI change. Gather more turns, check more spans, or use
  a stronger producer.

This keeps `internal/ctxplan/snfitness.go` honest: the RSI fitness path remains shadow-only
until correlation against exact-eviction LOO exists.

## Failure modes to guard

- **Adversarial or alternative attention.** A different attention distribution may preserve
  output behavior. Guard: never treat attention as causal without ablation validation.
- **Layer mixing and residual paths.** Raw per-layer attention does not track all information
  flow through a Transformer. Guard: aggregate all observed rows for the operational cache
  signal; use rollout/flow or gradient checks for explanation claims.
- **Attention sinks and position bias.** BOS, system preambles, prefix positions, recent
  tokens, and beginning/end spans can draw position-driven mass. Guard: #862's expected
  attention-by-position/length baseline is mandatory.
- **Length bias.** Longer spans can collect more mass because they own more key positions.
  Guard: subtract or divide by a length-aware expected mass before ranking.
- **Task failure.** A failed turn can strongly attend to the wrong span. Guard: positive
  reward requires an external success gate; failed-turn attention is diagnostic or negative
  evidence.
- **Head specialization.** Some heads attend for syntax, delimiters, sink behavior, or copy
  mechanics rather than semantic utility. Guard: aggregate for the cheap prior, and let LOO
  validate the decision-relevant extremes.
- **Prompt-injection focus.** Malicious or distracting spans can attract attention because
  they are salient. Guard: reward is downstream of the same capability and outcome gates;
  attention alone never clears a trust boundary.

## Decision for the reward epic

Use witnessed span-attention in the epic, but name it accurately:

- Accepted name before validation: **attention-derived candidate reward**.
- Accepted name after validation: **witnessed span-credit proxy validated against
  exact-eviction LOO**.
- Forbidden name: **causal attention reward**.

The implementation order is:

1. Keep the current post-softmax row witness and span attribution ledger.
2. Ensure reward scoring uses consumer/probe rows, not span-construction rows.
3. Apply #862 confound normalization before any correlation claim.
4. Run #866 exact-eviction LOO on top/bottom candidates.
5. Promote attention only for slices that return `CORRELATE`.

If this fails, #863 already defines the fallback menu: use exact LOO deltas directly, or use
gradient/context-attribution producers through the same consumer seam.

## Sources

Primary sources:

- Attention is not Explanation: <https://arxiv.org/abs/1902.10186>
- Attention is not not Explanation: <https://arxiv.org/abs/1908.04626>
- Is Attention Interpretable?: <https://arxiv.org/abs/1906.03731>
- Quantifying Attention Flow in Transformers: <https://arxiv.org/abs/2005.00928>

Local source files:

- `internal/model/attn_observer.go`
- `internal/kvmmu/attention.go`
- `internal/kvmmu/accumulator.go`
- `internal/model/span_reward_shadow.go`
- `internal/ctxplan/outcome_attention.go`
- `internal/ctxplan/snfitness.go`
- `docs/notes/RESEARCH-span-credit-signal-menu-2026-06-26.md`
