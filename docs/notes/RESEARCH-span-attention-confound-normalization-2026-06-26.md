---
title: "Span-attention confound normalization: sinks, position, recency, and length"
description: "Prior-art survey 2/5 for the reward-over-spans epic (#860): the confounds a span-attention reward must subtract before attention mass can become a reward proxy, including attention sinks, massive activations, position bias, lost-in-the-middle, recency, and span length."
slug: span-attention-confound-normalization
keywords:
  - span reward
  - attention sinks
  - positional bias
  - lost in the middle
  - massive activations
  - recency bias
  - confound normalization
  - exact eviction
date: 2026-06-26
---

# Span-attention confound normalization: sinks, position, recency, and length

This is the deliverable for
[#862](https://github.com/anthony-chaudhary/fak/issues/862), the second prior-art survey
under the reward-over-spans epic
([#860](https://github.com/anthony-chaudhary/fak/issues/860)). The question is not whether
attention has any signal; #861 answers that. The question here is what a span-attention
reward must subtract before the remaining mass can be treated as content evidence.

Verdict:

> A span reward must be **excess attention over a null model**, not raw attention. The null
> model must account for sink tokens, non-creditable prompt infrastructure, span length,
> relative position, and distance from the consuming query.

The shape already matches the local shadow scorer:

```text
reward(s,T) =
    success_gate(T)
  * recency_discount(s,T)
  * max(0, raw_attention_mass(s,T) - expected_attention_mass(s,T))
```

The work of #862 is defining `expected_attention_mass`.

## Confounds to subtract

| Confound | Prior-art basis | What goes wrong with raw attention | Required normalization |
|---|---|---|---|
| Attention sinks | StreamingLLM observes that initial tokens can receive strong attention even when they are not semantically important. Source: <https://arxiv.org/abs/2309.17453>. | A span that contains BOS, the first prompt token, or another sink-like prefix token receives fake credit. | Mark sink/preamble positions as non-creditable infrastructure; subtract a sink baseline or exclude them from positive reward. |
| Massive activations / outlier tokens | Massive Activations in LLMs finds very few activations can be extremely large and can concentrate attention probabilities on corresponding tokens. Source: <https://arxiv.org/abs/2402.17762>. | A token can attract attention because it is an architectural/bias carrier rather than task-relevant content. | Track outlier/special tokens as a sink class; cap per-token contribution or subtract an empirical per-position/per-token-class baseline. |
| Beginning/end position bias | Lost in the Middle shows performance often peaks when relevant information is at the beginning or end and degrades in the middle. Source: <https://arxiv.org/abs/2307.03172>. | Beginning/end spans look better than middle spans even when relevance is held constant. | Subtract an expected attention curve by relative position bucket. |
| U-shaped positional attention bias | Found in the Middle ties lost-in-the-middle behavior to a U-shaped attention bias favoring leading and ending contexts regardless of relevance. Source: <https://arxiv.org/abs/2406.16008>. | A raw attention reward learns the layout preference instead of content value. | Calibrate by position: compare a span only against same-length spans at the same relative position/distance bucket, or subtract the learned U-shaped baseline. |
| Recency / end-of-context bias | Attention Sorting reports that sorting highly attended documents later can improve long-context performance, exposing recency-sensitive use of context. Source: <https://arxiv.org/abs/2310.01427>. | Recently appended spans draw mass because they are near the query, not necessarily because they are useful. | Include distance-to-query or turn-age in the null model; keep semantic recency discount separate from raw recency attention. |
| Span length | A longer span owns more key positions and can collect more row mass by surface area. | Big spans look useful by size. | Expected mass must scale with visible token count; the fallback baseline is uniform-visible mass for the span length. |
| Prompt role / pinned roots | Local system, developer, policy, and pinned spans may be required infrastructure. | Required setup text gets rewarded because every turn depends on it. | Treat pinned/root spans as protected or separately accounted; do not let them compete with candidate working-memory spans unless the benchmark explicitly asks for root credit. |

## Normalization contract

For every candidate span `s` consumed by turn/task `T`, compute:

```text
raw_mass(s,T) =
  sum attention(l,h,q,p)
  for consumer query rows q in T
  for all observed layers l and heads h
  for key positions p owned by span s

expected_mass(s,T) =
  E[raw_mass | model, task_class, query_shape,
              span_length_bucket,
              relative_position_bucket,
              distance_to_query_bucket,
              token_class/sink_class,
              prompt_role]

excess_mass(s,T) = max(0, raw_mass(s,T) - expected_mass(s,T))
reward(s,T) = success_gate(T) * semantic_recency_discount(s,T) * excess_mass(s,T)
```

The null model can be implemented in rungs:

1. **Uniform-visible fallback.** If no empirical baseline exists, expected mass is the
   per-row uniform attention share for the span's visible token count. This is the existing
   safe fallback in `internal/model/span_reward_shadow.go`.
2. **Position-length baseline.** Bucket by relative position and span length, then subtract
   the mean attention mass for spans with the same geometry.
3. **Distance/age baseline.** Add distance from the consuming query and turn age. This
   separates "recent therefore attended" from "content useful despite age."
4. **Sink/preamble mask.** Mark BOS/special-prefix/system/policy/pinned roots as
   non-creditable or use a dedicated sink baseline that cannot produce positive reward.
5. **Per-head/layer calibration.** If a head or layer is sink-dominated, subtract its own
   baseline before summing, or downweight it in a documented calibration run.

The first rung is enough for a fail-closed shadow experiment. The second and third rungs are
the minimum for a quotable reward claim on long-context agent traces.

## Concrete spec for fak

Use this producer contract for the #866 experiment:

1. Score only consumer/probe rows, not rows emitted while the span is being constructed.
2. Attribute rows to semantic spans through the `From`/`Len` ledger.
3. Compute `raw_attention_mass` per span.
4. Compute `expected_attention_mass` from the best available baseline:
   - empirical same-position/same-length/same-distance baseline if present;
   - otherwise uniform-visible baseline;
   - sink/preamble spans are excluded from positive reward unless explicitly allowed.
5. Compute `normalized_reward = max(0, raw_attention_mass - expected_attention_mass) *
   success_gate * recency_discount`.
6. Run exact-eviction LOO on top/bottom spans and record whether normalization improved
   Spearman correlation versus raw attention.

The acceptance signal is already named locally:
`SpanRewardReport.ConfoundNormalizationImproved` must be true for an attention producer to
graduate. If raw attention beats normalized attention, the normalization is wrong or the null
model is too coarse; do not promote the reward.

## What not to normalize away

Not every correlation with position is a defect. Fresh tool output may genuinely matter more
than an old plan. A pinned system rule may genuinely be necessary. The rule is:

- **Subtract architectural/layout default.**
- **Keep task-specific excess above that default.**
- **Let exact-eviction LOO decide whether the residual mattered.**

This is why recency appears twice. A distance-to-query baseline removes content-agnostic
recency attention. A separate semantic recency discount can still be used as a product choice
when fresher evidence should matter more.

## Failure cases

- A sink span still wins after masking because it is bundled with real content. Remedy: split
  infrastructure spans from content spans before scoring.
- A middle span is useful but under-attended. Remedy: LOO can rescue it; attention should not
  be the only producer.
- A large tool result wins by length. Remedy: length-normalized expected mass and, where
  needed, per-byte/token cost in the planner utility.
- A model family has different sink behavior. Remedy: baselines are model/task-class
  artifacts, not global constants.
- A failed task attends strongly to a malicious instruction. Remedy: the success gate zeroes
  positive reward, and the security floor remains a separate capability gate.

## Mapping to current code

- `internal/kvmmu/attention.go` supplies the raw span mass by attributing post-softmax rows to
  `From`/`Len` spans.
- `internal/model/span_reward_shadow.go` already carries `ExpectedAttentionMass`,
  `NormalizedReward`, `SuccessGate`, `RecencyDiscount`, and
  `ConfoundNormalizationImproved`.
- `SpanRewardSegment.ExpectedAttentionMass` is the injection point for an empirical
  position/length/recency baseline.
- The current fallback baseline is deliberately conservative: when no explicit expectation is
  supplied, the scorer uses the uniform-visible expectation implied by the attention rows.
- `internal/ctxplan/snfitness.go` correctly holds live RSI/planner promotion until #866 proves
  correlation against exact-eviction leave-one-out.

## Downstream rule

Any downstream result that quotes a span reward must report four fields:

- `raw_attention_mass`
- `expected_attention_mass`
- `normalized_reward`
- the verifier verdict: `CORRELATE`, `REFUTE`, or `INSUFFICIENT`

If `expected_attention_mass` is omitted, call the number raw telemetry, not reward.

## Sources

Primary sources:

- StreamingLLM / Efficient Streaming Language Models with Attention Sinks: <https://arxiv.org/abs/2309.17453>
- Massive Activations in Large Language Models: <https://arxiv.org/abs/2402.17762>
- Lost in the Middle: <https://arxiv.org/abs/2307.03172>
- Found in the Middle / Calibrating Positional Attention Bias: <https://arxiv.org/abs/2406.16008>
- Attention Sorting Combats Recency Bias in Long Context Language Models: <https://arxiv.org/abs/2310.01427>

Local source files:

- `internal/kvmmu/attention.go`
- `internal/model/span_reward_shadow.go`
- `internal/ctxplan/snfitness.go`
- `docs/notes/RESEARCH-witnessed-span-attention-validity-2026-06-26.md`
- `docs/notes/RESEARCH-span-credit-signal-menu-2026-06-26.md`
