---
title: "Span-reward novelty boundary: H2O, ContextCite, PRMs, and the defensible claim"
description: "Prior-art synthesis 5/5 for the reward-over-spans epic (#860): the honest novelty boundary for attention-as-span-reward, including what H2O/SnapKV/Quest, ContextCite, influence methods, and PRMs already own; what fak adds; the one-sentence defensible claim; and the forbidden overclaims downstream docs must avoid."
slug: span-reward-novelty-boundary
keywords:
  - span reward
  - attention as importance
  - H2O
  - SnapKV
  - Quest
  - ContextCite
  - influence functions
  - leave-one-out attribution
  - process reward models
  - PRM
  - RSI
  - novelty boundary
date: 2026-06-26
---

# Span-reward novelty boundary: H2O, ContextCite, PRMs, and the defensible claim

This is the deliverable for
[#865](https://github.com/anthony-chaudhary/fak/issues/865), the final survey under the
reward-over-spans epic
([#860](https://github.com/anthony-chaudhary/fak/issues/860)). Its job is not to maximize
the claim. Its job is to keep the claim alive under hostile review.

The short version:

> fak does not claim attention proves causality; it claims a kernel can turn per-span
> context credit into a witnessed, replayable, falsifiable reward signal by measuring
> attention in its own forward pass, checking it against exact-eviction leave-one-out, and
> only letting RSI keep changes when an external witness confirms they improved the
> objective.

Everything downstream should cite this sentence or stay weaker.

## The hard boundary

Attention mass is a **candidate proxy**, not a causal proof. The shipped substrate should
therefore be described as a signal pipeline:

1. Measure a span-level signal from the model run itself.
2. Persist enough evidence to replay or audit where the signal came from.
3. Correlate the signal against an ablation-grade fallback, preferably exact-eviction
   leave-one-out when the backend supports it.
4. Gate any RSI/planner change on an independent outcome witness, not on the model saying a
   span was useful.

If step 3 fails, the attention producer is rejected for that model/task. The `Outcome`
interface still survives: it can carry leave-one-out, gradient, or another attribution signal
instead of attention. That is the point of the design, and it is the safety valve that keeps
this from being "attention is explanation" in new clothes.

Local seams this memo relies on:

- `internal/kvmmu/accumulator.go` records cumulative/EMA attention-style signals and already
  labels the `lambda < 1` shape as the H2O/heavy-hitter signal.
- `internal/ctxplan/outcome_attention.go` turns per-span attention into shadow
  `Hits`/`Wasted`/`Faults` outcomes and folds them into forecast/weight learners only when a
  caller adopts the revised values.
- `internal/model/span_reward_shadow.go` runs shadow leave-one-out checks and marks the signal
  `CORRELATE`, `REFUTE`, or `INSUFFICIENT`.
- `docs/explainers/addressable-kv-cache.md` is the exact-eviction boundary: claim exact span
  removal only at the evidence level that has actually been witnessed.
- `docs/industry-scorecard/memory.md` already fences fak from H2O/SnapKV/Quest style KV
  compression claims.

## Novelty table

| Prior art | What it already owns | Same-as-prior-art dimension | What fak may claim, if witnessed |
|---|---|---|---|
| H2O / Heavy-Hitter Oracle | Uses accumulated attention to identify important tokens and retain heavy-hitter plus recent tokens under a KV-cache budget. Source: <https://arxiv.org/abs/2306.14048>. | If fak only accumulates attention over raw tokens and ranks them for retention, that is H2O with new names. Accumulated attention, heavy-hitter ranking, and eviction under a fixed cache budget are not novel. | Span/tool-result granularity; a replayable kernel witness from fak's own forward pass; use as a reward/benchmark/RSI input rather than only an eviction heuristic; validation against exact span eviction or another ablation signal. |
| SnapKV | Uses prompt/observation-window attention patterns to select clustered important KV positions per head and compress long-context KV state. Source: <https://arxiv.org/abs/2404.14469>. | Attention-derived KV selection and cache compression are not novel. "The model already reveals useful prompt positions before generation" is not a fak claim. | fak is not claiming a better KV compressor here. The delta is semantic agent units plus persisted evidence plus falsification and reward use. |
| Quest | Uses query-aware KV page selection: estimate which cached pages are critical for a query, then load only top pages for attention. Source: <https://arxiv.org/abs/2406.10774>. | Query-aware criticality and sparse/page-level attention selection for inference speed are not novel. | fak may say it scores semantic context spans after, or during, a witnessed agent run. It should not say this beats Quest on self-attention speed or sparse-attention quality unless a direct benchmark exists. |
| ContextCite | Defines context attribution for language-model outputs and uses surrogate/ablation-style modeling to assign credit to context chunks. It applies that to verification, pruning, and poisoning analysis. Source: <https://arxiv.org/abs/2409.00729>. | Context attribution, chunk credit, and "remove important context to see output impact" are not novel. | fak may claim a kernel-owned implementation path: the signal is produced from fak's own forward/cache state, and exact-eviction leave-one-out can be used as the falsification witness for resident spans. ContextCite still owns the broad attribution framing. |
| Influence functions / perturbation attribution | Estimates how training examples or features affect predictions, including counterfactual-style reasoning about removals or perturbations. Source: <https://arxiv.org/abs/1703.04730>. | Counterfactual influence and leave-one-out attribution are not novel. | fak's narrower contribution is operational: resident context spans can be ablated through the kernel/cache path and tied to a replayable agent artifact. |
| Process Reward Models / PRM800K | Rewards intermediate reasoning steps rather than only final answers; process supervision can outperform outcome-only supervision on reasoning tasks. Source: <https://arxiv.org/abs/2305.20050>. | Process-level reward and outcome-vs-process framing are not novel. | fak transfers the shape from reasoning steps to context spans/tool results, with a kernel witness and task/outcome gate. |
| Math-Shepherd | Builds process-wise supervision automatically and uses it for verification and reinforcement learning. Source: <https://arxiv.org/abs/2312.08935>. | Outcome-gated or rollout-derived step credit is not novel. | fak may use the same honest pattern: credit spans only on verified-good outcomes, and treat attention on bad outcomes as anti-signal or diagnostic evidence. |
| Attention-as-explanation debate | Jain and Wallace show attention weights often fail as explanations; Wiegreffe and Pinter argue the conclusion depends on definition and test design. Sources: <https://arxiv.org/abs/1902.10186>, <https://arxiv.org/abs/1908.04626>. | "Attention means importance" is contested and cannot be asserted as a fact. | fak may call attention a witnessed proxy only after correlation against ablation evidence. Without that correlation, attention remains telemetry. |

## Exact deltas to preserve

| Dimension | Prior-art baseline | fak's defensible delta |
|---|---|---|
| Granularity | Tokens, KV positions, KV pages, document/context chunks, or reasoning steps. | Agent semantic units: tool results, messages, retrieved docs, sub-agent outputs, and other ctxmmu spans. |
| Persistence | Often transient serving state or an offline attribution report. | Persisted witness metadata that can be linked back to the run, span identity, and outcome gate. |
| Purpose | KV eviction/compression, sparse attention, verification, pruning, poisoning analysis, or reward modeling over reasoning steps. | Reward/benchmark/RSI over agent context units, subject to witness gates. |
| Producer | Attention heuristic, surrogate model, perturbation method, human labels, or rollout-generated process labels. | First producer can be fak's own forward-pass attention; fallback producers can be LOO/gradient/ablation signals through the same `Outcome` contract. |
| Falsification | Task accuracy under compression, attribution metrics, or step-label accuracy. | Exact-eviction leave-one-out when available; otherwise an explicitly named shadow/ablation witness with a `CORRELATE`/`REFUTE`/`INSUFFICIENT` verdict. |
| Non-forgeability | Usually not a systems boundary. | Only the keep/revert decision is non-forgeable: RSI can use the reward only after an independent outcome witness confirms the candidate improved the objective. The attention number itself is not trusted just because the model emitted it. |

## Allowed claim language

Use these phrases:

- "witnessed span-credit proxy"
- "attention-derived candidate reward, validated against leave-one-out"
- "semantic context-unit credit, not raw-token heavy-hitter eviction"
- "shadow reward until correlation is established"
- "RSI input gated by external outcome evidence"

Avoid stronger phrasing unless a later benchmark explicitly proves it.

## Forbidden overclaims

- "Attention proves the span caused the output."
- "This reward is the true value of a span."
- "fak invented accumulated attention, attention heavy hitters, or attention-based KV
  retention."
- "fak is H2O/SnapKV/Quest but better" unless a same-model, same-budget compression
  benchmark exists.
- "fak replaces ContextCite" or "fak invented context attribution."
- "fak invented process rewards" or "fak invented outcome-gated intermediate credit."
- "The kernel controls attention as a hardware/security boundary." The kernel can witness,
  store, ablate, and gate. It does not make attention intrinsically truthful.
- "The reward is non-forgeable because the model emitted it." Non-forgeability enters only at
  the witness-gated keep/revert boundary.
- "Exact eviction is proven for every production checkpoint, serving backend, and cache layout."
  Say exactly which backend/model/cache witness exists.
- "The reward already improves live planning or RSI" unless the improvement survives the
  repository's keep/revert gate and has a committed artifact.
- "Attention on a failed turn is positive training signal." Failed-turn attention is at best
  diagnostic and often anti-signal.

## Downstream rule

Any downstream doc that wants to say "span reward" must answer three questions in the same
paragraph or nearby evidence block:

1. What was the producer: attention, exact leave-one-out, gradient, surrogate, or another
   named signal?
2. What witness makes it replayable or falsifiable?
3. What outcome gate prevents the model from rewarding its own preferred context?

If those answers are absent, call it telemetry, not reward.

## Sources

Primary sources:

- H2O: <https://arxiv.org/abs/2306.14048>
- SnapKV: <https://arxiv.org/abs/2404.14469>
- Quest: <https://arxiv.org/abs/2406.10774>
- ContextCite: <https://arxiv.org/abs/2409.00729>
- Understanding Black-box Predictions via Influence Functions: <https://arxiv.org/abs/1703.04730>
- Let's Verify Step by Step / PRM800K: <https://arxiv.org/abs/2305.20050>
- Math-Shepherd: <https://arxiv.org/abs/2312.08935>
- Attention is not Explanation: <https://arxiv.org/abs/1902.10186>
- Attention is not not Explanation: <https://arxiv.org/abs/1908.04626>

Local source files:

- `docs/industry-scorecard/memory.md`
- `docs/explainers/addressable-kv-cache.md`
- `docs/notes/RESEARCH-credit-assignment-over-spans-2026-06-25.md`
- `internal/kvmmu/accumulator.go`
- `internal/ctxplan/outcome_attention.go`
- `internal/model/span_reward_shadow.go`
