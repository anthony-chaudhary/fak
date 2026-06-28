---
title: "idea-scout triage: 'EpiKV' — KV-cache eviction scored by the EPIPHANY SCORE (the change in the model's internal representation, read from the forward pass with no attention matrix) instead of attention weight; a fused-kernel-friendly importance POLICY that lands squarely on fak's policy-agnostic exact-eviction MECHANISM. Verdict: prior art to cite + validation of fak's attention-as-explanation demotion (#861/#862) in the EVICTION-POLICY domain + a candidate forward-pass-readable importance signal for fak's exact evictor — specifically a fused-kernel-compatible ALTERNATIVE to the attention-weight signal fak's already-shipped #856 'attention-informed exact eviction' (evict coldest-by-attention via the bit-exact evictor, kvmmu lane, fed by internal/model/attn_observer.go) uses today. The load-bearing fak-specific point: fak's exact-eviction MECHANISM (kvcache.go Evict / paged_evict.go PagedKV.Evict, #33) is POLICY-agnostic — it evicts a span handed to it — so importance scoring is a swappable input; the shipped #856 policy reads the post-softmax attention mass, which fak's CPU softmax seam already materializes (so #856 is ~free on CPU) but a FUSED GPU attention path (FlashAttention/FlashInfer, fak's production decode direction) does NOT materialize, so carrying #856 onto the fused path forces materializing the attention matrix — the exact 'prohibits fused kernels' tax EpiKV names — whereas EpiKV's representation-delta signal is readable from the forward pass without it. Not adopted as a mechanism: fak has an attention-mass observer (attn_observer.go) but NO hidden-state-delta observer (the EpiKV-score sibling), and validating that any importance policy preserves quality on long reasoning traces needs a served-model long-CoT eviction-quality benchmark this win32 dev box cannot run (host-gated). Not a threat (no adversary). One residual FILED not built: a forward-pass hidden-state-delta observer feeding the kvmmu exact evictor (the representation-delta twin of attn_observer→#856), gated on that served long-CoT eviction-quality witness. No code change; no change to tools/idea_scout.py (surfaced + scored correctly under topic kv-prefix-cache-reuse, score 44). (2026-06-28)"
description: "Triage of idea-scout candidate arXiv:2606.26472 (Steven Kolawole, Virginia Smith — 'Epiphany-Aware KV Cache Eviction Without the Attention Matrix' / EpiKV, submitted 2026-06-25): as reasoning models emit chains of thought tens of thousands of tokens long, KV cache becomes a deployment bottleneck; existing eviction methods rank tokens by ATTENTION WEIGHT, which the paper argues is (a) a noisy importance proxy in long reasoning traces and (b) prohibits fused kernels in production by forcing the model to materialize the attention matrix. EpiKV instead scores tokens with the 'epiphany score' — the change in the model's internal representation, read directly from the forward pass with no attention matrix and negligible extra state — yielding a training-free, classifier-free, custom-kernel-free eviction method. Verdict: this lands on fak's exact serving surface, unlike the recent off-mission training/pedagogy hits (#1008/#1009/#1122). (1) Prior art to cite + validation of fak's attention-as-explanation demotion (#861/#862, the witnessed-span-attention-validity / span-attention-confound notes) applied to the EVICTION-POLICY domain: EpiKV's 'attention weight is a noisy importance proxy, score off the forward-pass representation instead' is the eviction-policy sibling of fak's 'a self-generated attention signal is a proposal to verify, never an accepted causal account', and the field-level catalog row fak's docs/awesome-token-efficiency.md §B2 should carry next to H2O/SnapKV/Scissorhands. (2) A candidate importance POLICY for fak's POLICY-AGNOSTIC exact-eviction MECHANISM: fak's value-add is the bit-exact mid-run causal evictor (internal/model/kvcache.go Evict; the paged twin internal/model/paged_evict.go PagedKV.Evict, #33; the kvmmu governance evictor) that evicts an arbitrary span and leaves the cache byte-for-byte == a run that never saw it (max|Δ|=0). That mechanism is agnostic to WHICH span/tokens to drop, so an importance scorer is a swappable input — and fak already shipped ONE such policy as #856 ('attention-informed exact eviction — evict coldest-by-attention via the existing bit-exact evictor', kvmmu lane, CLOSED), fed by the post-softmax attention mass internal/model/attn_observer.go emits. EpiKV is a SECOND candidate policy with a sharp advantage on fak's production path: #856's attention mass is materialized for free on fak's CPU softmax seam (attn_observer copies the already-materialized post-softmax score row), but a FUSED GPU attention kernel (FlashAttention/FlashInfer — fak's production decode direction) does NOT materialize the attention matrix, so carrying coldest-by-attention onto the fused path forces materializing it (the exact tax EpiKV names), whereas the epiphany score (a hidden-state delta) is readable from the forward pass with no attention matrix. So EpiKV is the fused-kernel-compatible importance signal for the exact gap fak's #856 policy leaves open on the fused path. (3) NOT adopted as a mechanism: fak has an attention-mass observer (attn_observer.go) but NO hidden-state-delta observer — the epiphany score's carrier — and the load-bearing claim of ANY lossy eviction policy (that dropping these tokens preserves answer quality on long reasoning traces) needs a served-model long-CoT eviction-quality benchmark this win32 dev host cannot run (it serves no model; AVOID-TESTING-ON-THIS-MACHINE), so shipping a representation-delta scorer now would be an unwitnessed quality claim. Not a classic threat (a performance capability, no adversary). One residual, FILED not built: a forward-pass hidden-state-delta observer (the EpiKV-score sibling of attn_observer.go) feeding the kvmmu exact evictor — same shape as attn_observer→#856 but with a representation-delta signal instead of attention mass — gated on the served long-CoT eviction-quality witness fak lacks on this host. No capability code change; the one concrete artifact is the §B2 catalog row (cite as prior art). Scout calibration: surfaced under topic kv-prefix-cache-reuse (score 44) on a genuine kv cache (title) term + freshness (3 days old, ≤30d) — correctly on-topic AND close-to-mission (a serving-side KV-eviction result touching fak's exact eviction + attention-observer surfaces), the scout working as designed; no change to tools/idea_scout.py."
---

# idea-scout triage — EpiKV / attention-matrix-free KV eviction by the epiphany score (issue #1123)

> Closes the daily idea-scout candidate [#1123](https://github.com/anthony-chaudhary/fak/issues/1123)
> (`tools/idea_scout.py`, filed 2026-06-28). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a capability,
> defend against as a threat, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + validation of fak's attention-as-explanation demotion
> ([#861](https://github.com/anthony-chaudhary/fak/issues/861)/[#862](https://github.com/anthony-chaudhary/fak/issues/862))
> applied to the EVICTION-POLICY domain, AND a candidate fused-kernel-friendly importance
> POLICY for fak's policy-agnostic exact-eviction MECHANISM. fak's value-add is the
> bit-exact mid-run evictor (`kvcache.go` `Evict`, the paged twin `paged_evict.go`
> `PagedKV.Evict` [#33], the kvmmu governance evictor) — it evicts a span handed to it and
> leaves the cache byte-for-byte `== never-saw-it` (`max|Δ| = 0`). That mechanism is
> agnostic to WHICH tokens to drop, so importance scoring is a swappable input. fak already
> shipped ONE such policy — [#856](https://github.com/anthony-chaudhary/fak/issues/856)
> "attention-informed exact eviction — evict coldest-by-attention via the existing bit-exact
> evictor" (CLOSED, kvmmu lane), fed by the post-softmax attention mass
> `internal/model/attn_observer.go` emits. EpiKV is a SECOND candidate policy with a sharp
> advantage on fak's PRODUCTION path: #856's attention mass is materialized for free on
> fak's CPU softmax seam, but a FUSED GPU attention kernel (FlashAttention/FlashInfer — fak's
> production decode direction) does NOT materialize the attention matrix, so carrying
> coldest-by-attention onto the fused path forces materializing it (the exact "prohibits
> fused kernels" tax EpiKV names); the epiphany score — a hidden-state delta read from the
> forward pass — needs no attention matrix. NOT adopted as a mechanism: fak has an
> attention-mass observer but NO hidden-state-delta observer (the epiphany score's carrier),
> and proving any lossy eviction policy preserves answer quality on long reasoning traces
> needs a served-model long-CoT eviction-quality benchmark this win32 dev box cannot run.
> NOT a threat. One residual — a forward-pass hidden-state-delta observer feeding the kvmmu
> evictor — is FILED not built. The one concrete artifact: a §B2 catalog row in
> [`docs/awesome-token-efficiency.md`](../awesome-token-efficiency.md) citing EpiKV. No
> capability code change; no change to `tools/idea_scout.py`.**

**Source:** https://arxiv.org/abs/2606.26472 — "Epiphany-Aware KV Cache Eviction Without
the Attention Matrix" (EpiKV), Steven Kolawole, Virginia Smith (submitted 2026-06-25).
Read from the arXiv abstract as surfaced by the scout on 2026-06-28; this is a surface read
of the abstract, not a paper audit or a reproduction (the abstract names the metric and the
mechanism but reports compression/quality numbers this note does not re-measure).

## What it is

A **serving / inference** result — a KV-cache **eviction policy**, not a training method, an
attack, or a protocol. The motivation is the long-reasoning regime fak's KV work targets:
reasoning models emit chains of thought **tens of thousands of tokens long**, so the KV cache
becomes the deployment bottleneck. The paper's two claims against the incumbent approach:

- existing eviction methods rank tokens by **attention weight** (H2O / SnapKV /
  Scissorhands), which is a **noisy importance proxy in long reasoning traces**; and
- ranking by attention weight **prohibits fused kernels** in production inference, because it
  **forces the model to materialize the attention matrix** (a fused FlashAttention-style
  kernel never writes the full `S × S` score matrix out — it streams softmax-weighted values;
  reading per-token attention mass means breaking that fusion or paying to materialize it).

EpiKV instead scores each token by the **epiphany score**: *the change in the model's internal
representation* (a hidden-state delta), **read directly from the forward pass with no attention
matrix and negligible extra state**. The resulting method "requires no training, classifier, or
custom kernel" — a forward-pass-readable importance signal that drops into an existing decode
loop. (The abstract is truncated at the per-token definition and the compression/quality
results; this note triages the **mechanism and its fit**, not the headline numbers.)

## Where this lands on fak (surfaces verified against the tree)

fak is an **agent kernel** with an unusually relevant pair of surfaces here — an exact
eviction **mechanism** and an attention-mass **observer** that already feeds an importance
policy:

- **The exact-eviction MECHANISM (policy-agnostic).** `internal/model/kvcache.go` `Evict(from, n)`
  removes an arbitrary contiguous span and re-derives every shifted survivor's post-RoPE K in a
  single rotation from its pre-RoPE `Kraw`, leaving the cache byte-for-byte identical to a run
  that never saw the span (`TestKVQuarantineEqualsNeverSaw`, `max|Δ| = 0`). The paged twin
  `internal/model/paged_evict.go` `PagedKV.Evict` ([#33], see
  [`EVICT-ON-PAGED-KV-DESIGN-2026-06-28.md`](EVICT-ON-PAGED-KV-DESIGN-2026-06-28.md)) carries it
  onto the block-paged layout. The kvmmu governance evictor (`internal/kvmmu/`) drives it under
  policy. **Crucially, none of these decide WHICH span to evict** — they execute a span handed
  to them. The importance scorer is an input, not part of the mechanism.
- **The attention-mass OBSERVER that already feeds a policy.** `internal/model/attn_observer.go`
  (the attention-witness epic [#851](https://github.com/anthony-chaudhary/fak/issues/851)/#852)
  emits the **post-softmax attention weights** behind a default-off observer, "already
  materialized in the per-worker score scratch at the softmax seam" — and its own rung list names
  **"attention-informed eviction (#856)"** as a downstream consumer. [#856](https://github.com/anthony-chaudhary/fak/issues/856)
  ("attention-informed exact eviction — evict coldest-by-attention via the existing bit-exact
  evictor", kvmmu lane) is **CLOSED**: fak already ships an **attention-weight importance policy**
  feeding the exact evictor. *This is precisely the approach EpiKV argues against.*
- **The field catalog.** [`docs/awesome-token-efficiency.md`](../awesome-token-efficiency.md) §B2
  lists the eviction landscape — StreamingLLM, **H2O**, **Scissorhands**, **SnapKV**, FastGen,
  Quest, DuoAttention — all `fak: ❌` (engine-level importance *policies*), and ends with fak's
  own `✅⭐ UNIQUE` row: **addressable, bit-exact mid-run causal eviction** (the *mechanism*).
  EpiKV is a new entrant in the **policy** column — and a fused-kernel-friendly one, the axis the
  existing rows all share a weakness on.
- **The attention-as-explanation demotion.** fak's reward-over-spans epic
  ([#861](https://github.com/anthony-chaudhary/fak/issues/861)/#862;
  [witnessed-span-attention-validity](RESEARCH-witnessed-span-attention-validity-2026-06-26.md),
  [span-attention-confound-normalization](RESEARCH-span-attention-confound-normalization-2026-06-26.md))
  already treats raw attention weight as a **confounded** importance proxy to be verified, not
  trusted. EpiKV reaches the same conclusion in the eviction-policy domain.

## The three triage questions

### Adopt as a capability? — adopt the *thesis* (cite it, and catalog it); do NOT build the mechanism here.

The paper's thesis — **score token importance off the forward-pass representation, not off
attention weight, so the policy needs no attention matrix and survives fused kernels** — is
directly on fak's mission, for two reasons fak's own tree makes sharp:

1. **It validates fak's attention-as-explanation demotion in a new domain.** fak's #861/#862
   stance is that a raw attention signal is a *proposal to verify against a witness*, never an
   accepted causal account. EpiKV is the **eviction-policy sibling**: it argues the attention
   weight is too noisy to be the importance authority for *which token to drop* in a long
   reasoning trace, and reads a different forward-pass signal instead. That is independent,
   serving-side support for fak's "don't let the attention weight be the floor" bet — now about
   *cache value*, where #1008/#1122 were about *credit* and *step reward*.

2. **It is the fused-kernel-compatible answer to the gap fak's #856 policy leaves on the
   production path.** fak's exact evictor is policy-agnostic, and #856 already wires ONE
   importance policy into it: evict coldest-**by-attention**, fed by `attn_observer.go`. On fak's
   **CPU** softmax seam, the post-softmax score row is *already materialized* in worker-local
   scratch, so #856 reads it essentially for free — the attention-matrix tax does not bite there.
   But fak's **production decode direction is fused GPU attention** (FlashAttention/FlashInfer
   class; see the [backend SOTA matrix](RESEARCH-backend-sota-matrix-2026-06-26.md)), which by
   construction does **not** write the attention matrix out. Carrying coldest-by-attention onto
   that path forces materializing the matrix — the **exact "prohibits fused kernels" tax EpiKV
   names**. EpiKV's epiphany score (a hidden-state delta) is readable from the forward pass
   **without** the attention matrix, so it is the importance signal that would let fak's exact
   evictor run a learned-importance policy on the fused path where #856's attention-weight policy
   cannot follow for free. *This is the precise, fak-specific reason the paper is worth citing,
   not just filed.*

The concrete adoption in this increment is the **citation**: a §B2 catalog row in
`docs/awesome-token-efficiency.md` recording EpiKV as the attention-matrix-free importance
policy, with fak's honest position (the exact evictor is policy-agnostic; #856 ships the
attention-weight policy; EpiKV is the candidate fused-kernel-compatible alternative, filed not
built). The *mechanism* is not adopted (see the third question).

### Defend against as a threat? — no. A performance capability on the same side, no adversary.

EpiKV is a lossy-eviction *optimization*; there is no attacker and no security surface. The only
"defensive" note is the standard lossy-cache fence fak already carries: a lossy importance policy
that drops tokens is **not** fak's bit-exact path — fak's `✅⭐` differentiator is *exact*
eviction (`max|Δ| = 0`), and any importance-policy-driven drop is by definition lossy and must be
labelled as such (the same honesty the §B2 catalog and the
[KV-precision-tiers](EVICT-ON-PAGED-KV-DESIGN-2026-06-28.md) §6 work carry). EpiKV does not change
that boundary; it is a candidate *policy* on the lossy side of it.

### Why not build the epiphany-score policy now? — a missing observer AND a missing quality witness this host cannot reach.

Two gates, both honest:

1. **No hidden-state-delta observer.** fak has `attn_observer.go` — an **attention-mass**
   observer — but no **hidden-state-delta** observer, which is the epiphany score's carrier. The
   epiphany score is "the change in the model's internal representation"; emitting it needs a
   default-off forward-pass hook on the per-layer residual stream (the representation-delta twin
   of `attn_observer`'s post-softmax hook), which does not exist. That is buildable in principle,
   but —
2. **No served long-CoT eviction-quality witness on this host.** The load-bearing claim of *any*
   lossy eviction policy is that dropping the low-scored tokens **preserves answer quality** on
   long reasoning traces. That is exactly what EpiKV's compression/quality results assert and what
   fak would have to re-measure before shipping a policy — a served model emitting tens-of-
   thousands-of-token chains of thought, evicted under the policy, scored on a reasoning
   benchmark. This **win32 dev box serves no model and cannot run that** (see
   [`AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md)).
   Shipping a representation-delta scorer without that witness would be an **unwitnessed quality
   claim** — the precise thing fak's witness discipline refuses. So the mechanism is filed, not
   built.

## Recorded residual (FILED, not built)

The smallest honest follow-on is a **forward-pass hidden-state-delta observer feeding the kvmmu
exact evictor** — the representation-delta sibling of `attn_observer.go` → #856: a default-off,
zero-alloc-when-off hook that emits each token's representation change per layer (the epiphany
score), which the kvmmu policy can rank to choose the coldest-by-epiphany span and hand to the
existing bit-exact `Evict`. It would give fak a **fused-kernel-compatible** importance policy for
the production path where #856's attention-weight policy cannot follow for free. It is **not
shipped here** because (a) the observer hook does not exist and (b) its value depends on a served
long-CoT eviction-quality benchmark this host cannot run, so the "this policy preserves quality"
half would be unwitnessed. Filed as the next checkable step, gated on a model-serving host.

## Scout calibration

Surfaced under topic **`kv-prefix-cache-reuse`** (score 44) on a genuine `kv cache` (title) term
+ freshness (3 days old, ≤30d) — **correctly on-topic and close to mission**: unlike the recent
off-mission training/pedagogy hits ([#1008 CoT-training](RESEARCH-cot-training-gains-agents-triage-2026-06-27.md),
[#1009 SE-education](RESEARCH-llm-mcp-se-education-triage-2026-06-27.md),
[#1122 progress-advantage](RESEARCH-progress-advantage-rl-posttraining-triage-2026-06-28.md)),
this is a **serving-side** KV-eviction result that touches fak's **exact** eviction
(`internal/model/kvcache.go`, `paged_evict.go`, `internal/kvmmu/`) and attention-observer
(`internal/model/attn_observer.go`) surfaces — and counterpoints a policy fak **already shipped**
(#856). The scout judged new-and-on-topic and handed the worth-pursuing call to human triage —
**working as designed**. No change to `tools/idea_scout.py`.

## Disposition

**Cite as prior art** (the attention-matrix-free importance policy, recorded as a §B2 catalog row
and as validation of fak's attention-as-explanation demotion applied to eviction policy) and
**record the candidate-policy fit + the fused-path advantage over the shipped #856 attention-weight
policy** above. No mechanism adopted (fak lacks the hidden-state-delta observer the epiphany score
needs, and proving the policy preserves quality needs a served long-CoT eviction benchmark this
host cannot run), no threat to defend (no adversary), one observability residual filed behind a
model-serving gate fak cannot reach on this win32 box. The issue is resolved by this triage note +
the §B2 citation; the hidden-state-delta observer is the named next step.
