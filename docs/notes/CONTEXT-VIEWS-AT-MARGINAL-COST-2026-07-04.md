---
title: "Context views at marginal cost: on-demand quality views over KV + attention (2026-07-04)"
description: "The kernel already computes attention scores and holds the whole KV cache. This note works out how to turn that into MANY on-demand 'views' of the token history at marginal cost — spans as pointers into the ledger, a scalar side-car of witnessed signal per token, threshold/layer/expert views, and the token↔prose round-trip (which views must be re-run through attention). Grounds each idea in what fak already has (attn_observer, kvmmu/attention.go, memview, contextq) and in the SOTA (H2O/SnapKV/TOVA, retrieval heads, Prompt Cache/CacheBlend/PIC)."
date: 2026-07-04
keywords:
  - context views
  - attention scores
  - KV cache
  - side-car
  - on-demand context
  - marginal cost
  - retrieval heads
  - position-independent caching
  - MoE routing side channel
---

# Context views at marginal cost

## 0. The one-paragraph version

When the kernel runs a forward pass it *already* computes, for every token, the
thing most "views of the history" want: the post-softmax attention distribution
(what attended to what) and — on an MoE model — the router's top-k expert picks
and gate weights. Both are computed and then thrown away microseconds later. fak
already taps the attention half (`internal/model/attn_observer.go` → span
attribution in `internal/kvmmu/attention.go`), but only to drive one consumer
(coldest-span eviction) and one report (`report.go`). The claim of this note is
narrower and more useful than "store attention": **because the spans of the
history are already named, digest-bound cells in a ledger, a "view" of the
history is a cheap predicate over per-cell scalars that the forward pass hands us
for free — and only a small, bounded set of scalars needs to be *kept*; the rest
is either recomputed by re-filtering the ledger, or (for query-dependent views)
re-attended over a selected subset of tokens.** The cost model has three tiers:
free (already computed), keep-a-scalar (O(spans), a side-car), and re-attend
(O(selected tokens), a fresh prefill of a subset). Getting each view into the
right tier is the whole design.

## 1. What "a view" is here, precisely

A **view** is a *selection* over the history plus a *rendering* of the selected
cells. fak already has the rendering half as a typed contract:
`internal/memview.MemoryViewRecord` (snippet / summary / QA / fact, digest-bound,
taint-inherited, invalidate-on-digest-change) and `internal/contextq` (the
queryable materializer with the HIT/FAULT/RECOMPUTE/REFUSE/ABSTAIN verdict). What
is *missing* is the selection half being driven by **runtime signals the kernel
computes** rather than by lexical overlap or a caller's explicit pin list.

The selection is a predicate over per-cell descriptors. Some descriptors are
static (role, taint, digest, byte length, producer). The interesting ones are
**dynamic, per-run, and computed by attention/routing**:

- `a_s` — witnessed attention mass a span received this turn (already computed:
  `Segment.Attended` in `kvmmu/attention.go`).
- `EMA(a_s)`, `Cumulative(a_s)`, `Trajectory(a_s)` — the recency / overall /
  when-was-it-hot reductions (already computed: `CloseTurn`).
- (proposed) `expert_hist_s` — which experts the tokens of a span routed to, and
  the mean gate weight (available in the forward pass, **not yet tapped**).
- (proposed) `layer_profile_s` / `head_profile_s` — the same attention mass but
  resolved per-layer or per-head instead of summed to a scalar (the observer
  already receives `(layer, head)`; `attention.go` sums them away — see §4).

A view is then: *"the cells where `predicate(descriptor) == true`, rendered as
`kind`."* The naive examples from the goal are exactly this:

- "all tokens with attention score > θ" → `filter(a_s > θ)`, render as snippet.
- "scores within one layer / expert / section" → `filter(layer_profile_s[L] > θ)`
  or `filter(L ∈ expert_hist_s)`, which needs the per-layer/per-expert resolution
  §4 describes.

The reason this is *cheap* is the reason the addressable-KV note already gives:
the spans are pointers. A view's body does not copy tokens; it is a set of
`[From, Len)` refs into the KV/CAS ledger plus the scalar that selected them. A
1000-view fan-out over a session is 1000 predicates over the same O(spans) scalar
table — not 1000 passes over the tokens.

## 2. The cost tiers (the core of the design)

Every candidate view lands in exactly one tier. The engineering discipline is:
push each view to the cheapest tier it can correctly live in, and *label* the tier
so a consumer knows whether it is reading a free byproduct, a kept aggregate, or a
recomputed approximation.

### Tier 0 — FREE (already computed this forward pass)

The post-softmax weights and the router picks exist in worker-local scratch at
their respective seams. Emitting a *reduction* of them costs one add per
(row, span) — the `AttributeRow` cursor is already O(positions + spans). A view
that consumes only this turn's live signal (e.g. "what is hot *right now*") is
free: it reads `Segment.Attended` after the turn's attribution.

Invariant to preserve (from `attn_observer.go`): **nil observer == byte-identical
forward pass, zero alloc.** Any new side channel (expert routing, per-layer
attention) must keep this — default-off, emit-a-copy, never touch the math.

### Tier 1 — KEEP-A-SCALAR (a durable side-car, O(spans) not O(n²))

This is the load-bearing decision, and the SOTA settled it years ago: **you do
not keep the attention matrix; you keep a running reduction.** The full
per-(layer, head, query, key) attention is O(n²) per head per layer — for a 128k
context that is astronomically larger than the KV cache itself and pointless to
store. H2O / Scissorhands / TOVA / SnapKV all keep an O(n) reduction: an
accumulated or windowed attention score *per token* (or per span). fak already
does exactly this — `EMA` + `Cumulative` + a bounded `Trajectory` ring
(`trajCap = 64` turns), O(1) per span per turn. That is the side-car.

The side-car is the answer to "what info needs to be kept at run time": **one
struct per span** — `{Attended (this turn), EMA, Cumulative, traj[64], and
(proposed) expert_hist, layer_profile}`. Bounded, model-independent in shape,
and orders of magnitude smaller than the KV it describes. It is the sidecar
concept (`internal/sidecar`) applied to attention/routing rather than to fleet
posture. Store it next to the session image; it survives compaction because it is
about spans, not tokens.

The scalar side-car supports every view whose predicate is a function of the
*integrated* signal: hottest, dead-weight (cold-but-resident), "bloated since
turn K", "spans this expert cluster ever fired on." These are the `report.go`
outputs generalized into first-class queryable views.

### Tier 2 — RE-ATTEND (recompute at query time, O(selected tokens))

This is the tier the goal's "going back and forth between tokens and prose … maybe
some views have to be rerun through attention" intuition is pointing at, and it is
the subtle one. **Attention mass is query-dependent.** The `a_s` in the side-car
is the attention *the actual decode queries* placed on each span during the real
run. A view like "which of my 200 stored spans are relevant to *this new
question*" cannot be answered from the stored `a_s` — that mass was about the old
queries. To get it you must re-attend: run the new query against the (cached) keys
of the candidate spans and read the resulting distribution.

The cheap way to do this is the crux, and it is where the position-independent-
caching SOTA (Prompt Cache, CacheBlend, PIC/MiniPIC) becomes directly relevant:

- You do **not** re-prefill the tokens (that would recompute K/V from scratch).
  You reuse the cached K/V and compute only the *query* projection of the new
  question, then a single attention read against the candidate keys. That is one
  QK^T over `|query| × |candidate keys|` — cheap relative to a prefill.
- But cached K/V is bound to its *original position* (RoPE). To score a candidate
  span against a query sitting at a new position, you need the keys at a compatible
  position. fak already keeps the pre-RoPE key (`Kraw`, the thing that makes exact
  eviction possible) — so it can re-rotate candidate keys to a scoring position
  *once*, exactly, the same primitive `Evict` uses. This is a latent capability:
  the machinery that makes span removal bit-exact is the same machinery that makes
  *position-independent re-scoring* possible.

The decomposition that makes this cheap (the SOTA's central fact): **K and V are
functions of the context tokens — cacheable and re-positionable; Q is a function
of the new query — never cacheable.** So re-attending a retrieved subset under a
new query costs only the new query's Q projection + the QK^T·softmax·V read; you
*skip the K/V projection of the entire subset*. The QK^T matmul itself is the
irreducible floor — it is query-dependent by construction — so the whole game is
shrinking *what* you re-attend, never eliminating re-attention.

So Tier 2 has two sub-modes:
- **cheap re-score** (relevance of stored spans to a new query): reuse cached K/V,
  re-rotate `Kraw` to a probe position, one QK^T read. No full forward pass. This
  is the position-independent-caching path (PIC/MiniPIC store unrotated keys and
  apply RoPE at attention time — which is precisely what fak's `Kraw` already
  enables; §7).
- **full re-attend** (need deep-layer, post-mixing attention, not just layer-0
  similarity): re-prefill the selected subset as its own short context. Genuinely
  expensive, so only for a small, deliberately-selected set — Tier 1 exists to
  narrow the candidates first. When recomposing several cached spans, the SOTA
  result (EPIC) is that the error is localized to span *seams*: recompute only a
  small constant band (k<20 tokens) at each boundary rather than a percentage of
  the whole, making the recompute O(#spans) not O(#tokens).

The token↔prose round-trip the goal names is this Tier-1→Tier-2 handoff: Tier 1
(scalar filter) narrows thousands of spans to dozens using free/kept signal; the
prose rendering of those dozens is a `memview` snippet (already tokens→prose,
lossless); if the view needs *fresh* query-relevance it re-enters attention
(prose→tokens→attention) over just those dozens.

## 3. Pure side-car vs. materialized: the "score concept only" mode

The goal asks whether this can work "purely as a side-car, e.g. score concept
only." Yes — and that is the right V1. The score-only side-car is:

1. A per-span scalar table updated at each `CloseTurn` (Tier 1). No token copies,
   no view bodies. Just `{span_id → scalars}`.
2. Views are computed **lazily at query time** by filtering that table
   (`filter(a_s > θ)`), and the *result* is a set of span refs. The bytes are
   faulted in through the existing `contextq` / `recall` page-in gate only if and
   when a consumer actually renders them.

This is strictly cheaper than materializing views eagerly, and it matches the
on-demand-context note's "build the cheap frontier, lazily fault deeper." The
score table is the frontier; everything else faults.

The "filter by only values above a certain point" mode the goal describes is
important for a second reason beyond cost: **it is the natural way to keep the
side-car bounded.** You never need to store `a_s` for the long cold tail — a span
whose EMA has decayed below θ for K turns can drop its fine-grained trajectory and
keep only the cumulative scalar (or be a candidate for the coldest-eviction path
that already exists). The threshold is both a query predicate and a storage GC
policy.

## 4. The dimensions a view can slice (layer / head / expert / section)

`attention.go` today sums the observer's `(layer, head)` dimensions into one scalar
`Attended` per span. That is the right default (one number drives eviction), but it
throws away the axes the goal explicitly wants views over ("one layer or expert or
section"). The observer *already receives* `layer` and `head` (`attn_observer.go`
signature is `func(layer, queryPos, head, keyPositions, weights)`) — the
information is there; `attention.go` chooses to collapse it. Preserving a *reduced*
per-axis profile (not the full matrix) is a bounded extension:

- **Per-layer view.** Keep `layer_profile_s[L]` = mass span s received summed over
  heads at layer L. Storage: O(spans × layers) scalars. This is what lets "scores
  within one layer" be a view. It also connects to the **retrieval-head / attention-
  sink** literature: the useful long-context signal is concentrated in a *small
  subset* of heads/layers, so you likely keep the profile only for the handful of
  layers that carry retrieval signal, not all of them — a big storage win and a
  more discriminating view.
- **Per-head view.** Same idea at head granularity for the identified retrieval
  heads only. "Which spans do the retrieval heads attend to" is a sharper relevance
  signal than mean attention.
- **Per-expert view (MoE).** `route()` in `internal/model/moe.go` already computes,
  per token per layer, the top-k `(expert, gate_weight)` picks — and it is **not
  observed at all** (there is no `RouteObserver`, confirmed by grep). Adding one
  (mirroring `AttnObserver`, default-off, emit-a-copy) gives a per-span
  `expert_hist_s`: the multiset of experts the span's tokens routed to. Views:
  "spans that fired expert E" (a crude topical clustering — experts specialize),
  "spans whose routing entropy is high" (ambiguous/boundary content). This is the
  cheapest possible semantic-ish view because routing is a byproduct of every FFN.
- **Per-section view.** "Section" = a span or a contiguous run of spans in the
  ledger. Already free: the ledger *is* the section index. A section view is a
  predicate over the spans in `[From_a, From_b)`.

Honest caveat carried from the addressable-KV note: layer-0 / single-head attention
similarity is a *weak* relevance signal (it is mostly embedding+position); the
strong signal is deep-layer, post-mixing, and query-dependent (Tier 2). So the
per-layer/per-head views are best framed as *cheap candidate generators* feeding a
Tier-2 re-attend, not as final relevance verdicts.

## 5. What has to be kept vs. generated at query time (summary table)

| View | Runtime signal it needs | Tier | Keep or recompute | Approx cost |
|---|---|---|---|---|
| "hot right now" (recency) | this turn's `Attended` / `EMA` | 0/1 | keep scalar | O(spans), free at turn close |
| "mattered overall" (cumulative) | `Cumulative` | 1 | keep scalar | O(spans) |
| "when was span hot" (trajectory) | `traj[64]` ring | 1 | keep bounded ring | O(spans × 64) |
| "attention > θ" (threshold) | `a_s` (any reduction) | 1 | filter kept table | O(spans) predicate |
| "within layer L" | `layer_profile_s[L]` | 1 | keep per-retrieval-layer profile | O(spans × few layers) |
| "within head H" | `head_profile_s[H]` | 1 | keep per-retrieval-head profile | O(spans × few heads) |
| "fired expert E" (MoE) | `expert_hist_s` | 0/1 | keep multiset (needs RouteObserver) | O(spans × k) |
| "dead weight / bloated-since" | cumulative + cost + S/N curve | 1 | fold kept table (report.go) | O(spans) |
| "relevant to THIS new query" | fresh QK^T of new query vs candidate keys | 2 | recompute (reuse Kraw, re-rotate) | O(\|q\| × \|candidate keys\|) |
| "deep post-mixing view of a subset" | re-prefill selected subset | 2 | recompute (full) | O(selected tokens) prefill |

The rule the table encodes: **keep O(spans) scalars; never keep O(n²) matrices;
recompute anything query-dependent by re-attending over a Tier-1-narrowed subset,
reusing cached keys rather than re-prefilling when the view only needs a similarity
score.**

## 6. How this composes with what fak already ships

Nothing here is greenfield; it is wiring latent capabilities into the view layer:

- The attention side-car (Tier 1) already exists as `kvmmu/attention.go` +
  `report.go`. The gap is exposing it as *queryable views* (predicate → span refs)
  in `contextq`, not just as an eviction driver and a post-hoc report.
- The re-attend primitive (Tier 2 cheap mode) is latent in `Kraw` + `applyRopeRow`
  (`internal/model/kv.go`) — the same re-rotation that makes exact eviction work
  makes position-independent re-scoring work. Not yet exposed.
- The MoE side channel (per-expert view) needs a new `RouteObserver` in
  `internal/model` mirroring `AttnObserver`. This is the one genuinely new observer.
- The view contract, taint inheritance, staleness, and verdict stream
  (`memview` + `contextq`) already exist and are exactly where the new,
  runtime-signal-selected views should land — so a "hot span" view inherits taint
  and is refused if its source was sealed, for free.

## 7. SOTA grounding

Two research passes (attention-as-signal; position-independent/queryable KV)
converge on three load-bearing facts that shape the design above. Arxiv IDs from
2023–early-2025 are solid; a few late-2025/2026 IDs are flagged `(verify)` — the
*mechanism* is well-attested but the exact ID sits past reliable-verification
horizon and should be confirmed before any published claim.

**Fact 1 — everyone keeps an O(n) reduction, nobody keeps the O(n²) matrix.**
The full per-(layer, head, query, key) attention is what FlashAttention exists to
*avoid* materializing; at n=100k a single head's matrix is ~10¹⁰ entries. Every
attention-informed cache system reduces it to a running scalar per token (per head)
and, after eviction, to an O(B) surviving set (B ≪ n). This is exactly fak's
`{EMA, Cumulative, traj[64]}` reduction — so fak is already on the correct side of
the store/recompute line. What none of them do that fak's `report.go` does is *keep
the reduction as a durable post-hoc artifact* rather than consuming it inside the
pass; that durability is fak's differentiator, not the raw signal.

**Fact 2 — the useful long-context signal lives in a tiny head subset.**
"Retrieval heads" (Wu et al., [2404.15574]) are a sparse (<5% of heads), universal,
intrinsic set — concentrated in **middle/upper layers** — mechanistically
responsible for pulling information from long context; ablating just them collapses
retrieval, ablating an equal number of random heads barely dents it. DuoAttention
([2410.10819]) operationalizes this: retrieval heads get the full cache, "streaming"
heads get only sinks+recent. **Consequence for §4:** the per-layer/per-head view
should track the scalar only on the retrieval-head subset (a small, offline-computed
**head-importance map** — the one durable side-car worth persisting), not across all
L·H heads. Attention sinks ([2309.17453]; mechanistic cause in "Massive
Activations" [2402.17762]) are a *stability* signal on the first ~4 tokens, present
in all layers — useful to always-keep, but not a content-relevance signal.

Per-system reductions (all computed FREE in the forward pass unless noted):
H2O [2306.14048] — accumulated attention sum per token/head, ~20% budget; TOVA
[2401.06104] — the *latest* query's weight, head-averaged, argmin-evict; SnapKV
[2404.14469] — one-shot at end of prefill, pools attention from the last **N=32**
prompt tokens ("observation window") into one score per prompt token per head, then
max-pools (kernel 7) so a kept key drags its neighbors; PyramidKV [2406.02069] —
non-uniform budget *per layer* (broad early → peaked late); Ada-KV [2407.11550] —
adaptive budget *per head* by attention concentration; SqueezeAttention [2404.04793]
— budget *per layer* by how much the layer changes the representation; Keyformer
[2403.09054] — accumulated sum + Gumbel regularization to fix the post-eviction
distribution shift; Scissorhands [2305.17118] — frequency a token exceeds a
threshold; FastGen [2310.01801] — per-head policy label. The through-line: cheap
signal = running scalar per token per (retrieval) head, reduced immediately.

**Fact 3 — the re-attend floor is irreducible, but the marginal cost is small.**
K and V are functions of the *context* tokens (cacheable, position-encodable); Q is
a function of the *new query* (never cacheable). So "attention under a new query
over a retrieved subset" decomposes: reuse cached K/V, pay only the new query's Q
projection + the QKᵀ·softmax·V read. You skip the K/V projection of the whole
subset — that is the Tier-2a win. What you cannot skip is the QKᵀ matmul; it is
query-dependent by construction (O(|new q| × |candidate keys|)). Three cost tiers
match §2:

- **Prompt Cache** ([2311.04934]) — modular snippets in *pre-reserved position
  slots*, no re-attention at all; only correct when snippets are independent. The
  fast path: a ledger pointer can carry a reserved-slot annotation.
- **PIC family — MiniPIC / MEPIC / Irminsul** (`(verify)`: 2606.13126 / 2512.16822
  / 2605.05696) — store **unrotated (NoPE) keys**, apply RoPE inside the attention
  kernel at per-request logical positions. Re-placing a cached key is one complex
  multiply per element. **This is fak's `Kraw` exactly** — the pre-RoPE key fak
  already keeps to make eviction bit-exact is the same object that makes
  position-independent re-scoring possible. fak has the PIC primitive already; it
  is just not exposed as a re-score path (§6).
- **CacheBlend** ([2405.16444]) — reuse non-prefix KV, selectively recompute ~15%
  of tokens (by KV-deviation) to repair cross-attention. **EPIC** ([2410.15332])
  sharpens this: the damage is localized to span *seams*, so recompute only a
  constant band of **k<20 tokens per boundary** (not a percentage of the whole) —
  making re-attention O(#spans), not O(#tokens). This is the right model for
  Tier-2b over a pointer ledger: pay a small boundary band per span, nothing per
  interior token.

**Queryable KV as memory:** PagedAttention ([2309.06180]) already makes KV blocks
addressable pointers with copy-on-write — the primitive "spans as pointers into a
ledger" generalizes. LMCache ([2510.09665]) / Mooncake ([2407.00079]) show the
storage/transport/addressing layer should be *decoupled* from the reuse policy
(exactly fak's cachemeta-under, contextq-over split). CacheGen ([2310.07240])
shows KV bytes are ~4× compressible with a layer/locality-aware encoder — relevant
if the ledger spills off-GPU. MemGPT/Letta ([2310.08560]) has the right *control*
model (agent pages memory in/out by function call) but operates one level too high
(re-reads text, re-prefills); fak's contribution is pushing that paging model down
to the KV layer where a page-in is pointer-reuse + boundary-recompute. Cartridges
([2506.06266]) is the "cold, high-value span" escape hatch: a *learned* synthetic
KV (self-study distillation) matches ICL at ~38× less memory — worth an offline
compaction pass for spans queried many times.

**MoE routing view — a caution, not a win.** Router picks (`route()` in `moe.go`,
per token per layer, top-k `(expert, gate)`) are genuinely free to tap. But "The
Myth of Expert Specialization in MoEs" (`(verify)`: 2604.09780) finds routing is a
**linear projection of the hidden state — a coarse geometric hash, not a domain
label**: different models on the same problem overlap only ~60% in experts,
prompt-phase routing fails to predict generation-phase routing. So the expert-view
is usable as a *within-model, within-run* clustering/dedup key (§4), but must not
be sold as a portable semantic tag. When you want query-relevance, prefer
accumulated attention (Fact 2) over router logits — attention is what a new query
actually reads.

## 8. Open questions / kill criteria

- **Does layer/head profiling earn its storage?** If the retrieval-head subset is
  ~a dozen heads, the per-head profile is cheap and discriminating. If the signal
  is diffuse, this collapses back to the single scalar and the per-axis views are
  not worth keeping. Measure before building the general per-layer store.
- **Is the cheap re-score (Tier 2a) accurate enough** to narrow candidates, or does
  it always need the full re-prefill (Tier 2b)? This is the CacheBlend question in a
  new dress. Gate with an oracle: does the cheap re-score's top-k agree with a full
  re-attend's top-k?
- **Does the expert-routing view carry real topical signal**, or is expert
  specialization too entangled to cluster on? Cheap to test: tap `route()`, bucket
  spans by dominant expert, eyeball coherence.
- Stop if the side-car costs more to maintain than the views save, or if
  threshold-θ views are no better than the lexical-overlap selection ctxplan
  already has.

## 9. Filed next steps

The three actionable, code-grounded next steps from this note are tracked:

- **#2617** — `feat(contextq)`: expose the witnessed attention side-car as
  on-demand queryable views (threshold / hottest / dead-weight) over the kept
  per-span scalars. Tier 1; zero new attention compute.
- **#2623** — `feat(model)`: add `RouteObserver`, the free per-token MoE
  expert-routing side channel (the analogue of `AttnObserver` #852), with the
  honest "routing is a geometric hash, not a semantic label" caution.
- **#2626** — `feat(kvmmu)`: cheap query re-score over `Kraw` — position-
  independent relevance of stored spans to a *new* query, reusing the exact-
  eviction primitive. Tier 2a, gated by a top-k-vs-full-re-attend oracle.
