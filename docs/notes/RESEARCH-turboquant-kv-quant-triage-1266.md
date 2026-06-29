---
title: "idea-scout triage: TurboQuant (Google, ICLR 2026) — random-rotation + per-coordinate Lloyd-Max near-optimal scalar quantization of the KV cache with ASYMMETRIC key>value bit allocation (K4/V2 ≈ 3-bit avg), training/calibration-free, per-token. Verdict: prior art to cite as a new B1 KV-quantization entrant + independent validation of TWO fak choices fak's tree already makes — (1) the asymmetric 'keep Keys more precise than Values' layout fak's #1047 q8 KV-precision tier ships as `f32-Kraw + q8_0-K/V` (capacity.go), and (2) fak's witness discipline that a high attention-fidelity cosine is NOT a generation-quality witness — TurboQuant's OWN README admits its 99.5%+ attention-score cosine 'does not guarantee working generation' and that the headline 5x@3-bit (K4/V2) shows generation failures, the precise reason fak gates the lossy q8 tier behind a served quality witness it cannot run on this win32 box. Not adopted as a mechanism: TurboQuant is an engine-side LOSSY KV quantizer, the opposite axis from fak's ✅⭐ value-add (addressable BIT-EXACT mid-run eviction, max|Δ|=0) — and its random-rotation+quant DESTROYS the lossless pre-RoPE Kraw the exact evictor re-rotates survivors from (kvcache.go), so it cannot live under the exact path; it is a candidate quantizer for the lossy #1047 precision tier, which needs a GPU engine quant seam + a served generation-quality benchmark this host lacks. Not a threat (no adversary). One residual FILED not built: TurboQuant as a candidate quantizer for the #1047 q8/f32 KV-precision lossy tier, gated on that served generation-quality witness. One concrete artifact: the §B1 catalog row. No code change; no change to tools/idea_scout.py (surfaced + scored correctly under topic kv-prefix-cache-reuse, score 35). (2026-06-29)"
description: "Triage of idea-scout GitHub candidate github:tonbistudio/turboquant-pytorch (1021 stars, Python, pushed 2026-04-23): a from-scratch PyTorch implementation of Google's TurboQuant (ICLR 2026) for LLM KV-cache compression. The method: each KV vector gets a random orthogonal rotation to normalize its coordinate distributions, then independent per-coordinate Lloyd-Max optimal scalar quantization, storing quantized indices + a norm for reconstruction — training-free and calibration-free, applied post-hoc to captured KV tensors, per-token granularity. Its V3 uses ASYMMETRIC bit allocation — 'keys get more bits than values' because keys determine attention focus (need precision) while values are error-tolerant through averaging — e.g. TurboQuantV3(key_bits=4, value_bits=2) for a ~3-bit average. Claimed: 2x compression with perfect generation at K6/V4; 5x at K4/V2 but with quality degradation (generation failures); 99.5%+ cosine similarity on attention scores (which, the README is explicit, does NOT guarantee working generation); 94-99% top-1 match. Verdict: prior art to cite as a new entrant in fak's §B1 KV-cache-quantization landscape (next to KIVI/KVQuant), and a sharp independent validation of two choices fak's tree already makes: (1) fak's #1047 q8 KV-precision tier already keeps the Key path MORE precise than the rest — its layout is `f32-Kraw + q8_0-K/V` (internal/compute/capacity.go), i.e. the pre-RoPE Key is held in full f32 — the SAME 'keys need more bits' asymmetry TurboQuant lands, though fak's load-bearing reason is different (the bit-exact mid-run evictor in internal/model/kvcache.go re-derives every shifted survivor's post-RoPE K from the lossless pre-RoPE Kraw, leaving the cache byte-for-byte == never-saw-it, max|Δ|=0; TurboQuant keeps keys precise for attention fidelity); and (2) fak's witness discipline that an attention-cosine number is not a generation-quality witness — fak's own q8-vs-f32 Approx gate is exactly cosine ≥ 0.995 (internal/compute/compute_test.go), TurboQuant reports 99.5%+ attention cosine, and TurboQuant's OWN README says that cosine 'does not guarantee working generation' and that 5x@3-bit shows generation failures, which is precisely why fak's q8 tier is 🟡 half-shipped/GPU-gated behind a served quality witness this win32 dev box cannot run (AVOID-TESTING-ON-THIS-MACHINE). NOT adopted as a mechanism: TurboQuant is an engine-side LOSSY quantizer, the orthogonal axis from fak's ✅⭐ value-add (addressable bit-exact mid-run causal eviction) — and lossily rotating+quantizing the key DESTROYS the lossless pre-RoPE Kraw the exact evictor depends on, so it cannot sit under the exact path; it is a candidate quantizer for the lossy #1047 precision tier, gated on a GPU engine quant seam + a served generation-quality benchmark fak lacks on this host. Not a threat (a performance capability, no adversary). One residual FILED not built: TurboQuant as a candidate quantizer for the #1047 q8/f32 KV-precision lossy tier behind the served-generation-quality witness. No capability code change; no change to tools/idea_scout.py (surfaced + scored correctly under topic kv-prefix-cache-reuse, score 35)."
---

# idea-scout triage — TurboQuant / random-rotation near-optimal KV quantization (issue #1266)

> Closes the daily idea-scout candidate [#1266](https://github.com/anthony-chaudhary/fak/issues/1266)
> (`tools/idea_scout.py`, filed 2026-06-29). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a capability,
> defend against as a threat, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — TurboQuant is a new entrant in fak's
> [§B1 KV-cache quantization](../awesome-token-efficiency.md) landscape (random orthogonal
> rotation + per-coordinate Lloyd-Max scalar quant, training/calibration-free, per-token, with
> ASYMMETRIC key>value bit allocation), AND a sharp independent validation of two choices fak's
> tree already makes. (1) fak's [#1047](https://github.com/anthony-chaudhary/fak/issues/1047) q8
> KV-precision tier already keeps the Key path MORE precise than the rest — its layout is
> `f32-Kraw + q8_0-K/V` (`internal/compute/capacity.go`), i.e. the pre-RoPE Key in full f32 —
> the SAME "keys need more bits" asymmetry TurboQuant lands (K4/V2 ≈ 3-bit), though for a
> DIFFERENT load-bearing reason: fak keeps the pre-RoPE `Kraw` lossless so the bit-exact
> mid-run evictor (`internal/model/kvcache.go`) can re-derive every shifted survivor's
> post-RoPE K from it (`max|Δ| = 0`), whereas TurboQuant keeps keys precise for attention
> fidelity. (2) fak's witness discipline that a high attention-fidelity cosine is NOT a
> generation-quality witness: fak's own q8-vs-f32 gate is exactly cosine ≥ 0.995
> (`internal/compute/compute_test.go`), TurboQuant reports 99.5%+ attention cosine — and
> TurboQuant's OWN README says that cosine "does not guarantee working generation" and that the
> headline 5x@3-bit (K4/V2) shows generation failures. That is the precise reason fak's q8 tier
> is 🟡 half-shipped / GPU-gated behind a served quality witness this win32 box cannot run.
> NOT adopted as a mechanism: TurboQuant is an engine-side LOSSY quantizer, the orthogonal axis
> from fak's ✅⭐ value-add (addressable bit-exact mid-run eviction) — and lossily rotating +
> quantizing the key DESTROYS the lossless pre-RoPE `Kraw` the exact evictor depends on, so it
> cannot live under the exact path; it is a candidate quantizer for the lossy #1047 precision
> tier, gated on a GPU engine quant seam + a served generation-quality benchmark fak lacks on
> this host. NOT a threat. One residual — TurboQuant as a candidate #1047 quantizer — is FILED
> not built. The one concrete artifact: a §B1 catalog row in
> [`docs/awesome-token-efficiency.md`](../awesome-token-efficiency.md). No capability code change;
> no change to `tools/idea_scout.py`.**

**Source:** https://github.com/tonbistudio/turboquant-pytorch — a from-scratch PyTorch
implementation of Google's **TurboQuant** (ICLR 2026) for LLM KV-cache compression (1021 stars,
Python, last push 2026-04-23). Read from the repo's README via WebFetch on 2026-06-29; this is a
surface read of the implementation's description, **not** a paper audit or a reproduction (the
README names the method and the bit-width/compression/fidelity numbers; this note does not
re-measure them, and notes below where the README itself qualifies them).

## What it is

A **serving / inference** result — a KV-cache **quantizer** (a lossy precision-reduction
*policy*), not an eviction method, an attack, or a protocol. The method, in one pass:

- **Random orthogonal rotation per vector**, to normalize each KV vector's coordinate
  distributions before quantizing (the "Turbo" rotation — it spreads energy so a per-coordinate
  scalar quantizer is near-optimal regardless of the raw distribution's shape).
- **Per-coordinate Lloyd-Max optimal scalar quantization**, applied independently to each
  coordinate of the rotated vector, storing the quantized indices + a norm value for
  reconstruction.
- **Training-free and calibration-free** — applied post-hoc directly to captured KV tensors,
  at **per-token** granularity (each token's key and value vector compressed independently).
- **Asymmetric bit allocation** (the V3 refinement): *"keys get more bits than values"* because
  keys determine attention focus (so they need precision) while values are error-tolerant through
  averaging. `TurboQuantV3(key_bits=4, value_bits=2)` gives a ~3-bit average with better results
  than a symmetric 3/3 split. (The V2 "QJL" residual-correction stage was **dropped in V3** —
  community findings showed it degrades attention because softmax amplifies its added variance.)

The numbers the README reports — and explicitly qualifies:

- **2x compression with perfect generation** at K6/V4;
- **5x compression at K4/V2** — the headline 3-bit-average figure — **but with quality
  degradation (generation failures)**;
- **99.5%+ cosine similarity on attention scores**, with the README's own caveat that this
  **"does not guarantee working generation"**;
- **94–99% top-1 match rate** across configurations.

## Where this lands on fak (surfaces verified against the tree)

fak fronts an engine and, on the fused path, runs its own reference engine with an **addressable,
bit-exact KV cache**. The relevant surfaces:

- **The §B1 KV-quantization landscape.** [`docs/awesome-token-efficiency.md`](../awesome-token-efficiency.md)
  §B1 already lists **KIVI** (2-bit, Keys per-channel / Values per-token) and **KVQuant**
  (per-channel Key pre-RoPE + per-token Value) as `fak: 🟡`, and the engine-native INT8/INT4/FP8
  rows as `fak: ➖`. TurboQuant is a new entrant on the **same axis** — and its asymmetric
  key>value split is the same observation KIVI/KVQuant encode structurally (Keys get the
  finer-grained per-channel treatment; Values the coarser per-token one).
- **fak's #1047 q8 KV-precision tier already keeps Keys more precise.**
  [#1047](https://github.com/anthony-chaudhary/fak/issues/1047) ships the KV precision tiers as a
  planner half (🟡, engine GPU-gated). Its q8 tier's storage layout is **`f32-Kraw + q8_0-K/V`**
  (`internal/compute/capacity.go`: *"the q8 tier's f32-Kraw + q8_0-K/V layout"*) — the **pre-RoPE
  Key (`Kraw`) is held in full f32** while the rest narrows to q8_0. That is structurally
  TurboQuant's "keys get more bits than values," reached independently. **But fak's reason is not
  attention fidelity — it is exactness:** `internal/model/kvcache.go` keeps `Kraw` (the pre-RoPE
  K) *"so that a span can be repositioned"*, and the evictor re-derives every shifted survivor's
  post-RoPE K from the lossless `Kraw` in a single rotation, leaving the cache byte-for-byte
  identical to a run that never saw the evicted span (`max|Δ| = 0`). fak keeps the Key lossless
  because the **bit-exact evictor** depends on it; TurboQuant keeps the Key precise because
  **attention focus** depends on it. Same asymmetry, two different load-bearing reasons — and that
  difference is exactly why the next question answers *don't adopt the mechanism*.
- **fak's q8 quality gate is the same 0.995 cosine TurboQuant reports — and fak treats it as
  necessary, not sufficient.** fak's own q8-vs-f32 approximation gate is **cosine ≥ 0.995**
  (`internal/compute/compute_test.go`: *"Q8 vs f32 cosine … < 0.995 (the Approx gate)"*).
  TurboQuant reports **99.5%+ attention cosine** — and its README is explicit that this number
  **does not guarantee working generation**, with the 5x@3-bit config showing **generation
  failures**. This is independent, third-party confirmation of fak's witness discipline: an
  attention-cosine number is a *proposal to verify against a served generation witness*, never a
  generation-quality claim on its own.
- **The exact-eviction MECHANISM is the orthogonal axis (fak's ✅⭐).**
  [`docs/awesome-token-efficiency.md`](../awesome-token-efficiency.md) §B2 ends with fak's
  `✅⭐ UNIQUE` row: addressable, bit-exact mid-run causal eviction (`max|Δ| = 0`). TurboQuant is a
  **lossy** precision reduction; it lives on the lossy side of the fence fak draws (line 39–40:
  lossy methods *"trade fidelity for size and belong behind a flag with a witness"*). It is not a
  competitor to the exact path — it is a candidate occupant of the lossy precision tier.

## The three triage questions

### Adopt as a capability? — adopt nothing as a mechanism now; cite it, and catalog it.

The thesis is on-mission and the asymmetric key>value insight independently validates fak's q8
`f32-Kraw + q8_0-K/V` layout, but the **mechanism is not adopted in this increment**, for a
fak-specific structural reason plus a missing witness:

1. **It is the orthogonal axis to fak's value-add, and it conflicts with the exact path.** fak's
   `✅⭐` differentiator is *bit-exact* eviction, which is load-bearing on the **lossless pre-RoPE
   `Kraw`**. TurboQuant's random rotation + per-coordinate quantization is a **lossy** transform of
   the key; running it on the key destroys the exact `Kraw` the evictor re-rotates survivors from,
   so TurboQuant cannot sit *under* the exact path — it can only occupy the **lossy** #1047
   precision tier, alongside KIVI/KVQuant. There it is a *candidate quantizer*, not a fak
   primitive.
2. **The load-bearing quality claim is unwitnessed on this host.** The whole value of any lossy KV
   quantizer is that the compressed cache **still generates the right answer** — and TurboQuant's
   own README is the cautionary example: 99.5%+ attention cosine, yet generation failures at the
   headline 5x. Proving a quantizer's quality requires a **served model emitting real generations,
   scored end-to-end** — which this **win32 dev box cannot run** (it serves no model; see
   [`AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md)).
   Shipping a quantizer here would be an **unwitnessed quality claim** — the precise thing fak's
   witness discipline refuses.

The concrete adoption in this increment is the **citation**: a §B1 catalog row recording
TurboQuant as the random-rotation + per-coordinate near-optimal scalar quantizer with asymmetric
key>value bits, carrying fak's honest position (a candidate for the lossy #1047 tier; the exact
path is a different axis; the quality claim needs a served witness).

### Defend against as a threat? — no. A performance capability on the same side, no adversary.

TurboQuant is a lossy-compression *optimization*; there is no attacker and no security surface.
The only "defensive" note is the standard lossy-cache fence fak already carries: a quantizer that
narrows KV precision is **not** fak's bit-exact path, and any precision-reduced cache is by
definition lossy and must be labelled as such (the same honesty the §B1 catalog and the
[KV-precision-tiers](EVICT-ON-PAGED-KV-DESIGN-2026-06-28.md) work carry). TurboQuant does not move
that boundary; it is a candidate *policy* on the lossy side of it.

### Why not build it now? — wrong axis for the exact path, plus a served quality witness this host cannot reach.

Two gates, both honest, restating the above as the build decision:

1. **Structural: it is the lossy-precision axis, not the exact-eviction axis.** Adopting TurboQuant
   means wiring an engine-side KV quantizer into the lossy #1047 precision tier — a GPU engine
   quant seam — not extending the exact evictor (which it would break). That seam is the 🟡
   half-shipped, GPU-gated part of #1047, not landable on this CPU-only win32 host.
2. **Witness: no served generation-quality benchmark on this host.** Even with the seam, the
   load-bearing "this quantizer preserves answer quality" half needs a served-model generation
   benchmark this box cannot run. Until that witness exists, the quantizer stays filed.

## Recorded residual (FILED, not built)

The smallest honest follow-on is **TurboQuant (or its asymmetric key>value scalar-quant idea) as a
candidate quantizer for the lossy #1047 q8/f32 KV-precision tier** — an engine-side, default-off
precision policy that narrows the q8 tier further (e.g. a K4/V2-style asymmetric split) *below* the
exact path, never on the `Kraw` the exact evictor needs. It is **not shipped here** because (a) it
belongs to the GPU-gated engine quant seam of #1047, not the exact path, and (b) its value depends
on a served generation-quality benchmark this host cannot run — so the "this quantizer preserves
quality" half would be unwitnessed. Filed as the next checkable step, gated on a model-serving host.

## Scout calibration

Surfaced under topic **`kv-prefix-cache-reuse`** (score 35) on a genuine `kv cache` term +
recency (96 days) + a well-starred (1021), recently-pushed repo — **correctly on-topic**: a
serving-side KV-cache-compression result that touches fak's KV-precision (#1047, `f32-Kraw + q8_0`)
and exact-eviction (`internal/model/kvcache.go`) surfaces and slots cleanly into the §B1 catalog
next to KIVI/KVQuant. The scout judged new-and-on-topic and handed the worth-pursuing call to human
triage — **working as designed**. No change to `tools/idea_scout.py`.

## Disposition

**Cite as prior art** (a new §B1 KV-quantization entrant, and independent validation of fak's
asymmetric `f32-Kraw + q8_0-K/V` key-precision layout and of fak's attention-cosine-is-not-a-
generation-witness discipline) and **record the candidate-quantizer fit + the exact-path conflict**
above. No mechanism adopted (TurboQuant is the orthogonal lossy-precision axis; running it on the
key destroys the lossless pre-RoPE `Kraw` the exact evictor depends on, and proving it preserves
quality needs a served generation benchmark this host cannot run), no threat to defend (no
adversary), one residual filed behind a model-serving gate fak cannot reach on this win32 box. The
issue is resolved by this triage note + the §B1 citation; a TurboQuant-class quantizer for the
lossy #1047 tier is the named next step.
