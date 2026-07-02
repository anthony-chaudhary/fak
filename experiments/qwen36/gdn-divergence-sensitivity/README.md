<!-- RESULT NUMBERS BELOW ARE FILLED FROM result.json (full 24x48 run); do not edit by hand -->
# GDN reduction-order / precision sensitivity — can rounding explain the token-3 drift? (device-independent)

Host-runnable witness for the **correctness** half of Qwen3.6-27B↔llama.cpp parity — the
[token-3 drift](../token3-drift-investigation-2026-06-28.md). On the fixed 22-token ChatML
prompt, greedy decode, fak and llama.cpp agree for two tokens (`248068, 198`) then disagree on
the third: fak picks `8160` (logit **23.18**, runner-up `90700` at **21.43** — a **~1.75-logit
near-tie**) where llama.cpp picks `90700`. The arch math is bit-exact vs HF on the tiny fixture
and the drift survives the Q8→q4_k change on *both* engines, so it localizes to a **kernel-numerics
divergence at 27B scale on the hybrid Gated-DeltaNet path**. The top-ranked hypothesis (H1) is
that the delta-rule recurrent scan — the only **stateful** op — rounds differently from
llama.cpp's kernel and that per-step difference **compounds** across tokens and layers until it
tips the near-tie at token 3.

This artifact tests that hypothesis **without a Mac, without a GPU, and without the 27B
artifact**, by isolating the variable H1 names: it runs a faithful 48-GDN-layer residual stack
**twice**, where the two runs are bit-identical in every op *except* the recurrent scan's
numerics, and measures how fast the resulting relative hidden-state divergence ρ = ‖Δhidden‖/‖hidden‖
compounds with depth. Two brackets:

- **`reorder`** — reverse the scan's `i`-reduction order. Same exact arithmetic, different
  rounding; models a different SIMD/threadgroup reduction order (the 1-ULP floor).
- **`f16state`** — keep forward order but round-trip the recurrent state through IEEE half each
  step; models a kernel that stores the GDN state in f16 (a realistic worst case for the
  accumulator).

Both runs share the same seeded-random weights and inputs, so the **only** source of A↔B
divergence is the scan numerics — exactly the "same math, different rounding" H1 posits. The
recurrence body is copied verbatim from
[`internal/model/metal_prefill_hybrid_core.go:202-246`](../../../internal/model/metal_prefill_hybrid_core.go)
(the prefill twin of `qwen35.go:linearAttnStep`); only the `i`-loop direction / state dtype
differs.

```text
seeded-random weights + inputs
   |
   +--> run A: 48-GDN-layer stack, forward scan --+
   |                                              +--> ρ = ‖Δhidden‖/‖hidden‖ per layer
   +--> run B: same stack, scan numerics differ --+
        (`reorder` / `f16state` bracket)
```

## The question it answers (falsifiable)

> Is a reduction-order-only (or f16-state) numeric difference, compounded over 48 GDN layers and
> ~24 carried positions, large enough **on its own** to flip a ~1.75-logit near-tie?

A near-tie of margin `m` logits at logit scale `|logit| ≈ 20` flips when the hidden state moves
by ρ\* ≈ `m / |logit|` ≈ `1.75 / 20` ≈ **0.0875** (8.75%) in relative L2 (order-of-magnitude;
logits ≈ wᵥ·hidden so |Δlogit| ≈ |logit|·ρ). The experiment reports the measured ρ for each
bracket and compares.

## Run

```
go run ./experiments/qwen36/gdn-divergence-sensitivity            # human table (default 24 positions, 48 layers)
go run ./experiments/qwen36/gdn-divergence-sensitivity -json      # machine result (result.json)
go run ./experiments/qwen36/gdn-divergence-sensitivity -tokens 6 -layers 6   # quick smoke
```

No build tags, no GPU, no model file, no `internal/model` dependency — pure stdlib Go, so it
builds and runs even when the working tree's `internal/model` is mid-refactor. Verified green on
this `windows/amd64`, `CGO_ENABLED=0` host (`go1.26.3`).

## Result (full 24-position × 48-GDN-layer run, this box)

Relative hidden divergence ρ = ‖Δhidden‖/‖hidden‖ at the measured (token-3) position, the
**only** difference between the two runs being the scan numerics:

| bracket (scan-numerics difference only) | ρ after layer 1 | ρ after layer 48 | implied \|Δlogit\| (≈ ρ·20) | flips the 1.75-logit near-tie? | margin below threshold |
|---|---:|---:|---:|:---:|---:|
| `reorder` — reverse-`i` reduction (1-ULP, different SIMD order) | 1.75e-08 | **1.95e-07** | 3.9e-06 | **no** | ~4.5×10⁵× too small |
| `f16state` — f16 state round-trip each step (f16 accumulator) | 3.46e-06 | **3.13e-05** | 6.3e-04 | **no** | ~2.8×10³× too small |

Flip threshold: ρ\* ≈ 1.75 / 20 ≈ **0.0875**. The full per-layer ρ curve (all 48 layers, both
brackets) is in [`result.json`](result.json). ρ grows **sub-linearly** with depth (≈×11 over 48
layers for `reorder`, ≈×9 for `f16state`) — the residual stream attenuates relative perturbations
rather than amplifying them — so extrapolating deeper or to more tokens does not close the gap.

## What the number means

Even the **f16-state** bracket — an *aggressive* ~5e-4 relative per-step perturbation of the
recurrent accumulator — compounds to a final relative hidden divergence **orders of magnitude
below** the ~8.75% needed to flip the observed near-tie, and the **reorder** (1-ULP) bracket is
smaller still. The residual stream is contractive in *relative* perturbation (each layer's output
is a shrinking fraction of the growing residual norm), so reduction-order/precision rounding
**attenuates** rather than blows up with depth.

**Verdict: numerics-as-pure-rounding is INSUFFICIENT to explain the token-3 flip.** The flip
implies an **anomalous (algorithmic / ordering) divergence** between fak's and llama.cpp's GDN
kernels — a genuinely different formulation or op, not mere accumulation noise. This **sharpens**
the per-layer divergence probe of the
[investigation doc §3](../token3-drift-investigation-2026-06-28.md): its first-divergence
threshold must be set to catch an **anomalous** cosine drop (a real op mismatch), because a
1-ULP-floor threshold would correctly see the rounding-class layers as "agreeing" and so would
*not* be tripped by the rounding modeled here — the diverging layer is where something
*algorithmically* differs.

## What this does NOT do

It does not run llama.cpp, does not load the 27B artifact, and does not claim to have found the
diverging `(layer, op)` — those are the Mac/artifact-gated steps
([investigation §5 steps 4–5](../token3-drift-investigation-2026-06-28.md)). It is a synthetic
residual stack with seeded-random (not trained) weights and a representative near-1 GDN decay; its
claim is about **order of magnitude** — that rounding-class divergences are ~10³–10⁴× too small to
flip the observed near-tie — which a ~4-orders-of-magnitude margin makes robust to those modeling
choices. The on-device per-layer capture that names the actual diverging op remains the honest
`not yet` of the investigation doc.
