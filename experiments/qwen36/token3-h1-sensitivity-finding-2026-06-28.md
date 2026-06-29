# Qwen3.6-27B token-3 drift — H1 sensitivity result: rounding bounds, a systematic bias integrates (2026-06-28)

**Status: EXECUTED host-independent experiment.** This is the *result* of the
experiment the investigation note
([`token3-drift-investigation-2026-06-28.md`](token3-drift-investigation-2026-06-28.md)
§4, §5-step-2) named as "the highest-value host-independent experiment short of the
Mac capture" but left un-run. The runnable model is
[`h1_recurrence_sensitivity.py`](h1_recurrence_sensitivity.py); the gradeable witness is
[`h1-recurrence-sensitivity-20260628.json`](h1-recurrence-sensitivity-20260628.json).

Honesty rider ([`../../docs/proofs/00-METHOD.md`](../../docs/proofs/00-METHOD.md)): this
is a **synthetic plausibility / sensitivity model with assumed magnitudes**, run on a
`win32` box — **no 27B weights, no llama.cpp, no GPU**. It does **not** reproduce the real
fak↔llama.cpp drift (that needs llama.cpp in the loop, Mac-gated). What it *can* settle is
the **integrate-or-bound dynamics** of a per-step error in the GDN recurrent scan, which is
exactly H1's open question.

---

## 1 — What was run

One representative Qwen3.6-27B GDN delta-rule head (kHd=vHd=128), stacked over **L=48**
layers with a residual stream, run over **P=22** prefill + **D=3** decode positions —
matching the fixed 22-token ChatML prompt and the token-3 decode step. The modeled per-step
recurrence is faithful to `qwen35.go:481-517`:

```
st[i,d] *= g                  # decay the persistent state (g near 1)
kvmem[d]  = SUM_i st[i,d]*k[i] # reduction #1  (order/dtype-sensitive)
delta[d]  = (v[d] - kvmem[d]) * b
st[i,d]  += k[i]*delta[d]      # rank-1 state update
out[d]    = SUM_i st[i,d]*q[i] # reduction #2  (order/dtype-sensitive)
x += out                      # residual add (logit proxy)
```

Three arms differ **only** in a per-reduction perturbation injected into the two
reductions, against a clean float64 reference:

| arm | per-reduction factor | models |
|---|---|---|
| REF | exact (f64) | the ideal |
| RANDOM | `× (1 + eps·N(0,1))` | reduction-**ORDER** rounding — a different but valid summation order, uncorrelated step to step |
| SYSTEMATIC | `× (1 + eps)` constant | a genuine **algorithmic / FMA-fusion / ordering BIAS** that pushes the same way every step |

`eps = 6e-8` (f32 ULP class). The measured quantity is the relative L2 divergence of the
final-layer residual (the logit proxy) vs REF, at each decode token. The flip bar is the
observed near-tie `1.75 logits / ~20 logit-scale = 0.0875`.

---

## 2 — The result (seed 20260628, decay swept 0.9 → 0.9999)

| decay g | RANDOM relΔ t1→t3 | grow | integrates? | SYSTEMATIC relΔ t1→t3 | grow | integrates? | sys/rand @t3 |
|---:|---|---:|:--:|---|---:|:--:|---:|
| 0.9000 | 3.15e-7 → 2.48e-7 | 0.79× | no | 8.47e-7 → 9.02e-7 | 1.07× | yes | 3.6× |
| 0.9900 | 3.19e-7 → 2.48e-7 | 0.78× | no | 8.48e-7 → 9.08e-7 | 1.07× | yes | 3.7× |
| 0.9990 | 3.19e-7 → 2.48e-7 | 0.78× | no | 8.48e-7 → 9.09e-7 | 1.07× | yes | 3.7× |
| 0.9999 | 3.19e-7 → 2.48e-7 | 0.78× | no | 8.48e-7 → 9.09e-7 | 1.07× | yes | 3.7× |

The depth×position **amplification** of a systematic per-reduction bias is ~**15×** (and
linear in eps — confirmed: at `eps=6e-3` the systematic arm reaches relΔ **0.095**, just
over the 0.0875 bar). So the systematic per-reduction relative bias that reaches an
argmax-flip-plausible magnitude by token 3 is:

> **eps_flip ≈ 5.8e-3** — about **~10⁵× larger than f32 rounding** (`6e-8`).

---

## 3 — The finding (what it sharpens)

1. **Pure reduction-ORDER rounding (the literal H1) is BOUNDED.** A random, uncorrelated
   per-step error does a decaying random walk and reaches a small stationary magnitude
   (~2.5e-7) under decay `g < 1`; it does **not** integrate, and it is ~10⁵× too small to
   tip a token-3 near-tie. So *reduction order in the scan, on its own, is an unlikely sole
   cause* of the flip.
2. **A SYSTEMATIC (correlated) per-step bias INTEGRATES.** The near-1 decay barely damps a
   consistently-signed push, so it accumulates across the 48 layers × 25 positions and
   reaches the flip bar at an **algorithmic-scale** per-reduction bias (~6e-3), not a
   float-noise one.
3. **Directive for the Mac per-layer probe** ([`…investigation…`](token3-drift-investigation-2026-06-28.md) §3):
   hunt for a **systematic, consistently-signed** kernel difference — a wrong/missing eps, a
   scale-ordering difference, a normalization-form mismatch, or a consistent FMA-fusion — **not
   mere reduction noise**. The experiment distinguishes **random vs systematic error**, *not*
   one suspect op vs another, so it does **not** demote H1: the scan's own reduction can itself
   be *systematic* (a consistently-signed serial-f32 / f16→f32 / FMA-fusion rounding direction),
   in which case H1 integrates exactly as modeled. What it **does** establish is that the
   culprit — wherever it sits (H1's accumulation, H2 q/k L2-norm eps+order, H3 gated RMSNorm) —
   must have a **systematic component**; pure uncorrelated reduction-order noise is ~10⁵× too
   small. The scan is the **amplifier** that integrates whatever systematic bias is injected
   upstream; the state-feeding normalization ops (H2/H3) are *candidate sources* of such a
   bias, not a proven ranking over H1.

---

## 4 — Caveats (what this does NOT establish)

*This section incorporates a 4-lane adversarial review (dynamics / faithfulness / inference /
honesty). Lane verdicts: C1 dynamics SOUND_WITH_CAVEAT, C2 faithfulness flagged the H2
omission below, C3 inference VALID-WITH-NARROWED-SCOPE (folded into §3), C4 honesty HONEST.*

- It is a synthetic model with **assumed magnitudes** (q/k/v ∼ 0.5, logit scale ∼20). The
  *qualitative* random-bounded / systematic-integrates split is robust — it is the standard
  bounded-variance behaviour of an AR(1)-class recurrence with `|g|<1`, and the rank-1 state
  feedback `st += outer(k,(v−kvmem)·b)` does **not** amplify a zero-mean random error (the
  state-dependent noise stays zero-mean in steady state) — but the *exact* eps_flip number is
  **order-of-magnitude only**.
- **It models the recurrence's error-PROPAGATION dynamics abstractly, not the specific
  suspect ops.** It omits the q/k L2-norm + 1/√kHd scale (**H2**, the faithfulness lane's
  flagged critical omission), the gated RMSNorm readout (H3), the depthwise conv1d + SiLU
  (H4), and the (1+w) norms. So it shows *that* a systematic bias integrates and *how large*
  it must be — it does **not** prove *which* op supplies it. An omitted op that injects a
  correlated bias would *strengthen* the systematic story; this is exactly what the Mac probe
  must localize.
- It distinguishes **random vs systematic** error, **not** H1 vs H2/H3 — H1's own reduction
  can be systematic, so H1 is **not** demoted (see §3).
- 3 decode tokens is a short window; the distinguishing signal is the magnitude gap
  (systematic 3.7× random at equal eps) and the bounded-vs-linear-growth shape, not a large
  token-over-token ratio.
- It does **not** reproduce the fak↔llama.cpp drift and makes **no** claim about the real
  27B logits. The faithful capture remains the Mac/artifact-gated per-layer probe
  ([#7](https://github.com/anthony-chaudhary/fak/issues/7) per-layer hidden capture).

---

## 5 — Reproduce

```sh
python experiments/qwen36/h1_recurrence_sensitivity.py \
  --out experiments/qwen36/h1-recurrence-sensitivity-20260628.json \
  --markdown experiments/qwen36/h1-recurrence-sensitivity-20260628.md
# sweep the per-step bias magnitude to see the flip threshold:
python experiments/qwen36/h1_recurrence_sensitivity.py --eps 6e-3 --decays 0.999
```

Deterministic (seeded); needs only CPU + numpy. The conclusion is in the witness JSON's
`verdict.reading` and `verdict.systematic_eps_flip_threshold`.
