# Llama3 RoPE-scaling — the long-context positional-quality lever (issue #19)

> Lane `model`. The positional-quality long-context arch axis: a piecewise rescale
> of the RoPE inverse frequencies that lets a model trained at 8K attend coherently
> at 128K **without retraining** — the mechanism Llama-3.1/3.2/3.3 ship. This is the
> *non-windowed* counterpart to sliding-window attention (#20): SWA bounds attention
> COMPUTE and the bounded-memory cache bounds MEMORY; rope-scaling fixes the
> remaining wall — the rotational range a full-attention model sees at a position far
> past its trained context. Proven against HuggingFace's own reference; byte-for-byte
> identical to the base RoPE when unset.

## What shipped

An optional `Config.RopeScaling` (`internal/model/rope_scaling.go`) loaded straight
from HF `config.json`'s `rope_scaling` object, plus one transform — `scaleInvFreq` —
applied at the **single** inverse-frequency builder `invFreq` (`internal/model/kv.go`).
Because `newRope` (the from-scratch forward reference) and `cachedInvFreq` (decode +
batch) both route through `invFreq`, the whole CPU prefill+decode+batch path inherits
the rescale from one site:

| consumer | file | path |
|---|---|---|
| `newRope` | `forward.go` | full-prefill reference (HF-oracle path) |
| `cachedInvFreq` → `ropeRow` | `kv.go` | serial f32 decode + `Evict` reposition |
| `cachedInvFreq` (`inv := …`) | `batch.go` | multi-user decode + batched prefill RoPE |

### The NTK-by-parts rescale

RoPE encodes position as a rotation whose per-dimension angle is `pos · inv_freq[j]`.
For each band the wavelength is `2π / inv_freq[j]`; with `old_ctx = original_max_position`:

- **high frequency** (`wavelen < old_ctx/high_freq_factor`) — local position, already
  inside the trained range → **left untouched** (bit-exact).
- **low frequency** (`wavelen > old_ctx/low_freq_factor`) — global position, would wrap
  past anything seen in training → `inv_freq /= factor` (wavelength stretched ×`factor`).
- **medium band** (between the two) — smoothly interpolated, `f·((1-s)/factor + s)`,
  so `inv_freq` stays continuous across the regime boundaries.

For the real Llama-3.1 values (`factor 8 / low_freq 1 / high_freq 4 / orig 8192`,
θ=500000, head_dim=128) this stretches the long-wavelength bands by 8×, so a position
at 64K–128K lands inside the 8K trained rotational range instead of aliasing.

## The two load-bearing invariants

1. **Unset == default == byte-identical.** A model with no `rope_scaling` (SmolLM2,
   Qwen2.5) takes the exact instruction stream it ran before this axis existed:
   `scaleInvFreq(nil, inv)` returns `inv` on the *same backing slice*, no copy, no
   transform. Not asserted — proven: the real-model rungs stay green (R2 cached-decode
   `max|Δ|=0`, R14 prefix-reuse, the HF oracle, the Q8 gate), and
   `TestRopeScalingUnsetIsByteIdentical` shows `invFreq` is `math.Float64bits`-equal to
   an independently-recomputed `1/θ^(2j/hd)` table. An unrecognized `rope_type` (yarn,
   longrope — not wired here) is the same no-op pass-through, so an unknown tag can
   never silently corrupt the table; it just runs base RoPE.

2. **Static in position ⇒ re-rotation-safe.** The rescale is applied once at table
   build, independent of sequence length or live cache size. So the rotation angle for
   band `j` at position `p` is still `p · inv'[j]` — LINEAR in `p` — and RoPE stays
   RELATIVE. `KVCache.Evict`'s single re-rotation of a renumbered survivor (the
   primitive the bounded-memory decode depends on) is therefore still bit-exact under
   scaling: `TestLlama3RopeIsPositionPureUnderEvict` evicts a middle span from a scaled
   cache and gets a `math.Float32bits`-identical result to a cache that never saw the
   span. This is the property longrope/#23 has to *work* to preserve (pin its short/long
   regime at session start, never to the live cache length); llama3 has no regime switch
   and gets it for free.

## Witnesses (`go test ./internal/model -run Rope`, green)

| test | proves |
|---|---|
| `TestRopeScalingUnsetIsByteIdentical` | nil/unknown scaling ⇒ base table, same slice, bit-exact |
| `TestLlama3RopeScalingMatchesHFReference` | production rescale == a literal port of HF `_compute_llama3_parameters` (rel ≤ 1e-12) on the real Llama-3.1 values, and the table actually changed |
| `TestLlama3RopeScalingRegimes` | all three frequency regimes fire: high-freq untouched (bit-exact), low-freq `/factor`, a non-empty medium band strictly between |
| `TestLlama3RopeScalingCacheKeyDistinct` | the inv-freq cache key is fingerprinted by the scaling, so a scaled config never collides onto an unscaled table of equal θ/head_dim (and vice-versa) |
| `TestLlama3RopeIsPositionPureUnderEvict` | `Evict`'s single-rotation reposition is bit-exact under scaling — RoPE stays relative |
| `TestLlama3RopeScalingBoundaries` | the strict-`<` vs `<=` regime boundaries (wavelen == high/low wavelen) match HF exactly — pins the one place a `<`→`<=` edit could silently diverge |
| `TestRopeScalingValidate` | the loud-failure predicate: malformed llama3 params (missing field, factor≤0, low≥high, orig≤0) are rejected; nil / unwired type / real values pass |
| `TestLlama3MalformedFallsBackToBase` | a malformed config never poisons the cached table with NaN/Inf — `scaleInvFreq` returns the base table |
| `TestLoadRejectsMalformedLlama3RopeScaling` | `Load` fails loudly (error naming `rope_scaling`) on a `llama3` config missing numeric fields, instead of silently emitting NaN logits |

The HF cross-check has teeth because the test's reference is the literal two-step
`torch.where` form (`step-1 where → smoothed → is_medium where`), structurally
different from the production three-way switch — matching it is a genuine second
implementation, not the code under test rephrased.

## Robustness — fail loud, never silent (hardened after an adversarial review)

An adversarial verification pass found that a `rope_type:"llama3"` config *missing*
its numeric fields (a truncated or hand-written `config.json`) unmarshals to Go
zero-values, which built an **all-NaN** `inv_freq` table — and because the table is
memoized in a process-global cache, that NaN was sticky and silently turned every
logit into NaN (argmax → token 0 forever). HF raises a loud `KeyError` on the same
config; fak now matches that:

- `RopeScaling.validate()` enforces HF's parameter requirements (`factor>0`,
  `0<low_freq_factor<high_freq_factor`, `original_max_position_embeddings>0`). The
  `low<high` ordering also keeps the production high/medium/low switch a faithful
  collapse of HF's `torch.where` (it is *not* for an inverted `low>high` config).
- `Load` calls it right after reading `config.json` and returns a descriptive error
  — the real-checkpoint path fails **loudly at load**, not silently at decode.
- `scaleInvFreq` has a defense-in-depth guard: a directly-constructed malformed
  `Config` falls back to the base table (valid, unscaled numerics) rather than
  poisoning the cache with NaN/±Inf.

## Honest scope

The **mechanism** is proven against HF's own reference and against the bit-exact rungs
on the SmolLM2 path. What is *not* claimed here: a re-exported real **Llama-3.1-8B HF
oracle** that would additionally prove a checkpoint's `rope_scaling` object flows
through `Load` into argmax-identical long-context logits end-to-end. That needs the 8B
weights on the box and is the separable, checkpoint-bound follow-up — exactly as the
SWA family-window-value oracle is for #20.

**Device lanes are NOT yet wired (known gap).** The scaling reaches every CPU lane —
`Forward`/`newRope`, serial + batched decode, batched prefill, the Q8 path, and
`KVCache.Evict` — because they all route through the single `invFreq`/`cachedInvFreq`
builder. The two *device* RoPE paths reconstruct `inv_freq` from `θ` alone, so they
**bypass** the rescale: the compute-HAL backend (`hal.go`, `be.RoPE(q, pos, …, θ)`,
peer GPU lane) and the Metal GPU-resident prefill (`metal_prefill.go`, `forward.m`
`rope_k`). A `llama3`-scaled checkpoint run through either today *silently* gets base
RoPE — and within a Metal session the resident prefill would even disagree with its
own scaled hybrid/CPU fallback. Both are opt-in / build-tagged (`fakmetal`), not the
default lanes, so no default path is affected; but the device wiring (thread the
scaled table through `compute.KVConfig` + `FwdConfig`) — and, until then, a guard that
*errors* when a scaled `Config` enters the device path instead of running base RoPE —
is the peer-lane follow-up. Tracked honestly rather than papered over.

## Where this sits

| wall (to long, non-windowed context) | before | now |
|---|---|---|
| compute (per-token attention) | dense O(N) | O(window) read-mask — SWA, #20 (windowed only) |
| memory | O(N) cache | O(window) bounded decode — #20 follow-on (windowed only) |
| **positional quality (full attention)** | **RoPE aliases past the trained range** | **llama3 inv_freq rescale — 8K→128K, this increment** |

longrope (Phi, #23) and the remaining #19 mechanical bits (qk-norm, attn/logit
soft-caps, per-projection bias, embed/logit scale) are the next separable rungs.

## longrope (#23) — Part A (inv_freq) shipped; attention_factor is Part B

longrope (Phi-3/3.5/4, 4K→128K) has **two** outputs in HF
`_compute_longrope_parameters`, both pinned at session start. **Part A — the per-dim
`inv_freq` rescale — is shipped here**; Part B (the `attention_factor` mscale) is the
next sub-rung.

**Part A (shipped).** `scaleInvFreq`'s switch gains a `"longrope"` case:
`inv_freq'[j] = inv_freq[j] / ext[j]`, where `ext` is a length-(head_dim/2) multiplier
array — `long_factor[]` when the model's *static* max context exceeds
`original_max_position_embeddings`, else `short_factor[]`. The pure per-dim divide
(Phi3 form), **not** a yarn-style extrapolation/interpolation blend. It routes through
the single `invFreq` builder, so every CPU lane inherits it for free, exactly like
llama3, and the no-window/no-scaling default stays byte-identical. Proven against a
from-scratch port of HF's `inv_freq = 1/(ext·θ^(2j/dim))` (`internal/model/rope_scaling.go`,
`longropeInvFreq`).

**The #23 guardrail (implemented).** The short-vs-long regime is pinned to the model's
*static* max-context (`MaxPositionEmbeddings > original`), **never** the live cache
length — else a mid-session `Evict`/`TrimToWindow` shrink below `orig` would flip the
regime, rebuild `inv_freq` with `short_factor`, and make `Evict`'s single-rotation
reposition rotate survivors with a different table than encoded them (breaking R2/R3
bit-exactness). Because the choice is derived from immutable `Config`, which `Evict`
(holds `cfg`) and every forward path share, the regime is fixed for the session — the
`swa.go` discipline (key off a static quantity, not a live index). The resolved regime
+ an FNV hash of the `short/long_factor` slices are folded into `ropeScaleKey` so a
scaled/short/long/plain config never shares a cached table.

longrope witnesses (`go test ./internal/model -run Longrope`, green):

| test | proves |
|---|---|
| `TestLongropeInvFreqMatchesHFReference` | `longropeInvFreq` == HF's `1/(ext·θ^(2j/dim))` (rel ≤ 1e-12) in the long regime, table changed |
| `TestLongropeRegimePinnedByStaticContext` | the short/long choice tracks static max-vs-orig, not the cache — short_factor=1 ⇒ base, bit-exact |
| `TestLongropeIsPositionPureUnderEvict` | `Evict` reposition bit-exact under a long-regime config |
| `TestLongropeCacheKeyDistinct` | short / long / plain configs get distinct cached tables (slice-hash + regime fingerprint) |
| `TestLongropeValidateAndLoad` | absent/mis-sized factor arrays or non-positive context fail loud at `validateRopeScaling` and `Load` |

**Part B (next sub-rung) — `attention_factor` (mscale).** A scalar `= sqrt(1 +
ln(factor)/ln(orig))` (factor = max_pos/orig; `1.0` when factor ≤ 1; explicit config
value overrides). HF **multiplies cos AND sin** by it, so the q·k score carries
`attention_factor²` — folding it once onto the logit scale would NOT match the oracle.
For Phi-3.5-mini-128k (factor 32) it is ≈1.19 → ~1.42× on the score, so it is **not**
optional for full Phi oracle fidelity. Implementation (deferred to keep this rung
inv_freq-only and low-risk): a `cfg.ropeAttentionFactor()` (1.0 default) multiplied
into the cos/sin at the model-package builders (`newRope`, `ropeRowForLayer`, the
`batch.go` `ropeRowInto` sites), **gated on `!= 1.0`** so the default path is literally
untouched and byte-identical; the device lanes stay unwired (consistent with the gap
above). The G-S6 done-witness — a real Phi-128k HF oracle proving inv_freq + attention
end-to-end — is checkpoint-bound, deferred like the llama3 8B oracle.
