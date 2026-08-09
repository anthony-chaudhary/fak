package kvbudget

// This file adds the OPTIONAL per-layer refinement of a Shape's cache geometry
// (issue #5498). Everything above sizes a UNIFORM architecture: one scalar
// Layers count, one head geometry, and every layer attending over the whole
// context — so the KV footprint of a stream is exactly ctx × KV-bytes/token.
//
// That is exact for a uniformly-global model and an OVER-count for any
// architecture whose layers cap their attention extent. A sliding-window layer
// with a window of W holds at most W tokens of KV no matter how long the context
// grows, so its contribution is FLAT in ctx, not linear. Recent Gemma and
// Mistral families interleave such local layers with a few global ones: the
// public `google/gemma-4-26B-A4B-it` config declares 30 layers whose layer_types
// repeat five `sliding_attention` then one `full_attention` (25 sliding, 5
// global) with `sliding_window: 1024`. Its true worst case is therefore
// CONSTANT + LINEAR — 25 layers bounded at 1024 tokens plus 5 that grow with ctx
// — where the uniform formula charges all 30 for the full ctx.
//
// The over-count propagates: MaxStreams(budget, perStream) under-reports how
// many streams fit, which is the conservative-but-wrong direction for an
// admission gate — it refuses streams that would have fit.
//
// # The zero value changes nothing
//
// The refinement hangs off ONE new optional field, Shape.PerLayer, a *nil*
// pointer by default. A Shape that declares no profile (every Shape that exists
// today, including GLM52DSA and everything model.Config.KVCacheShape builds)
// takes the same expression it always took, so its answer is bit-for-bit
// unchanged — see uniform() below and TestPerLayerZeroValueIsBitIdentical. The
// pointer also keeps Shape COMPARABLE (`==`), which callers rely on.
//
// The per-token methods above are deliberately left alone: with a window there
// is no single per-token figure, so the ctx-dependent truth lives in the new
// *PerStream methods and the per-token ones keep their uniform meaning.

// LayerProfile is a Shape's optional per-layer geometry: the layers that differ
// from the Shape's uniform scalars. Every slice is independently optional and
// indexed by layer (0-based); a layer past the end of a slice, or one whose
// entry is non-positive, falls back to the Shape's scalar. An all-empty profile
// therefore means exactly the same thing as no profile at all.
//
// The field names and the "absent means uniform" convention mirror
// model.Config.{Window, NumKVHeadsPerLayer, HeadDimPerLayer}, which the loader
// already fills per layer for the interleaved-attention families
// (applyGemma4Config in internal/ggufload). This package does not import that
// one — it stays a stdlib-only leaf — so the values are passed in by whoever
// builds the Shape.
type LayerProfile struct {
	// Window is the per-layer sliding-window cap in TOKENS: layer l retains at
	// most Window[l] tokens of KV however long the context grows, so its cache
	// contribution is min(Window[l], ctx) rather than ctx. A non-positive entry
	// (and any layer past the end) means FULL causal attention — the same
	// "absent or ≤ 0 ⇒ full" convention model.Config.Window uses.
	Window []int
	// NumKVHeads, HeadDim, and VHeadDim override the Shape's uniform MHA head
	// geometry for a layer (Kind==MHA only; an MLA layer caches a latent whose
	// width is head-independent). Interleaved-attention families need these
	// TOGETHER with Window because their local and global layers differ in head
	// WIDTH as well as extent — gemma-4's `head_dim: 256` vs
	// `global_head_dim: 512`. Declaring a window without the wider global head
	// would under-count the global layers, which is the dangerous direction for
	// an admission gate, so a caller that sets one should set both.
	NumKVHeads []int
	HeadDim    []int
	VHeadDim   []int
}

// perLayerOr returns the per-layer override for layer l, or the uniform scalar
// when the layer declares none. "Declares none" is a layer at or past the end of
// the slice, or a non-positive entry — the model.Config convention, which uses
// an empty/short slice for a uniform model and a sentinel ≤ 0 for a layer that
// opts out.
func perLayerOr(xs []int, l, uniform int) int {
	if l < 0 || l >= len(xs) || xs[l] <= 0 {
		return uniform
	}
	return xs[l]
}

// uniform reports whether the Shape declares no per-layer refinement at all —
// no profile, or a profile with every slice empty. When it does, the byte
// figures below take the untouched pre-refinement expression (ctx × bytes per
// token) rather than an algebraically-equal rearrangement, so an unrefined
// Shape's float answer is identical bit-for-bit at ANY quant, not merely equal
// to within a rounding.
func (s Shape) uniform() bool {
	p := s.PerLayer
	return p == nil ||
		(len(p.Window) == 0 && len(p.NumKVHeads) == 0 &&
			len(p.HeadDim) == 0 && len(p.VHeadDim) == 0)
}

// LayerTokens is the number of KV token-slots layer l holds once a stream has
// reached ctx tokens: min(Window[l], ctx) for a window-capped layer, and ctx for
// a layer that attends over the whole context. This is the single place the
// window bound is applied.
func (s Shape) LayerTokens(l, ctx int) int {
	if s.PerLayer == nil {
		return ctx
	}
	if w := perLayerOr(s.PerLayer.Window, l, 0); w > 0 && w < ctx {
		return w
	}
	return ctx
}

// cachedTokens is the total per-layer token-slots the first n layers hold at
// ctx: Σ_{l<n} LayerTokens(l, ctx). For an unwindowed Shape it is exactly
// n × ctx, which is what makes every uniform figure below reduce to the
// pre-refinement one.
func (s Shape) cachedTokens(n, ctx int) int {
	if s.PerLayer == nil || len(s.PerLayer.Window) == 0 {
		return n * ctx
	}
	total := 0
	for l := 0; l < n; l++ {
		total += s.LayerTokens(l, ctx)
	}
	return total
}

// MLAElemsPerStream is the MLA latent + decoupled rope key a whole stream of ctx
// tokens holds: Σ over layers of min(window, ctx) × (KVLoraRank + QKRopeHeadDim).
// Without windows it is ctx × MLAElemsPerToken().
func (s Shape) MLAElemsPerStream(ctx int) int {
	return s.cachedTokens(s.Layers, ctx) * (s.KVLoraRank + s.QKRopeHeadDim)
}

// IndexElemsPerStream is the DSA indexer key a whole stream holds, summed over
// the index layers under the same window bound. IndexLayers is an upper bound at
// Layers (doc §3.3), so the first IndexLayers windows are the ones that apply.
func (s Shape) IndexElemsPerStream(ctx int) int {
	return s.cachedTokens(s.IndexLayers, ctx) * s.IndexHeadDim
}

// MHAElemsPerStream is the full per-head K+V a whole stream holds for standard
// multi-head / grouped-query attention, layer by layer: Σ_l min(window_l, ctx) ×
// kvHeads_l × (headDim_l + vHeadDim_l), each per-layer term falling back to the
// Shape's uniform scalar. Without any profile it is ctx × MHAElemsPerToken().
func (s Shape) MHAElemsPerStream(ctx int) int {
	if s.uniform() {
		return ctx * s.MHAElemsPerToken()
	}
	p := s.PerLayer
	total := 0
	for l := 0; l < s.Layers; l++ {
		heads := perLayerOr(p.NumKVHeads, l, s.NumKVHeads)
		k := perLayerOr(p.HeadDim, l, s.HeadDim)
		v := perLayerOr(p.VHeadDim, l, s.VHeadDim)
		total += s.LayerTokens(l, ctx) * heads * (k + v)
	}
	return total
}

// KVElemsPerStream is the total KV elements a stream of ctx tokens holds,
// branched on attention arch exactly as KVElemsPerToken is. This — not
// ctx × KVElemsPerToken() — is the ctx-dependent truth once any layer caps its
// window, and it equals ctx × KVElemsPerToken() whenever no layer does.
func (s Shape) KVElemsPerStream(ctx int) int {
	if s.Kind == MHA {
		return s.MHAElemsPerStream(ctx)
	}
	return s.MLAElemsPerStream(ctx) + s.IndexElemsPerStream(ctx)
}

// KVBytesPerStream is the full KV footprint of one stream of ctx tokens at the
// given quant. An unrefined Shape takes the original ctx × KV-bytes/token
// expression untouched (bit-identical at any quant); a refined one sums its
// layers under the window bound.
func (s Shape) KVBytesPerStream(ctx int, q Quant) float64 {
	if s.uniform() {
		return float64(ctx) * s.KVBytesPerToken(q)
	}
	return float64(s.KVElemsPerStream(ctx)) * q.BytesPerElem
}

// MLABytesPerStream is the MLA-only footprint of one stream (the doc's "MLA
// only" column), under the same uniform/refined split as KVBytesPerStream.
func (s Shape) MLABytesPerStream(ctx int, q Quant) float64 {
	if s.uniform() {
		return float64(ctx) * s.MLABytesPerToken(q)
	}
	return float64(s.MLAElemsPerStream(ctx)) * q.BytesPerElem
}
