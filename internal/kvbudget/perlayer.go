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
// pointer by default. A Shape that declares no profile (GLM52DSA and every
// uniform-attention Shape model.Config.KVCacheShape builds) takes the same
// expression it always took, so its answer is bit-for-bit unchanged — see
// uniform() below and TestPerLayerZeroValueIsBitIdentical. The pointer also
// keeps Shape COMPARABLE (`==`), which callers rely on. The compression axis
// below keeps the same promise for every profile that does not declare it: a
// window-only profile sizes exactly as it did before that axis existed.
//
// The per-token methods above are deliberately left alone: with a window there
// is no single per-token figure, so the ctx-dependent truth lives in the new
// *PerStream methods and the per-token ones keep their uniform meaning.
//
// # Compressed schedules
//
// DeepSeek-V4 Flash adds a second per-layer axis, compress_ratios (#13555). A
// layer with ratio r > 0 keeps, besides its sliding window, one compressed row
// per r positions for the whole context, so its row count at ctx is
//
//	min(window, ctx) + ceil(ctx / r)
//
// — linear in ctx at slope 1/r (r = 4 or 128), with a flat window term. A
// window alone UNDER-counts such a layer, the unsafe direction for admission,
// so a profile that declares the window for these families must declare the
// ratios too. The V4 lightning indexer rides the compressed rows: an indexing
// layer keeps one key of its per-layer index width per compressed row, not per
// token.
//
// This is PLANNING geometry — what the native V4 attention state holds, sized
// for admission and context fit — derived by reading that state's code, not an
// allocator's layout and not a hardware measurement.

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
	// CompressRatio is the per-layer compression ratio (DeepSeek-V4 Flash
	// compress_ratios): a layer with r > 0 additionally retains
	// CompressedRows(l, ctx) = ceil(ctx / r) rows at the layer's row width, ON
	// TOP of its window rows. A non-positive entry (and any layer past the end)
	// compresses nothing — there is no scalar to fall back to.
	CompressRatio []int
	// IndexHeadDim is the per-layer indexer key width. A positive entry makes
	// layer l cache one key of that width per compressed row when it compresses
	// (ratio > 0), else one per retained token (LayerTokens). A non-positive
	// entry (and any layer past the end) falls back to the Shape's scalar
	// IndexHeadDim for l < IndexLayers, else no indexer.
	IndexHeadDim []int
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
			len(p.HeadDim) == 0 && len(p.VHeadDim) == 0 &&
			len(p.CompressRatio) == 0 && len(p.IndexHeadDim) == 0)
}

// compresses reports whether layer l declares a positive compression ratio.
func (s Shape) compresses(l int) bool {
	return s.PerLayer != nil && perLayerOr(s.PerLayer.CompressRatio, l, 0) > 0
}

// CompressedRows is the number of compressed rows layer l retains once a
// stream has reached ctx tokens: ceil(ctx / r) for a layer whose CompressRatio
// r is positive, and 0 for a layer that compresses nothing or a non-positive
// ctx. Ceil, not floor: a trailing partial group is charged a whole row, so the
// figure never under-counts (ctx = r·k + 1 is where the two differ).
func (s Shape) CompressedRows(l, ctx int) int {
	if !s.compresses(l) || ctx <= 0 {
		return 0
	}
	r := s.PerLayer.CompressRatio[l]
	rows := ctx / r // split form: ctx + r - 1 could overflow near MaxInt
	if ctx%r != 0 {
		rows++
	}
	return rows
}

// LayerTokens is the number of KV rows layer l holds once a stream has reached
// ctx tokens: min(Window[l], ctx) for a window-capped layer, ctx for a layer
// that attends over the whole context, plus CompressedRows(l, ctx) on a layer
// that compresses. This is the single place the window bound is applied.
func (s Shape) LayerTokens(l, ctx int) int {
	if s.PerLayer == nil {
		return ctx
	}
	rows := ctx
	if w := perLayerOr(s.PerLayer.Window, l, 0); w > 0 && w < ctx {
		rows = w
	}
	return rows + s.CompressedRows(l, ctx)
}

// cachedTokens is the total per-layer token-slots the first n layers hold at
// ctx: Σ_{l<n} LayerTokens(l, ctx). For a Shape with neither a window nor a
// compression ratio it is exactly n × ctx, which is what makes every uniform
// figure below reduce to the pre-refinement one.
func (s Shape) cachedTokens(n, ctx int) int {
	if s.PerLayer == nil || (len(s.PerLayer.Window) == 0 && len(s.PerLayer.CompressRatio) == 0) {
		return n * ctx
	}
	total := 0
	for l := 0; l < n; l++ {
		total += s.LayerTokens(l, ctx)
	}
	return total
}

// MLAElemsPerStream is the MLA latent + decoupled rope key a whole stream of ctx
// tokens holds: Σ over layers of LayerTokens(l, ctx) × (KVLoraRank +
// QKRopeHeadDim) — min(window, ctx), plus ceil(ctx / r) on a compressing layer.
// Without windows or ratios it is ctx × MLAElemsPerToken().
func (s Shape) MLAElemsPerStream(ctx int) int {
	return s.cachedTokens(s.Layers, ctx) * (s.KVLoraRank + s.QKRopeHeadDim)
}

// IndexElemsPerStream is the DSA indexer key a whole stream holds, summed over
// the index layers under the same window bound. IndexLayers is an upper bound at
// Layers (doc §3.3), so the first IndexLayers windows are the ones that apply.
//
// A profile that declares CompressRatio or IndexHeadDim is summed layer by layer
// over [0, Layers) instead: each layer's key width (its IndexHeadDim entry, else
// the scalar for l < IndexLayers, else none) times its compressed rows when it
// compresses — the V4 indexer keys compressed rows, not tokens — else its
// retained tokens. Any other Shape takes the original expression untouched.
func (s Shape) IndexElemsPerStream(ctx int) int {
	p := s.PerLayer
	if p == nil || (len(p.CompressRatio) == 0 && len(p.IndexHeadDim) == 0) {
		return s.cachedTokens(s.IndexLayers, ctx) * s.IndexHeadDim
	}
	total := 0
	for l := 0; l < s.Layers; l++ {
		width := perLayerOr(p.IndexHeadDim, l, 0)
		if width <= 0 {
			width = 0
			if l < s.IndexLayers {
				width = s.IndexHeadDim
			}
		}
		rows := s.LayerTokens(l, ctx)
		if s.compresses(l) {
			rows = s.CompressedRows(l, ctx)
		}
		total += rows * width
	}
	return total
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
// window or compresses, and it equals ctx × KVElemsPerToken() whenever the
// Shape declares no profile.
func (s Shape) KVElemsPerStream(ctx int) int {
	if s.Kind == MHA {
		return s.MHAElemsPerStream(ctx)
	}
	return s.MLAElemsPerStream(ctx) + s.IndexElemsPerStream(ctx)
}

// KVBytesPerStream is the full KV footprint of one stream of ctx tokens at the
// given quant. An unrefined Shape takes the original ctx × KV-bytes/token
// expression untouched (bit-identical at any quant); a refined one sums its
// layers under the window bound plus any compressed rows.
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
