package model

import "github.com/anthony-chaudhary/fak/internal/kvbudget"

// KVCacheShape reads this model's attention architecture from its loaded header
// and returns the kvbudget.Shape that sizes its KV cache — the general branch
// over attention arch (MLA / MLA+NSA-indexer / MHA) that ktransformers'
// kv_cache_calculator makes from a ModelConfig (kv_cache_calculator.py:34-121
// @0c2912a). It replaces kvbudget's GLM-5.2 GLM52DSA estimate with the values
// this config actually carries, so kvbudget can size an arbitrary served model
// instead of the GLM-5.2 hardcode (#5242).
//
// Branch predicate: a header that declares an MLA compressed latent
// (KVLoraRank > 0 — DeepSeek-V2/V3, GLM-5.2) sizes as MLA; every other header
// (Llama, Qwen, and every standard multi-head / grouped-query model) sizes as
// MHA. The DeepSeek-NSA / GLM-DSA lightning indexer contributes its IndexHeadDim
// key per layer only when the header declares one (IndexNHeads > 0); because the
// indexer is shared across a group of layers, NumLayers is the (≤) upper bound
// kvbudget documents, not the exact index-layer count.
//
// This is header-driven METADATA sizing (gen/next): it reads the same exported
// config fields the loader already parses and computes no cache — nothing in the
// decode path consumes the Shape yet. Its invalidating assumption is that a
// resident GGUF caches the compressed MLA latent (not a decompressed per-head
// K/V) at the header's declared widths; if a served checkpoint materializes a
// different cache layout, the numbers move (the branch stands).
//
// # Per-layer refinement
//
// When the header carries heterogeneous per-layer attention — the interleaved
// local/global families the loader already parses into Config.{Window,
// HeadDimPerLayer, NumKVHeadsPerLayer} — the Shape also carries the optional
// kvbudget.LayerProfile that bounds each windowed layer at min(window, ctx)
// instead of ctx (#5498). Without it a window-capped checkpoint sized as if every
// layer attended over the whole context, and MaxStreams under-reported how many
// streams fit, which for an admission gate means refusing streams that would have
// fit. A header with no per-layer data (every model that is uniformly global)
// leaves PerLayer nil and keeps its previous numbers bit-for-bit.
func (c Config) KVCacheShape() kvbudget.Shape {
	if c.KVLoraRank > 0 {
		s := kvbudget.Shape{
			Kind:          kvbudget.MLA,
			Layers:        c.NumLayers,
			KVLoraRank:    c.KVLoraRank,
			QKRopeHeadDim: c.QKRopeHeadDim,
		}
		if c.IndexNHeads > 0 && c.IndexHeadDim > 0 {
			s.IndexLayers = c.NumLayers
			s.IndexHeadDim = c.IndexHeadDim
		}
		// An MLA layer caches a latent whose width is head-independent, so the
		// only per-layer axis that can refine it is the window.
		if w := c.kvWindowPerLayer(); w != nil {
			s.PerLayer = &kvbudget.LayerProfile{Window: w}
		}
		return s
	}
	// MHA / GQA. A square head (no separate value-head width in the header)
	// defaults VHeadDim to HeadDim, matching the loader's attention geometry.
	vHeadDim := c.VHeadDim
	if vHeadDim == 0 {
		vHeadDim = c.HeadDim
	}
	s := kvbudget.Shape{
		Kind:       kvbudget.MHA,
		Layers:     c.NumLayers,
		NumKVHeads: c.NumKVHeads,
		HeadDim:    c.HeadDim,
		VHeadDim:   vHeadDim,
	}
	window := c.kvWindowPerLayer()
	kvHeads := kvPerLayerOverride(c.NumKVHeadsPerLayer, c.NumLayers, c.NumKVHeads)
	headDim := kvPerLayerOverride(c.HeadDimPerLayer, c.NumLayers, c.HeadDim)
	// The interleaved-attention families that carry a per-layer head_dim carry no
	// per-layer VALUE width: their layers are square (gemma4's global layers have
	// no v_proj at all — V is the raw k_proj output, gemma4.go:150-154), so V is
	// exactly as wide as K layer by layer. Mirroring headDim into VHeadDim is the
	// per-layer form of the scalar squaring above; a header that DOES declare a
	// rectangular head (VHeadDim != 0) has no per-layer V data, so those layers
	// fall back to the scalar rather than silently inheriting K's width.
	var vPerLayer []int
	if c.VHeadDim == 0 {
		vPerLayer = headDim
	}
	if window != nil || kvHeads != nil || headDim != nil {
		s.PerLayer = &kvbudget.LayerProfile{
			Window:     window,
			NumKVHeads: kvHeads,
			HeadDim:    headDim,
			VHeadDim:   vPerLayer,
		}
	}
	return s
}

// kvWindowPerLayer translates Config.Window into the per-layer token bound
// kvbudget.LayerProfile.Window expects, or nil when no layer caps its attention.
//
// The two slices agree on shape but NOT on how they spell "no window": the loader
// writes the sentinel -1 for a full-attention layer (applyGemma4Config in
// internal/ggufload, and Config.windowForLayer's default), while kvbudget reads
// any NON-POSITIVE entry as full attention. Both spellings mean the same thing to
// kvbudget's perLayerOr today, but mapping the sentinel here explicitly is what
// keeps a -1 from ever reaching LayerTokens' min(window, ctx) as a real bound if
// either convention drifts; the copy also stops a Shape from aliasing — and a
// Shape consumer from mutating — the Config's own slice.
//
// Returning nil when NO layer declares a positive window is load-bearing rather
// than an optimization: an all-(-1) Window slice (a model whose every layer is
// global) would otherwise make Shape.uniform() false and route the byte figures
// through the summed per-layer expression instead of the untouched
// ctx × bytes-per-token one — algebraically equal, but no longer bit-identical at
// every quant, which is the property perlayer.go's zero value promises.
func (c Config) kvWindowPerLayer() []int {
	n := c.NumLayers
	if n <= 0 || len(c.Window) == 0 {
		return nil
	}
	if n > len(c.Window) {
		n = len(c.Window)
	}
	out, windowed := make([]int, n), false
	for l := 0; l < n; l++ {
		if w := c.Window[l]; w > 0 {
			out[l], windowed = w, true
		}
	}
	if !windowed {
		return nil
	}
	return out
}

// kvPerLayerOverride returns a copy of a per-layer geometry slice (head width,
// kv-head count) when some layer really overrides the Shape's uniform scalar, and
// nil when the slice is absent or agrees with the scalar everywhere. Non-positive
// entries are carried through untouched — kvbudget reads those as "this layer
// declares nothing", the same convention Config.headDimForLayer uses.
//
// The uniform-agrees case must return nil for the same bit-identity reason as
// kvWindowPerLayer: a profile that says nothing new must not exist at all, or an
// unrefined Shape stops taking its untouched ctx × bytes-per-token expression.
func kvPerLayerOverride(xs []int, layers, uniform int) []int {
	if layers <= 0 || len(xs) == 0 {
		return nil
	}
	n := layers
	if n > len(xs) {
		n = len(xs)
	}
	out, differs := make([]int, n), false
	for l := 0; l < n; l++ {
		out[l] = xs[l]
		if xs[l] > 0 && xs[l] != uniform {
			differs = true
		}
	}
	if !differs {
		return nil
	}
	return out
}
