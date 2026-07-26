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
		return s
	}
	// MHA / GQA. A square head (no separate value-head width in the header)
	// defaults VHeadDim to HeadDim, matching the loader's attention geometry.
	vHeadDim := c.VHeadDim
	if vHeadDim == 0 {
		vHeadDim = c.HeadDim
	}
	return kvbudget.Shape{
		Kind:       kvbudget.MHA,
		Layers:     c.NumLayers,
		NumKVHeads: c.NumKVHeads,
		HeadDim:    c.HeadDim,
		VHeadDim:   vHeadDim,
	}
}
