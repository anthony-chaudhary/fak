package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// ContextSizeConfig projects this model Config into the compute-level geometry the
// context auto-sizer consumes (compute.AutoSizeContextPlan, #1049). It is the one
// projection both the serve boot path (cmd/fak/serve.go) and the in-kernel per-request
// planner (internal/agent/inkernel_planner.go) share, so they size a context plan from
// a single source of geometry and cannot drift apart on the same model.
//
// # Per-layer window (#5520)
//
// It is also the FIRST consumer of the corrected KV geometry KVCacheShape computes: an
// interleaved local/global checkpoint (gemma-4's five sliding layers per global one)
// hands compute.KVConfig.WindowPerLayer the same per-layer bound the Shape's
// kvbudget.LayerProfile carries, so EstimateKVStoreBytes charges a windowed layer
// min(window, tokens) positions instead of the whole context. Before this the Shape was
// computed and discarded — every number a user or the auto-sizer saw came from the flat
// NumLayers x tokens product, which over-charges the pinned gemma-4-26B cadence (30
// layers, 25 windowed at 1024, 5 global) by 32/7 ~= 4.57x at a 16384-token context.
//
// GATED, and deliberately so. The discount is only TRUE if the cache actually drops
// aged-out positions on a windowed layer, and today it does not: Session.TrimToWindow
// (swa.go:48) has no production caller — swa.go:14 says so and a repo-wide grep agrees —
// and Config.MaxWindow (swa.go:25) returns 0 unless EVERY layer declares a window, which
// an interleaved model never does. Charging the discount unconditionally would turn a
// conservative over-reservation into an UNDER-reservation and trade a refusal for an OOM.
// So it rides FAK_HYBRID_KV, the flag kvgroups.go:220 already gates this exact
// grouped-vs-uniform saving behind ("dogfood-gated until promotion evidence lands"). Off
// (the default) the projection is byte-for-byte what it was; on, the planner reads the
// corrected shape. Flipping it on by default is the follow-on, and it is blocked on the
// cache realizing the window, not on this projection.
func (c Config) ContextSizeConfig() compute.ContextSizeConfig {
	kv := compute.KVConfig{
		NumLayers:  c.NumLayers,
		NumKVHeads: c.NumKVHeads,
		HeadDim:    c.HeadDim,
		RopeTheta:  c.RopeTheta,
	}
	if HybridKVGroupsEnabled() {
		// KVCacheShape already normalizes the loader's -1 full-attention sentinel to
		// kvbudget's non-positive "no window" spelling, which is the spelling
		// compute.KVConfig.WindowPerLayer reads — so the slice crosses verbatim. A
		// uniformly-global model leaves PerLayer (or PerLayer.Window) nil and the
		// projection stays exactly the uniform one.
		if p := c.KVCacheShape().PerLayer; p != nil {
			kv.WindowPerLayer = p.Window
		}
	}
	return compute.ContextSizeConfig{
		KV: kv,
		Scratch: compute.TransformerScratchConfig{
			HiddenSize:       c.HiddenSize,
			IntermediateSize: c.IntermediateSize,
			VocabSize:        c.VocabSize,
			NumLayers:        c.NumLayers,
			NumHeads:         c.NumHeads,
			NumKVHeads:       c.NumKVHeads,
			HeadDim:          c.HeadDim,
			IncludeLogits:    true,
		},
		MaxContext: c.MaxPositionEmbeddings,
	}
}
