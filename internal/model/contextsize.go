package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// ContextSizeConfig projects this model Config into the compute-level geometry the
// context auto-sizer consumes (compute.AutoSizeContextPlan, #1049). It is the one
// projection both the serve boot path (cmd/fak/serve.go) and the in-kernel per-request
// planner (internal/agent/inkernel_planner.go) share, so they size a context plan from
// a single source of geometry and cannot drift apart on the same model.
func (c Config) ContextSizeConfig() compute.ContextSizeConfig {
	return compute.ContextSizeConfig{
		KV: compute.KVConfig{
			NumLayers:  c.NumLayers,
			NumKVHeads: c.NumKVHeads,
			HeadDim:    c.HeadDim,
			RopeTheta:  c.RopeTheta,
		},
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
