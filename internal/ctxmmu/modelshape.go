package ctxmmu

// modelshape.go — Per-model architecture-shape profile for sequence descriptors.
//
// The context-MMU sizes checkpoint sequence descriptors (attention geometry,
// hybrid linear-attention state, RoPE offsets, MTP draft state) from facts
// about the served model's architecture. Those facts used to be hardcoded
// literals inside SaveInPlaceCheckpoint; they now route through ModelShape so
// a caller serving a specific model injects that model's shape — resolved from
// the model catalog / provider abstraction at the engine layer — while any
// zero-valued field falls back to DefaultModelShape, which preserves the
// historical defaults byte-for-byte.

// ModelShape bundles the per-model architecture-shape facts ctxmmu needs when
// building sequence descriptors. Every field is optional: a zero value falls
// back field-by-field to DefaultModelShape (see normalized).
type ModelShape struct {
	// AttentionHeadCount is the number of attention heads the served model uses.
	AttentionHeadCount int

	// SlidingWindowSize is the sliding-window size recorded on attention masks.
	SlidingWindowSize int

	// RoPEBaseFrequency is the RoPE base frequency of the served model.
	RoPEBaseFrequency float64

	// MTPDraftDepth is the speculative multi-token-prediction draft depth.
	MTPDraftDepth int

	// DeltaNet* fields describe the hybrid linear-attention (Gated-DeltaNet) stack.
	DeltaNetLayers     int
	DeltaNetValueHeads int
	DeltaNetKeyHeadDim int
	DeltaNetValHeadDim int
	DeltaNetConvKernel int
}

// DefaultModelShape is the single source for the historical default shape used
// when a caller does not supply per-model values. The DeltaNet fields alias
// the exported DefaultDeltaNet* constants so both surfaces stay one-sourced.
var DefaultModelShape = ModelShape{
	AttentionHeadCount: 32,
	SlidingWindowSize:  4096,
	RoPEBaseFrequency:  1000000.0,
	MTPDraftDepth:      4,
	DeltaNetLayers:     DefaultDeltaNetLayers,
	DeltaNetValueHeads: DefaultDeltaNetValueHeads,
	DeltaNetKeyHeadDim: DefaultDeltaNetKeyHeadDim,
	DeltaNetValHeadDim: DefaultDeltaNetValHeadDim,
	DeltaNetConvKernel: DefaultDeltaNetConvKernel,
}

// normalized returns s with every unset (<= 0) field filled from
// DefaultModelShape, matching the zero-fallback convention used by
// NewDeltaNetStateDescriptor and GatherOptions normalization.
func (s ModelShape) normalized() ModelShape {
	d := DefaultModelShape
	if s.AttentionHeadCount <= 0 {
		s.AttentionHeadCount = d.AttentionHeadCount
	}
	if s.SlidingWindowSize <= 0 {
		s.SlidingWindowSize = d.SlidingWindowSize
	}
	if s.RoPEBaseFrequency <= 0 {
		s.RoPEBaseFrequency = d.RoPEBaseFrequency
	}
	if s.MTPDraftDepth <= 0 {
		s.MTPDraftDepth = d.MTPDraftDepth
	}
	if s.DeltaNetLayers <= 0 {
		s.DeltaNetLayers = d.DeltaNetLayers
	}
	if s.DeltaNetValueHeads <= 0 {
		s.DeltaNetValueHeads = d.DeltaNetValueHeads
	}
	if s.DeltaNetKeyHeadDim <= 0 {
		s.DeltaNetKeyHeadDim = d.DeltaNetKeyHeadDim
	}
	if s.DeltaNetValHeadDim <= 0 {
		s.DeltaNetValHeadDim = d.DeltaNetValHeadDim
	}
	if s.DeltaNetConvKernel <= 0 {
		s.DeltaNetConvKernel = d.DeltaNetConvKernel
	}
	return s
}
