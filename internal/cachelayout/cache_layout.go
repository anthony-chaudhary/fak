package cachelayout

// Per-layer hybrid cache-type registry (#942). Hybrid token-mixer checkpoints
// (Qwen3.5/3.6 Gated-DeltaNet and later families) interleave full-attention layers
// (token-indexed, block-sliceable K/V) with linear-attention layers (an O(1)-per-
// sequence recurrent state). Consumers previously hard-coded this layout
// (qwen35_paged_swap.go, checkpoint.go); this registry derives it once at load from
// layer_types, dependency-free, avoiding an import cycle with internal/model.

// LayerKind names the token-mixer storage class of one decoder layer. It is the
// storage-granularity axis, orthogonal to which decode kernel the layer runs.
type LayerKind uint8

const (
	// LayerKindFullAttention is a full-causal-attention layer: token-indexed K/Kraw/V,
	// one row per position; block-sliceable, paged/evicted by token index.
	LayerKindFullAttention LayerKind = iota
	// LayerKindLinearAttention is a recurrent state-space layer (Gated-DeltaNet / Mamba):
	// an accumulated recurrent state plus a short causal-conv window, O(1) per sequence;
	// boundary-only, carried whole, never sliced by token index.
	LayerKindLinearAttention
	// LayerKindSlidingAttention is a sliding-window-attention layer: token-indexed like
	// full attention, but only the trailing window is attendable; sliceable with a
	// resident window cap.
	LayerKindSlidingAttention
	// LayerKindUnknown is an unclassified/absent layer-type entry, treated conservatively
	// as boundary-only so an unrecognised hybrid cannot be sliced incorrectly.
	LayerKindUnknown
)

// String renders the kind for witnesses and operator readouts.
func (k LayerKind) String() string {
	switch k {
	case LayerKindFullAttention:
		return "full-attention"
	case LayerKindLinearAttention:
		return "linear-attention"
	case LayerKindSlidingAttention:
		return "sliding-attention"
	default:
		return "unknown"
	}
}

// SequenceAxis describes the token-indexed axis of a layer state, if it has one
// (a linear-attention layer has none).
type SequenceAxis struct {
	// TokenIndexed is true when this layer holds one state row per absolute position.
	TokenIndexed bool
	// Planes is the per-position plane count (K, Kraw, V => 3) for a token-indexed layer, 0 otherwise.
	Planes int
	// Stride is the per-position width in float32 for a token-indexed layer, else 0.
	Stride int
}

// StateAxis describes the recurrent/boundary-only state of a layer, if it has one
// (a full/sliding-attention layer has none).
type StateAxis struct {
	// Present is true when this layer carries an O(1) accumulated recurrent state.
	Present bool
	// ValueHeads, KeyHeadDim, ValueHeadDim, ConvKernel are the recurrent geometry; zero when absent.
	ValueHeads   int
	KeyHeadDim   int
	ValueHeadDim int
	ConvKernel   int
}

// LayerCacheDescriptor is the per-layer storage descriptor the block-store/swap paths
// query instead of re-deriving a layout from constants.
type LayerCacheDescriptor struct {
	LayerIndex int       `json:"layer_index"`
	LayerType  string    `json:"layer_type"`
	Kind       LayerKind `json:"kind"`
	// Sliceable is true iff a token span of this layer state can be paged/evicted/swapped by token index.
	Sliceable bool `json:"sliceable"`
	// Sequence is the token-indexed axis metadata (empty for the linear class).
	Sequence SequenceAxis `json:"sequence"`
	// State is the recurrent/boundary-only axis metadata (empty for the attention classes).
	State StateAxis `json:"state"`
}

// ModelCacheLayout is the per-layer cache-type registry derived once at load from a
// config layer_types, plus the shared recurrent geometry; its Kind/Sliceable/axis
// fields are the single source of truth the swap and checkpoint seams route through.
type ModelCacheLayout struct {
	// Layers is indexed by decoder layer; Layers[l].LayerIndex == l.
	Layers []LayerCacheDescriptor
	// Recurrent is the shared linear-attention geometry; zero when the model has no linear layers.
	Recurrent StateAxis
	// Stride is the token-indexed per-position width (NumKVHeads*HeadDim); 0 if unknown.
	Stride int
}

// CacheLayoutGeometry is the model geometry the registry needs; a plain struct (not a
// model.Config) so the registry stays dependency-free; callers adapt their config into it.
type CacheLayoutGeometry struct {
	NumKVHeads          int
	HeadDim             int
	LinearNumValueHeads int
	LinearKeyHeadDim    int
	LinearValueHeadDim  int
	LinearConvKernelDim int
}

// NewModelCacheLayout derives the per-layer registry from a config layer_types and
// geometry. Unknown/empty entries classify as LayerKindUnknown (boundary-only, not
// sliceable), so an unrecognised hybrid degrades per layer; same inputs yield an identical layout.
func NewModelCacheLayout(layerTypes []string, geom CacheLayoutGeometry) ModelCacheLayout {
	recurrent := StateAxis{
		Present:      false,
		ValueHeads:   geom.LinearNumValueHeads,
		KeyHeadDim:   geom.LinearKeyHeadDim,
		ValueHeadDim: geom.LinearValueHeadDim,
		ConvKernel:   geom.LinearConvKernelDim,
	}
	stride := geom.NumKVHeads * geom.HeadDim
	if stride < 0 {
		stride = 0
	}

	layout := ModelCacheLayout{
		Layers:    make([]LayerCacheDescriptor, len(layerTypes)),
		Recurrent: recurrent,
		Stride:    stride,
	}
	for l, t := range layerTypes {
		d := LayerCacheDescriptor{LayerIndex: l, LayerType: t}
		switch t {
		case "full_attention":
			d.Kind = LayerKindFullAttention
			d.Sliceable = true
			d.Sequence = SequenceAxis{TokenIndexed: true, Planes: 3, Stride: stride}
		case "sliding_attention":
			d.Kind = LayerKindSlidingAttention
			d.Sliceable = true
			d.Sequence = SequenceAxis{TokenIndexed: true, Planes: 3, Stride: stride}
		case "linear_attention":
			d.Kind = LayerKindLinearAttention
			d.Sliceable = false
			d.State = recurrent
			d.State.Present = true
		default:
			d.Kind = LayerKindUnknown
			d.Sliceable = false
		}
		layout.Layers[l] = d
	}
	return layout
}

// Descriptor returns the storage descriptor for decoder layer l, or a zero-value
// boundary-only descriptor when l is out of range; an unknown layer is never
// reported sliceable.
func (m ModelCacheLayout) Descriptor(l int) LayerCacheDescriptor {
	if l < 0 || l >= len(m.Layers) {
		return LayerCacheDescriptor{LayerIndex: l, Kind: LayerKindUnknown, Sliceable: false}
	}
	return m.Layers[l]
}

// IsSliceable reports whether layer l state is token-indexed and block-sliceable.
func (m ModelCacheLayout) IsSliceable(l int) bool {
	return m.Descriptor(l).Sliceable
}

// KindOf returns layer l storage class (LayerKindUnknown when out of range).
func (m ModelCacheLayout) KindOf(l int) LayerKind {
	return m.Descriptor(l).Kind
}

// FullAttentionLayers returns the indices of the token-indexed full-attention layers, in
// order. Sliding-window layers are NOT included (they are sliceable but window-capped);
// callers that need every sliceable layer use SliceableLayers.
func (m ModelCacheLayout) FullAttentionLayers() []int {
	var out []int
	for _, d := range m.Layers {
		if d.Kind == LayerKindFullAttention {
			out = append(out, d.LayerIndex)
		}
	}
	return out
}

// SliceableLayers returns the indices of every token-indexed block-sliceable layer, in
// order.
func (m ModelCacheLayout) SliceableLayers() []int {
	var out []int
	for _, d := range m.Layers {
		if d.Sliceable {
			out = append(out, d.LayerIndex)
		}
	}
	return out
}

// RecurrentLayers returns the indices of the boundary-only linear-attention layers, in
// order.
func (m ModelCacheLayout) RecurrentLayers() []int {
	var out []int
	for _, d := range m.Layers {
		if d.Kind == LayerKindLinearAttention {
			out = append(out, d.LayerIndex)
		}
	}
	return out
}
