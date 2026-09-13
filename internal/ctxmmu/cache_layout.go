package ctxmmu

// cache_layout.go - the per-layer hybrid cache-type registry (#942).
//
// fak serves hybrid token-mixer families whose decoder layers do NOT all hold the
// same kind of attention state: a Qwen3.5/3.6 Gated-DeltaNet checkpoint (and the
// next non-qwen35 hybrid family) interleaves gated FULL-attention layers (a
// token-indexed, block-sliceable K/V span) with linear-attention layers (an
// accumulated recurrent state plus a short causal-conv window - O(1) per sequence,
// not token-indexed at all). Before this file the classification was bespoke inside
// each consumer: qwen35_paged_swap.go:21-23 hard-coded the layout and
// checkpoint.go:38-41 bound DeltaNet geometry to compile-time constants, so a
// non-conforming hybrid could only fail as a typed UnsupportedArchError rather than
// degrade storage granularity per layer.
//
// ModelCacheLayout is the ONE query that answers, for a loaded config layer_types,
// which layers produce token-indexed block-sliceable state vs boundary-only state,
// and what storage axes that state rides. It is a pure derivation from the layer-type
// strings (no dependency on internal/model), so it can be built once at load and
// consulted from the swap/checkpoint/block-store seams without an import cycle.

// LayerKind names the token-mixer storage class of one decoder layer. It is the
// storage-granularity axis, orthogonal to which decode kernel the layer runs.
type LayerKind uint8

const (
	// LayerKindFullAttention is a full-causal-attention layer: token-indexed K/Kraw/V,
	// one row per position, resident footprint growing linearly with context. It is the
	// block-SLICEABLE class - a span of its state can be paged, evicted and swapped by
	// token index.
	LayerKindFullAttention LayerKind = iota
	// LayerKindLinearAttention is a recurrent state-space layer (Gated-DeltaNet / Mamba):
	// an accumulated recurrent state plus a short causal-conv window, O(1) per sequence.
	// It is boundary-only - carried whole, never sliced by token index.
	LayerKindLinearAttention
	// LayerKindSlidingAttention is a sliding-window-attention layer: token-indexed K/Kraw/V
	// like full attention, but only the trailing window is attendable. It is sliceable
	// with a resident window cap.
	LayerKindSlidingAttention
	// LayerKindUnknown is an unclassified/absent layer-type entry. It is treated
	// conservatively as boundary-only so an unrecognised hybrid cannot be sliced
	// incorrectly.
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

// SequenceAxis describes the token-indexed axis of a layer state, if it has one. A
// linear-attention layer has none (Sliceable=false, Stride=0).
type SequenceAxis struct {
	// TokenIndexed is true when this layer holds one state row per absolute position.
	TokenIndexed bool
	// Planes is the number of per-position planes (K, Kraw, V => 3) for a token-indexed
	// layer, 0 otherwise.
	Planes int
	// Stride is the per-position width in float32 for a token-indexed layer, else 0.
	Stride int
}

// StateAxis describes the recurrent/boundary-only state of a layer, if it has one. A
// full/sliding-attention layer has none (Present=false).
type StateAxis struct {
	// Present is true when this layer carries an O(1) accumulated recurrent state
	// rather than (or in addition to) a token-indexed span.
	Present bool
	// ValueHeads, KeyHeadDim, ValueHeadDim, ConvKernel are the recurrent geometry; zero
	// when absent.
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
	// Sliceable is true iff a token span of this layer state can be paged/evicted/swapped
	// by token index (the full and sliding-window classes).
	Sliceable bool `json:"sliceable"`
	// Sequence is the token-indexed axis metadata (empty for the linear class).
	Sequence SequenceAxis `json:"sequence"`
	// State is the recurrent/boundary-only axis metadata (empty for the attention classes).
	State StateAxis `json:"state"`
}

// ModelCacheLayout is the per-layer cache-type registry derived once at load from a
// config layer_types, plus the recurrent geometry the linear layers share. Its
// LayerKind/Sliceable/axis fields are the single source of truth the swap and
// checkpoint seams route their classification through.
type ModelCacheLayout struct {
	// Layers is indexed by decoder layer; Layers[l].LayerIndex == l.
	Layers []LayerCacheDescriptor
	// Recurrent is the shared linear-attention geometry (StateAxis present on every
	// LayerKindLinearAttention layer). Zero when the model has no linear layers.
	Recurrent StateAxis
	// Stride is the token-indexed per-position width (NumKVHeads*HeadDim); 0 if unknown.
	Stride int
}

// CacheLayoutGeometry is the model geometry the registry needs to fill in the
// per-layer storage axes. It is a plain struct (not a model.Config) so the registry
// stays dependency-free; callers adapt their config into it once.
type CacheLayoutGeometry struct {
	NumKVHeads          int
	HeadDim             int
	LinearNumValueHeads int
	LinearKeyHeadDim    int
	LinearValueHeadDim  int
	LinearConvKernelDim int
}

// NewModelCacheLayout derives the per-layer registry from a config layer_types and
// geometry. Unknown/empty layer-type entries classify as LayerKindUnknown (boundary-only,
// not sliceable) so an unrecognised hybrid degrades per layer rather than being sliced
// incorrectly. Deriving twice from the same inputs yields an identical layout.
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
// boundary-only descriptor when l is out of range (fail-closed: an unknown layer is
// never reported sliceable).
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
// order - the layers whose K/Kraw/V rows occupy token-indexed pages. Sliding-window layers
// are NOT included (they are sliceable but window-capped); callers that need every
// sliceable layer use SliceableLayers.
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
