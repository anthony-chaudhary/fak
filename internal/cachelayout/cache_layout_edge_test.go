package cachelayout

import (
	"encoding/json"
	"testing"
)

// TestSlidingAttentionClassification pins the sliding-window class: token-indexed
// and sliceable like full attention, but excluded from FullAttentionLayers so a
// window-capped layer never rides the full-attention paged path.
func TestSlidingAttentionClassification(t *testing.T) {
	layout := NewModelCacheLayout([]string{"sliding_attention", "full_attention"}, CacheLayoutGeometry{NumKVHeads: 2, HeadDim: 64})

	d := layout.Descriptor(0)
	if d.Kind != LayerKindSlidingAttention || !d.Sliceable {
		t.Fatalf("layer 0 kind=%v sliceable=%t, want sliding-attention sliceable", d.Kind, d.Sliceable)
	}
	if !d.Sequence.TokenIndexed || d.Sequence.Planes != 3 || d.Sequence.Stride != 128 {
		t.Fatalf("layer 0 sequence axis=%+v", d.Sequence)
	}
	if d.State.Present {
		t.Fatalf("layer 0 must carry no recurrent state")
	}

	if got := layout.SliceableLayers(); !equalInts(got, []int{0, 1}) {
		t.Fatalf("SliceableLayers=%v want [0 1]", got)
	}
	if got := layout.FullAttentionLayers(); !equalInts(got, []int{1}) {
		t.Fatalf("FullAttentionLayers=%v want [1] (sliding layers excluded)", got)
	}
	if got := layout.RecurrentLayers(); len(got) != 0 {
		t.Fatalf("RecurrentLayers=%v want none", got)
	}
}

// TestEmptyLayerTypesYieldsEmptyRegistry pins the degenerate input: an empty
// layer_types still yields a coherent registry whose boundary-only queries report
// unknown/not-sliceable even at index 0.
func TestEmptyLayerTypesYieldsEmptyRegistry(t *testing.T) {
	layout := NewModelCacheLayout([]string{}, CacheLayoutGeometry{NumKVHeads: 4, HeadDim: 128})

	if len(layout.Layers) != 0 {
		t.Fatalf("layers=%d want 0", len(layout.Layers))
	}
	if layout.Stride != 512 {
		t.Fatalf("stride=%d want 512", layout.Stride)
	}
	if got := layout.FullAttentionLayers(); len(got) != 0 {
		t.Fatalf("FullAttentionLayers=%v want empty", got)
	}
	if got := layout.SliceableLayers(); len(got) != 0 {
		t.Fatalf("SliceableLayers=%v want empty", got)
	}
	if got := layout.RecurrentLayers(); len(got) != 0 {
		t.Fatalf("RecurrentLayers=%v want empty", got)
	}
	d := layout.Descriptor(0)
	if d.Kind != LayerKindUnknown || d.Sliceable {
		t.Fatalf("empty-registry descriptor kind=%v sliceable=%t, want unknown boundary-only", d.Kind, d.Sliceable)
	}
}

// TestNegativeGeometryClampsStrideToZero pins the fail-safe clamp: a negative
// head count or dim must not leak negative widths into descriptors.
func TestNegativeGeometryClampsStrideToZero(t *testing.T) {
	layout := NewModelCacheLayout([]string{"full_attention"}, CacheLayoutGeometry{NumKVHeads: -4, HeadDim: 128})

	if layout.Stride != 0 {
		t.Fatalf("stride=%d want 0", layout.Stride)
	}
	if got := layout.Descriptor(0).Sequence.Stride; got != 0 {
		t.Fatalf("descriptor stride=%d want 0", got)
	}
}

// TestDescriptorJSONWireContract pins the exact JSON field names consumers parse;
// LayerCacheDescriptor carries json tags while its nested axis structs do not.
func TestDescriptorJSONWireContract(t *testing.T) {
	layout := NewModelCacheLayout([]string{"full_attention"}, CacheLayoutGeometry{NumKVHeads: 4, HeadDim: 128})
	got, err := json.Marshal(layout.Descriptor(0))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"layer_index":0,"layer_type":"full_attention","kind":0,"sliceable":true,"sequence":{"TokenIndexed":true,"Planes":3,"Stride":512},"state":{"Present":false,"ValueHeads":0,"KeyHeadDim":0,"ValueHeadDim":0,"ConvKernel":0}}`
	if string(got) != want {
		t.Fatalf("json=%s\nwant %s", got, want)
	}
}
