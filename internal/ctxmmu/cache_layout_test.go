package ctxmmu

import "testing"

// TestHybridCacheTypeRegistrySpine is the #942 witness: the per-layer cache-type
// registry classifies a mixed GDN/full-attention layer_types into the right
// LayerKind, sliceability, and storage-axis metadata, and answers the full /
// sliceable / recurrent layer queries the swap and checkpoint seams route through.
func TestHybridCacheTypeRegistrySpine(t *testing.T) {
	// The qwen35-family cadence: three linear (recurrent) layers then one full.
	types := []string{
		"linear_attention", "linear_attention", "linear_attention", "full_attention",
		"linear_attention", "linear_attention", "linear_attention", "full_attention",
	}
	geom := CacheLayoutGeometry{
		NumKVHeads:          4,
		HeadDim:             128,
		LinearNumValueHeads: 36,
		LinearKeyHeadDim:    128,
		LinearValueHeadDim:  128,
		LinearConvKernelDim: 4,
	}
	layout := NewModelCacheLayout(types, geom)

	if len(layout.Layers) != len(types) {
		t.Fatalf("layers=%d want %d", len(layout.Layers), len(types))
	}
	if layout.Stride != geom.NumKVHeads*geom.HeadDim {
		t.Fatalf("stride=%d want %d", layout.Stride, geom.NumKVHeads*geom.HeadDim)
	}

	wantKinds := []LayerKind{
		LayerKindLinearAttention, LayerKindLinearAttention, LayerKindLinearAttention, LayerKindFullAttention,
		LayerKindLinearAttention, LayerKindLinearAttention, LayerKindLinearAttention, LayerKindFullAttention,
	}
	for l, want := range wantKinds {
		d := layout.Descriptor(l)
		if d.LayerIndex != l {
			t.Fatalf("layer %d descriptor index=%d", l, d.LayerIndex)
		}
		if d.Kind != want {
			t.Fatalf("layer %d kind=%v want %v", l, d.Kind, want)
		}
		switch want {
		case LayerKindFullAttention:
			if !d.Sliceable {
				t.Fatalf("full-attention layer %d must be sliceable", l)
			}
			if !d.Sequence.TokenIndexed || d.Sequence.Planes != 3 || d.Sequence.Stride != layout.Stride {
				t.Fatalf("full-attention layer %d sequence axis=%+v", l, d.Sequence)
			}
			if d.State.Present {
				t.Fatalf("full-attention layer %d must carry no recurrent state", l)
			}
		case LayerKindLinearAttention:
			if d.Sliceable {
				t.Fatalf("linear-attention layer %d must NOT be sliceable", l)
			}
			if d.Sequence.TokenIndexed || d.Sequence.Planes != 0 || d.Sequence.Stride != 0 {
				t.Fatalf("linear-attention layer %d must have no token axis: %+v", l, d.Sequence)
			}
			if !d.State.Present {
				t.Fatalf("linear-attention layer %d must carry recurrent state", l)
			}
			if d.State.ValueHeads != geom.LinearNumValueHeads ||
				d.State.KeyHeadDim != geom.LinearKeyHeadDim ||
				d.State.ValueHeadDim != geom.LinearValueHeadDim ||
				d.State.ConvKernel != geom.LinearConvKernelDim {
				t.Fatalf("linear-attention layer %d state geometry=%+v", l, d.State)
			}
		}
	}

	// The three queries the swap / checkpoint seams consume.
	if got := layout.FullAttentionLayers(); !equalInts(got, []int{3, 7}) {
		t.Fatalf("FullAttentionLayers=%v want [3 7]", got)
	}
	if got := layout.SliceableLayers(); !equalInts(got, []int{3, 7}) {
		t.Fatalf("SliceableLayers=%v want [3 7]", got)
	}
	if got := layout.RecurrentLayers(); !equalInts(got, []int{0, 1, 2, 4, 5, 6}) {
		t.Fatalf("RecurrentLayers=%v want [0 1 2 4 5 6]", got)
	}

	// Idempotence: deriving twice yields an identical registry.
	again := NewModelCacheLayout(types, geom)
	if !equalInts(again.FullAttentionLayers(), layout.FullAttentionLayers()) ||
		!equalInts(again.RecurrentLayers(), layout.RecurrentLayers()) {
		t.Fatalf("layout derivation is not deterministic")
	}
}

// TestHybridCacheTypeRegistryDegradesUnknownLayers pins the fail-closed rule: an
// unrecognised/absent layer-type entry classifies boundary-only (not sliceable) so a
// non-conforming hybrid degrades per layer instead of being sliced incorrectly, and an
// out-of-range layer query is never reported sliceable.
func TestHybridCacheTypeRegistryDegradesUnknownLayers(t *testing.T) {
	layout := NewModelCacheLayout([]string{"full_attention", "", "minimax_m3_sparse"}, CacheLayoutGeometry{NumKVHeads: 2, HeadDim: 64})
	for _, l := range []int{1, 2} {
		d := layout.Descriptor(l)
		if d.Kind != LayerKindUnknown || d.Sliceable {
			t.Fatalf("layer %d kind=%v sliceable=%t, want unknown boundary-only", l, d.Kind, d.Sliceable)
		}
	}
	if layout.IsSliceable(-1) || layout.IsSliceable(99) {
		t.Fatalf("out-of-range layer must not be sliceable")
	}
	if got := layout.Descriptor(99).Kind; got != LayerKindUnknown {
		t.Fatalf("out-of-range kind=%v want unknown", got)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
