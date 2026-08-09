package compute

import "testing"

// capacity_window_test.go — EstimateKVStoreBytes' optional per-layer window (#5520).
//
// The estimator gained a WindowPerLayer axis so an interleaved local/global checkpoint is
// charged min(window, tokens) positions on a windowed layer instead of the whole context.
// The load-bearing property is the IDENTITY half: every config that declares no usable
// window must produce byte-for-byte the number it produced before the axis existed, at
// both precision tiers — otherwise the axis is a silent repricing of every served model.

// TestEstimateKVStoreBytesWindowCapsWindowedLayers witnesses the charge itself on a small
// hand-checkable geometry: 4 layers, 2 kv heads of a 4-wide head, layers 0 and 2 capped at
// a 3-token window, layers 1 and 3 full attention.
//
//	elemsPerRow      = 2 * 4                  = 8
//	perTokenPerLayer = 8 * 3 rows * 4 bytes   = 96
//	uniform  slots   = 4 * 10                 = 40
//	windowed slots   = 3 + 10 + 3 + 10        = 26
func TestEstimateKVStoreBytesWindowCapsWindowedLayers(t *testing.T) {
	cfg := KVConfig{NumLayers: 4, NumKVHeads: 2, HeadDim: 4, WindowPerLayer: []int{3, -1, 3, 0}}
	const perTokenPerLayer = 2 * 4 * 3 * 4 // 96

	if got, want := EstimateKVStoreBytes(cfg, 10), int64(26*perTokenPerLayer); got != want {
		t.Errorf("EstimateKVStoreBytes(ctx=10) = %d, want %d", got, want)
	}
	// Both "no window" spellings must behave alike: the loader writes -1, kvbudget writes
	// a non-positive entry. Layers 1 and 3 use one each above and both held 10 slots.
	uncapped := KVConfig{NumLayers: 4, NumKVHeads: 2, HeadDim: 4}
	if got, want := EstimateKVStoreBytes(uncapped, 10), int64(40*perTokenPerLayer); got != want {
		t.Fatalf("uncapped baseline = %d, want %d", got, want)
	}
	// The classed form must carry the same discount into the MemoryKVCache demand row.
	plan := EstimateKVStoreMemoryPlan(cfg, 10)
	if len(plan) != 1 || plan[0].Bytes != int64(26*perTokenPerLayer) {
		t.Errorf("EstimateKVStoreMemoryPlan = %+v, want one %d-byte kv_cache demand", plan, 26*perTokenPerLayer)
	}
}

// TestEstimateKVStoreBytesWindowIdentityCases pins the bit-identity contract: each config
// below declares a window that cannot bite, so each must equal the untouched
// NumLayers x tokens product at every tier and context.
func TestEstimateKVStoreBytesWindowIdentityCases(t *testing.T) {
	base := KVConfig{NumLayers: 4, NumKVHeads: 2, HeadDim: 4}
	// Every case pairs a window declaration with the token counts at which that
	// declaration provably cannot bite — a window only caps a layer once the context
	// grows PAST it, so the identity claim has to name the contexts it holds at.
	cases := []struct {
		name   string
		window []int
		tokens []int
	}{
		{"nil slice (the uniform zero value)", nil, []int{0, 1, 63, 64, 65, 100000}},
		{"empty slice", []int{}, []int{0, 1, 63, 64, 65, 100000}},
		{"every layer full-attention (-1 sentinel)", []int{-1, -1, -1, -1}, []int{0, 1, 64, 100000}},
		{"every layer full-attention (0 spelling)", []int{0, 0, 0, 0}, []int{0, 1, 64, 100000}},
		{"slice shorter than NumLayers, no positive entry", []int{-1}, []int{0, 1, 64, 100000}},
		{"windows wider than the context", []int{4096, 4096, 4096, 4096}, []int{0, 1, 63, 4095, 4096}},
		{"window exactly the context (min is a no-op)", []int{64, 64, 64, 64}, []int{0, 1, 63, 64}},
		{"mixed windows, all at or above the context", []int{64, -1, 4096, 0}, []int{0, 1, 63, 64}},
	}
	for _, tc := range cases {
		for _, tokens := range tc.tokens {
			for _, prec := range []KVPrecision{KVPrecisionF32, KVPrecisionQ8} {
				want := base
				want.Precision = prec
				got := want
				got.WindowPerLayer = tc.window
				if a, b := EstimateKVStoreBytes(got, tokens), EstimateKVStoreBytes(want, tokens); a != b {
					t.Errorf("%s: tokens=%d prec=%v gave %d, want the uniform %d", tc.name, tokens, prec, a, b)
				}
			}
		}
	}
}

// TestEstimateKVStoreBytesWindowFailsOpenOnBadGeometry keeps the estimator's fail-open
// floor: incomplete geometry still reports 0 rather than inventing a windowed plan, and a
// window on a zero-layer config cannot resurrect a nonzero charge.
func TestEstimateKVStoreBytesWindowFailsOpenOnBadGeometry(t *testing.T) {
	for _, cfg := range []KVConfig{
		{NumLayers: 4, HeadDim: 4, WindowPerLayer: []int{2, 2, 2, 2}},    // NumKVHeads 0
		{NumLayers: 4, NumKVHeads: 2, WindowPerLayer: []int{2, 2, 2, 2}}, // HeadDim 0
		{NumKVHeads: 2, HeadDim: 4, WindowPerLayer: []int{2}},            // NumLayers 0
	} {
		if got := EstimateKVStoreBytes(cfg, 100); got != 0 {
			t.Errorf("EstimateKVStoreBytes(%+v) = %d, want 0 (fail open)", cfg, got)
		}
	}
}
