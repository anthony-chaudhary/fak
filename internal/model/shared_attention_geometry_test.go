package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// TestSharedAttentionGeometryRefusals pins the shared-attention geometry contract ADAPTED
// from ashhart/TensorFold zig/src/core/shared_attention.zig:17@9e8c368b0b4c2c7892a0c9f0f302d7c164016209:
// valid GQA/RoPE passes; qh=3/kh=2, odd rotary_dim, misaligned head_dim, dup KV slots, and
// aliased buffers refuse via sentinel errors.
func TestSharedAttentionGeometryRefusals(t *testing.T) {
	good := SharedAttentionShape{QueryHeads: 4, KVHeads: 2, HeadDim: 64, RotaryDim: 32, Capacity: 128}
	if err := good.Validate(); err != nil {
		t.Fatalf("valid GQA/RoPE shape must pass, got %v", err)
	}
	noRope := good
	noRope.RotaryDim = 0
	if err := noRope.Validate(); err != nil {
		t.Fatalf("zero rotary_dim (no RoPE) must pass, got %v", err)
	}

	shapeCases := []struct {
		name string
		mut  func(*SharedAttentionShape)
	}{
		{"qh=3/kh=2 non-dividing", func(s *SharedAttentionShape) { s.QueryHeads = 3 }},
		{"zero query heads", func(s *SharedAttentionShape) { s.QueryHeads = 0 }},
		{"zero kv heads", func(s *SharedAttentionShape) { s.KVHeads = 0 }},
		{"query heads above bound", func(s *SharedAttentionShape) { s.QueryHeads = 129 }},
		{"odd rotary_dim", func(s *SharedAttentionShape) { s.RotaryDim = 33 }},
		{"rotary_dim above head_dim", func(s *SharedAttentionShape) { s.RotaryDim = 65 }},
		{"negative rotary_dim", func(s *SharedAttentionShape) { s.RotaryDim = -2 }},
		{"misaligned head_dim 31", func(s *SharedAttentionShape) { s.HeadDim = 31 }},
		{"head_dim 48 not 32-aligned", func(s *SharedAttentionShape) { s.HeadDim = 48 }},
		{"head_dim above 256", func(s *SharedAttentionShape) { s.HeadDim = 288 }},
		{"zero capacity", func(s *SharedAttentionShape) { s.Capacity = 0 }},
		{"capacity above u32", func(s *SharedAttentionShape) { s.Capacity = math.MaxUint32 + 1 }},
	}
	for _, c := range shapeCases {
		s := good
		c.mut(&s)
		if err := s.Validate(); !errors.Is(err, ErrSharedAttentionShape) {
			t.Errorf("%s: Validate() = %v, want ErrSharedAttentionShape", c.name, err)
		}
	}

	// Append: unique in-capacity slots on disjoint stores pass.
	rowBytes := good.KVHeads * good.HeadDim * sharedAttentionF32Bytes // 512
	cacheBytes := good.Capacity * rowBytes                            // 65536
	appendGood := SharedAttentionAppend{
		Shape:  good,
		Slots:  []int{3, 1},
		K:      SharedAttentionBuffer{ID: "k", Bytes: 2 * rowBytes},
		V:      SharedAttentionBuffer{ID: "v", Bytes: 2 * rowBytes},
		CacheK: SharedAttentionBuffer{ID: "cacheK", Bytes: cacheBytes},
		CacheV: SharedAttentionBuffer{ID: "cacheV", Bytes: cacheBytes},
	}
	if err := appendGood.Validate(); err != nil {
		t.Fatalf("unique-slot disjoint append must pass, got %v", err)
	}

	appendCases := []struct {
		name string
		mut  func(*SharedAttentionAppend)
		want error
	}{
		{"dup KV slot", func(a *SharedAttentionAppend) { a.Slots = []int{3, 1, 3} }, ErrSharedAttentionDuplicateSlot},
		{"slot at capacity", func(a *SharedAttentionAppend) { a.Slots = []int{good.Capacity} }, ErrSharedAttentionAppendSlot},
		{"negative slot", func(a *SharedAttentionAppend) { a.Slots = []int{-1} }, ErrSharedAttentionAppendSlot},
		{"empty batch", func(a *SharedAttentionAppend) { a.Slots = nil }, ErrSharedAttentionAppendView},
		{"batch above 16", func(a *SharedAttentionAppend) { a.Slots = make([]int, 17) }, ErrSharedAttentionAppendView},
		{"K aliases cacheK", func(a *SharedAttentionAppend) { a.K.ID = "cacheK" }, ErrSharedAttentionAliased},
		{"V aliases cacheV", func(a *SharedAttentionAppend) { a.V.ID = "cacheV" }, ErrSharedAttentionAliased},
		{"cacheK aliases cacheV", func(a *SharedAttentionAppend) { a.CacheV.ID = "cacheK" }, ErrSharedAttentionAliased},
		{"K window too small", func(a *SharedAttentionAppend) { a.K.Bytes = rowBytes }, ErrSharedAttentionBufferSmall},
	}
	for _, c := range appendCases {
		a := appendGood
		c.mut(&a)
		if err := a.Validate(); !errors.Is(err, c.want) {
			t.Errorf("%s: Validate() = %v, want %v", c.name, err, c.want)
		}
	}

	// View: the byte mask[row*stride+key] alone decides visible keys; bounds and
	// output-vs-input disjointness are refused by sentinel.
	viewGood := SharedAttentionView{
		Shape:      good,
		Rows:       2,
		Keys:       8,
		MaskStride: 8,
		Scale:      0.125,
		Q:          SharedAttentionBuffer{ID: "q", Bytes: 2 * good.QueryHeads * good.HeadDim * sharedAttentionF32Bytes},
		K:          SharedAttentionBuffer{ID: "vk", Bytes: 8 * good.KVHeads * good.HeadDim * sharedAttentionF32Bytes},
		V:          SharedAttentionBuffer{ID: "vv", Bytes: 8 * good.KVHeads * good.HeadDim * sharedAttentionF32Bytes},
		Out:        SharedAttentionBuffer{ID: "out", Bytes: 2 * good.QueryHeads * good.HeadDim * sharedAttentionF32Bytes},
		Mask:       SharedAttentionBuffer{ID: "mask", Bytes: 2 * 8},
	}
	if err := viewGood.Validate(); err != nil {
		t.Fatalf("valid byte-mask view must pass, got %v", err)
	}

	viewCases := []struct {
		name string
		mut  func(*SharedAttentionView)
		want error
	}{
		{"zero rows", func(v *SharedAttentionView) { v.Rows = 0 }, ErrSharedAttentionView},
		{"rows above 16", func(v *SharedAttentionView) { v.Rows = 17 }, ErrSharedAttentionView},
		{"keys past capacity", func(v *SharedAttentionView) { v.Keys = good.Capacity + 1 }, ErrSharedAttentionView},
		{"stride below keys", func(v *SharedAttentionView) { v.MaskStride = 7 }, ErrSharedAttentionView},
		{"NaN scale", func(v *SharedAttentionView) { v.Scale = float32(math.NaN()) }, ErrSharedAttentionView},
		{"infinite scale", func(v *SharedAttentionView) { v.Scale = float32(math.Inf(1)) }, ErrSharedAttentionView},
		{"mask one byte short", func(v *SharedAttentionView) { v.Mask.Bytes = 2*8 - 1 }, ErrSharedAttentionBufferSmall},
		{"out aliases q", func(v *SharedAttentionView) { v.Out.ID = "q" }, ErrSharedAttentionAliased},
		{"out aliases mask", func(v *SharedAttentionView) { v.Out.ID = "mask" }, ErrSharedAttentionAliased},
		{"q offset misaligned", func(v *SharedAttentionView) { v.Q.Off = 2; v.Q.Bytes += 2 }, ErrSharedAttentionMisaligned},
	}
	for _, c := range viewCases {
		v := viewGood
		c.mut(&v)
		if err := v.Validate(); !errors.Is(err, c.want) {
			t.Errorf("%s: Validate() = %v, want %v", c.name, err, c.want)
		}
	}

	// ForwardTP seam: with the contract flag set, a non-dividing GQA shape refuses via
	// the sentinel before any sharded-attention dispatch; compliant geometry runs.
	badCfg := Config{
		HiddenSize: 96, NumLayers: 1, NumHeads: 3, NumKVHeads: 2, HeadDim: 32,
		IntermediateSize: 96, VocabSize: 40, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	}
	if _, err := (&Model{Cfg: badCfg}).ForwardTP([]int{1, 2}, TPConfig{SharedAttentionGeometry: true}); !errors.Is(err, ErrSharedAttentionShape) {
		t.Fatalf("ForwardTP gate: qh=3/kh=2 must refuse with ErrSharedAttentionShape, got %v", err)
	}
	okCfg := badCfg
	okCfg.HiddenSize = 128
	okCfg.NumHeads = 4
	if _, err := NewSynthetic(okCfg).ForwardTP([]int{1, 2, 3}, TPConfig{AttnRanks: 2, FFNRanks: 2, Coll: LocalCollective{}, SharedAttentionGeometry: true}); err != nil {
		t.Fatalf("ForwardTP gate: compliant GQA/RoPE geometry must run, got %v", err)
	}
}
