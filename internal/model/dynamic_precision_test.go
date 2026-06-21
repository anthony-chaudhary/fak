package model

import (
	"math"
	"testing"
)

func dynamicPrecisionSyntheticModel() *Model {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
	return NewSynthetic(cfg)
}

func TestDynamicPrecisionAcceptsQ8WhenConfidenceGatePasses(t *testing.T) {
	m := dynamicPrecisionSyntheticModel()
	m.Quantize()
	prompt := []int{3, 17, 5, 23, 41}

	want := m.NewSession()
	want.Quant = true
	wantLogits := want.Prefill(prompt)

	got := m.NewSession()
	got.PrecisionPolicy = &DynamicPrecisionPolicy{}
	gotLogits := got.Prefill(prompt)
	assertFloat32BitsEqual(t, "dynamic accepted-q8 prefill logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "dynamic accepted-q8 prefill", want.Cache, got.Cache)
	if got.PrecisionStats.Q8Attempts != 1 || got.PrecisionStats.Q8Calls != 1 || got.PrecisionStats.Fallbacks != 0 {
		t.Fatalf("prefill stats = %+v, want one accepted q8 attempt and no fallback", got.PrecisionStats)
	}

	id := 11
	wantLogits = want.Step(id)
	gotLogits = got.Step(id)
	assertFloat32BitsEqual(t, "dynamic accepted-q8 step logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "dynamic accepted-q8 step", want.Cache, got.Cache)
	if got.PrecisionStats.Q8Attempts != 2 || got.PrecisionStats.Q8Calls != 2 || got.PrecisionStats.Q8Tokens != len(prompt)+1 {
		t.Fatalf("step stats = %+v, want two accepted q8 calls over prompt+step tokens", got.PrecisionStats)
	}
	if got.PrecisionStats.LastTier != PrecisionQ8 || !got.PrecisionStats.LastAccepted {
		t.Fatalf("last decision = %+v, want accepted q8", got.PrecisionStats)
	}
}

func TestDynamicPrecisionFallbackRestoresF32Cache(t *testing.T) {
	m := dynamicPrecisionSyntheticModel()
	m.Quantize()
	prompt := []int{3, 17, 5, 23, 41}
	policy := &DynamicPrecisionPolicy{MinTop1Margin: float32(math.MaxFloat32)}

	want := m.NewSession()
	wantLogits := want.Prefill(prompt)

	got := m.NewSession()
	got.Quant = true // fallback must still recompute through f32 when policy rejects Q8.
	got.PrecisionPolicy = policy
	gotLogits := got.Prefill(prompt)
	assertFloat32BitsEqual(t, "dynamic fallback prefill logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "dynamic fallback prefill", want.Cache, got.Cache)
	if got.PrecisionStats.Q8Attempts != 1 || got.PrecisionStats.Q8Calls != 0 || got.PrecisionStats.Fallbacks != 1 {
		t.Fatalf("prefill fallback stats = %+v, want one rejected q8 attempt", got.PrecisionStats)
	}

	id := 11
	wantLogits = want.Step(id)
	gotLogits = got.Step(id)
	assertFloat32BitsEqual(t, "dynamic fallback step logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "dynamic fallback step", want.Cache, got.Cache)
	if got.PrecisionStats.Q8Attempts != 2 || got.PrecisionStats.F32Calls != 2 || got.PrecisionStats.F32Tokens != len(prompt)+1 || got.PrecisionStats.Fallbacks != 2 {
		t.Fatalf("step fallback stats = %+v, want two f32 fallbacks over prompt+step tokens", got.PrecisionStats)
	}
	if got.PrecisionStats.LastTier != PrecisionF32 || got.PrecisionStats.LastAccepted {
		t.Fatalf("last decision = %+v, want rejected q8 followed by f32", got.PrecisionStats)
	}
}

func TestDynamicPrecisionWithoutQ8WeightsFallsBackToF32(t *testing.T) {
	m := dynamicPrecisionSyntheticModel()
	prompt := []int{3, 17, 5, 23, 41}

	want := m.NewSession()
	wantLogits := want.Prefill(prompt)

	got := m.NewSession()
	got.PrecisionPolicy = &DynamicPrecisionPolicy{}
	gotLogits := got.Prefill(prompt)
	assertFloat32BitsEqual(t, "dynamic no-q8 prefill logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "dynamic no-q8 prefill", want.Cache, got.Cache)
	if got.PrecisionStats.Q8Attempts != 0 || got.PrecisionStats.F32Calls != 1 || got.PrecisionStats.F32Tokens != len(prompt) {
		t.Fatalf("no-q8 stats = %+v, want direct f32", got.PrecisionStats)
	}
}
