package model

import (
	"math"
	"testing"
	"unsafe"
)

// TestQ8PrefillOutputProjectionReuseAndParity witnesses issue #11599:
// In prefillBatchedQ, self_attn.o_proj destination buffer is pre-allocated once
// for the request (P*H) and reused across all NumLayers, eliminating redundant
// slice allocations per layer while preserving exact numerical parity with the
// per-token decode reference.
func TestQ8PrefillOutputProjectionReuseAndParity(t *testing.T) {
	oldMode := qgemmMode
	qgemmMode = qgemmModeLegacy
	defer func() { qgemmMode = oldMode }()

	cfg := llamaArchConfig()
	cfg.NumLayers = 4
	m := NewSynthetic(cfg)
	m.Quantize()

	prompt := []int{3, 17, 5, 23, 41, 2, 19, 11}
	P := len(prompt)
	H := cfg.HiddenSize

	type dstObservation struct {
		layer  int
		ptr    uintptr
		length int
		cap    int
	}
	var observations []dstObservation

	q8PrefillOProjDstObserver = func(layer int, dst []float32) {
		ptr := uintptr(0)
		if len(dst) > 0 {
			ptr = uintptr(unsafe.Pointer(&dst[0]))
		}
		observations = append(observations, dstObservation{
			layer:  layer,
			ptr:    ptr,
			length: len(dst),
			cap:    cap(dst),
		})
	}
	defer func() { q8PrefillOProjDstObserver = nil }()

	// 1. Run batched Q8 prefill.
	batSession := m.NewSession()
	batSession.Quant = true
	gotHidden := batSession.prefillBatchedQ(prompt)
	gotLogits := batSession.headQ(gotHidden)

	// Verify buffer reuse across all layers.
	if len(observations) != cfg.NumLayers {
		t.Fatalf("expected %d observer invocations, got %d", cfg.NumLayers, len(observations))
	}
	firstPtr := observations[0].ptr
	if firstPtr == 0 {
		t.Fatal("observed nil/empty destination buffer pointer at layer 0")
	}
	for l, obs := range observations {
		if obs.length != P*H {
			t.Fatalf("layer %d: destination length = %d, want %d (P*H)", l, obs.length, P*H)
		}
		if obs.ptr != firstPtr {
			t.Fatalf("layer %d: destination buffer was not reused (ptr %x != layer 0 ptr %x)", l, obs.ptr, firstPtr)
		}
	}

	// 2. Second prefill run to verify idempotence and bit-level determinism.
	batSession2 := m.NewSession()
	batSession2.Quant = true
	gotHidden2 := batSession2.prefillBatchedQ(prompt)
	for i := range gotHidden {
		if math.Float32bits(gotHidden[i]) != math.Float32bits(gotHidden2[i]) {
			t.Fatalf("hidden[%d] mismatch between two prefill runs: %08x vs %08x", i, math.Float32bits(gotHidden[i]), math.Float32bits(gotHidden2[i]))
		}
	}

	// 3. Reference per-token decode loop.
	refSession := m.NewSession()
	refSession.Quant = true
	var refHidden []float32
	for _, id := range prompt {
		refHidden = refSession.tokenHiddenQ(id, refSession.Cache.Len())
	}
	refLogits := refSession.headQ(refHidden)

	// Verify numerical parity with per-token decode (drift bound <= 1e-5 for float32 reduction order).
	if d, _ := maxAbsDiff(gotHidden, refHidden); d > 1e-5 {
		t.Fatalf("batched Q8 prefill hidden != per-token decode hidden: max abs diff %.3e > 1e-5", d)
	}
	if d, _ := maxAbsDiff(gotLogits, refLogits); d > 1e-5 {
		t.Fatalf("batched Q8 prefill logits != per-token decode logits: max abs diff %.3e > 1e-5", d)
	}

	// Verify full KV cache state equality across all layers.
	if batSession.Cache.Len() != refSession.Cache.Len() {
		t.Fatalf("cache len mismatch: batched=%d ref=%d", batSession.Cache.Len(), refSession.Cache.Len())
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for name, pair := range map[string][2][]float32{
			"K":    {refSession.Cache.K[l], batSession.Cache.K[l]},
			"Kraw": {refSession.Cache.Kraw[l], batSession.Cache.Kraw[l]},
			"V":    {refSession.Cache.V[l], batSession.Cache.V[l]},
		} {
			if len(pair[0]) != len(pair[1]) {
				t.Fatalf("layer %d %s len mismatch: ref=%d bat=%d", l, name, len(pair[0]), len(pair[1]))
			}
			if d, _ := maxAbsDiff(pair[0], pair[1]); d > 1e-5 {
				t.Fatalf("layer %d %s max abs diff %.3e > 1e-5", l, name, d)
			}
		}
	}
}

// TestQ8PrefillOutputProjectionReuseWithBias witnesses that destination buffer reuse
// maintains exact numerical parity and does not leak in-place bias addition across layers
// on models that carry self_attn.o_proj.bias.
func TestQ8PrefillOutputProjectionReuseWithBias(t *testing.T) {
	oldMode := qgemmMode
	qgemmMode = qgemmModeLegacy
	defer func() { qgemmMode = oldMode }()

	cfg := llamaArchConfig()
	cfg.NumLayers = 3
	m := newProjBiasModel(t, cfg)
	m.Quantize()

	prompt := []int{12, 34, 56, 78}
	P := len(prompt)
	H := cfg.HiddenSize

	var observedPtrs []uintptr
	q8PrefillOProjDstObserver = func(layer int, dst []float32) {
		if len(dst) > 0 {
			observedPtrs = append(observedPtrs, uintptr(unsafe.Pointer(&dst[0])))
		}
	}
	defer func() { q8PrefillOProjDstObserver = nil }()

	// Batched prefill with reused o_proj buffer.
	batSession := m.NewSession()
	batSession.Quant = true
	gotHidden := batSession.prefillBatchedQ(prompt)
	gotLogits := batSession.headQ(gotHidden)

	if len(observedPtrs) != cfg.NumLayers {
		t.Fatalf("expected %d observer invocations, got %d", cfg.NumLayers, len(observedPtrs))
	}
	for l := 1; l < len(observedPtrs); l++ {
		if observedPtrs[l] != observedPtrs[0] {
			t.Fatalf("layer %d: destination ptr %x != layer 0 ptr %x", l, observedPtrs[l], observedPtrs[0])
		}
	}

	// Reference per-token decode loop.
	refSession := m.NewSession()
	refSession.Quant = true
	var refHidden []float32
	for _, id := range prompt {
		refHidden = refSession.tokenHiddenQ(id, refSession.Cache.Len())
	}
	refLogits := refSession.headQ(refHidden)

	if d, _ := maxAbsDiff(gotHidden, refHidden); d > 1e-5 {
		t.Fatalf("biased Q8 prefill hidden != per-token decode: max abs diff %.3e > 1e-5", d)
	}
	if d, _ := maxAbsDiff(gotLogits, refLogits); d > 1e-5 {
		t.Fatalf("biased Q8 prefill logits != per-token decode: max abs diff %.3e > 1e-5", d)
	}
	_ = P
	_ = H
}

// TestQ8PrefillOutputProjectionReuseDefaultTileMode verifies destination buffer reuse
// and numerical stability under the default production tile GEMM mode.
func TestQ8PrefillOutputProjectionReuseDefaultTileMode(t *testing.T) {
	cfg := llamaArchConfig()
	cfg.NumLayers = 3
	m := NewSynthetic(cfg)
	m.Quantize()

	prompt := []int{5, 11, 23, 42}
	P := len(prompt)
	H := cfg.HiddenSize

	var observedPtrs []uintptr
	q8PrefillOProjDstObserver = func(layer int, dst []float32) {
		if len(dst) > 0 {
			observedPtrs = append(observedPtrs, uintptr(unsafe.Pointer(&dst[0])))
		}
	}
	defer func() { q8PrefillOProjDstObserver = nil }()

	s1 := m.NewSession()
	s1.Quant = true
	out1 := s1.prefillBatchedQ(prompt)

	if len(observedPtrs) != cfg.NumLayers {
		t.Fatalf("expected %d layer observations, got %d", cfg.NumLayers, len(observedPtrs))
	}
	for l := 1; l < len(observedPtrs); l++ {
		if observedPtrs[l] != observedPtrs[0] {
			t.Fatalf("layer %d: destination ptr %x != layer 0 ptr %x", l, observedPtrs[l], observedPtrs[0])
		}
	}

	// Second run produces bit-identical results.
	s2 := m.NewSession()
	s2.Quant = true
	out2 := s2.prefillBatchedQ(prompt)
	for i := range out1 {
		if math.Float32bits(out1[i]) != math.Float32bits(out2[i]) {
			t.Fatalf("out1[%d] (%08x) != out2[%d] (%08x)", i, math.Float32bits(out1[i]), i, math.Float32bits(out2[i]))
		}
	}
	_ = P
	_ = H
}
