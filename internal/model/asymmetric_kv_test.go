package model

import (
	"math"
	"math/rand"
	"testing"
)

// asymmetric_kv_test.go — unit tests for asymmetric K/V bit allocation codec and memory budgeting (#10731).

// TestAsymmetricKVByteSavings262KReduction verifies the exact byte accounting for 262K context tokens,
// proving the 61.4 GB -> ~20.08 GB footprint reduction.
func TestAsymmetricKVByteSavings262KReduction(t *testing.T) {
	const contextTokens = 262144 // 262K tokens

	// Config 1: 26 layers, 23 KV heads, HeadDim 64 -> kvDim = 1472.
	// K=Q8_0, V=Q4_0 -> 2944 bytes/token/layer * 26 * 262144 = 20,065,550,336 bytes (~20.08 GB).
	numLayers1 := 26
	kvDim1 := 23 * 64 // 1472
	bytes1 := AsymmetricKVCacheBytes(contextTokens, numLayers1, kvDim1, KVPrecisionQ8_0, KVPrecisionQ4_0)
	wantBytes1 := int64(26) * 2944 * int64(contextTokens)
	if bytes1 != wantBytes1 {
		t.Fatalf("Config 1 bytes = %d, want %d", bytes1, wantBytes1)
	}

	gb1 := float64(bytes1) / 1e9
	if math.Abs(gb1-20.07) > 0.1 {
		t.Fatalf("Config 1 in GB = %.2f, want ~20.08 GB", gb1)
	}
	t.Logf("Config 1 (262K, 26 layers, kvDim 1472): %d bytes (%.2f GB / %.2f GiB)",
		bytes1, gb1, float64(bytes1)/(1<<30))

	// Config 2: 29 layers, 8 KV heads, HeadDim 128 -> kvDim = 1024.
	// FP32 baseline: 62,276,730,880 bytes (~62.28 GB / 58.0 GiB)
	// Asymmetric K=8, V=4: 15,569,182,720 bytes (~15.57 GB / 14.5 GiB) -> 4.0x reduction!
	numLayers2 := 29
	kvDim2 := 8 * 128 // 1024
	fp32Bytes := AsymmetricKVCacheBytes(contextTokens, numLayers2, kvDim2, KVPrecisionFP32, KVPrecisionFP32)
	asymBytes := AsymmetricKVCacheBytes(contextTokens, numLayers2, kvDim2, KVPrecisionQ8_0, KVPrecisionQ4_0)

	wantFP32 := int64(contextTokens) * int64(numLayers2) * int64(1024*8) // 62,276,730,880
	if fp32Bytes != wantFP32 {
		t.Fatalf("Config 2 FP32 bytes = %d, want %d", fp32Bytes, wantFP32)
	}

	wantAsym := int64(contextTokens) * int64(numLayers2) * int64(2048) // 15,569,182,720
	if asymBytes != wantAsym {
		t.Fatalf("Config 2 Asym bytes = %d, want %d", asymBytes, wantAsym)
	}

	if fp32Bytes != 4*asymBytes {
		t.Fatalf("Config 2 reduction ratio: got %v vs 4*asym %v", fp32Bytes, 4*asymBytes)
	}

	// FP16 Keys with Q4_0 Values
	asymFP16Bytes := AsymmetricKVCacheBytes(contextTokens, numLayers2, kvDim2, KVPrecisionFP16, KVPrecisionQ4_0)
	wantFP16Asym := int64(contextTokens) * int64(numLayers2) * int64(2048+768) // 2816 B/pos
	if asymFP16Bytes != wantFP16Asym {
		t.Fatalf("Config 2 FP16-Keys bytes = %d, want %d", asymFP16Bytes, wantFP16Asym)
	}

	t.Logf("Config 2 (262K, 29 layers, kvDim 1024):")
	t.Logf("  FP32 baseline: %.2f GB", float64(fp32Bytes)/1e9)
	t.Logf("  K=Q8_0, V=Q4_0: %.2f GB (4.0x reduction)", float64(asymBytes)/1e9)
	t.Logf("  K=FP16, V=Q4_0: %.2f GB (2.91x reduction)", float64(asymFP16Bytes)/1e9)
}

// TestAsymmetricKVBudgetReportAndMaxTokens tests the budget report and context ceiling solver.
func TestAsymmetricKVBudgetReportAndMaxTokens(t *testing.T) {
	const contextTokens = 131072 // 128K
	const numLayers = 32
	const kvDim = 1024

	report := AsymmetricKVCacheBudgetReport(contextTokens, numLayers, kvDim)
	if report.SavingsVsFP32 < 3.99 || report.SavingsVsFP32 > 4.01 {
		t.Fatalf("SavingsVsFP32 = %v, want ~4.0x", report.SavingsVsFP32)
	}
	if report.AsymQ8Q4Bytes >= report.FP32Bytes {
		t.Fatalf("asym bytes %d >= fp32 bytes %d", report.AsymQ8Q4Bytes, report.FP32Bytes)
	}

	// Test MaxContextTokensForBudget with 24 GB VRAM budget
	const budget24GB = 24 * 1000 * 1000 * 1000 // 24 GB
	maxFP32 := MaxContextTokensForBudget(budget24GB, numLayers, kvDim, KVPrecisionFP32, KVPrecisionFP32)
	maxAsym := MaxContextTokensForBudget(budget24GB, numLayers, kvDim, KVPrecisionQ8_0, KVPrecisionQ4_0)

	if maxAsym < 4*maxFP32-10 || maxAsym > 4*maxFP32+10 {
		t.Fatalf("maxAsym tokens (%d) should be ~4x maxFP32 tokens (%d)", maxAsym, maxFP32)
	}
	t.Logf("24 GB VRAM context limits: FP32=%d tokens, Asym(K=8,V=4)=%d tokens", maxFP32, maxAsym)

	// Edge cases
	if AsymmetricKVCacheBytes(0, 32, 1024, KVPrecisionQ8_0, KVPrecisionQ4_0) != 0 {
		t.Fatal("0 tokens should return 0 bytes")
	}
	if AsymmetricKVCacheBytes(1024, -1, 1024, KVPrecisionQ8_0, KVPrecisionQ4_0) != 0 {
		t.Fatal("-1 layers should return 0 bytes")
	}
	if MaxContextTokensForBudget(0, 32, 1024, KVPrecisionQ8_0, KVPrecisionQ4_0) != 0 {
		t.Fatal("0 budget should return 0 tokens")
	}
}

// TestAsymmetricKVPrecisionRetention verifies that Keys retain significantly higher precision
// than Values when quantized asymmetrically (e.g. K=Q8_0 or FP16 vs V=Q4_0).
func TestAsymmetricKVPrecisionRetention(t *testing.T) {
	rng := rand.New(rand.NewSource(20260903))
	const dim = 256
	kSrc := make([]float32, dim)
	vSrc := make([]float32, dim)
	for i := 0; i < dim; i++ {
		kSrc[i] = float32(rng.NormFloat64() * 2.0)
		vSrc[i] = float32(rng.NormFloat64() * 2.0)
	}

	// Test with K=Q8_0
	pairQ8, err := QuantizeAsymmetricKV(kSrc, vSrc, KVPrecisionQ8_0)
	if err != nil {
		t.Fatalf("QuantizeAsymmetricKV(Q8_0) failed: %v", err)
	}
	kOutQ8, vOutQ8, err := DequantizeAsymmetricKV(pairQ8)
	if err != nil {
		t.Fatalf("DequantizeAsymmetricKV(Q8_0) failed: %v", err)
	}

	repQ8, err := EvaluatePrecisionRetention(kSrc, vSrc, kOutQ8, vOutQ8)
	if err != nil {
		t.Fatalf("EvaluatePrecisionRetention failed: %v", err)
	}

	// Keys (8-bit) must have significantly higher cosine similarity and SNR than Values (4-bit)
	if repQ8.KeyCosineSimilarity <= repQ8.ValCosineSimilarity {
		t.Fatalf("Key cosine similarity (%.6f) not higher than Value (%.6f)",
			repQ8.KeyCosineSimilarity, repQ8.ValCosineSimilarity)
	}
	if repQ8.KeyCosineSimilarity < 0.9999 {
		t.Fatalf("Key cosine similarity %.6f < 0.9999", repQ8.KeyCosineSimilarity)
	}
	if repQ8.ValCosineSimilarity < 0.99 {
		t.Fatalf("Value cosine similarity %.6f < 0.99", repQ8.ValCosineSimilarity)
	}
	if repQ8.KeySNRdB <= repQ8.ValSNRdB {
		t.Fatalf("Key SNR (%.2f dB) not higher than Value SNR (%.2f dB)",
			repQ8.KeySNRdB, repQ8.ValSNRdB)
	}

	t.Logf("Q8/Q4 Fidelity: Key Cos=%.6f SNR=%.2fdB, Val Cos=%.6f SNR=%.2fdB",
		repQ8.KeyCosineSimilarity, repQ8.KeySNRdB, repQ8.ValCosineSimilarity, repQ8.ValSNRdB)

	// Test with K=FP16
	pairFP16, err := QuantizeAsymmetricKV(kSrc, vSrc, KVPrecisionFP16)
	if err != nil {
		t.Fatalf("QuantizeAsymmetricKV(FP16) failed: %v", err)
	}
	kOutFP16, vOutFP16, err := DequantizeAsymmetricKV(pairFP16)
	if err != nil {
		t.Fatalf("DequantizeAsymmetricKV(FP16) failed: %v", err)
	}

	repFP16, err := EvaluatePrecisionRetention(kSrc, vSrc, kOutFP16, vOutFP16)
	if err != nil {
		t.Fatalf("EvaluatePrecisionRetention(FP16) failed: %v", err)
	}

	if repFP16.KeyCosineSimilarity < 0.99999 {
		t.Fatalf("FP16 Key cosine similarity %.7f < 0.99999", repFP16.KeyCosineSimilarity)
	}
	if repFP16.KeySNRdB < 50.0 {
		t.Fatalf("FP16 Key SNR %.2fdB < 50 dB", repFP16.KeySNRdB)
	}
	t.Logf("FP16/Q4 Fidelity: Key Cos=%.7f SNR=%.2fdB, Val Cos=%.6f SNR=%.2fdB",
		repFP16.KeyCosineSimilarity, repFP16.KeySNRdB, repFP16.ValCosineSimilarity, repFP16.ValSNRdB)
}

// TestAsymmetricKVErrorBounds verifies that round-trip errors respect theoretical bounds.
func TestAsymmetricKVErrorBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(20260904))
	const dim = 128
	kSrc := make([]float32, dim)
	vSrc := make([]float32, dim)
	for i := 0; i < dim; i++ {
		kSrc[i] = float32(rng.NormFloat64() * 3.0)
		vSrc[i] = float32(rng.NormFloat64() * 3.0)
	}

	pair, err := QuantizeAsymmetricKV(kSrc, vSrc, KVPrecisionQ8_0)
	if err != nil {
		t.Fatalf("QuantizeAsymmetricKV failed: %v", err)
	}

	kBound := pair.KeyErrorBound()
	vBound := pair.ValueErrorBound()

	if kBound <= 0 || vBound <= 0 {
		t.Fatalf("error bounds must be positive, got kBound=%v, vBound=%v", kBound, vBound)
	}
	// Key (8-bit) bound should be substantially tighter than Value (4-bit) bound
	if kBound >= vBound {
		t.Fatalf("Key error bound (%v) should be tighter than Value bound (%v)", kBound, vBound)
	}

	kOut, vOut, err := DequantizeAsymmetricKV(pair)
	if err != nil {
		t.Fatalf("DequantizeAsymmetricKV failed: %v", err)
	}

	tol := float32(1.001) // floating-point slack
	for i := 0; i < dim; i++ {
		kErr := float32(math.Abs(float64(kOut[i] - kSrc[i])))
		if kErr > kBound*tol {
			t.Fatalf("Key element %d error %v > bound %v", i, kErr, kBound)
		}

		vErr := float32(math.Abs(float64(vOut[i] - vSrc[i])))
		if vErr > vBound*tol {
			t.Fatalf("Value element %d error %v > bound %v", i, vErr, vBound)
		}
	}
}

// TestAsymmetricKVCacheMultiLayer verifies multi-layer caching and token appends.
func TestAsymmetricKVCacheMultiLayer(t *testing.T) {
	const numLayers = 4
	const kvDim = 64
	cache, err := NewAsymmetricKVCache(numLayers, kvDim, KVPrecisionQ8_0)
	if err != nil {
		t.Fatalf("NewAsymmetricKVCache failed: %v", err)
	}

	const numTokens = 10
	rng := rand.New(rand.NewSource(42))

	// Populate tokens
	for token := 0; token < numTokens; token++ {
		for layer := 0; layer < numLayers; layer++ {
			k := make([]float32, kvDim)
			v := make([]float32, kvDim)
			for i := range k {
				k[i] = float32(rng.Float64()*10 - 5)
				v[i] = float32(rng.Float64()*10 - 5)
			}
			if err := cache.Append(layer, k, v); err != nil {
				t.Fatalf("Append token %d layer %d failed: %v", token, layer, err)
			}
		}
	}

	// Verify lengths
	for l := 0; l < numLayers; l++ {
		if cache.Len(l) != numTokens {
			t.Fatalf("layer %d len=%d, want %d", l, cache.Len(l), numTokens)
		}
	}

	// Verify Get
	for l := 0; l < numLayers; l++ {
		for pos := 0; pos < numTokens; pos++ {
			k, v, err := cache.Get(l, pos)
			if err != nil {
				t.Fatalf("Get(l=%d, pos=%d) failed: %v", l, pos, err)
			}
			if len(k) != kvDim || len(v) != kvDim {
				t.Fatalf("retrieved dimensions wrong: len(k)=%d, len(v)=%d", len(k), len(v))
			}
		}
	}

	totalBytes := cache.TotalBytes()
	if totalBytes <= 0 {
		t.Fatalf("totalBytes should be > 0, got %d", totalBytes)
	}
}

// TestAsymmetricKVPrecisionParsingAndRejection tests precision parsing and invalid inputs.
func TestAsymmetricKVPrecisionParsingAndRejection(t *testing.T) {
	valid := []struct {
		input string
		want  KVPrecision
	}{
		{"fp32", KVPrecisionFP32},
		{"32", KVPrecisionFP32},
		{"FP16", KVPrecisionFP16},
		{"16", KVPrecisionFP16},
		{"Q8_0", KVPrecisionQ8_0},
		{"q8", KVPrecisionQ8_0},
		{"8", KVPrecisionQ8_0},
		{"Q4_0", KVPrecisionQ4_0},
		{"4", KVPrecisionQ4_0},
	}

	for _, tc := range valid {
		got, err := ParseKVPrecision(tc.input)
		if err != nil {
			t.Fatalf("ParseKVPrecision(%q) failed: %v", tc.input, err)
		}
		if got != tc.want {
			t.Fatalf("ParseKVPrecision(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}

	if _, err := ParseKVPrecision("int2"); err == nil {
		t.Fatal("ParseKVPrecision('int2') should fail")
	}

	// Quantize with empty inputs
	if _, err := QuantizeAsymmetricKV(nil, []float32{1, 2}, KVPrecisionQ8_0); err == nil {
		t.Fatal("empty K should fail")
	}
	if _, err := QuantizeAsymmetricKV([]float32{1, 2}, nil, KVPrecisionQ8_0); err == nil {
		t.Fatal("empty V should fail")
	}
	if _, err := QuantizeAsymmetricKV([]float32{1, 2}, []float32{1, 2}, "bad_precision"); err == nil {
		t.Fatal("unsupported precision should fail")
	}

	// Cache creation with invalid params
	if _, err := NewAsymmetricKVCache(0, 64, KVPrecisionQ8_0); err == nil {
		t.Fatal("0 layers should fail")
	}
	if _, err := NewAsymmetricKVCache(4, 0, KVPrecisionQ8_0); err == nil {
		t.Fatal("0 kvDim should fail")
	}
}

// TestAsymmetricKVCacheTruncateClearAndNil tests cache clearing, truncation, and nil receiver safety.
func TestAsymmetricKVCacheTruncateClearAndNil(t *testing.T) {
	cache, err := NewAsymmetricKVCache(2, 32, KVPrecisionQ8_0)
	if err != nil {
		t.Fatalf("NewAsymmetricKVCache failed: %v", err)
	}

	for i := 0; i < 10; i++ {
		k := make([]float32, 32)
		v := make([]float32, 32)
		_ = cache.Append(0, k, v)
		_ = cache.Append(1, k, v)
	}

	if cache.Len(0) != 10 || cache.Len(1) != 10 {
		t.Fatalf("expected 10 tokens per layer, got %d, %d", cache.Len(0), cache.Len(1))
	}

	cache.Truncate(5)
	if cache.Len(0) != 5 || cache.Len(1) != 5 {
		t.Fatalf("expected 5 tokens per layer after truncate, got %d, %d", cache.Len(0), cache.Len(1))
	}

	cache.Clear()
	if cache.Len(0) != 0 || cache.Len(1) != 0 {
		t.Fatalf("expected 0 tokens per layer after clear, got %d, %d", cache.Len(0), cache.Len(1))
	}

	// Nil receiver safety
	var nilCache *AsymmetricKVCache
	if nilCache.Len(0) != 0 {
		t.Fatal("nilCache.Len should return 0")
	}
	if nilCache.TotalBytes() != 0 {
		t.Fatal("nilCache.TotalBytes should return 0")
	}
	if err := nilCache.Append(0, nil, nil); err == nil {
		t.Fatal("nilCache.Append should return error")
	}
	if _, _, err := nilCache.Get(0, 0); err == nil {
		t.Fatal("nilCache.Get should return error")
	}
	nilCache.Truncate(5) // should not panic
	nilCache.Clear()     // should not panic

	var nilPair *AsymmetricKVPair
	if nilPair.Bytes() != 0 {
		t.Fatal("nilPair.Bytes should return 0")
	}
	if nilPair.KeyErrorBound() != 0 || nilPair.ValueErrorBound() != 0 {
		t.Fatal("nilPair error bounds should return 0")
	}
	if _, ok := nilPair.AsKVQuantAsymmetric(); ok {
		t.Fatal("nilPair.AsKVQuantAsymmetric should return false")
	}
}

// TestAsymmetricKVBridgeConversion tests interop between AsymmetricKVPair and KVQuantAsymmetric.
func TestAsymmetricKVBridgeConversion(t *testing.T) {
	kSrc := make([]float32, 64)
	vSrc := make([]float32, 64)
	for i := range kSrc {
		kSrc[i] = float32(i) * 0.1
		vSrc[i] = float32(i) * 0.2
	}

	pair, err := QuantizeAsymmetricKV(kSrc, vSrc, KVPrecisionQ8_0)
	if err != nil {
		t.Fatalf("QuantizeAsymmetricKV failed: %v", err)
	}

	// Bridge to KVQuantAsymmetric
	asym, ok := pair.AsKVQuantAsymmetric()
	if !ok {
		t.Fatal("expected AsKVQuantAsymmetric to return true for Q8_0")
	}
	if asym.Bytes() <= 0 {
		t.Fatalf("expected positive asym.Bytes, got %d", asym.Bytes())
	}

	// Bridge back from KVQuantAsymmetric
	pairRestored := FromKVQuantAsymmetric(asym)
	if pairRestored.KeyPrecision != KVPrecisionQ8_0 || pairRestored.ValPrecision != KVPrecisionQ4_0 {
		t.Fatalf("unexpected precisions: k=%v v=%v", pairRestored.KeyPrecision, pairRestored.ValPrecision)
	}

	kOut, vOut, err := DequantizeAsymmetricKV(pairRestored)
	if err != nil {
		t.Fatalf("DequantizeAsymmetricKV failed: %v", err)
	}
	if len(kOut) != 64 || len(vOut) != 64 {
		t.Fatalf("unexpected dimensions: len(k)=%d, len(v)=%d", len(kOut), len(vOut))
	}
}

// TestFromKVQuantAsymmetric_OddDimensions verifies that FromKVQuantAsymmetric preserves odd dimensions (#10911).
func TestFromKVQuantAsymmetric_OddDimensions(t *testing.T) {
	oddDims := []int{31, 33, 65, 127}
	for _, n := range oddDims {
		kSrc := make([]float32, n)
		vSrc := make([]float32, n)
		for i := 0; i < n; i++ {
			kSrc[i] = float32(i)*0.05 - 1.0
			vSrc[i] = float32(i)*0.02 + 0.5
		}

		asym := QuantizeKVAsymmetric(kSrc, vSrc)
		pair := FromKVQuantAsymmetric(asym)

		if pair.KeyDim != n {
			t.Errorf("dim %d: KeyDim = %d, want %d", n, pair.KeyDim, n)
		}
		if pair.ValDim != n {
			t.Errorf("dim %d: ValDim = %d, want %d", n, pair.ValDim, n)
		}

		kOut, vOut, err := DequantizeAsymmetricKV(pair)
		if err != nil {
			t.Fatalf("dim %d: DequantizeAsymmetricKV failed: %v", n, err)
		}
		if len(kOut) != n {
			t.Errorf("dim %d: len(kOut) = %d, want %d", n, len(kOut), n)
		}
		if len(vOut) != n {
			t.Errorf("dim %d: len(vOut) = %d, want %d", n, len(vOut), n)
		}
	}
}
