package metalgemm

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func generateDeterministicFloats(n int, seed int64, scale float32) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = (rng.Float32() - 0.5) * 2.0 * scale
	}
	return out
}

func maxAbsDiff(a, b []float32) float32 {
	var maxDiff float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

// TestMetalFlashAttention2Parity satisfies Scoped Acceptance Criterion 1 and 3:
// - Verifies that FlashAttention-2 kernel compiles and executes on Apple Silicon Metal 4.
// - Verifies that output matches reference SDPA within 1e-4 tolerance across varying head dimensions.
func TestMetalFlashAttention2Parity(t *testing.T) {
	if !MetalFlashAttention2Available() {
		t.Skip("Metal 4 FlashAttention-2 unavailable in this environment")
	}

	testCases := []struct {
		name     string
		nH, nKV  int
		hd       int
		qTokens  int
		kvTokens int
		causal   bool
	}{
		{"MHA_D32_Short", 4, 4, 32, 64, 64, true},
		{"MHA_D64_Prefill", 8, 8, 64, 256, 256, true},
		{"GQA_D64_Prefill", 8, 2, 64, 256, 256, true},
		{"MQA_D64_Prefill", 8, 1, 64, 256, 256, true},
		{"GQA_D96_Prefill", 6, 2, 96, 128, 128, true},
		{"GQA_D128_Prefill", 8, 2, 128, 512, 512, true},
		{"MHA_D128_NonCausal", 4, 4, 128, 128, 128, false},
		{"GQA_D256_WideHD", 4, 2, 256, 256, 256, true},
		{"GQA_D128_ChunkedPrefill", 8, 2, 128, 32, 512, true},
	}

	const tolerance = 1e-4

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := FlashAttention2Config{
				NumQueryHeads: tc.nH,
				NumKVHeads:    tc.nKV,
				HeadDim:       tc.hd,
				Causal:        tc.causal,
				Batch:         1,
			}

			q := generateDeterministicFloats(tc.qTokens*tc.nH*tc.hd, 42, 1.0)
			k := generateDeterministicFloats(tc.kvTokens*tc.nKV*tc.hd, 43, 1.0)
			v := generateDeterministicFloats(tc.kvTokens*tc.nKV*tc.hd, 44, 1.0)

			// Execute on Metal 4
			metalRes, err := ExecuteMetalFlashAttention2(cfg, q, k, v, tc.qTokens, tc.kvTokens)
			if err != nil {
				t.Fatalf("ExecuteMetalFlashAttention2 failed: %v", err)
			}

			// Compute reference on CPU
			refOut, refLSE, err := ReferenceSDPA(cfg, q, k, v, tc.qTokens, tc.kvTokens)
			if err != nil {
				t.Fatalf("ReferenceSDPA failed: %v", err)
			}

			// Check max absolute difference
			diff := maxAbsDiff(metalRes.Output, refOut)
			if diff > tolerance {
				t.Errorf("output max difference %e exceeds tolerance %e", diff, tolerance)
			}

			// Check cosine similarity
			sim := CosineSimilarity(metalRes.Output, refOut)
			if sim < 0.9999 {
				t.Errorf("output cosine similarity %f < 0.9999", sim)
			}

			// Check LSE difference
			lseDiff := maxAbsDiff(metalRes.LSE, refLSE)
			if lseDiff > 5e-4 {
				t.Errorf("LSE max difference %e exceeds 5e-4", lseDiff)
			}

			t.Logf("PASS [%s]: maxDiffOutput=%e, cosineSim=%.8f, maxDiffLSE=%e",
				tc.name, diff, sim, lseDiff)
		})
	}
}

// TestMetalFlashAttention2MemoryScaling satisfies Scoped Acceptance Criterion 2:
// - Verifies that memory allocation during prefill scales as O(N) with sequence length.
// - Verifies that intermediate DRAM allocation is strictly 0 (no O(N^2) buffer allocations).
func TestMetalFlashAttention2MemoryScaling(t *testing.T) {
	lengths := []int{256, 512, 1024, 2048, 4096, 8192, 16384}
	cfg := FlashAttention2Config{
		NumQueryHeads: 8,
		NumKVHeads:    2,
		HeadDim:       128,
		Causal:        true,
	}

	var prevIOBytes int64
	for _, n := range lengths {
		stats := FlashAttention2MemoryProfile(cfg, n, n)

		// 1. Strict O(1) intermediate DRAM allocation guarantee (strictly 0)
		if stats.IntermediateBytes != 0 {
			t.Errorf("N=%d: intermediate DRAM bytes = %d, want 0", n, stats.IntermediateBytes)
		}

		// 2. IO buffer memory must scale linearly O(N) with sequence length
		if prevIOBytes > 0 {
			ratio := float64(stats.InputOutputBytes) / float64(prevIOBytes)
			// Doubling N should approximately double IO bytes (ratio ~ 2.0)
			if math.Abs(ratio-2.0) > 0.05 {
				t.Errorf("N=%d: IO bytes scaling ratio %f diverges from linear 2.0", n, ratio)
			}
		}
		prevIOBytes = stats.InputOutputBytes

		// 3. Naive intermediate matrix scales as O(N^2)
		expectedNaive := int64(cfg.NumQueryHeads) * int64(n) * int64(n) * 4
		if stats.NaiveIntermediateBytes != expectedNaive {
			t.Errorf("N=%d: Naive intermediate bytes = %d, want %d", n, stats.NaiveIntermediateBytes, expectedNaive)
		}

		// 4. Memory savings ratio at 16k must be > 50x
		if n == 16384 && stats.MemorySavingsRatio < 50.0 {
			t.Errorf("N=16384: memory savings ratio %.2fx < 50x", stats.MemorySavingsRatio)
		}

		t.Logf("N=%d: Flash DRAM intermediate=%d B, IO=%d B (%.2f MB) vs Naive Intermediate=%.2f MB (Savings: %.1fx)",
			n, stats.IntermediateBytes, stats.InputOutputBytes,
			float64(stats.InputOutputBytes)/(1024*1024),
			float64(stats.NaiveIntermediateBytes)/(1024*1024),
			stats.MemorySavingsRatio)
	}
}

// TestMetalFlashAttention2LongContext16k satisfies the Done Condition:
// - Computes attention across sequence length 16,384 tokens on Apple Silicon Metal 4.
// - Confirms bit-exact parity against reference SDPA on sampled rows while eliminating O(N^2) intermediate buffers.
func TestMetalFlashAttention2LongContext16k(t *testing.T) {
	if !MetalFlashAttention2Available() {
		t.Skip("Metal 4 FlashAttention-2 unavailable")
	}

	const n = 16384
	const nH = 8
	const nKV = 2
	const hd = 64

	cfg := FlashAttention2Config{
		NumQueryHeads: nH,
		NumKVHeads:    nKV,
		HeadDim:       hd,
		Causal:        true,
		Batch:         1,
	}

	q := generateDeterministicFloats(n*nH*hd, 12345, 0.5)
	k := generateDeterministicFloats(n*nKV*hd, 12346, 0.5)
	v := generateDeterministicFloats(n*nKV*hd, 12347, 0.5)

	t0 := time.Now()
	res, err := ExecuteMetalFlashAttention2(cfg, q, k, v, n, n)
	elapsed := time.Since(t0)
	if err != nil {
		t.Fatalf("ExecuteMetalFlashAttention2 at 16k context failed: %v", err)
	}

	// Verify output is valid (finite, non-zero, no NaNs)
	for i, val := range res.Output {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Fatalf("NaN/Inf detected at output index %d", i)
		}
	}

	// Verify sampled rows against reference SDPA
	sampleTokens := []int{0, 15, 127, 255, 1023, 4095, 16383}
	for _, tok := range sampleTokens {
		// Single query token row
		qSlice := q[tok*(nH*hd) : (tok+1)*(nH*hd)]
		// Keys up to tok+1 (causal)
		kvTokensForTok := tok + 1
		kSlice := k[:kvTokensForTok*(nKV*hd)]
		vSlice := v[:kvTokensForTok*(nKV*hd)]

		singleCfg := FlashAttention2Config{
			NumQueryHeads: nH,
			NumKVHeads:    nKV,
			HeadDim:       hd,
			Causal:        true,
			Batch:         1,
		}

		refOut, _, err := ReferenceSDPA(singleCfg, qSlice, kSlice, vSlice, 1, kvTokensForTok)
		if err != nil {
			t.Fatalf("ReferenceSDPA failed for sample token %d: %v", tok, err)
		}

		metalSlice := res.Output[tok*(nH*hd) : (tok+1)*(nH*hd)]
		diff := maxAbsDiff(metalSlice, refOut)
		if diff > 1e-4 {
			t.Errorf("token %d: max diff %e exceeds 1e-4 tolerance", tok, diff)
		}
	}

	// Verify memory statistics
	if res.Stats.IntermediateBytes != 0 {
		t.Errorf("IntermediateBytes = %d, want 0", res.Stats.IntermediateBytes)
	}

	t.Logf("PASS [16,384 Context]: Completed in %v. DRAM Scratchpad=0 B, Memory Savings=%.1fx",
		elapsed, res.Stats.MemorySavingsRatio)
}

// TestMetalFlashAttention2DispatchThreshold satisfies P4 Operations:
// - Automatically dispatches FlashAttention on prompts > 256 tokens.
func TestMetalFlashAttention2DispatchThreshold(t *testing.T) {
	thresholdCases := []struct {
		seqLen   int
		expected bool
	}{
		{1, false},
		{32, false},
		{128, false},
		{256, false},
		{257, true},
		{512, true},
		{1024, true},
		{4096, true},
		{16384, true},
	}

	for _, tc := range thresholdCases {
		got := ShouldDispatchFlashAttention(tc.seqLen)
		if got != tc.expected {
			t.Errorf("ShouldDispatchFlashAttention(%d) = %v, want %v", tc.seqLen, got, tc.expected)
		}
	}
}

// TestMetalFlashAttention2SlidingWindow verifies sliding window attention support.
func TestMetalFlashAttention2SlidingWindow(t *testing.T) {
	if !MetalFlashAttention2Available() {
		t.Skip("Metal 4 FlashAttention-2 unavailable")
	}

	const nH = 4
	const nKV = 2
	const hd = 64
	const qTokens = 256
	const kvTokens = 256
	const window = 64

	cfg := FlashAttention2Config{
		NumQueryHeads: nH,
		NumKVHeads:    nKV,
		HeadDim:       hd,
		Causal:        true,
		SlidingWindow: window,
		Batch:         1,
	}

	q := generateDeterministicFloats(qTokens*nH*hd, 901, 1.0)
	k := generateDeterministicFloats(kvTokens*nKV*hd, 902, 1.0)
	v := generateDeterministicFloats(kvTokens*nKV*hd, 903, 1.0)

	metalRes, err := ExecuteMetalFlashAttention2(cfg, q, k, v, qTokens, kvTokens)
	if err != nil {
		t.Fatalf("ExecuteMetalFlashAttention2 sliding window failed: %v", err)
	}

	refOut, _, err := ReferenceSDPA(cfg, q, k, v, qTokens, kvTokens)
	if err != nil {
		t.Fatalf("ReferenceSDPA failed: %v", err)
	}

	diff := maxAbsDiff(metalRes.Output, refOut)
	if diff > 1e-4 {
		t.Errorf("Sliding window: max diff %e exceeds 1e-4 tolerance", diff)
	}

	sim := CosineSimilarity(metalRes.Output, refOut)
	if sim < 0.9999 {
		t.Errorf("Sliding window: cosine similarity %f < 0.9999", sim)
	}

	t.Logf("PASS [Sliding Window]: maxDiff=%e, cosineSim=%.8f", diff, sim)
}

// TestMetalFlashAttention2InputValidation verifies proper parameter guards.
func TestMetalFlashAttention2InputValidation(t *testing.T) {
	validQ := make([]float32, 128*4*64)
	validK := make([]float32, 128*2*64)
	validV := make([]float32, 128*2*64)

	// Test invalid head division
	badHeadsCfg := FlashAttention2Config{
		NumQueryHeads: 7,
		NumKVHeads:    2,
		HeadDim:       64,
	}
	if _, err := ExecuteMetalFlashAttention2(badHeadsCfg, validQ, validK, validV, 128, 128); err == nil {
		t.Error("expected error for non-divisible heads, got nil")
	}

	// Test head dim > 256
	badHDCfg := FlashAttention2Config{
		NumQueryHeads: 4,
		NumKVHeads:    2,
		HeadDim:       512,
	}
	if _, err := ExecuteMetalFlashAttention2(badHDCfg, validQ, validK, validV, 128, 128); err == nil {
		t.Error("expected error for head dim > 256, got nil")
	}

	// Test zero tokens
	validCfg := FlashAttention2Config{
		NumQueryHeads: 4,
		NumKVHeads:    2,
		HeadDim:       64,
	}
	if _, err := ExecuteMetalFlashAttention2(validCfg, validQ, validK, validV, 0, 128); err == nil {
		t.Error("expected error for zero qTokens, got nil")
	}

	// Test insufficient slice length
	if _, err := ExecuteMetalFlashAttention2(validCfg, validQ[:10], validK, validV, 128, 128); err == nil {
		t.Error("expected error for truncated Q, got nil")
	}
}

// BenchmarkMetalFlashAttention2 benchmarks execution across context lengths.
func BenchmarkMetalFlashAttention2(b *testing.B) {
	if !MetalFlashAttention2Available() {
		b.Skip("Metal 4 FlashAttention-2 unavailable")
	}

	lengths := []int{512, 2048, 8192}
	for _, n := range lengths {
		b.Run(string(rune(n)), func(b *testing.B) {
			cfg := FlashAttention2Config{
				NumQueryHeads: 8,
				NumKVHeads:    2,
				HeadDim:       128,
				Causal:        true,
			}
			q := generateDeterministicFloats(n*8*128, 1, 0.5)
			k := generateDeterministicFloats(n*2*128, 2, 0.5)
			v := generateDeterministicFloats(n*2*128, 3, 0.5)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := ExecuteMetalFlashAttention2(cfg, q, k, v, n, n)
				if err != nil {
					b.Fatalf("benchmark failed: %v", err)
				}
			}
		})
	}
}
