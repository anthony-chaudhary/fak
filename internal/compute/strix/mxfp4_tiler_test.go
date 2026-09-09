// package strix validates OCP MXFP4 micro-scaling, 32MB MALL Infinity Cache tiling,
// attention sink protection, and GFX1151 hardware performance physics on AMD Strix Halo.
package strix

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

// TestMXFP4Tiler is the root test runner executing all scoped acceptance criteria subtests.
func TestMXFP4Tiler(t *testing.T) {
	t.Run("ExactSizing", testExactSizing)
	t.Run("CapacityCalculation", testCapacityCalculation)
	t.Run("RoundTripFidelity", testRoundTripFidelity)
	t.Run("AttentionSinkProtection", testAttentionSinkProtection)
	t.Run("FallbackMode", testFallbackMode)
	t.Run("CachePolicyHints", testCachePolicyHints)
	t.Run("ConcurrentSafety", testConcurrentSafety)
	t.Run("HardwareWitnessSimulation", testHardwareWitnessSimulation)
}

// testExactSizing verifies exact 1,088-byte per-token footprint under OCP MXFP4 micro-scaling.
// Criteria 2: [SW-VERIFIED] MXFP4 tiler achieves exactly 1,088 bytes per token for 8 KV heads,
// head dimension 128 (3.764x compression vs 4,096 bytes FP16).
func testExactSizing(t *testing.T) {
	// 1. Structural constants
	if GQAKVHeads != 8 {
		t.Fatalf("expected 8 KV heads, got %d", GQAKVHeads)
	}
	if GQAHeadDim != 128 {
		t.Fatalf("expected 128 head dim, got %d", GQAHeadDim)
	}
	if KeyElementsPerToken != 1024 {
		t.Fatalf("expected 1024 key elements, got %d", KeyElementsPerToken)
	}
	if ValueElementsPerToken != 1024 {
		t.Fatalf("expected 1024 value elements, got %d", ValueElementsPerToken)
	}
	if ElementsPerToken != 2048 {
		t.Fatalf("expected 2048 elements per token, got %d", ElementsPerToken)
	}

	// 2. OCP MXFP4 block-32 dimensions
	if MicroBlockElements != 32 {
		t.Fatalf("expected 32 elements per micro-block, got %d", MicroBlockElements)
	}
	if MicroBlockPayloadBytes != 16 {
		t.Fatalf("expected 16 payload bytes per micro-block, got %d", MicroBlockPayloadBytes)
	}
	if MicroBlockScaleBytes != 1 {
		t.Fatalf("expected 1 scale byte per micro-block, got %d", MicroBlockScaleBytes)
	}
	if MicroBlockSizeBytes != 17 {
		t.Fatalf("expected 17 bytes per micro-block, got %d", MicroBlockSizeBytes)
	}

	// 3. Blocks per token
	if KeyBlocksPerToken != 32 {
		t.Fatalf("expected 32 key blocks per token, got %d", KeyBlocksPerToken)
	}
	if ValueBlocksPerToken != 32 {
		t.Fatalf("expected 32 value blocks per token, got %d", ValueBlocksPerToken)
	}
	if BlocksPerToken != 64 {
		t.Fatalf("expected 64 blocks per token, got %d", BlocksPerToken)
	}

	// 4. Exact per-token footprint
	expectedBytesPerToken := 64 * 17 // 1,088
	if MXFP4BytesPerToken != expectedBytesPerToken {
		t.Fatalf("expected %d bytes per token, got %d", expectedBytesPerToken, MXFP4BytesPerToken)
	}

	// 5. Compression ratio vs FP16 (4,096 bytes)
	expectedFP16Bytes := 2 * 8 * 128 * 2 // 4,096
	if FP16BytesPerToken != expectedFP16Bytes {
		t.Fatalf("expected %d FP16 bytes per token, got %d", expectedFP16Bytes, FP16BytesPerToken)
	}
	ratio := float64(FP16BytesPerToken) / float64(MXFP4BytesPerToken)
	if math.Abs(ratio-3.76470588) > 0.001 {
		t.Fatalf("expected compression ratio ~3.7647x, got %.6fx", ratio)
	}

	// 6. Effective bits per weight
	bpw := float64(MXFP4BytesPerToken*8) / float64(ElementsPerToken)
	if math.Abs(bpw-4.25) > 1e-9 {
		t.Fatalf("expected 4.25 bits/weight, got %.4f", bpw)
	}

	// 7. Token frame serialization sizing
	key := make([]float32, KeyElementsPerToken)
	val := make([]float32, ValueElementsPerToken)
	frame, err := PackTokenFrame(0, key, val)
	if err != nil {
		t.Fatalf("PackTokenFrame failed: %v", err)
	}
	raw := frame.RawBytes()
	if len(raw) != 1088 {
		t.Fatalf("expected serialized frame length 1088, got %d", len(raw))
	}
}

// testCapacityCalculation proves exactly 30,840 tokens fit within physical 33,554,432 bytes (32MB).
// Criteria 3: [SW-VERIFIED] Capacity calculator proves exactly 30,840 tokens fit within physical
// 33,554,432 bytes (32MB) without buffer overflow.
func testCapacityCalculation(t *testing.T) {
	mallSize := MALLSizeBytes // 33,554,432 bytes

	// 1. Hardware topology verification
	expectedTopologyBytes := int64(MALLSets) * int64(MALLWays) * int64(MALLLineBytes)
	if expectedTopologyBytes != mallSize {
		t.Fatalf("MALL topology calculation mismatch: %d != %d", expectedTopologyBytes, mallSize)
	}

	// 2. Pure MXFP4 capacity
	maxTokens := int(mallSize / int64(MXFP4BytesPerToken))
	if maxTokens != 30840 {
		t.Fatalf("expected exactly 30,840 tokens in 32MB MALL under MXFP4, got %d", maxTokens)
	}
	if MXFP4MaxMALLTokens != 30840 {
		t.Fatalf("expected MXFP4MaxMALLTokens == 30840, got %d", MXFP4MaxMALLTokens)
	}

	// 3. Exact byte allocation and headroom check
	allocatedBytes := int64(maxTokens) * int64(MXFP4BytesPerToken) // 30,840 * 1,088 = 33,553,920
	if allocatedBytes != 33553920 {
		t.Fatalf("expected 33,553,920 allocated bytes, got %d", allocatedBytes)
	}
	headroom := mallSize - allocatedBytes
	if headroom != 512 {
		t.Fatalf("expected exactly 512 bytes headroom, got %d", headroom)
	}
	if MALLHeadroomBytes != 512 {
		t.Fatalf("expected MALLHeadroomBytes == 512, got %d", MALLHeadroomBytes)
	}

	// 4. Overflow proof: 30,841 tokens MUST exceed 32MB
	overflowBytes := int64(maxTokens+1) * int64(MXFP4BytesPerToken) // 33,555,008
	if overflowBytes <= mallSize {
		t.Fatalf("expected 30,841 tokens to overflow 32MB MALL, but %d <= %d", overflowBytes, mallSize)
	}
	excess := overflowBytes - mallSize
	if excess != 576 {
		t.Fatalf("expected 576 bytes overflow for 30,841 tokens, got %d", excess)
	}

	// 5. Uncompressed FP16 capacity verification
	fp16Max := int(mallSize / int64(FP16BytesPerToken))
	if fp16Max != 8192 {
		t.Fatalf("expected 8,192 tokens in 32MB MALL under FP16, got %d", fp16Max)
	}
	if FP16MaxMALLTokens != 8192 {
		t.Fatalf("expected FP16MaxMALLTokens == 8192, got %d", FP16MaxMALLTokens)
	}

	// 6. Capacity expansion multiplier
	expansion := float64(maxTokens) / float64(fp16Max)
	if math.Abs(expansion-3.764648) > 0.001 {
		t.Fatalf("expected capacity expansion ~3.765x, got %.6fx", expansion)
	}

	// 7. Tiler capacity reporting
	cfgPure := DefaultMXFP4Config()
	cfgPure.SinkProtection = false
	tilerPure := NewMXFP4Tiler(cfgPure)
	if tilerPure.CapacityTokens() != 30840 {
		t.Fatalf("expected tiler capacity 30840, got %d", tilerPure.CapacityTokens())
	}
	if tilerPure.HeadroomBytes() != 512 {
		t.Fatalf("expected tiler headroom 512, got %d", tilerPure.HeadroomBytes())
	}

	// 8. Tiler with sink protection (128 FP16 tokens + remaining MXFP4)
	cfgSink := DefaultMXFP4Config()
	cfgSink.SinkProtection = true
	cfgSink.SinkTokens = 128
	tilerSink := NewMXFP4Tiler(cfgSink)
	// 128 * 4,096 = 524,288 bytes
	// Remaining: 33,554,432 - 524,288 = 33,030,144 bytes
	// MXFP4 tokens: floor(33,030,144 / 1,088) = 30,358 tokens
	// Total: 128 + 30,358 = 30,486 tokens
	expectedSinkCap := 128 + int((mallSize-128*4096)/1088)
	if tilerSink.CapacityTokens() != expectedSinkCap {
		t.Fatalf("expected sink-protected capacity %d, got %d", expectedSinkCap, tilerSink.CapacityTokens())
	}
	if tilerSink.CapacityTokens() < 30000 {
		t.Fatalf("expected sink-protected capacity >= 30,000, got %d", tilerSink.CapacityTokens())
	}
}

// testRoundTripFidelity tests quantization and dequantization numerical accuracy.
// Checks cosine similarity >= 0.99 for typical transformer KV activations.
func testRoundTripFidelity(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	generateActivations := func(dim int, scale float64) []float32 {
		v := make([]float32, dim)
		for i := range v {
			// Gaussian-like activation distribution
			u1 := rng.Float64()
			u2 := rng.Float64()
			for u1 <= 1e-15 {
				u1 = rng.Float64()
			}
			z0 := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
			v[i] = float32(z0 * scale)
		}
		return v
	}

	testScales := []float64{0.1, 0.5, 1.0, 2.5, 8.0, 32.0}

	for _, s := range testScales {
		keyOrig := generateActivations(KeyElementsPerToken, s)
		valOrig := generateActivations(ValueElementsPerToken, s)

		frame, err := PackTokenFrame(0, keyOrig, valOrig)
		if err != nil {
			t.Fatalf("scale %.2f: PackTokenFrame failed: %v", s, err)
		}

		keyDequant, valDequant, err := UnpackTokenFrame(frame)
		if err != nil {
			t.Fatalf("scale %.2f: UnpackTokenFrame failed: %v", s, err)
		}

		simK, err := ComputeCosineSimilarity(keyOrig, keyDequant)
		if err != nil {
			t.Fatalf("scale %.2f: ComputeCosineSimilarity Key failed: %v", s, err)
		}
		if simK < 0.99 {
			t.Fatalf("scale %.2f: Key cosine similarity %.6f < 0.99 threshold", s, simK)
		}

		simV, err := ComputeCosineSimilarity(valOrig, valDequant)
		if err != nil {
			t.Fatalf("scale %.2f: ComputeCosineSimilarity Value failed: %v", s, err)
		}
		if simV < 0.99 {
			t.Fatalf("scale %.2f: Value cosine similarity %.6f < 0.99 threshold", s, simV)
		}

		// Verify binary serialization round-trip fidelity
		raw := frame.RawBytes()
		var restoredFrame MXFP4TokenFrame
		if err := restoredFrame.FromBytes(raw); err != nil {
			t.Fatalf("scale %.2f: FromBytes failed: %v", s, err)
		}
		kRestored, vRestored, err := UnpackTokenFrame(&restoredFrame)
		if err != nil {
			t.Fatalf("scale %.2f: UnpackTokenFrame on restored failed: %v", s, err)
		}
		simRestoredK, _ := ComputeCosineSimilarity(keyOrig, kRestored)
		simRestoredV, _ := ComputeCosineSimilarity(valOrig, vRestored)
		if simRestoredK != simK || simRestoredV != simV {
			t.Fatalf("scale %.2f: binary serialization round-trip altered dequantized values", s)
		}
	}
}

// testAttentionSinkProtection tests that initial 128 tokens are preserved in unquantized FP16.
func testAttentionSinkProtection(t *testing.T) {
	cfg := DefaultMXFP4Config()
	cfg.SinkProtection = true
	cfg.SinkTokens = 128
	tiler := NewMXFP4Tiler(cfg)

	rng := rand.New(rand.NewSource(101))
	makeRandom := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * 2.0)
		}
		return v
	}

	// 1. Verify attention sink tagging for tokens 0..127
	for i := 0; i < 128; i++ {
		if !tiler.IsSinkToken(i) {
			t.Fatalf("token %d should be identified as attention sink", i)
		}
	}
	if tiler.IsSinkToken(128) {
		t.Fatalf("token 128 should NOT be an attention sink")
	}

	// 2. Tile sink token (token 0)
	key0 := makeRandom(KeyElementsPerToken)
	val0 := makeRandom(ValueElementsPerToken)
	if err := tiler.TileToken(0, key0, val0); err != nil {
		t.Fatalf("TileToken for sink token 0 failed: %v", err)
	}

	// Read sink token back: must be marked sink and have 100% exact numerical match (cosine similarity = 1.0)
	kRead, vRead, isSink, err := tiler.ReadToken(0)
	if err != nil {
		t.Fatalf("ReadToken for sink token 0 failed: %v", err)
	}
	if !isSink {
		t.Fatalf("expected token 0 to be marked as sink")
	}
	simK, _ := ComputeCosineSimilarity(key0, kRead)
	simV, _ := ComputeCosineSimilarity(val0, vRead)
	if math.Abs(simK-1.0) > 1e-6 || math.Abs(simV-1.0) > 1e-6 {
		t.Fatalf("attention sink token had quantization error: simK=%.8f, simV=%.8f", simK, simV)
	}

	// 3. Tile non-sink token (token 128): should be packed into MXFP4
	key128 := makeRandom(KeyElementsPerToken)
	val128 := makeRandom(ValueElementsPerToken)
	if err := tiler.TileToken(128, key128, val128); err != nil {
		t.Fatalf("TileToken for token 128 failed: %v", err)
	}

	kRead128, vRead128, isSink128, err := tiler.ReadToken(128)
	if err != nil {
		t.Fatalf("ReadToken for token 128 failed: %v", err)
	}
	if isSink128 {
		t.Fatalf("expected token 128 to NOT be marked as sink")
	}
	simK128, _ := ComputeCosineSimilarity(key128, kRead128)
	simV128, _ := ComputeCosineSimilarity(val128, vRead128)
	if simK128 < 0.99 || simV128 < 0.99 {
		t.Fatalf("token 128 similarity below 0.99: simK=%.6f, simV=%.6f", simK128, simV128)
	}

	// 4. Verify telemetry attention sink count
	tel := tiler.Telemetry()
	if tel.AttentionSinkCount != 1 {
		t.Fatalf("expected 1 attention sink in telemetry, got %d", tel.AttentionSinkCount)
	}
	if tel.ResidencyTokens != 2 {
		t.Fatalf("expected 2 residency tokens, got %d", tel.ResidencyTokens)
	}
}

// testFallbackMode verifies smooth fallback to FP16 linear tiling (8,192 token limit).
func testFallbackMode(t *testing.T) {
	tiler := NewMXFP4Tiler()
	tiler.SetFallback(true)

	if !tiler.IsFallback() {
		t.Fatalf("expected fallback active")
	}
	if tiler.CapacityTokens() != 8192 {
		t.Fatalf("expected capacity 8192 in fallback mode, got %d", tiler.CapacityTokens())
	}
	if tiler.BytesPerToken() != 4096 {
		t.Fatalf("expected 4,096 bytes/token in fallback mode, got %d", tiler.BytesPerToken())
	}

	rng := rand.New(rand.NewSource(202))
	key := make([]float32, KeyElementsPerToken)
	val := make([]float32, ValueElementsPerToken)
	for i := range key {
		key[i] = float32(rng.NormFloat64())
		val[i] = float32(rng.NormFloat64())
	}

	// Tile token within 8k window
	if err := tiler.TileToken(8191, key, val); err != nil {
		t.Fatalf("TileToken(8191) failed in fallback mode: %v", err)
	}

	// Read token back
	kRead, vRead, _, err := tiler.ReadToken(8191)
	if err != nil {
		t.Fatalf("ReadToken(8191) failed: %v", err)
	}
	simK, _ := ComputeCosineSimilarity(key, kRead)
	simV, _ := ComputeCosineSimilarity(val, vRead)
	if math.Abs(simK-1.0) > 1e-6 || math.Abs(simV-1.0) > 1e-6 {
		t.Fatalf("fallback mode should preserve exact FP16 float fidelity")
	}

	// Attempting to tile beyond 8,192 must return ErrCapacityExceeded in fallback mode
	errOverflow := tiler.TileToken(8192, key, val)
	if errOverflow == nil {
		t.Fatalf("expected ErrCapacityExceeded when tiling token 8192 in fallback mode")
	}

	// Telemetry check
	tel := tiler.Telemetry()
	if !tel.FallbackActive {
		t.Fatalf("telemetry should report fallback active")
	}
	if tel.CompressionRatio != 1.0 {
		t.Fatalf("expected compression ratio 1.0 in fallback mode, got %.2f", tel.CompressionRatio)
	}
	if tel.EffectiveBPW != 16.0 {
		t.Fatalf("expected 16 bpw in fallback mode, got %.2f", tel.EffectiveBPW)
	}

	// Smooth reversion back to MXFP4
	tiler.SetFallback(false)
	if tiler.IsFallback() {
		t.Fatalf("expected fallback inactive after reset")
	}
	if tiler.CapacityTokens() <= 8192 {
		t.Fatalf("expected capacity > 8192 after reverting to MXFP4, got %d", tiler.CapacityTokens())
	}
}

// testCachePolicyHints verifies RDNA 3.5 cache allocation directives.
func testCachePolicyHints(t *testing.T) {
	tiler := NewMXFP4Tiler()
	capTokens := tiler.CapacityTokens() // 30,486 with sink or 30,840 pure

	// 1. Token within capacity: CacheHintTemporalPinned (SLC=0, GLC=0, NT=0)
	hintWithin := tiler.ClassifyToken(100)
	if !hintWithin.Temporal || hintWithin.Bypass || hintWithin.SLC != 0 || hintWithin.NT != 0 {
		t.Fatalf("expected CacheHintTemporalPinned for token 100: %+v", hintWithin)
	}

	// 2. Token beyond capacity: CacheHintStreamingBypass (SLC=1, GLC=1, NT=1)
	hintBeyond := tiler.ClassifyToken(capTokens + 500)
	if hintBeyond.Temporal || !hintBeyond.Bypass || hintBeyond.SLC != 1 || hintBeyond.NT != 1 {
		t.Fatalf("expected CacheHintStreamingBypass for token %d: %+v", capTokens+500, hintBeyond)
	}

	// 3. Span within capacity
	spanWithin := tiler.ClassifyTokenSpan(0, 1000)
	if !spanWithin.Temporal || spanWithin.Bypass {
		t.Fatalf("expected temporal hint for span [0, 1000): %+v", spanWithin)
	}

	// 4. Span crossing capacity boundary
	spanCross := tiler.ClassifyTokenSpan(capTokens-500, capTokens+500)
	if spanCross.PolicyName != "PARTITIONED" || !spanCross.Temporal || !spanCross.Bypass {
		t.Fatalf("expected PARTITIONED hint for span crossing capacity: %+v", spanCross)
	}
}

// testConcurrentSafety tests race-free concurrent tiling and retrieval across multiple goroutines.
func testConcurrentSafety(t *testing.T) {
	tiler := NewMXFP4Tiler()
	numWorkers := 16
	tokensPerWorker := 32

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(workerID * 1000)))

			for i := 0; i < tokensPerWorker; i++ {
				tokenIndex := workerID*tokensPerWorker + i
				k := make([]float32, KeyElementsPerToken)
				v := make([]float32, ValueElementsPerToken)
				for j := range k {
					k[j] = float32(rng.Float64())
					v[j] = float32(rng.Float64())
				}

				if err := tiler.TileToken(tokenIndex, k, v); err != nil {
					t.Errorf("worker %d: TileToken(%d) error: %v", workerID, tokenIndex, err)
					return
				}

				kRead, vRead, _, err := tiler.ReadToken(tokenIndex)
				if err != nil {
					t.Errorf("worker %d: ReadToken(%d) error: %v", workerID, tokenIndex, err)
					return
				}
				if len(kRead) != KeyElementsPerToken || len(vRead) != ValueElementsPerToken {
					t.Errorf("worker %d: unexpected vector lengths", workerID)
					return
				}

				_ = tiler.Telemetry()
				_ = tiler.ClassifyToken(tokenIndex)
			}
		}()
	}

	wg.Wait()

	tel := tiler.Telemetry()
	expectedResident := numWorkers * tokensPerWorker
	if tel.ResidencyTokens != expectedResident {
		t.Fatalf("expected %d resident tokens, got %d", expectedResident, tel.ResidencyTokens)
	}
}

// testHardwareWitnessSimulation tests the GFX1151 appliance model for 30,000-token contexts.
// Scoped Acceptance Criteria 4:
// - [SW-VERIFIED] Simulator confirms >= 90% residency and >= 3x effective bandwidth vs DRAM-spilled FP16.
// - [HW-WITNESSED] Clearly marks physical criteria as requiring live appliance probe.
func testHardwareWitnessSimulation(t *testing.T) {
	tiler := NewMXFP4Tiler()
	const contextTokens = 30000

	eval := tiler.EvaluateGFX1151Residency(contextTokens)

	// Verify SW-VERIFIED metrics:
	// 1. Context tokens
	if eval.ContextTokens != 30000 {
		t.Fatalf("expected 30,000 context tokens, got %d", eval.ContextTokens)
	}

	// 2. FP16 Baseline: 8,192 resident, 21,808 spilled (~27.3% residency)
	if eval.FP16ResidentTokens != 8192 {
		t.Fatalf("expected 8,192 FP16 resident tokens, got %d", eval.FP16ResidentTokens)
	}
	if eval.FP16SpilledTokens != 21808 {
		t.Fatalf("expected 21,808 FP16 spilled tokens, got %d", eval.FP16SpilledTokens)
	}
	if math.Abs(eval.FP16ResidencyRatio-0.273067) > 0.001 {
		t.Fatalf("expected FP16 residency ~27.3%%, got %.2f%%", eval.FP16ResidencyRatio*100)
	}

	// 3. MXFP4: 30,000 resident in 32MB MALL, 0 spilled (100% residency >= 90%)
	if eval.MXFP4ResidentTokens != 30000 {
		t.Fatalf("expected 30,000 MXFP4 resident tokens, got %d", eval.MXFP4ResidentTokens)
	}
	if eval.MXFP4SpilledTokens != 0 {
		t.Fatalf("expected 0 MXFP4 spilled tokens, got %d", eval.MXFP4SpilledTokens)
	}
	if eval.MXFP4ResidencyRatio < 0.90 {
		t.Fatalf("MXFP4 residency %.2f%% < 90%% required threshold", eval.MXFP4ResidencyRatio*100)
	}

	// 4. Bandwidth speedup ratio >= 3.0x vs DRAM-spilled FP16
	if eval.BandwidthSpeedupRatio < 3.0 {
		t.Fatalf("Bandwidth speedup %.2fx < 3.0x required threshold", eval.BandwidthSpeedupRatio)
	}

	// 5. Effective bandwidth checks
	if eval.MXFP4EffectiveBWGBs < 1000.0 {
		t.Fatalf("expected MXFP4 effective bandwidth >= 1000 GB/s, got %.2f GB/s", eval.MXFP4EffectiveBWGBs)
	}

	// 6. Verify bipartite separation discipline
	if !eval.SWVerified {
		t.Fatalf("expected SWVerified == true")
	}
	if !eval.HWWitnessedRequired {
		t.Fatalf("expected HWWitnessedRequired == true")
	}

	t.Logf("[SW-VERIFIED] Simulation passed: %s", eval.StatusMessage)
	t.Logf("[HW-WITNESSED] Live hardware verification pending on Strix Halo appliance (GFX1151)")
}
