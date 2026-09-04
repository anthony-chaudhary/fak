package deepseekv4moe

import (
	"errors"
	"math"
	"testing"
)

// TestPartitionContiguousSpansCorrectness tests partition correctness across 2, 4, and 8 ranks
// for both divisible and non-divisible expert counts.
func TestPartitionContiguousSpansCorrectness(t *testing.T) {
	testCases := []struct {
		name         string
		totalExperts int
		worldSize    int
	}{
		// Divisible configurations
		{"256 experts / 2 ranks (dual Strix Halo)", 256, 2},
		{"256 experts / 4 ranks", 256, 4},
		{"256 experts / 8 ranks", 256, 8},
		{"384 experts / 2 ranks (V4 Pro 2-node TP)", 384, 2},
		{"384 experts / 4 ranks", 384, 4},
		{"384 experts / 8 ranks", 384, 8},
		{"16 experts / 4 ranks", 16, 4},
		{"8 experts / 8 ranks", 8, 8},

		// Non-divisible configurations
		{"256 experts / 3 ranks", 256, 3},
		{"384 experts / 7 ranks", 384, 7},
		{"257 experts / 2 ranks", 257, 2},
		{"257 experts / 4 ranks", 257, 4},
		{"257 experts / 8 ranks", 257, 8},
		{"250 experts / 8 ranks", 250, 8},
		{"13 experts / 4 ranks", 13, 4},
		{"1 expert / 1 rank", 1, 1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			spans, err := PartitionContiguousSpans(tc.totalExperts, tc.worldSize)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(spans) != tc.worldSize {
				t.Fatalf("expected %d spans, got %d", tc.worldSize, len(spans))
			}

			seen := make([]bool, tc.totalExperts)
			expectedBase := 0
			totalAssigned := 0

			for r, span := range spans {
				if err := span.Validate(); err != nil {
					t.Fatalf("rank %d span invalid: %v", r, err)
				}
				if span.Rank != r {
					t.Errorf("expected Rank %d, got %d", r, span.Rank)
				}
				if span.WorldSize != tc.worldSize {
					t.Errorf("expected WorldSize %d, got %d", tc.worldSize, span.WorldSize)
				}
				if span.TotalExperts != tc.totalExperts {
					t.Errorf("expected TotalExperts %d, got %d", tc.totalExperts, span.TotalExperts)
				}

				// Check contiguous property: Base must match end of previous span
				if span.Base != expectedBase {
					t.Errorf("rank %d: expected Base %d, got %d (non-contiguous)", r, expectedBase, span.Base)
				}
				expectedBase += span.Count
				totalAssigned += span.Count

				// Check disjoint coverage and bidirectional indexing
				for localIdx := 0; localIdx < span.Count; localIdx++ {
					globalExp := span.Base + localIdx

					// Contains check
					if !span.Contains(globalExp) {
						t.Errorf("rank %d: Contains(%d) should be true", r, globalExp)
					}

					// LocalIndex check
					lIdx, ok := span.LocalIndex(globalExp)
					if !ok || lIdx != localIdx {
						t.Errorf("rank %d: LocalIndex(%d) = (%d, %v), want (%d, true)", r, globalExp, lIdx, ok, localIdx)
					}

					// GlobalIndex check
					gIdx, ok := span.GlobalIndex(localIdx)
					if !ok || gIdx != globalExp {
						t.Errorf("rank %d: GlobalIndex(%d) = (%d, %v), want (%d, true)", r, localIdx, gIdx, ok, globalExp)
					}

					// Disjointness check
					if seen[globalExp] {
						t.Fatalf("expert %d assigned multiple times", globalExp)
					}
					seen[globalExp] = true
				}

				// Check bounds outside this span
				if span.Base > 0 && span.Contains(span.Base-1) {
					t.Errorf("rank %d: Contains(%d) should be false (below base)", r, span.Base-1)
				}
				if span.Contains(span.Base + span.Count) {
					t.Errorf("rank %d: Contains(%d) should be false (above span)", r, span.Base+span.Count)
				}
				if _, ok := span.LocalIndex(span.Base + span.Count); ok {
					t.Errorf("rank %d: LocalIndex(%d) should return ok=false", r, span.Base+span.Count)
				}
			}

			// Verify total coverage
			if totalAssigned != tc.totalExperts {
				t.Fatalf("total assigned experts %d != totalExperts %d", totalAssigned, tc.totalExperts)
			}
			for exp, covered := range seen {
				if !covered {
					t.Fatalf("expert %d was not covered in partition", exp)
				}
			}
		})
	}
}

// TestPartitionContiguousSpansInvalidInputs tests validation and error reporting.
func TestPartitionContiguousSpansInvalidInputs(t *testing.T) {
	cases := []struct {
		name         string
		totalExperts int
		worldSize    int
		wantErr      error
	}{
		{"zero total experts", 0, 2, ErrExpertCount},
		{"negative total experts", -10, 2, ErrExpertCount},
		{"zero world size", 256, 0, ErrInvalidWorldSize},
		{"negative world size", 256, -4, ErrInvalidWorldSize},
		{"world size exceeds total experts", 8, 16, ErrInvalidWorldSize},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PartitionContiguousSpans(tc.totalExperts, tc.worldSize)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("PartitionContiguousSpans(%d, %d) error = %v, want %v", tc.totalExperts, tc.worldSize, err, tc.wantErr)
			}
		})
	}
}

// TestZeroByteAllocationForUnownedExperts proves that backing storage is allocated ONLY
// for Count resident experts, with strictly 0 bytes allocated for unowned experts.
func TestZeroByteAllocationForUnownedExperts(t *testing.T) {
	const (
		totalExperts   = 256
		worldSize      = 2
		bytesPerExpert = 600 * 1024 * 1024 // ~600 MiB per expert (for arithmetic checks)
		testBufBytes   = 1024              // 1 KiB per expert for actual memory buffer allocation
	)

	spans, err := PartitionContiguousSpans(totalExperts, worldSize)
	if err != nil {
		t.Fatal(err)
	}

	// Rank 0 owns [0, 128); Rank 1 owns [128, 256).
	for _, span := range spans {
		plan, err := LayoutExpertStorage(span, bytesPerExpert)
		if err != nil {
			t.Fatalf("rank %d: failed to plan storage: %v", span.Rank, err)
		}

		// 1. Structural assertions
		if plan.ResidentExperts != 128 {
			t.Errorf("rank %d: expected 128 resident experts, got %d", span.Rank, plan.ResidentExperts)
		}
		if plan.UnownedExperts != 128 {
			t.Errorf("rank %d: expected 128 unowned experts, got %d", span.Rank, plan.UnownedExperts)
		}
		expectedResidentBytes := int64(128) * bytesPerExpert
		if plan.ResidentBytes != expectedResidentBytes {
			t.Errorf("rank %d: resident bytes %d != expected %d", span.Rank, plan.ResidentBytes, expectedResidentBytes)
		}
		if plan.UnownedBytes != 0 {
			t.Errorf("rank %d: unowned bytes must be strictly 0, got %d", span.Rank, plan.UnownedBytes)
		}
		if plan.SavedBytes != expectedResidentBytes {
			t.Errorf("rank %d: saved bytes %d != expected %d", span.Rank, plan.SavedBytes, expectedResidentBytes)
		}
		if math.Abs(plan.FootprintRatio-0.5) > 1e-9 {
			t.Errorf("rank %d: footprint ratio %f != 0.5", span.Rank, plan.FootprintRatio)
		}
		if math.Abs(plan.SavingsRatio-0.5) > 1e-9 {
			t.Errorf("rank %d: savings ratio %f != 0.5", span.Rank, plan.SavingsRatio)
		}

		// 2. Strict zero-allocation verification
		if err := plan.VerifyZeroUnownedAllocation(); err != nil {
			t.Fatalf("rank %d zero allocation verification failed: %v", span.Rank, err)
		}

		for e := 0; e < totalExperts; e++ {
			alloc := plan.AllocatedBytesForExpert(e)
			if span.Contains(e) {
				if alloc != bytesPerExpert {
					t.Errorf("rank %d: resident expert %d allocated %d != %d", span.Rank, e, alloc, bytesPerExpert)
				}
				offset, err := plan.ResidentOffset(e)
				if err != nil {
					t.Errorf("rank %d: unexpected offset error for resident expert %d: %v", span.Rank, e, err)
				}
				expectedOffset := int64(e-span.Base) * bytesPerExpert
				if offset != expectedOffset {
					t.Errorf("rank %d: expert %d offset = %d, want %d", span.Rank, e, offset, expectedOffset)
				}
			} else {
				// Must be exactly 0 bytes allocated
				if alloc != 0 {
					t.Fatalf("rank %d: unowned expert %d allocated %d bytes (MUST BE 0)", span.Rank, e, alloc)
				}
				// Offsets for unowned experts must fail
				if _, err := plan.ResidentOffset(e); !errors.Is(err, ErrUnownedExpert) {
					t.Errorf("rank %d: ResidentOffset(%d) error = %v, want %v", span.Rank, e, err, ErrUnownedExpert)
				}
			}
		}

		// 3. Slicing and buffer allocation test (using testBufBytes to avoid real 75 GB allocation)
		bufPlan, err := LayoutExpertStorage(span, testBufBytes)
		if err != nil {
			t.Fatalf("rank %d: failed to plan buffer storage: %v", span.Rank, err)
		}
		expectedBufBytes := int64(128) * testBufBytes
		residentBuf := bufPlan.AllocateResidentBuffer()
		if int64(len(residentBuf)) != expectedBufBytes {
			t.Fatalf("allocated resident buffer length = %d, want %d", len(residentBuf), expectedBufBytes)
		}

		// Resident experts can be sliced
		for e := span.Base; e < span.Base+span.Count; e++ {
			slice, err := bufPlan.ExpertSlice(residentBuf, e)
			if err != nil {
				t.Fatalf("rank %d: failed to slice resident expert %d: %v", span.Rank, e, err)
			}
			if int64(len(slice)) != testBufBytes {
				t.Errorf("rank %d: slice length for expert %d = %d, want %d", span.Rank, e, len(slice), testBufBytes)
			}
		}

		// Unowned experts cannot be sliced from resident buffer
		for e := 0; e < totalExperts; e++ {
			if !span.Contains(e) {
				slice, err := bufPlan.ExpertSlice(residentBuf, e)
				if !errors.Is(err, ErrUnownedExpert) {
					t.Errorf("rank %d: ExpertSlice on unowned expert %d returned error %v, want %v", span.Rank, e, err, ErrUnownedExpert)
				}
				if slice != nil {
					t.Fatalf("rank %d: ExpertSlice on unowned expert %d returned non-nil slice", span.Rank, e)
				}
			}
		}
	}

	// 4. Test CalculateShardMemorySavings
	full, sharded, ratio := CalculateShardMemorySavings(totalExperts, 128, bytesPerExpert)
	if full != int64(totalExperts)*bytesPerExpert {
		t.Errorf("fullBytes = %d, want %d", full, int64(totalExperts)*bytesPerExpert)
	}
	if sharded != int64(128)*bytesPerExpert {
		t.Errorf("shardedBytes = %d, want %d", sharded, int64(128)*bytesPerExpert)
	}
	if math.Abs(ratio-0.5) > 1e-9 {
		t.Errorf("ratio = %f, want 0.5", ratio)
	}
}

// TestRouterRemapping validates router output remapping for both branchless GPU execution
// (mapping unowned to index 0 with weight 0.0f) and host skip-based execution (sentinel -1).
func TestRouterRemapping(t *testing.T) {
	const (
		totalExperts = 256
		worldSize    = 2
	)

	spans, err := PartitionContiguousSpans(totalExperts, worldSize)
	if err != nil {
		t.Fatal(err)
	}
	span0 := spans[0] // [0, 128)
	span1 := spans[1] // [128, 256)

	// Top-6 router selections: experts [10, 45, 127, 128, 200, 255]
	selected := []int{10, 45, 127, 128, 200, 255}
	weights := []float32{0.35, 0.25, 0.15, 0.10, 0.10, 0.05}

	// --- Test Rank 0 (GPU branchless mode: default index 0, weight 0.0f) ---
	localIdx0, localW0, isLocal0 := RemapRoutedTokens(selected, weights, span0)

	expectedIdx0 := []int{10, 45, 127, 0, 0, 0}
	expectedW0 := []float32{0.35, 0.25, 0.15, 0.0, 0.0, 0.0}
	expectedIsLocal0 := []bool{true, true, true, false, false, false}

	for i := range selected {
		if localIdx0[i] != expectedIdx0[i] {
			t.Errorf("Rank 0 localIdx[%d] = %d, want %d", i, localIdx0[i], expectedIdx0[i])
		}
		if localW0[i] != expectedW0[i] {
			t.Errorf("Rank 0 localW[%d] = %f, want %f", i, localW0[i], expectedW0[i])
		}
		if isLocal0[i] != expectedIsLocal0[i] {
			t.Errorf("Rank 0 isLocal[%d] = %v, want %v", i, isLocal0[i], expectedIsLocal0[i])
		}
	}

	// --- Test Rank 0 with Sentinel (-1 for skipping) ---
	sentinelIdx0, sentinelW0, sentinelIsLocal0 := RemapRoutedTokensWithSentinel(selected, weights, span0, UnownedExpertSentinel)
	expectedSentinelIdx0 := []int{10, 45, 127, -1, -1, -1}
	for i := range selected {
		if sentinelIdx0[i] != expectedSentinelIdx0[i] {
			t.Errorf("Rank 0 sentinelIdx[%d] = %d, want %d", i, sentinelIdx0[i], expectedSentinelIdx0[i])
		}
		if sentinelW0[i] != expectedW0[i] {
			t.Errorf("Rank 0 sentinelW[%d] = %f, want %f", i, sentinelW0[i], expectedW0[i])
		}
		if sentinelIsLocal0[i] != expectedIsLocal0[i] {
			t.Errorf("Rank 0 sentinelIsLocal[%d] = %v, want %v", i, sentinelIsLocal0[i], expectedIsLocal0[i])
		}
	}

	// --- Test Rank 1 (GPU branchless mode: default index 0, weight 0.0f) ---
	localIdx1, localW1, isLocal1 := RemapRoutedTokens(selected, weights, span1)

	// In span1 [128, 256), experts 128, 200, 255 map to 0, 72, 127
	expectedIdx1 := []int{0, 0, 0, 0, 72, 127}
	expectedW1 := []float32{0.0, 0.0, 0.0, 0.10, 0.10, 0.05}
	expectedIsLocal1 := []bool{false, false, false, true, true, true}

	for i := range selected {
		if localIdx1[i] != expectedIdx1[i] {
			t.Errorf("Rank 1 localIdx[%d] = %d, want %d", i, localIdx1[i], expectedIdx1[i])
		}
		if localW1[i] != expectedW1[i] {
			t.Errorf("Rank 1 localW[%d] = %f, want %f", i, localW1[i], expectedW1[i])
		}
		if isLocal1[i] != expectedIsLocal1[i] {
			t.Errorf("Rank 1 isLocal[%d] = %v, want %v", i, isLocal1[i], expectedIsLocal1[i])
		}
	}

	// --- Test Rank 1 with Sentinel ---
	sentinelIdx1, sentinelW1, sentinelIsLocal1 := RemapRoutedTokensWithSentinel(selected, weights, span1, UnownedExpertSentinel)
	expectedSentinelIdx1 := []int{-1, -1, -1, 0, 72, 127}
	for i := range selected {
		if sentinelIdx1[i] != expectedSentinelIdx1[i] {
			t.Errorf("Rank 1 sentinelIdx[%d] = %d, want %d", i, sentinelIdx1[i], expectedSentinelIdx1[i])
		}
		if sentinelW1[i] != expectedW1[i] {
			t.Errorf("Rank 1 sentinelW[%d] = %f, want %f", i, sentinelW1[i], expectedW1[i])
		}
		if sentinelIsLocal1[i] != expectedIsLocal1[i] {
			t.Errorf("Rank 1 sentinelIsLocal[%d] = %v, want %v", i, sentinelIsLocal1[i], expectedIsLocal1[i])
		}
	}
}

// TestShardedMoEMathematicalParity proves that combining outputs from all sharded spans
// produces 100% mathematical parity with un-sharded execution across various rank counts
// and expert configurations.
func TestShardedMoEMathematicalParity(t *testing.T) {
	configs := []struct {
		name         string
		totalExperts int
		worldSize    int
		topK         int
		hiddenDim    int
		numTokens    int
	}{
		{"2 ranks TP / 256 experts (dual Strix Halo)", 256, 2, 6, 32, 20},
		{"4 ranks TP / 256 experts", 256, 4, 6, 32, 20},
		{"8 ranks TP / 256 experts", 256, 8, 8, 32, 20},
		{"2 ranks TP / 384 experts (DeepSeek V4)", 384, 2, 6, 32, 20},
		{"4 ranks TP / 384 experts", 384, 4, 6, 32, 20},
		{"8 ranks TP / 384 experts", 384, 8, 6, 32, 20},
		{"3 ranks TP / 256 experts (non-divisible)", 256, 3, 6, 32, 20},
		{"7 ranks TP / 384 experts (non-divisible)", 384, 7, 6, 32, 20},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			spans, err := PartitionContiguousSpans(cfg.totalExperts, cfg.worldSize)
			if err != nil {
				t.Fatalf("partition error: %v", err)
			}

			// Generate deterministic synthetic weights for all experts: shape [totalExperts][hiddenDim]
			// We model each expert as a deterministic linear transform: y[d] = x[d] * expertScale[e][d].
			expertScales := make([][]float32, cfg.totalExperts)
			for e := 0; e < cfg.totalExperts; e++ {
				expertScales[e] = make([]float32, cfg.hiddenDim)
				for d := 0; d < cfg.hiddenDim; d++ {
					// Deterministic weight derived from expert ID and dimension
					h := mix(uint64(e+1), uint64(d+1))
					expertScales[e][d] = float32(h%1000) / 100.0 // [0.0, 9.99]
				}
			}

			// Resident weights per rank: rank r stores ONLY its resident experts!
			// Backing storage on rank r has size spans[r].Count, not cfg.totalExperts.
			rankResidentScales := make([][][]float32, cfg.worldSize)
			for r := 0; r < cfg.worldSize; r++ {
				rankResidentScales[r] = make([][]float32, spans[r].Count)
				for localIdx := 0; localIdx < spans[r].Count; localIdx++ {
					globalExp := spans[r].Base + localIdx
					rankResidentScales[r][localIdx] = expertScales[globalExp]
				}
			}

			// Run forward pass over tokens
			for tok := 0; tok < cfg.numTokens; tok++ {
				// Synthetic input token vector
				x := make([]float32, cfg.hiddenDim)
				for d := 0; d < cfg.hiddenDim; d++ {
					x[d] = float32(mix(uint64(tok+1), uint64(d*7+1))%500) / 50.0
				}

				// Select top-k experts and softmax-normalized routing weights deterministically
				selected := make([]int, cfg.topK)
				weights := make([]float32, cfg.topK)
				var weightSum float32
				for k := 0; k < cfg.topK; k++ {
					// Pseudo-random expert selection across [0, totalExperts)
					selected[k] = int(mix(uint64(tok+1), uint64(k*31+1)) % uint64(cfg.totalExperts))
					rawW := float32((mix(uint64(tok+7), uint64(k+3))%900)+100) / 1000.0 // (0.1, 1.0]
					weights[k] = rawW
					weightSum += rawW
				}
				for k := 0; k < cfg.topK; k++ {
					weights[k] /= weightSum
				}

				// --- 1. Un-sharded reference execution ---
				unshardedOut := make([]float32, cfg.hiddenDim)
				for k := 0; k < cfg.topK; k++ {
					exp := selected[k]
					w := weights[k]
					scale := expertScales[exp]
					for d := 0; d < cfg.hiddenDim; d++ {
						unshardedOut[d] += w * (x[d] * scale[d])
					}
				}

				// --- 2. Sharded execution across ranks (GPU branchless remap mode) ---
				shardedCombinedGPU := make([]float32, cfg.hiddenDim)
				for r := 0; r < cfg.worldSize; r++ {
					span := spans[r]
					localIndices, localWeights, _ := RemapRoutedTokens(selected, weights, span)
					rankOut := make([]float32, cfg.hiddenDim)

					for k := 0; k < cfg.topK; k++ {
						localIdx := localIndices[k]
						localW := localWeights[k]
						// Even for unowned experts (where localIdx is 0 and localW is 0.0f),
						// the branchless GPU kernel executes without bounds errors.
						scale := rankResidentScales[r][localIdx]
						for d := 0; d < cfg.hiddenDim; d++ {
							rankOut[d] += localW * (x[d] * scale[d])
						}
					}

					// AllReduceSum across ranks
					for d := 0; d < cfg.hiddenDim; d++ {
						shardedCombinedGPU[d] += rankOut[d]
					}
				}

				// --- 3. Sharded execution across ranks (Host skip mode with isLocal) ---
				shardedCombinedHost := make([]float32, cfg.hiddenDim)
				for r := 0; r < cfg.worldSize; r++ {
					span := spans[r]
					localIndices, localWeights, isLocal := RemapRoutedTokensWithSentinel(selected, weights, span, UnownedExpertSentinel)
					rankOut := make([]float32, cfg.hiddenDim)

					for k := 0; k < cfg.topK; k++ {
						if !isLocal[k] {
							continue // host loop skips unowned experts
						}
						localIdx := localIndices[k]
						localW := localWeights[k]
						scale := rankResidentScales[r][localIdx]
						for d := 0; d < cfg.hiddenDim; d++ {
							rankOut[d] += localW * (x[d] * scale[d])
						}
					}

					// AllReduceSum across ranks
					for d := 0; d < cfg.hiddenDim; d++ {
						shardedCombinedHost[d] += rankOut[d]
					}
				}

				// --- 4. Mathematical Parity Verification ---
				// Float32 addition is commutative but non-associative across different rank partition orders;
				// standard float32 tolerance for sum of products is 1e-4 absolute / 1e-5 relative (~1-2 ULP).
				for d := 0; d < cfg.hiddenDim; d++ {
					diffGPU := math.Abs(float64(shardedCombinedGPU[d] - unshardedOut[d]))
					relGPU := diffGPU / (math.Abs(float64(unshardedOut[d])) + 1e-6)
					if diffGPU > 1e-4 && relGPU > 1e-5 {
						t.Fatalf("token %d dim %d: GPU sharded output %f differs from unsharded %f (diff %e, rel %e)",
							tok, d, shardedCombinedGPU[d], unshardedOut[d], diffGPU, relGPU)
					}

					diffHost := math.Abs(float64(shardedCombinedHost[d] - unshardedOut[d]))
					relHost := diffHost / (math.Abs(float64(unshardedOut[d])) + 1e-6)
					if diffHost > 1e-4 && relHost > 1e-5 {
						t.Fatalf("token %d dim %d: Host sharded output %f differs from unsharded %f (diff %e, rel %e)",
							tok, d, shardedCombinedHost[d], unshardedOut[d], diffHost, relHost)
					}
				}
			}
		})
	}
}

// TestShardedMoEExactIntegerBitParity proves that sharded expert dispatch is an exact,
// zero-loss identity: when evaluated in associative integer arithmetic, sharded sum
// equals un-sharded sum with 0 bit difference.
func TestShardedMoEExactIntegerBitParity(t *testing.T) {
	const (
		totalExperts = 256
		worldSize    = 4
		topK         = 6
		hiddenDim    = 16
		numTokens    = 10
	)

	spans, err := PartitionContiguousSpans(totalExperts, worldSize)
	if err != nil {
		t.Fatal(err)
	}

	// Integer expert weights [totalExperts][hiddenDim]
	expertWeights := make([][]int64, totalExperts)
	for e := 0; e < totalExperts; e++ {
		expertWeights[e] = make([]int64, hiddenDim)
		for d := 0; d < hiddenDim; d++ {
			expertWeights[e][d] = int64(mix(uint64(e+1), uint64(d+1)) % 1000)
		}
	}

	// Resident weights per rank
	residentWeights := make([][][]int64, worldSize)
	for r := 0; r < worldSize; r++ {
		residentWeights[r] = make([][]int64, spans[r].Count)
		for lIdx := 0; lIdx < spans[r].Count; lIdx++ {
			residentWeights[r][lIdx] = expertWeights[spans[r].Base+lIdx]
		}
	}

	for tok := 0; tok < numTokens; tok++ {
		x := make([]int64, hiddenDim)
		for d := 0; d < hiddenDim; d++ {
			x[d] = int64(mix(uint64(tok+1), uint64(d+5)) % 50)
		}

		selected := make([]int, topK)
		weights := make([]float32, topK)
		intWeights := make([]int64, topK)
		for k := 0; k < topK; k++ {
			selected[k] = int(mix(uint64(tok+1), uint64(k*17+1)) % uint64(totalExperts))
			intWeights[k] = int64((mix(uint64(tok+3), uint64(k+7)) % 100) + 1)
			weights[k] = float32(intWeights[k])
		}

		// Un-sharded reference
		unsharded := make([]int64, hiddenDim)
		for k := 0; k < topK; k++ {
			exp := selected[k]
			w := intWeights[k]
			for d := 0; d < hiddenDim; d++ {
				unsharded[d] += w * (x[d] * expertWeights[exp][d])
			}
		}

		// Sharded execution across ranks
		shardedCombined := make([]int64, hiddenDim)
		for r := 0; r < worldSize; r++ {
			span := spans[r]
			localIndices, _, isLocal := RemapRoutedTokens(selected, weights, span)
			rankOut := make([]int64, hiddenDim)

			for k := 0; k < topK; k++ {
				if !isLocal[k] {
					continue
				}
				lIdx := localIndices[k]
				w := intWeights[k]
				for d := 0; d < hiddenDim; d++ {
					rankOut[d] += w * (x[d] * residentWeights[r][lIdx][d])
				}
			}

			for d := 0; d < hiddenDim; d++ {
				shardedCombined[d] += rankOut[d]
			}
		}

		// Bit-exact check (must be exact 0 delta)
		for d := 0; d < hiddenDim; d++ {
			if shardedCombined[d] != unsharded[d] {
				t.Fatalf("token %d dim %d: exact integer mismatch: sharded=%d, unsharded=%d",
					tok, d, shardedCombined[d], unsharded[d])
			}
		}
	}
}
