package compute

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// topk_greedy_test.go — correctness gates for candidate-filtered top-k logit reduction
// for tensor-parallel greedy decode (borrowed from DS4_TP_FEATURE_GREEDY_TOP2; Issue #10756).
//
// GATES:
// 1. 100% numerical parity and zero divergence between candidate-filtered top-2 reduction
//    and exhaustive AllGather argmax across 32k, 64k, and 128k vocabulary sizes across 2, 4, and 8 ranks.
// 2. Exact 24-byte wire packet structure and deterministic binary serialization.
// 3. Concurrent multi-rank simulation with goroutines and channel transmission.
// 4. Fail-closed contract for invalid backends, dtypes, unready tensors, and canceled contexts.
// 5. Deterministic tie-breaking (lower global token ID wins on equal logit values).

// TestTopKGreedyParityVsExhaustiveAllGather verifies 100% numerical parity and zero divergence
// between candidate-filtered top-2 reduction and exhaustive AllGather argmax over 32k, 64k, and 128k
// vocabulary sizes across 2, 4, and 8 ranks.
func TestTopKGreedyParityVsExhaustiveAllGather(t *testing.T) {
	c, cb := asCollective(t)
	ctx := context.Background()

	vocabSizes := []int{
		32000, 32768, // 32k
		64000, 65536, // 64k
		128000, 131072, // 128k
	}
	rankCounts := []int{2, 4, 8}

	for _, vocab := range vocabSizes {
		for _, ranks := range rankCounts {
			testName := fmt.Sprintf("vocab=%d/ranks=%d", vocab, ranks)
			t.Run(testName, func(t *testing.T) {
				if vocab%ranks != 0 {
					t.Skipf("vocab %d not divisible by ranks %d", vocab, ranks)
				}
				shardLen := vocab / ranks

				// Subtest 1: Deterministic pseudo-random logits
				t.Run("RandomLogits", func(t *testing.T) {
					var s lcg = lcg(uint64(vocab)*31 + uint64(ranks)*17 + 101)
					parts := make([]Tensor, ranks)
					for r := 0; r < ranks; r++ {
						raw := randVec(&s, shardLen)
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}

					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					if gotID != wantID {
						t.Fatalf("Argmax token ID divergence: got %d, want %d (exhaustive oracle)", gotID, wantID)
					}
					if gotVal != wantVal {
						t.Fatalf("Argmax logit value mismatch: got %v, want %v", gotVal, wantVal)
					}
				})

				// Subtest 2: Winner placed on Rank 0
				t.Run("WinnerOnRank0", func(t *testing.T) {
					parts := make([]Tensor, ranks)
					for r := 0; r < ranks; r++ {
						raw := make([]float32, shardLen)
						for i := range raw {
							raw[i] = -10.0 + float32(r)*0.01
						}
						if r == 0 {
							raw[shardLen/2] = 42.5 // winner
						}
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}
					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					if gotID != wantID || gotVal != wantVal {
						t.Fatalf("WinnerOnRank0 mismatch: got (%d, %v), want (%d, %v)", gotID, gotVal, wantID, wantVal)
					}
					if wantID != shardLen/2 {
						t.Fatalf("Winner position wrong: want %d, got %d", shardLen/2, wantID)
					}
				})

				// Subtest 3: Winner placed on last rank (Rank P-1)
				t.Run("WinnerOnLastRank", func(t *testing.T) {
					parts := make([]Tensor, ranks)
					lastRank := ranks - 1
					targetLocalIdx := shardLen - 7
					for r := 0; r < ranks; r++ {
						raw := make([]float32, shardLen)
						for i := range raw {
							raw[i] = -5.0
						}
						if r == lastRank {
							raw[targetLocalIdx] = 99.125 // winner
						}
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}
					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					expectedGlobalID := lastRank*shardLen + targetLocalIdx
					if gotID != expectedGlobalID || gotID != wantID {
						t.Fatalf("WinnerOnLastRank token ID mismatch: got %d, want %d (expected %d)", gotID, wantID, expectedGlobalID)
					}
					if gotVal != wantVal || gotVal != 99.125 {
						t.Fatalf("WinnerOnLastRank value mismatch: got %v, want %v", gotVal, wantVal)
					}
				})

				// Subtest 4: Winner placed on middle rank
				t.Run("WinnerOnMiddleRank", func(t *testing.T) {
					parts := make([]Tensor, ranks)
					midRank := ranks / 2
					targetLocalIdx := 13
					for r := 0; r < ranks; r++ {
						raw := make([]float32, shardLen)
						for i := range raw {
							raw[i] = -1.0
						}
						if r == midRank {
							raw[targetLocalIdx] = 88.0
						}
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}
					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					expectedGlobalID := midRank*shardLen + targetLocalIdx
					if gotID != expectedGlobalID || gotID != wantID {
						t.Fatalf("WinnerOnMiddleRank mismatch: got %d, want %d", gotID, wantID)
					}
					if gotVal != wantVal {
						t.Fatalf("WinnerOnMiddleRank value mismatch: got %v, want %v", gotVal, wantVal)
					}
				})

				// Subtest 5: Very close runner-up across different ranks (1e-5 margin)
				t.Run("CloseMarginRunnerUp", func(t *testing.T) {
					parts := make([]Tensor, ranks)
					for r := 0; r < ranks; r++ {
						raw := make([]float32, shardLen)
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					// Set runner-up on Rank 0 and winner on Rank 1
					r0Raw, _ := c.Host(parts[0])
					r0Raw[0] = 50.00000

					r1Raw, _ := c.Host(parts[1])
					r1Raw[0] = 50.00001 // winner by 1e-5

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}
					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					expectedGlobalID := shardLen // Rank 1 index 0
					if gotID != expectedGlobalID || gotID != wantID {
						t.Fatalf("CloseMarginRunnerUp mismatch: got %d, want %d", gotID, wantID)
					}
					if gotVal != wantVal {
						t.Fatalf("CloseMarginRunnerUp value mismatch: got %v, want %v", gotVal, wantVal)
					}
				})

				// Subtest 6: All negative logits
				t.Run("AllNegativeLogits", func(t *testing.T) {
					parts := make([]Tensor, ranks)
					for r := 0; r < ranks; r++ {
						raw := make([]float32, shardLen)
						for i := range raw {
							raw[i] = -1000.0 - float32(i)
						}
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					// Set highest negative value on Rank 0, index 5 (-50.0)
					r0Raw, _ := c.Host(parts[0])
					r0Raw[5] = -50.0

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}
					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					if gotID != 5 || gotID != wantID || gotVal != -50.0 || gotVal != wantVal {
						t.Fatalf("AllNegativeLogits mismatch: got (%d, %v), want (%d, %v)", gotID, gotVal, wantID, wantVal)
					}
				})

				// Subtest 7: Ties across ranks (deterministic tie-breaking: lower token ID wins)
				t.Run("TieBreakingLowerIDWins", func(t *testing.T) {
					parts := make([]Tensor, ranks)
					for r := 0; r < ranks; r++ {
						raw := make([]float32, shardLen)
						for i := range raw {
							raw[i] = 10.0 // all equal
						}
						parts[r] = NewF32(c, []int{shardLen}, raw)
					}

					wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
					if err != nil {
						t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
					}
					gotID, gotVal, err := SimulateMultiRankTopKGreedy(ctx, parts, 2)
					if err != nil {
						t.Fatalf("SimulateMultiRankTopKGreedy: %v", err)
					}

					// Under standard argmax, first index (0) wins
					if gotID != 0 || gotID != wantID {
						t.Fatalf("TieBreakingLowerIDWins token ID mismatch: got %d, want %d", gotID, wantID)
					}
					if gotVal != 10.0 || gotVal != wantVal {
						t.Fatalf("TieBreakingLowerIDWins value mismatch: got %v, want %v", gotVal, wantVal)
					}
				})
			})
		}
	}
}

// TestTop2LogitsPacketSerialization verifies that Top2LogitsPacket has exact 24-byte size
// and round-trips bit-for-bit through binary marshaling.
func TestTop2LogitsPacketSerialization(t *testing.T) {
	if Top2LogitsPacketSize != 24 {
		t.Fatalf("Top2LogitsPacketSize = %d, want 24 bytes", Top2LogitsPacketSize)
	}

	pkt := Top2LogitsPacket{
		Candidates: [2]TopKLogit{
			{ID: 104523, Value: 18.75},
			{ID: 4201, Value: 17.125},
		},
		Rank:  3,
		SeqID: 42,
	}

	wire := pkt.MarshalBinary()
	if len(wire) != 24 {
		t.Fatalf("wire length = %d, want 24", len(wire))
	}

	recovered := UnmarshalTop2LogitsPacket(wire)
	if recovered.Rank != pkt.Rank {
		t.Fatalf("Rank: got %d, want %d", recovered.Rank, pkt.Rank)
	}
	if recovered.SeqID != pkt.SeqID {
		t.Fatalf("SeqID: got %d, want %d", recovered.SeqID, pkt.SeqID)
	}
	for i := 0; i < 2; i++ {
		if recovered.Candidates[i].ID != pkt.Candidates[i].ID {
			t.Fatalf("Candidate[%d].ID: got %d, want %d", i, recovered.Candidates[i].ID, pkt.Candidates[i].ID)
		}
		if recovered.Candidates[i].Value != pkt.Candidates[i].Value {
			t.Fatalf("Candidate[%d].Value: got %v, want %v", i, recovered.Candidates[i].Value, pkt.Candidates[i].Value)
		}
	}

	// Test slice unmarshaler
	fromBytes, err := UnmarshalTop2LogitsBytes(wire[:])
	if err != nil {
		t.Fatalf("UnmarshalTop2LogitsBytes: %v", err)
	}
	if fromBytes != recovered {
		t.Fatalf("UnmarshalTop2LogitsBytes mismatch vs UnmarshalTop2LogitsPacket")
	}

	// Test slice length error
	if _, err := UnmarshalTop2LogitsBytes(wire[:20]); err == nil {
		t.Fatalf("UnmarshalTop2LogitsBytes should reject short slice")
	}
}

// TestConcurrentMultiRankSimulator verifies the concurrent multi-rank simulator
// communicating over channels with 24-byte wire packets across ranks.
func TestConcurrentMultiRankSimulator(t *testing.T) {
	c, cb := asCollective(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	for _, ranks := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("ranks=%d", ranks), func(t *testing.T) {
			const vocab = 64000
			shardLen := vocab / ranks
			var s lcg = lcg(uint64(ranks)*997 + 13)

			parts := make([]Tensor, ranks)
			for r := 0; r < ranks; r++ {
				parts[r] = NewF32(c, []int{shardLen}, randVec(&s, shardLen))
			}

			sim, err := NewMultiRankGreedySimulator(ranks, 2)
			if err != nil {
				t.Fatalf("NewMultiRankGreedySimulator: %v", err)
			}

			winnerID, winnerVal, bytesSent, err := sim.Run(ctx, parts)
			if err != nil {
				t.Fatalf("sim.Run: %v", err)
			}

			wantID, wantVal, err := ExhaustiveAllGatherArgmax(cb, parts)
			if err != nil {
				t.Fatalf("ExhaustiveAllGatherArgmax: %v", err)
			}

			if winnerID != wantID {
				t.Fatalf("sim.Run winnerID=%d != AllGather argmax=%d", winnerID, wantID)
			}
			if winnerVal != wantVal {
				t.Fatalf("sim.Run winnerVal=%v != AllGather val=%v", winnerVal, wantVal)
			}

			expectedBytes := (ranks - 1) * Top2LogitsPacketSize
			if bytesSent != expectedBytes {
				t.Fatalf("bytesSent = %d, want %d ((ranks-1) * 24 bytes)", bytesSent, expectedBytes)
			}
		})
	}
}

// TestReduceTopKGreedyCpuBackend pins the cpuBackend.ReduceTopKGreedy method contract
// and fail-closed validation.
func TestReduceTopKGreedyCpuBackend(t *testing.T) {
	c, _ := asCollective(t)
	ctx := context.Background()

	// 1. Happy path on local partition
	raw := []float32{1.0, 5.5, 2.0, 9.25, 3.0}
	ten := NewF32(c, []int{len(raw)}, raw)

	winnerID, winnerVal, err := c.ReduceTopKGreedy(ctx, ten, 2, 1000)
	if err != nil {
		t.Fatalf("ReduceTopKGreedy happy path: %v", err)
	}
	if winnerID != 1003 { // offset 1000 + index 3 (9.25)
		t.Fatalf("winnerID = %d, want 1003", winnerID)
	}
	if winnerVal != 9.25 {
		t.Fatalf("winnerVal = %v, want 9.25", winnerVal)
	}

	// 2. Canceled context
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := c.ReduceTopKGreedy(canceledCtx, ten, 2, 0); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on canceled context")
	}

	// 3. k <= 0
	if _, _, err := c.ReduceTopKGreedy(ctx, ten, 0, 0); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on k=0")
	}

	// 4. vocabOffset < 0
	if _, _, err := c.ReduceTopKGreedy(ctx, ten, 2, -1); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on negative vocabOffset")
	}

	// 5. Empty tensor
	empty := NewF32(c, []int{0}, []float32{})
	if _, _, err := c.ReduceTopKGreedy(ctx, empty, 2, 0); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on empty tensor")
	}

	// 6. Unready tensor
	unready := makeTensor(c, F32, RowMajor, []int{5}, nil, nil)
	if _, _, err := c.ReduceTopKGreedy(ctx, unready, 2, 0); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on unready tensor")
	}

	// 7. Foreign backend
	foreign := NewF32(foreignBackend{c}, []int{5}, []float32{1, 2, 3, 4, 5})
	if _, _, err := c.ReduceTopKGreedy(ctx, foreign, 2, 0); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on foreign backend")
	}

	// 8. Non-F32 tensor (Q8)
	q := QuantizeQ8(c, []int{1, 32}, randVecN(32), 32)
	if _, _, err := c.ReduceTopKGreedy(ctx, q, 2, 0); err == nil {
		t.Fatalf("ReduceTopKGreedy should fail on non-F32 tensor")
	}
}

// TestTopKGreedyWireSavings validates the wire volume calculations and confirms
// that candidate-filtered top-2 reduction slashes wire traffic by >99.9% across
// standard vocabulary sizes and rank counts.
func TestTopKGreedyWireSavings(t *testing.T) {
	cases := []struct {
		vocab int
		ranks int
	}{
		{32000, 2},
		{32000, 4},
		{64000, 2},
		{64000, 4},
		{128000, 2},
		{128000, 4},
		{128000, 8},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("vocab=%d/ranks=%d", tc.vocab, tc.ranks), func(t *testing.T) {
			report := ComputeGreedyWireSavings(tc.vocab, tc.ranks)
			if report.AllGatherBytes != tc.vocab*4 {
				t.Fatalf("AllGatherBytes = %d, want %d", report.AllGatherBytes, tc.vocab*4)
			}
			expectedTop2Bytes := (tc.ranks - 1) * 24
			if report.Top2ReductionBytes != expectedTop2Bytes {
				t.Fatalf("Top2ReductionBytes = %d, want %d", report.Top2ReductionBytes, expectedTop2Bytes)
			}
			if report.SavingsPercentage < 99.9 {
				t.Fatalf("SavingsPercentage = %.4f%%, want >= 99.9%%", report.SavingsPercentage)
			}
		})
	}
}

// TestExtractTopKEdgeCases tests ExtractTopK for various k values and slice sizes.
func TestExtractTopKEdgeCases(t *testing.T) {
	// Empty slice
	if cands := ExtractTopK(nil, 2, 0); len(cands) != 0 {
		t.Fatalf("ExtractTopK(nil): want len 0, got %d", len(cands))
	}

	// k <= 0
	if cands := ExtractTopK([]float32{1.0}, 0, 0); len(cands) != 0 {
		t.Fatalf("ExtractTopK(k=0): want len 0, got %d", len(cands))
	}

	// len < k
	cands := ExtractTopK([]float32{1.0}, 2, 100)
	if len(cands) != 1 || cands[0].ID != 100 || cands[0].Value != 1.0 {
		t.Fatalf("ExtractTopK(len < k): got %v", cands)
	}

	// k = 1
	cands1 := ExtractTopK([]float32{1.0, 5.0, 3.0}, 1, 50)
	if len(cands1) != 1 || cands1[0].ID != 51 || cands1[0].Value != 5.0 {
		t.Fatalf("ExtractTopK(k=1): got %v", cands1)
	}

	// General k = 3
	cands3 := ExtractTopK([]float32{10.0, 40.0, 20.0, 50.0, 30.0}, 3, 0)
	if len(cands3) != 3 {
		t.Fatalf("ExtractTopK(k=3): len = %d, want 3", len(cands3))
	}
	if cands3[0].ID != 3 || cands3[0].Value != 50.0 {
		t.Fatalf("cands3[0] = %v, want (3, 50.0)", cands3[0])
	}
	if cands3[1].ID != 1 || cands3[1].Value != 40.0 {
		t.Fatalf("cands3[1] = %v, want (1, 40.0)", cands3[1])
	}
	if cands3[2].ID != 4 || cands3[2].Value != 30.0 {
		t.Fatalf("cands3[2] = %v, want (4, 30.0)", cands3[2])
	}
}

// TestMergeTopK verifies merging candidate lists into top-k globally.
func TestMergeTopK(t *testing.T) {
	rank0 := []TopKLogit{
		{ID: 10, Value: 100.0},
		{ID: 11, Value: 80.0},
	}
	rank1 := []TopKLogit{
		{ID: 50, Value: 95.0},
		{ID: 51, Value: 70.0},
	}

	merged := MergeTopK([][]TopKLogit{rank0, rank1}, 3)
	if len(merged) != 3 {
		t.Fatalf("len(merged) = %d, want 3", len(merged))
	}
	if merged[0].ID != 10 || merged[0].Value != 100.0 {
		t.Fatalf("merged[0] = %v, want (10, 100.0)", merged[0])
	}
	if merged[1].ID != 50 || merged[1].Value != 95.0 {
		t.Fatalf("merged[1] = %v, want (50, 95.0)", merged[1])
	}
	if merged[2].ID != 11 || merged[2].Value != 80.0 {
		t.Fatalf("merged[2] = %v, want (11, 80.0)", merged[2])
	}
}
