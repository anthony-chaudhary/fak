package compute

import (
	"math"
	"math/rand"
	"testing"
)

func createMockLinearExpertFn(rng *rand.Rand, numExperts, hiddenDim int) ExpertBatchFn {
	expertWeights := make([][]float32, numExperts)
	scale := 1.0 / float32(math.Sqrt(float64(hiddenDim)))
	for e := 0; e < numExperts; e++ {
		w := make([]float32, hiddenDim*hiddenDim)
		for i := range w {
			w[i] = (rng.Float32()*2 - 1) * scale
		}
		expertWeights[e] = w
	}

	return func(expertID int, inputs []float32, numTokens, dim int) []float32 {
		w := expertWeights[expertID]
		out := make([]float32, numTokens*dim)
		for t := 0; t < numTokens; t++ {
			x := inputs[t*dim : (t+1)*dim]
			y := out[t*dim : (t+1)*dim]
			for col := 0; col < dim; col++ {
				wRow := w[col*dim : (col+1)*dim]
				var sum float32
				for row := 0; row < dim; row++ {
					sum += x[row] * wRow[row]
				}
				y[col] = sum
			}
		}
		return out
	}
}

func generateMockRouting(rng *rand.Rand, batchSize, topK, numTotalExperts int, clusterFactor float64) ([][]int, [][]float32) {
	indices := make([][]int, batchSize)
	weights := make([][]float32, batchSize)

	numCluster := int(float64(numTotalExperts) * 0.1)
	if numCluster < topK {
		numCluster = topK
	}

	for b := 0; b < batchSize; b++ {
		seen := make(map[int]bool, topK)
		idxRow := make([]int, 0, topK)
		weightRow := make([]float32, topK)
		var totalWeight float32

		for len(idxRow) < topK {
			var cand int
			if rng.Float64() < clusterFactor {
				cand = rng.Intn(numCluster)
			} else {
				cand = rng.Intn(numTotalExperts)
			}
			if !seen[cand] {
				seen[cand] = true
				idxRow = append(idxRow, cand)
			}
		}

		for k := 0; k < topK; k++ {
			w := rng.Float32() + 0.1
			weightRow[k] = w
			totalWeight += w
		}
		for k := 0; k < topK; k++ {
			weightRow[k] /= totalWeight
		}

		indices[b] = idxRow
		weights[b] = weightRow
	}
	return indices, weights
}

func TestMoeUnionPlanConstruction(t *testing.T) {
	rng := rand.New(rand.NewSource(11113))
	const batchSize = 4
	const topK = 8
	const numExperts = 512

	indices, weights := generateMockRouting(rng, batchSize, topK, numExperts, 0.5)

	plan, err := BuildExpertUnion(indices, weights, numExperts)
	if err != nil {
		t.Fatalf("BuildExpertUnion failed: %v", err)
	}

	if plan.BatchSize != batchSize {
		t.Errorf("plan.BatchSize = %d, want %d", plan.BatchSize, batchSize)
	}
	if plan.NumTotalExperts != numExperts {
		t.Errorf("plan.NumTotalExperts = %d, want %d", plan.NumTotalExperts, numExperts)
	}
	if plan.TopK != topK {
		t.Errorf("plan.TopK = %d, want %d", plan.TopK, topK)
	}
	if plan.NaiveLaunches != batchSize*topK {
		t.Errorf("plan.NaiveLaunches = %d, want %d", plan.NaiveLaunches, batchSize*topK)
	}
	if plan.GroupedLaunches != plan.NumUniqueExperts {
		t.Errorf("plan.GroupedLaunches = %d != NumUniqueExperts %d", plan.GroupedLaunches, plan.NumUniqueExperts)
	}
	if plan.NumUniqueExperts > 40 {
		t.Errorf("plan.NumUniqueExperts = %d > 40 budget for batch=4", plan.NumUniqueExperts)
	}
	if !plan.IsWithinSmallBatchBudget() {
		t.Errorf("plan.IsWithinSmallBatchBudget() = false, want true")
	}

	// Verify unique active experts are strictly sorted and unique
	for i := 1; i < len(plan.ActiveExperts); i++ {
		if plan.ActiveExperts[i] <= plan.ActiveExperts[i-1] {
			t.Errorf("active experts not strictly sorted at index %d: %d <= %d", i, plan.ActiveExperts[i], plan.ActiveExperts[i-1])
		}
	}

	// Verify group lookup
	for _, expID := range plan.ActiveExperts {
		group, found := plan.ExpertGroup(expID)
		if !found {
			t.Errorf("ExpertGroup(%d) not found, want true", expID)
		}
		if group.ExpertID != expID {
			t.Errorf("group.ExpertID = %d, want %d", group.ExpertID, expID)
		}
		if len(group.TokenIndices) == 0 {
			t.Errorf("group %d has empty token indices", expID)
		}
		if len(group.TokenIndices) != len(group.Weights) {
			t.Errorf("group %d token indices count %d != weights count %d", expID, len(group.TokenIndices), len(group.Weights))
		}
	}

	// Verify CSR offsets and flattened indices
	if len(plan.Offsets) != plan.NumUniqueExperts+1 {
		t.Errorf("Offsets length = %d, want %d", len(plan.Offsets), plan.NumUniqueExperts+1)
	}
	if plan.Offsets[0] != 0 {
		t.Errorf("Offsets[0] = %d, want 0", plan.Offsets[0])
	}
	if plan.Offsets[plan.NumUniqueExperts] != len(plan.FlatTokenIndices) {
		t.Errorf("Offsets[end] = %d != FlatTokenIndices len %d", plan.Offsets[plan.NumUniqueExperts], len(plan.FlatTokenIndices))
	}
	if len(plan.FlatTokenIndices) != batchSize*topK {
		t.Errorf("FlatTokenIndices len = %d, want %d", len(plan.FlatTokenIndices), batchSize*topK)
	}

	// Also verify BuildExpertUnionFromRouting equivalence
	routings := make([]TokenRouting, batchSize)
	for b := 0; b < batchSize; b++ {
		routings[b] = TokenRouting{
			TokenIndex: b,
			ExpertIDs:  indices[b],
			Weights:    weights[b],
		}
	}
	planFromRouting, err := BuildExpertUnionFromRouting(routings, numExperts)
	if err != nil {
		t.Fatalf("BuildExpertUnionFromRouting failed: %v", err)
	}
	if planFromRouting.NumUniqueExperts != plan.NumUniqueExperts {
		t.Errorf("planFromRouting unique experts = %d, want %d", planFromRouting.NumUniqueExperts, plan.NumUniqueExperts)
	}
}

func TestMoeUnionBuildValidation(t *testing.T) {
	const numExperts = 512

	tests := []struct {
		name       string
		indices    [][]int
		weights    [][]float32
		numExperts int
		wantErr    bool
	}{
		{
			name:       "empty batch",
			indices:    [][]int{},
			weights:    nil,
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "non-positive total experts",
			indices:    [][]int{{1, 2}},
			weights:    nil,
			numExperts: 0,
			wantErr:    true,
		},
		{
			name:       "expert ID out of bounds high",
			indices:    [][]int{{1, 512}},
			weights:    nil,
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "expert ID out of bounds negative",
			indices:    [][]int{{-1, 10}},
			weights:    nil,
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "duplicate expert ID in single token",
			indices:    [][]int{{5, 5}},
			weights:    nil,
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "mismatched batch size in weights",
			indices:    [][]int{{1, 2}, {3, 4}},
			weights:    [][]float32{{0.5, 0.5}},
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "mismatched token weights length",
			indices:    [][]int{{1, 2}},
			weights:    [][]float32{{0.5}},
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "token with zero routed experts",
			indices:    [][]int{{}},
			weights:    nil,
			numExperts: numExperts,
			wantErr:    true,
		},
		{
			name:       "valid with nil weights",
			indices:    [][]int{{1, 2}, {2, 3}},
			weights:    nil,
			numExperts: numExperts,
			wantErr:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildExpertUnion(tc.indices, tc.weights, tc.numExperts)
			if (err != nil) != tc.wantErr {
				t.Fatalf("BuildExpertUnion() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestMoeUnionEvaluationValidation(t *testing.T) {
	plan, err := BuildExpertUnion([][]int{{1, 2}}, [][]float32{{0.5, 0.5}}, 512)
	if err != nil {
		t.Fatalf("failed to build plan: %v", err)
	}

	dummyFn := func(expertID int, inputs []float32, numTokens, dim int) []float32 {
		return make([]float32, numTokens*dim)
	}

	// 1. Nil plan
	if _, _, err := EvaluateExpertUnion(nil, []float32{1, 2}, 2, dummyFn); err == nil {
		t.Errorf("EvaluateExpertUnion with nil plan expected error, got nil")
	}

	// 2. Non-positive hiddenDim
	if _, _, err := EvaluateExpertUnion(plan, []float32{1, 2}, 0, dummyFn); err == nil {
		t.Errorf("EvaluateExpertUnion with non-positive hiddenDim expected error, got nil")
	}

	// 3. Activations length mismatch
	if _, _, err := EvaluateExpertUnion(plan, []float32{1}, 2, dummyFn); err == nil {
		t.Errorf("EvaluateExpertUnion with activations length mismatch expected error, got nil")
	}

	// 4. Nil expert function
	if _, _, err := EvaluateExpertUnion(plan, []float32{1, 2}, 2, nil); err == nil {
		t.Errorf("EvaluateExpertUnion with nil expert function expected error, got nil")
	}
}

func TestMoeUnionMathematicalEquivalence(t *testing.T) {
	configs := []struct {
		name      string
		batchSize int
		topK      int
		hiddenDim int
		cluster   float64
	}{
		{name: "Batch4_TopK8_Dim64", batchSize: 4, topK: 8, hiddenDim: 64, cluster: 0.6},
		{name: "Batch8_TopK8_Dim128", batchSize: 8, topK: 8, hiddenDim: 128, cluster: 0.5},
		{name: "Batch16_TopK8_Dim128", batchSize: 16, topK: 8, hiddenDim: 128, cluster: 0.5},
		{name: "Batch4_TopK10_Dim64_Qwen4Exp", batchSize: 4, topK: 10, hiddenDim: 64, cluster: 0.7},
		{name: "Batch16_TopK10_Dim64", batchSize: 16, topK: 10, hiddenDim: 64, cluster: 0.4},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(cfg.batchSize*1000 + cfg.topK*100 + cfg.hiddenDim)))
			const numExperts = 512

			indices, weights := generateMockRouting(rng, cfg.batchSize, cfg.topK, numExperts, cfg.cluster)

			plan, err := BuildExpertUnion(indices, weights, numExperts)
			if err != nil {
				t.Fatalf("BuildExpertUnion failed: %v", err)
			}

			// Generate random token activation inputs
			tokens := make([]float32, cfg.batchSize*cfg.hiddenDim)
			for i := range tokens {
				tokens[i] = rng.Float32()*2 - 1
			}

			expertFn := createMockLinearExpertFn(rng, numExperts, cfg.hiddenDim)

			// 1. Evaluate via EvaluateExpertUnion (runs grouped + naive, tests equivalence)
			groupedOut, stats, err := EvaluateExpertUnion(plan, tokens, cfg.hiddenDim, expertFn)
			if err != nil {
				t.Fatalf("EvaluateExpertUnion failed: %v", err)
			}

			if !stats.MathematicallyEqual {
				t.Fatalf("grouped and naive MoE outputs not mathematically equal: maxDiff=%.6e, cosine=%.8f",
					stats.EquivalenceMaxDiff, stats.CosineSimilarity)
			}

			if stats.EquivalenceMaxDiff > 1e-4 {
				t.Fatalf("max difference %.6e > threshold 1e-4", stats.EquivalenceMaxDiff)
			}

			if stats.CosineSimilarity < 0.99999 {
				t.Fatalf("cosine similarity %.8f < threshold 0.99999", stats.CosineSimilarity)
			}

			if stats.GroupedLaunches != plan.NumUniqueExperts {
				t.Fatalf("grouped launches %d != unique experts %d", stats.GroupedLaunches, plan.NumUniqueExperts)
			}

			if stats.NaiveLaunches != cfg.batchSize*cfg.topK {
				t.Fatalf("naive launches %d != batch*topk %d", stats.NaiveLaunches, cfg.batchSize*cfg.topK)
			}

			if stats.GroupedLaunches >= stats.NaiveLaunches {
				t.Fatalf("grouped launches %d not strictly less than naive %d (cluster=%.2f)",
					stats.GroupedLaunches, stats.NaiveLaunches, cfg.cluster)
			}

			t.Logf("[%s] Equivalence OK: maxDiff=%.3e, cosine=%.8f, launches: naive=%d -> grouped=%d (reduction=%.1f%%)",
				cfg.name, stats.EquivalenceMaxDiff, stats.CosineSimilarity,
				stats.NaiveLaunches, stats.GroupedLaunches, stats.LaunchReduction*100)

			// 2. Direct separate calls to EvaluateGroupedMoE and EvaluateNaiveMoE
			directGrouped, gStats, err := EvaluateGroupedMoE(plan, tokens, cfg.hiddenDim, expertFn)
			if err != nil {
				t.Fatalf("EvaluateGroupedMoE failed: %v", err)
			}
			directNaive, nStats, err := EvaluateNaiveMoE(plan, tokens, cfg.hiddenDim, expertFn)
			if err != nil {
				t.Fatalf("EvaluateNaiveMoE failed: %v", err)
			}

			if len(directGrouped) != len(groupedOut) {
				t.Fatalf("directGrouped len %d != groupedOut len %d", len(directGrouped), len(groupedOut))
			}
			if gStats.GroupedLaunches != plan.NumUniqueExperts {
				t.Errorf("gStats.GroupedLaunches %d != %d", gStats.GroupedLaunches, plan.NumUniqueExperts)
			}
			if nStats.NaiveLaunches != cfg.batchSize*cfg.topK {
				t.Errorf("nStats.NaiveLaunches %d != %d", nStats.NaiveLaunches, cfg.batchSize*cfg.topK)
			}

			for i := range directGrouped {
				diff := float32(math.Abs(float64(directGrouped[i] - directNaive[i])))
				if diff > 1e-4 {
					t.Fatalf("index %d directGrouped %.6f != directNaive %.6f (diff=%.6e)",
						i, directGrouped[i], directNaive[i], diff)
				}
			}
		})
	}
}

func TestMoeUnionSmallBatchBudget(t *testing.T) {
	rng := rand.New(rand.NewSource(4040))
	const numExperts = 512

	// Test 1: B=4, K=8 -> max possible unique experts is 32 <= 40
	indices8, weights8 := generateMockRouting(rng, 4, 8, numExperts, 0.0) // zero clustering = worst case spread
	plan8, err := BuildExpertUnion(indices8, weights8, numExperts)
	if err != nil {
		t.Fatalf("BuildExpertUnion failed: %v", err)
	}
	if plan8.NumUniqueExperts > 40 {
		t.Fatalf("B=4, K=8: unique experts %d > 40", plan8.NumUniqueExperts)
	}
	if !plan8.IsWithinSmallBatchBudget() {
		t.Fatalf("B=4, K=8: IsWithinSmallBatchBudget() = false")
	}

	// Test 2: B=4, K=10 -> max possible unique experts is 40 <= 40
	indices10, weights10 := generateMockRouting(rng, 4, 10, numExperts, 0.0)
	plan10, err := BuildExpertUnion(indices10, weights10, numExperts)
	if err != nil {
		t.Fatalf("BuildExpertUnion failed: %v", err)
	}
	if plan10.NumUniqueExperts > 40 {
		t.Fatalf("B=4, K=10: unique experts %d > 40", plan10.NumUniqueExperts)
	}
	if !plan10.IsWithinSmallBatchBudget() {
		t.Fatalf("B=4, K=10: IsWithinSmallBatchBudget() = false")
	}
}

// Benchmarks demonstrating reduction in expert evaluation launches and improved throughput on B=4..16

func runBenchmarkGrouped(b *testing.B, batchSize int) {
	const topK = 8
	const numExperts = 512
	const hiddenDim = 128
	rng := rand.New(rand.NewSource(int64(batchSize * 777)))

	indices, weights := generateMockRouting(rng, batchSize, topK, numExperts, 0.5)
	plan, err := BuildExpertUnion(indices, weights, numExperts)
	if err != nil {
		b.Fatalf("BuildExpertUnion failed: %v", err)
	}

	tokens := make([]float32, batchSize*hiddenDim)
	for i := range tokens {
		tokens[i] = rng.Float32()
	}
	expertFn := createMockLinearExpertFn(rng, numExperts, hiddenDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := EvaluateGroupedMoE(plan, tokens, hiddenDim, expertFn)
		if err != nil {
			b.Fatalf("EvaluateGroupedMoE failed: %v", err)
		}
	}
	b.ReportMetric(float64(plan.GroupedLaunches), "launches/op")
	b.ReportMetric(float64(batchSize)*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
}

func runBenchmarkNaive(b *testing.B, batchSize int) {
	const topK = 8
	const numExperts = 512
	const hiddenDim = 128
	rng := rand.New(rand.NewSource(int64(batchSize * 777)))

	indices, weights := generateMockRouting(rng, batchSize, topK, numExperts, 0.5)
	plan, err := BuildExpertUnion(indices, weights, numExperts)
	if err != nil {
		b.Fatalf("BuildExpertUnion failed: %v", err)
	}

	tokens := make([]float32, batchSize*hiddenDim)
	for i := range tokens {
		tokens[i] = rng.Float32()
	}
	expertFn := createMockLinearExpertFn(rng, numExperts, hiddenDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := EvaluateNaiveMoE(plan, tokens, hiddenDim, expertFn)
		if err != nil {
			b.Fatalf("EvaluateNaiveMoE failed: %v", err)
		}
	}
	b.ReportMetric(float64(plan.NaiveLaunches), "launches/op")
	b.ReportMetric(float64(batchSize)*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
}

func BenchmarkMoeUnionDispatchGrouped_B4(b *testing.B)  { runBenchmarkGrouped(b, 4) }
func BenchmarkMoeUnionDispatchGrouped_B8(b *testing.B)  { runBenchmarkGrouped(b, 8) }
func BenchmarkMoeUnionDispatchGrouped_B16(b *testing.B) { runBenchmarkGrouped(b, 16) }

func BenchmarkMoeUnionDispatchNaive_B4(b *testing.B)  { runBenchmarkNaive(b, 4) }
func BenchmarkMoeUnionDispatchNaive_B8(b *testing.B)  { runBenchmarkNaive(b, 8) }
func BenchmarkMoeUnionDispatchNaive_B16(b *testing.B) { runBenchmarkNaive(b, 16) }

func BenchmarkMoeUnionBuildPlan_B4(b *testing.B) {
	const topK, numExperts = 8, 512
	rng := rand.New(rand.NewSource(42))
	indices, weights := generateMockRouting(rng, 4, topK, numExperts, 0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = BuildExpertUnion(indices, weights, numExperts)
	}
}

func BenchmarkMoeUnionBuildPlan_B8(b *testing.B) {
	const topK, numExperts = 8, 512
	rng := rand.New(rand.NewSource(42))
	indices, weights := generateMockRouting(rng, 8, topK, numExperts, 0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = BuildExpertUnion(indices, weights, numExperts)
	}
}

func BenchmarkMoeUnionBuildPlan_B16(b *testing.B) {
	const topK, numExperts = 8, 512
	rng := rand.New(rand.NewSource(42))
	indices, weights := generateMockRouting(rng, 16, topK, numExperts, 0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = BuildExpertUnion(indices, weights, numExperts)
	}
}
