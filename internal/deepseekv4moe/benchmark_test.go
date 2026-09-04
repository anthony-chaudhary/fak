package deepseekv4moe

import "testing"

// BenchmarkDeepSeekV4MoE measures synthetic top-k routing and per-layer scheduling cost
// evaluation across representative token counts and dispatch modes.
func BenchmarkDeepSeekV4MoE(b *testing.B) {
	cfg := V4Config()
	tokenCounts := []int{16, 64, 256, 1024}
	modes := []ScheduleMode{NaivePerExpert, GroupedFused}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		tokens := tokenCounts[i%len(tokenCounts)]
		mode := modes[i%len(modes)]
		routing, err := Route(tokens, cfg, 0)
		if err != nil {
			b.Fatalf("route failed: %v", err)
		}
		metrics := routing.ComputeMetrics(mode, 4, 1)
		if metrics.WorkRows <= 0 {
			b.Fatalf("expected positive work rows, got %d", metrics.WorkRows)
		}
	}
}

// BenchmarkExpertParallelPlan measures contiguous expert-parallel rank partitioning.
func BenchmarkExpertParallelPlan(b *testing.B) {
	ranks := []int{1, 2, 4, 8, 16}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rk := ranks[i%len(ranks)]
		bands, err := ExpertParallelPlan(NumRoutedExperts, rk)
		if err != nil {
			b.Fatalf("expert parallel plan failed: %v", err)
		}
		if len(bands) != rk {
			b.Fatalf("expected %d bands, got %d", rk, len(bands))
		}
	}
}

// BenchmarkPartitionContiguousSpans measures contiguous expert sharding partitioning
// across dual-node and multi-node TP configurations.
func BenchmarkPartitionContiguousSpans(b *testing.B) {
	cases := []struct {
		experts int
		ranks   int
	}{
		{NumRoutedExperts, 2},
		{NumRoutedExperts, 4},
		{NumRoutedExperts, 8},
		{256, 2},
		{256, 4},
		{256, 8},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c := cases[i%len(cases)]
		spans, err := PartitionContiguousSpans(c.experts, c.ranks)
		if err != nil {
			b.Fatalf("partition failed: %v", err)
		}
		if len(spans) != c.ranks {
			b.Fatalf("expected %d spans, got %d", c.ranks, len(spans))
		}
	}
}

// BenchmarkRemapRoutedTokens measures router output remapping and sentinel assignment
// for resident versus unowned experts on a sharded rank.
func BenchmarkRemapRoutedTokens(b *testing.B) {
	spans, err := PartitionContiguousSpans(NumRoutedExperts, 4)
	if err != nil {
		b.Fatalf("partition failed: %v", err)
	}
	span := spans[1]

	const tokenCount = 64
	totalSelections := tokenCount * TopK
	selections := make([]int, totalSelections)
	weights := make([]float32, totalSelections)
	for i := 0; i < totalSelections; i++ {
		selections[i] = (i * 7) % NumRoutedExperts
		weights[i] = 0.166
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		localIndices, localWeights, isLocal := RemapRoutedTokens(selections, weights, span)
		if len(localIndices) != totalSelections || len(localWeights) != totalSelections || len(isLocal) != totalSelections {
			b.Fatal("unexpected remap length")
		}
	}
}

// BenchmarkSimulateExpertCache measures deterministic LRU cache simulation over
// synthetic routed-expert sequences.
func BenchmarkSimulateExpertCache(b *testing.B) {
	const (
		layers          = 4
		expertsPerLayer = NumRoutedExperts
		topK            = TopK
		capacity        = 32
	)
	routes := make([]ExpertRoute, 64)
	for i := range routes {
		exps := make([]int, topK)
		for k := 0; k < topK; k++ {
			exps[k] = (i*topK + k*13) % expertsPerLayer
		}
		routes[i] = ExpertRoute{
			Layer:   i % layers,
			Experts: exps,
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		trace, err := SimulateExpertCache(routes, capacity, layers, expertsPerLayer, topK)
		if err != nil {
			b.Fatalf("simulation failed: %v", err)
		}
		if trace.PeakResident <= 0 || trace.PeakResident > capacity {
			b.Fatalf("unexpected peak resident: %d", trace.PeakResident)
		}
	}
}

// BenchmarkSimulateExpertCacheBatch measures cross-agent decode-step coalescing
// and cache trace simulation.
func BenchmarkSimulateExpertCacheBatch(b *testing.B) {
	const (
		layers          = 2
		expertsPerLayer = NumRoutedExperts
		topK            = TopK
		agents          = 4
		capacity        = 48
	)
	steps := make([]BatchStep, 32)
	for s := range steps {
		perAgent := make([][]int, agents)
		for a := 0; a < agents; a++ {
			exps := make([]int, topK)
			for k := 0; k < topK; k++ {
				exps[k] = (s*topK + a*3 + k*17) % expertsPerLayer
			}
			perAgent[a] = exps
		}
		steps[s] = BatchStep{
			Layer:    s % layers,
			PerAgent: perAgent,
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		trace, err := SimulateExpertCacheBatch(steps, capacity, layers, expertsPerLayer, topK, agents)
		if err != nil {
			b.Fatalf("batch simulation failed: %v", err)
		}
		if trace.PeakResident <= 0 || trace.PeakResident > capacity {
			b.Fatalf("unexpected peak resident: %d", trace.PeakResident)
		}
	}
}

// BenchmarkPlanExpertCache measures static byte-admission budget planning for HBM limits.
func BenchmarkPlanExpertCache(b *testing.B) {
	const (
		totalHBM          = 96 * 1024 * 1024 * 1024
		nonRoutedResident = 16 * 1024 * 1024 * 1024
		runtimeReserve    = 8 * 1024 * 1024 * 1024
		expertGroupBytes  = 64 * 1024 * 1024
		layers            = 61
		expertsPerLayer   = NumRoutedExperts
	)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan, err := PlanExpertCache(totalHBM, nonRoutedResident, runtimeReserve, expertGroupBytes, layers, expertsPerLayer)
		if err != nil {
			b.Fatalf("plan failed: %v", err)
		}
		if plan.ResidentRoutedGroups <= 0 {
			b.Fatalf("expected positive resident groups: %d", plan.ResidentRoutedGroups)
		}
	}
}
