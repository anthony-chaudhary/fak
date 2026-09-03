package polymodel

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestRunTreeVerifyMicrobenchmarkDefaults(t *testing.T) {
	// Empty config should populate sensible defaults
	cfg := TreeVerifyBenchConfig{}
	res := RunTreeVerifyMicrobenchmark(cfg)

	if res.Backend != "simulated" {
		t.Errorf("expected backend 'simulated', got %s", res.Backend)
	}
	if res.BatchSize != 1 {
		t.Errorf("expected batch_size 1, got %d", res.BatchSize)
	}
	if res.TreeSize != 16 {
		t.Errorf("expected tree_size 16, got %d", res.TreeSize)
	}
	if res.Topology != "wide" {
		t.Errorf("expected topology 'wide', got %s", res.Topology)
	}
	if res.TreeDepth <= 0 {
		t.Errorf("expected tree_depth > 0, got %d", res.TreeDepth)
	}
	if res.SingleTokenUs <= 0 {
		t.Errorf("expected single_token_us > 0, got %f", res.SingleTokenUs)
	}
	if res.TreeVerifyUs <= 0 {
		t.Errorf("expected tree_verify_us > 0, got %f", res.TreeVerifyUs)
	}
	if res.OverheadRatio <= 0 {
		t.Errorf("expected overhead_ratio > 0, got %f", res.OverheadRatio)
	}
	if res.BreakEven != res.OverheadRatio {
		t.Errorf("expected break_even == overhead_ratio (%f != %f)", res.BreakEven, res.OverheadRatio)
	}
}

func TestRunTreeVerifyMicrobenchmarkMatrixAndOverhead(t *testing.T) {
	// Requirements:
	// - Verify RunTreeVerifyMicrobenchmark for K in {4, 8, 16, 32, 64} and batches B in {1, 2, 4, 8}.
	// - Verify that at batch_size = 1 for tree sizes K <= 32, simulated overhead stays <= 1.15x.
	treeSizes := []int{4, 8, 16, 32, 64}
	batchSizes := []int{1, 2, 4, 8}

	for _, bSize := range batchSizes {
		for _, k := range treeSizes {
			testName := fmt.Sprintf("B=%d/K=%d", bSize, k)
			t.Run(testName, func(t *testing.T) {
				cfg := TreeVerifyBenchConfig{
					Backend:     "simulated",
					BatchSize:   bSize,
					TreeSize:    k,
					Topology:    "wide",
					WarmupRuns:  50,
					MeasureRuns: 500,
				}
				res := RunTreeVerifyMicrobenchmark(cfg)

				if res.BatchSize != bSize {
					t.Errorf("expected batch_size %d, got %d", bSize, res.BatchSize)
				}
				if res.TreeSize != k {
					t.Errorf("expected tree_size %d, got %d", k, res.TreeSize)
				}
				if res.SingleTokenUs <= 0 {
					t.Errorf("expected positive single_token_us, got %f", res.SingleTokenUs)
				}
				if res.TreeVerifyUs <= 0 {
					t.Errorf("expected positive tree_verify_us, got %f", res.TreeVerifyUs)
				}
				if res.OverheadRatio <= 0 {
					t.Errorf("expected positive overhead_ratio, got %f", res.OverheadRatio)
				}
				if res.BreakEven != res.OverheadRatio {
					t.Errorf("expected break_even == overhead_ratio (%f != %f)", res.BreakEven, res.OverheadRatio)
				}

				// Key invariant: at batch_size = 1 for K <= 32, overhead ratio must not exceed 1.15x
				if bSize == 1 && k <= 32 {
					if res.OverheadRatio > 1.15 {
						t.Errorf("batch_size=1, K=%d: simulated overhead ratio %.4f exceeded 1.15x limit",
							k, res.OverheadRatio)
					}
				}
			})
		}
	}
}

func TestRunTreeVerifyMicrobenchmarkWithExplicitLatencies(t *testing.T) {
	cfg := TreeVerifyBenchConfig{
		Backend:       "cuda",
		BatchSize:     2,
		TreeSize:      32,
		Topology:      "linear",
		SingleTokenUs: 500.0,
		TreeVerifyUs:  550.0,
	}

	res := RunTreeVerifyMicrobenchmark(cfg)

	if res.Backend != "cuda" {
		t.Errorf("expected backend 'cuda', got %s", res.Backend)
	}
	if res.BatchSize != 2 {
		t.Errorf("expected batch_size 2, got %d", res.BatchSize)
	}
	if res.TreeSize != 32 {
		t.Errorf("expected tree_size 32, got %d", res.TreeSize)
	}
	if res.SingleTokenUs != 500.0 {
		t.Errorf("expected single_token_us 500.0, got %f", res.SingleTokenUs)
	}
	if res.TreeVerifyUs != 550.0 {
		t.Errorf("expected tree_verify_us 550.0, got %f", res.TreeVerifyUs)
	}
	expectedOverhead := 550.0 / 500.0
	if res.OverheadRatio != expectedOverhead {
		t.Errorf("expected overhead_ratio %f, got %f", expectedOverhead, res.OverheadRatio)
	}
	if res.BreakEven != expectedOverhead {
		t.Errorf("expected break_even %f, got %f", expectedOverhead, res.BreakEven)
	}
}

func TestRunTreeVerifyMicrobenchmarkTopologies(t *testing.T) {
	topologies := []string{"linear", "wide", "deep"}
	for _, top := range topologies {
		cfg := TreeVerifyBenchConfig{
			Backend:     "cpu",
			BatchSize:   1,
			TreeSize:    8,
			Topology:    top,
			WarmupRuns:  20,
			MeasureRuns: 200,
		}
		res := RunTreeVerifyMicrobenchmark(cfg)

		if res.TreeSize != 8 {
			t.Errorf("topology %s: expected tree size 8, got %d", top, res.TreeSize)
		}
		if res.Topology != top {
			t.Errorf("expected topology %s, got %s", top, res.Topology)
		}
		if res.OverheadRatio <= 0 {
			t.Errorf("topology %s: expected positive overhead ratio, got %f", top, res.OverheadRatio)
		}
	}
}

func TestTreeVerifyBenchResultJSONSchema(t *testing.T) {
	cfg := TreeVerifyBenchConfig{
		Backend:       "metal",
		BatchSize:     1,
		TreeSize:      16,
		TreeDepth:     3,
		Topology:      "wide",
		SingleTokenUs: 1000.0,
		TreeVerifyUs:  1080.0,
	}

	res := RunTreeVerifyMicrobenchmark(cfg)

	data, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	// 1. Verify exact required schema keys are present
	var rawMap map[string]any
	if err := json.Unmarshal(data, &rawMap); err != nil {
		t.Fatalf("json.Unmarshal into map failed: %v", err)
	}

	requiredKeys := []string{
		"backend",
		"batch_size",
		"tree_size",
		"tree_depth",
		"topology",
		"single_token_us",
		"tree_verify_us",
		"overhead_ratio",
		"break_even",
	}

	for _, key := range requiredKeys {
		if _, ok := rawMap[key]; !ok {
			t.Errorf("missing required JSON schema key: %q", key)
		}
	}

	// 2. Verify roundtrip unmarshaling back to TreeVerifyBenchResult
	var decoded TreeVerifyBenchResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal into struct failed: %v", err)
	}

	if decoded.Backend != res.Backend ||
		decoded.BatchSize != res.BatchSize ||
		decoded.TreeSize != res.TreeSize ||
		decoded.TreeDepth != res.TreeDepth ||
		decoded.Topology != res.Topology ||
		decoded.SingleTokenUs != res.SingleTokenUs ||
		decoded.TreeVerifyUs != res.TreeVerifyUs ||
		decoded.OverheadRatio != res.OverheadRatio ||
		decoded.BreakEven != res.BreakEven {
		t.Errorf("roundtrip mismatch: got %+v, want %+v", decoded, res)
	}

	// 3. Verify FormatTreeVerifyBenchJSON formatting
	formatted, err := FormatTreeVerifyBenchJSON(res)
	if err != nil {
		t.Fatalf("FormatTreeVerifyBenchJSON failed: %v", err)
	}
	if len(formatted) == 0 {
		t.Fatal("FormatTreeVerifyBenchJSON returned empty output")
	}
}

// ---------------------------------------------------------------------------
// Benchmarks for RunTreeVerifyMicrobenchmark across K and B
// ---------------------------------------------------------------------------

func BenchmarkRunTreeVerifyMicrobenchmark(b *testing.B) {
	treeSizes := []int{4, 8, 16, 32, 64}
	batchSizes := []int{1, 2, 4, 8}

	for _, bSize := range batchSizes {
		for _, k := range treeSizes {
			benchName := fmt.Sprintf("B=%d/K=%d", bSize, k)
			b.Run(benchName, func(b *testing.B) {
				cfg := TreeVerifyBenchConfig{
					Backend:     "simulated",
					BatchSize:   bSize,
					TreeSize:    k,
					Topology:    "wide",
					WarmupRuns:  10,
					MeasureRuns: 50,
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = RunTreeVerifyMicrobenchmark(cfg)
				}
			})
		}
	}
}
