package modelperfobs

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestRunSubagentFanoutMatrix_ConcurrencyMatrixExecution(t *testing.T) {
	cfg := MatrixConfig{
		Repetitions: 5,
		Seed:        42,
	}

	receipt, err := RunSubagentFanoutMatrix(cfg)
	if err != nil {
		t.Fatalf("RunSubagentFanoutMatrix failed: %v", err)
	}

	if receipt == nil {
		t.Fatal("expected non-nil receipt")
	}

	if receipt.Schema != SubagentFanoutSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, SubagentFanoutSchema)
	}
	if receipt.Issue != IssueSubagentFanoutMatrix {
		t.Errorf("issue = %d, want %d", receipt.Issue, IssueSubagentFanoutMatrix)
	}
	if receipt.Hardware.SustainableCeilingGBS != 224.0 {
		t.Errorf("sustainable ceiling = %.2f, want 224.0", receipt.Hardware.SustainableCeilingGBS)
	}
	if receipt.Hardware.PeakDRAMBandwidthGBps != 273.056 {
		t.Errorf("peak dram bandwidth = %.3f, want 273.056", receipt.Hardware.PeakDRAMBandwidthGBps)
	}
	if receipt.Hardware.MALLCacheSizeMB != 32 {
		t.Errorf("mall cache size = %d, want 32", receipt.Hardware.MALLCacheSizeMB)
	}

	// The concurrency matrix must have 6 configurations:
	// 1. Cold start (B=1, 0% cache hit)
	// 2. Warm same-prefix (B=1, 100% prefix hit)
	// 3. Shared-prefix forked (B=1)
	// 4. Shared-prefix forked (B=2)
	// 5. Shared-prefix forked (B=4)
	// 6. Shared-prefix forked (B=8)
	if len(receipt.MatrixResults) != 6 {
		t.Fatalf("expected 6 matrix scenario results, got %d", len(receipt.MatrixResults))
	}

	scenarioMap := make(map[string]MatrixScenarioResult)
	for _, res := range receipt.MatrixResults {
		key := res.Scenario
		if res.Scenario == MatrixScenarioSharedPrefixForked {
			key = key + fmtConcurrency(res.Concurrency)
		}
		scenarioMap[key] = res
	}

	// 1. Cold start verification (B=1, 0% cache hit)
	cold, ok := scenarioMap[MatrixScenarioCold]
	if !ok {
		t.Fatal("cold start scenario missing from matrix results")
	}
	if cold.Concurrency != 1 {
		t.Errorf("cold concurrency = %d, want 1", cold.Concurrency)
	}
	if cold.TotalUsefulTokens != 64 {
		t.Errorf("cold useful tokens = %d, want 64", cold.TotalUsefulTokens)
	}
	if cold.MALLHitRatioStats.Mean != 0.0 {
		t.Errorf("cold MALL hit ratio mean = %f, want 0.0", cold.MALLHitRatioStats.Mean)
	}
	if cold.PhysicalDRAMBytesStats.Mean <= 0 {
		t.Errorf("cold physical DRAM bytes transferred = %f, want > 0", cold.PhysicalDRAMBytesStats.Mean)
	}
	if cold.QueueLatencyMSStats.Mean <= 0 {
		t.Errorf("cold queue latency = %f, want > 0", cold.QueueLatencyMSStats.Mean)
	}
	if cold.AchievedBandwidthStats.Mean <= 0 {
		t.Errorf("cold achieved bandwidth = %f, want > 0", cold.AchievedBandwidthStats.Mean)
	}

	// 2. Warm same-prefix verification (B=1, 100% prefix hit)
	warm, ok := scenarioMap[MatrixScenarioWarmSamePrefix]
	if !ok {
		t.Fatal("warm same-prefix scenario missing from matrix results")
	}
	if warm.Concurrency != 1 {
		t.Errorf("warm concurrency = %d, want 1", warm.Concurrency)
	}
	if warm.TotalUsefulTokens != 64 {
		t.Errorf("warm useful tokens = %d, want 64", warm.TotalUsefulTokens)
	}
	if warm.MALLHitRatioStats.Mean != 1.0 {
		t.Errorf("warm MALL hit ratio mean = %f, want 1.0", warm.MALLHitRatioStats.Mean)
	}
	if !warm.FitsInMALL {
		t.Error("warm working set (2048 tokens = 256 KB) must fit in 32 MB MALL cache")
	}

	// 3. Shared-prefix forked subagents (B in {1, 2, 4, 8})
	forkedConcurrencies := []int{1, 2, 4, 8}
	var prevQueueLatency float64
	var prevAchievedBW float64

	for _, b := range forkedConcurrencies {
		key := MatrixScenarioSharedPrefixForked + fmtConcurrency(b)
		forked, ok := scenarioMap[key]
		if !ok {
			t.Fatalf("shared-prefix forked B=%d missing from matrix results", b)
		}
		if forked.Concurrency != b {
			t.Errorf("forked concurrency = %d, want %d", forked.Concurrency, b)
		}
		expectedTokens := b * 64
		if forked.TotalUsefulTokens != expectedTokens {
			t.Errorf("forked B=%d total useful tokens = %d, want %d", b, forked.TotalUsefulTokens, expectedTokens)
		}
		if !forked.FitsInMALL {
			t.Errorf("forked B=%d working set (30000 tokens = 3.84 MB) must fit in 32 MB MALL cache", b)
		}
		// MALL cache hit ratio for working set <= 32 MB must be >= 90%
		if forked.MALLHitRatioStats.Mean < 0.90 {
			t.Errorf("forked B=%d MALL hit ratio = %f, want >= 0.90", b, forked.MALLHitRatioStats.Mean)
		}
		if forked.QueueLatencyMSStats.Mean <= 0 {
			t.Errorf("forked B=%d queue latency = %f, want > 0", b, forked.QueueLatencyMSStats.Mean)
		}
		if forked.AchievedBandwidthStats.Mean <= 0 {
			t.Errorf("forked B=%d achieved bandwidth = %f, want > 0", b, forked.AchievedBandwidthStats.Mean)
		}
		if forked.AttainmentCeilingStats.Mean <= 0 {
			t.Errorf("forked B=%d attainment ceiling pct = %f, want > 0", b, forked.AttainmentCeilingStats.Mean)
		}
		if forked.AttainmentPeakStats.Mean <= 0 {
			t.Errorf("forked B=%d attainment peak pct = %f, want > 0", b, forked.AttainmentPeakStats.Mean)
		}

		// Queue latency should be tracked and positive; verify monotonic trend or positive delta
		if prevQueueLatency > 0 && forked.QueueLatencyMSStats.Mean < prevQueueLatency {
			t.Logf("note: queue latency B=%d (%.3f) vs B_prev (%.3f)", b, forked.QueueLatencyMSStats.Mean, prevQueueLatency)
		}
		prevQueueLatency = forked.QueueLatencyMSStats.Mean

		// Achieved bandwidth scales with batch size B toward ceiling (224 GB/s)
		if prevAchievedBW > 0 && forked.AchievedBandwidthStats.Mean < prevAchievedBW {
			t.Errorf("expected achieved bandwidth to scale with concurrency B=%d: got %.2f < %.2f",
				b, forked.AchievedBandwidthStats.Mean, prevAchievedBW)
		}
		prevAchievedBW = forked.AchievedBandwidthStats.Mean
	}

	// Verify global roofline attainment
	if !receipt.RooflineAttainment.Achieved {
		t.Errorf("expected roofline attainment to achieve >= 80%% floor, got %.2f%%",
			receipt.RooflineAttainment.AttainmentPercentage)
	}
	if receipt.RooflineAttainment.AchievedBandwidthGBps <= 180.0 {
		t.Errorf("expected B=8 achieved bandwidth > 180 GB/s, got %.2f",
			receipt.RooflineAttainment.AchievedBandwidthGBps)
	}
}

func TestRunSubagentFanoutMatrix_StatisticalRigor(t *testing.T) {
	// 1. Exact deterministic verification of ComputeDistributionStats
	t.Run("known sample distribution", func(t *testing.T) {
		samples := []float64{10.0, 20.0, 30.0, 40.0, 50.0}
		stats := ComputeDistributionStats(samples)

		if stats.Mean != 30.0 {
			t.Errorf("mean = %f, want 30.0", stats.Mean)
		}
		// Population variance: ((10-30)^2 + (20-30)^2 + 0 + (40-30)^2 + (50-30)^2)/5 = (400+100+0+100+400)/5 = 200
		if stats.Variance != 200.0 {
			t.Errorf("variance = %f, want 200.0", stats.Variance)
		}
		expectedStdDev := math.Sqrt(200.0)
		if math.Abs(stats.StdDev-expectedStdDev) > 1e-9 {
			t.Errorf("stddev = %f, want %f", stats.StdDev, expectedStdDev)
		}
		if stats.P50 != 30.0 {
			t.Errorf("p50 = %f, want 30.0", stats.P50)
		}
		if stats.P95 != 50.0 {
			t.Errorf("p95 = %f, want 50.0", stats.P95)
		}
		if stats.Min != 10.0 || stats.Max != 50.0 {
			t.Errorf("min/max = %f/%f, want 10.0/50.0", stats.Min, stats.Max)
		}
	})

	t.Run("single and even sample distributions", func(t *testing.T) {
		single := ComputeDistributionStats([]float64{42.0})
		if single.Mean != 42.0 || single.P50 != 42.0 || single.P95 != 42.0 || single.Variance != 0 {
			t.Errorf("unexpected single stats: %+v", single)
		}

		even := ComputeDistributionStats([]float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0})
		if even.Mean != 35.0 || even.P50 != 35.0 {
			t.Errorf("unexpected even stats: %+v", even)
		}

		empty := ComputeDistributionStats(nil)
		if empty.Mean != 0 || empty.P50 != 0 || empty.Variance != 0 {
			t.Errorf("unexpected empty stats: %+v", empty)
		}
	})

	// 2. Repetition runs count >= 5
	t.Run("runs count >= 5 verification", func(t *testing.T) {
		cfg := MatrixConfig{
			Repetitions: 7, // 7 runs >= 5
			Seed:        101,
		}

		receipt, err := RunSubagentFanoutMatrix(cfg)
		if err != nil {
			t.Fatalf("RunSubagentFanoutMatrix failed: %v", err)
		}

		for _, res := range receipt.MatrixResults {
			if res.Repetitions != 7 {
				t.Errorf("scenario %s repetitions = %d, want 7", res.Scenario, res.Repetitions)
			}
			if len(res.Runs) != 7 {
				t.Errorf("scenario %s runs count = %d, want 7", res.Scenario, len(res.Runs))
			}

			// Invariance checks: Min <= P50 <= P95 <= Max
			if res.TokensPerSecStats.Min > res.TokensPerSecStats.P50 {
				t.Errorf("%s: min %f > p50 %f", res.Scenario, res.TokensPerSecStats.Min, res.TokensPerSecStats.P50)
			}
			if res.TokensPerSecStats.P50 > res.TokensPerSecStats.P95 {
				t.Errorf("%s: p50 %f > p95 %f", res.Scenario, res.TokensPerSecStats.P50, res.TokensPerSecStats.P95)
			}
			if res.TokensPerSecStats.P95 > res.TokensPerSecStats.Max {
				t.Errorf("%s: p95 %f > max %f", res.Scenario, res.TokensPerSecStats.P95, res.TokensPerSecStats.Max)
			}

			// Variance and stddev non-negative
			if res.AchievedBandwidthStats.Variance < 0 {
				t.Errorf("%s: negative variance: %f", res.Scenario, res.AchievedBandwidthStats.Variance)
			}
			if res.DurationMSStats.Variance < 0 {
				t.Errorf("%s: negative duration variance: %f", res.Scenario, res.DurationMSStats.Variance)
			}
			if res.QueueLatencyMSStats.Variance < 0 {
				t.Errorf("%s: negative queue latency variance: %f", res.Scenario, res.QueueLatencyMSStats.Variance)
			}
		}
	})
}

func TestRunSubagentFanoutMatrix_SerializationAndReceiptStructure(t *testing.T) {
	cfg := MatrixConfig{
		Repetitions: 5,
		Seed:        999,
	}

	receipt, err := RunSubagentFanoutMatrix(cfg)
	if err != nil {
		t.Fatalf("RunSubagentFanoutMatrix failed: %v", err)
	}

	// 1. Structure invariants
	if receipt.Schema != "fak.benchmark.subagent_fanout/v1" {
		t.Errorf("unexpected schema %q", receipt.Schema)
	}
	if !strings.HasPrefix(receipt.Digest, "sha256:") {
		t.Errorf("digest %q does not start with sha256:", receipt.Digest)
	}
	if !receipt.Verified {
		t.Error("expected receipt to be verified")
	}

	// 2. JSON Serialization
	data, err := receipt.JSON()
	if err != nil {
		t.Fatalf("receipt.JSON failed: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("empty JSON output")
	}

	// 3. Unmarshaling round-trip
	unmarshaled, err := UnmarshalFanoutMatrixReceipt(data)
	if err != nil {
		t.Fatalf("UnmarshalFanoutMatrixReceipt failed: %v", err)
	}

	if unmarshaled.Schema != receipt.Schema {
		t.Errorf("unmarshaled schema = %q, want %q", unmarshaled.Schema, receipt.Schema)
	}
	if unmarshaled.Issue != receipt.Issue {
		t.Errorf("unmarshaled issue = %d, want %d", unmarshaled.Issue, receipt.Issue)
	}
	if unmarshaled.Digest != receipt.Digest {
		t.Errorf("unmarshaled digest = %q, want %q", unmarshaled.Digest, receipt.Digest)
	}
	if len(unmarshaled.MatrixResults) != len(receipt.MatrixResults) {
		t.Errorf("unmarshaled matrix results count = %d, want %d",
			len(unmarshaled.MatrixResults), len(receipt.MatrixResults))
	}

	// 4. Bad schema rejection in unmarshaling
	var rawMap map[string]interface{}
	if err := json.Unmarshal(data, &rawMap); err != nil {
		t.Fatal(err)
	}
	rawMap["schema"] = "fak.benchmark.subagent_fanout/v0"
	tamperedData, _ := json.Marshal(rawMap)
	if _, err := UnmarshalFanoutMatrixReceipt(tamperedData); err == nil {
		t.Error("expected error unmarshaling tampered schema, got nil")
	}

	// 5. Corrupted JSON rejection
	if _, err := UnmarshalFanoutMatrixReceipt([]byte("{bad-json")); err == nil {
		t.Error("expected error unmarshaling invalid JSON, got nil")
	}

	// 6. Summary table printer verification
	table := receipt.SummaryTable()
	if table == "" {
		t.Fatal("expected non-empty summary table")
	}
	if !strings.Contains(table, "fak.benchmark.subagent_fanout/v1") {
		t.Error("summary table missing schema")
	}
	if !strings.Contains(table, "224.00 GB/s") {
		t.Error("summary table missing sustainable ceiling")
	}
	if !strings.Contains(table, "273.056 GB/s") {
		t.Error("summary table missing theoretical peak")
	}
	if !strings.Contains(table, "MALL Cache: 32 MB") {
		t.Error("summary table missing MALL cache spec")
	}
	if !strings.Contains(table, "cold") {
		t.Error("summary table missing cold scenario")
	}
	if !strings.Contains(table, "warm_same_prefix") {
		t.Error("summary table missing warm scenario")
	}
	if !strings.Contains(table, "shared_prefix_forked") {
		t.Error("summary table missing shared prefix scenario")
	}

	// 7. PrintSummaryTable
	var buf bytes.Buffer
	if err := receipt.PrintSummaryTable(&buf); err != nil {
		t.Fatalf("PrintSummaryTable failed: %v", err)
	}
	if buf.String() != table {
		t.Error("PrintSummaryTable output does not match SummaryTable")
	}

	// 8. String() implements fmt.Stringer
	if receipt.String() != table {
		t.Error("receipt.String() does not match SummaryTable")
	}
}

func TestRunSubagentFanoutMatrix_NegativeParameterHandling(t *testing.T) {
	testCases := []struct {
		name string
		cfg  MatrixConfig
	}{
		{
			name: "negative repetitions",
			cfg:  MatrixConfig{Repetitions: -1},
		},
		{
			name: "repetitions less than 5",
			cfg:  MatrixConfig{Repetitions: 4},
		},
		{
			name: "negative theoretical peak",
			cfg:  MatrixConfig{TheoreticalPeakGBps: -10.0},
		},
		{
			name: "negative sustainable ceiling",
			cfg:  MatrixConfig{SustainableCeilingGBps: -5.0},
		},
		{
			name: "negative mall cache capacity",
			cfg:  MatrixConfig{MALLCacheCapacityMB: -32},
		},
		{
			name: "negative shared prefix tokens",
			cfg:  MatrixConfig{SharedPrefixTokens: -100},
		},
		{
			name: "negative warm prefix tokens",
			cfg:  MatrixConfig{WarmPrefixTokens: -100},
		},
		{
			name: "negative cold prompt tokens",
			cfg:  MatrixConfig{ColdPromptTokens: -100},
		},
		{
			name: "negative generated tokens per subagent",
			cfg:  MatrixConfig{GeneratedTokensPerSubagent: -64},
		},
		{
			name: "negative bytes per token",
			cfg:  MatrixConfig{BytesPerToken: -128},
		},
		{
			name: "zero concurrency in forked",
			cfg:  MatrixConfig{ForkedConcurrencies: []int{0}},
		},
		{
			name: "negative concurrency in forked",
			cfg:  MatrixConfig{ForkedConcurrencies: []int{1, -2, 4}},
		},
		{
			name: "scenario with unknown type",
			cfg: MatrixConfig{
				Scenarios: []MatrixScenarioConfig{
					{Scenario: "invalid_scenario", Concurrency: 1},
				},
			},
		},
		{
			name: "scenario with negative concurrency",
			cfg: MatrixConfig{
				Scenarios: []MatrixScenarioConfig{
					{Scenario: MatrixScenarioCold, Concurrency: -1},
				},
			},
		},
		{
			name: "scenario with negative prefix tokens",
			cfg: MatrixConfig{
				Scenarios: []MatrixScenarioConfig{
					{Scenario: MatrixScenarioCold, Concurrency: 1, PrefixTokens: -50},
				},
			},
		},
		{
			name: "scenario with negative generated tokens",
			cfg: MatrixConfig{
				Scenarios: []MatrixScenarioConfig{
					{Scenario: MatrixScenarioCold, Concurrency: 1, GeneratedTokensPerSubagent: -10},
				},
			},
		},
		{
			name: "scenario with invalid cache hit expectation",
			cfg: MatrixConfig{
				Scenarios: []MatrixScenarioConfig{
					{Scenario: MatrixScenarioCold, Concurrency: 1, CacheHitExpectation: 1.5},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			receipt, err := RunSubagentFanoutMatrix(tc.cfg)
			if err == nil {
				t.Fatalf("expected error for %s, got nil (receipt: %+v)", tc.name, receipt)
			}
		})
	}
}

func TestRunSubagentFanoutMatrix_MALLWorkingSetBoundary(t *testing.T) {
	// Verify behavior when working set <= 32 MB vs > 32 MB
	// 32 MB = 33,554,432 bytes. At 128 bytes/token, 32 MB corresponds to 262,144 tokens.

	t.Run("working set <= 32 MB fits in MALL", func(t *testing.T) {
		cfg := MatrixConfig{
			Repetitions:         5,
			SharedPrefixTokens:  30000, // 30000 * 128 = 3.84 MB <= 32 MB
			ForkedConcurrencies: []int{4},
			Seed:                777,
		}
		receipt, err := RunSubagentFanoutMatrix(cfg)
		if err != nil {
			t.Fatal(err)
		}

		for _, res := range receipt.MatrixResults {
			if res.Scenario == MatrixScenarioSharedPrefixForked {
				if !res.FitsInMALL {
					t.Errorf("expected 30k tokens (3.84 MB) to fit in MALL, got fitsInMALL=false")
				}
				if res.MALLHitRatioStats.Mean < 0.90 {
					t.Errorf("MALL hit ratio = %f, want >= 0.90", res.MALLHitRatioStats.Mean)
				}
			}
		}
	})

	t.Run("working set > 32 MB exceeds MALL and spills", func(t *testing.T) {
		// 300,000 tokens * 128 bytes = 38,400,000 bytes = 38.4 MB > 32 MB MALL
		cfg := MatrixConfig{
			Repetitions: 5,
			Scenarios: []MatrixScenarioConfig{
				{
					Scenario:                   MatrixScenarioSharedPrefixForked,
					Concurrency:                4,
					PrefixTokens:               300000,
					GeneratedTokensPerSubagent: 64,
					CacheHitExpectation:        1.0,
				},
			},
			Seed: 888,
		}
		receipt, err := RunSubagentFanoutMatrix(cfg)
		if err != nil {
			t.Fatal(err)
		}

		if len(receipt.MatrixResults) != 1 {
			t.Fatalf("expected 1 result, got %d", len(receipt.MatrixResults))
		}
		res := receipt.MatrixResults[0]
		if res.FitsInMALL {
			t.Errorf("expected 300k tokens (38.4 MB) to exceed 32 MB MALL, got fitsInMALL=true")
		}
		// Hit ratio should be capped by capacity ratio (32 MB / 38.4 MB ≈ 0.833)
		if res.MALLHitRatioStats.Mean > 0.86 {
			t.Errorf("expected spilled hit ratio <= 0.86, got %f", res.MALLHitRatioStats.Mean)
		}
	})
}

func fmtConcurrency(b int) string {
	return fmt.Sprintf("_b%d", b)
}
