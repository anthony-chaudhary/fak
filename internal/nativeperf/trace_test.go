package nativeperf

import (
	"encoding/json"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestTurnTrace_AllFiveBucketsCapturedAndPopulated verifies that all 5 nanosecond-resolution
// phase buckets are captured and correctly populated.
func TestTurnTrace_AllFiveBucketsCapturedAndPopulated(t *testing.T) {
	tracer := NewTurnLatencyTracer("test-turn-1")

	// Timings in nanoseconds:
	// HostDispatch: 150 µs = 150,000 ns
	// PrefixLookup:  40 µs =  40,000 ns
	// KVAllocation:  30 µs =  30,000 ns
	// GPUKernel:    700 µs = 700,000 ns
	// TokenSampling: 80 µs =  80,000 ns
	// Total:       1000 µs = 1,000,000 ns
	const (
		hostDispatchNs  int64 = 150_000
		prefixLookupNs  int64 = 40_000
		kvAllocationNs  int64 = 30_000
		gpuKernelNs     int64 = 700_000
		tokenSamplingNs int64 = 80_000
		totalWallNs     int64 = 1_000_000
	)

	tracer.RecordPhaseNs(TurnPhaseHostDispatch, hostDispatchNs)
	tracer.RecordPhaseNs(TurnPhasePrefixLookup, prefixLookupNs)
	tracer.RecordPhaseNs(TurnPhaseKVAllocation, kvAllocationNs)
	tracer.RecordPhaseNs(TurnPhaseGPUKernel, gpuKernelNs)
	tracer.RecordPhaseNs(TurnPhaseTokenSampling, tokenSamplingNs)
	tracer.SetTotalWallNs(totalWallNs)

	report, err := tracer.Report()
	if err != nil {
		t.Fatalf("expected valid report, got error: %v", err)
	}

	// 1. Verify schema
	if report.Schema != TurnTraceSchema {
		t.Errorf("schema = %q, want %q", report.Schema, TurnTraceSchema)
	}
	if report.TurnID != "test-turn-1" {
		t.Errorf("turnID = %q, want %q", report.TurnID, "test-turn-1")
	}

	// 2. Verify all 5 nanosecond-resolution phase buckets
	if report.HostDispatchNs != hostDispatchNs {
		t.Errorf("HostDispatchNs = %d, want %d", report.HostDispatchNs, hostDispatchNs)
	}
	if report.PrefixLookupNs != prefixLookupNs {
		t.Errorf("PrefixLookupNs = %d, want %d", report.PrefixLookupNs, prefixLookupNs)
	}
	if report.KVAllocationNs != kvAllocationNs {
		t.Errorf("KVAllocationNs = %d, want %d", report.KVAllocationNs, kvAllocationNs)
	}
	if report.GPUKernelNs != gpuKernelNs {
		t.Errorf("GPUKernelNs = %d, want %d", report.GPUKernelNs, gpuKernelNs)
	}
	if report.TokenSamplingNs != tokenSamplingNs {
		t.Errorf("TokenSamplingNs = %d, want %d", report.TokenSamplingNs, tokenSamplingNs)
	}

	// 3. Verify microsecond values
	if report.HostDispatchUS != 150.0 {
		t.Errorf("HostDispatchUS = %.2f, want 150.0", report.HostDispatchUS)
	}
	if report.PrefixLookupUS != 40.0 {
		t.Errorf("PrefixLookupUS = %.2f, want 40.0", report.PrefixLookupUS)
	}
	if report.KVAllocationUS != 30.0 {
		t.Errorf("KVAllocationUS = %.2f, want 30.0", report.KVAllocationUS)
	}
	if report.GPUKernelUS != 700.0 {
		t.Errorf("GPUKernelUS = %.2f, want 700.0", report.GPUKernelUS)
	}
	if report.TokenSamplingUS != 80.0 {
		t.Errorf("TokenSamplingUS = %.2f, want 80.0", report.TokenSamplingUS)
	}

	// 4. Verify percentage breakdown
	const tolerance = 0.001
	if math.Abs(report.HostDispatchPct-15.0) > tolerance {
		t.Errorf("HostDispatchPct = %.2f%%, want 15.0%%", report.HostDispatchPct)
	}
	if math.Abs(report.PrefixLookupPct-4.0) > tolerance {
		t.Errorf("PrefixLookupPct = %.2f%%, want 4.0%%", report.PrefixLookupPct)
	}
	if math.Abs(report.KVAllocationPct-3.0) > tolerance {
		t.Errorf("KVAllocationPct = %.2f%%, want 3.0%%", report.KVAllocationPct)
	}
	if math.Abs(report.GPUKernelPct-70.0) > tolerance {
		t.Errorf("GPUKernelPct = %.2f%%, want 70.0%%", report.GPUKernelPct)
	}
	if math.Abs(report.TokenSamplingPct-8.0) > tolerance {
		t.Errorf("TokenSamplingPct = %.2f%%, want 8.0%%", report.TokenSamplingPct)
	}

	// 5. Verify total wall time
	if report.TotalWallNs != totalWallNs {
		t.Errorf("TotalWallNs = %d, want %d", report.TotalWallNs, totalWallNs)
	}
	if report.TotalWallUS != 1000.0 {
		t.Errorf("TotalWallUS = %.2f, want 1000.0", report.TotalWallUS)
	}

	// 6. Verify detailed breakdown map and descriptions
	for _, p := range CanonicalTurnPhases {
		m, ok := report.Phases[p]
		if !ok {
			t.Errorf("missing phase %q in report.Phases map", p)
			continue
		}
		if m.Phase != p {
			t.Errorf("phase mismatch: got %q, want %q", m.Phase, p)
		}
		if m.Nanoseconds <= 0 {
			t.Errorf("expected positive nanoseconds for %q, got %d", p, m.Nanoseconds)
		}
		if m.Microseconds <= 0 {
			t.Errorf("expected positive microseconds for %q, got %.2f", p, m.Microseconds)
		}
		if m.Percentage <= 0 {
			t.Errorf("expected positive percentage for %q, got %.2f%%", p, m.Percentage)
		}
		if m.Description == "" {
			t.Errorf("expected non-empty description for phase %q", p)
		}
	}
}

// TestTurnTrace_SumInvariantMatchesTotalDuration verifies that Sum(Phases) matches
// TotalWallClock within 0.5% tolerance.
func TestTurnTrace_SumInvariantMatchesTotalDuration(t *testing.T) {
	t.Run("exact_match_passes", func(t *testing.T) {
		phases := map[TurnPhase]int64{
			TurnPhaseHostDispatch:  200_000,
			TurnPhasePrefixLookup:  100_000,
			TurnPhaseKVAllocation:  100_000,
			TurnPhaseGPUKernel:     500_000,
			TurnPhaseTokenSampling: 100_000,
		}
		const totalWallNs = 1_000_000 // exact sum: 200k + 100k + 100k + 500k + 100k = 1M

		report, err := NewTurnTraceReport("exact", totalWallNs, phases, 0)
		if err != nil {
			t.Fatalf("expected valid report, got: %v", err)
		}
		if !report.SumValid {
			t.Errorf("expected SumValid = true")
		}
		if report.SumDeltaNs != 0 {
			t.Errorf("expected SumDeltaNs = 0, got %d", report.SumDeltaNs)
		}
		if report.SumDeltaPct != 0 {
			t.Errorf("expected SumDeltaPct = 0, got %f", report.SumDeltaPct)
		}
		if err := report.AssertSumInvariant(); err != nil {
			t.Errorf("AssertSumInvariant failed: %v", err)
		}
	})

	t.Run("within_tolerance_0.2_percent_passes", func(t *testing.T) {
		phases := map[TurnPhase]int64{
			TurnPhaseHostDispatch:  200_000,
			TurnPhasePrefixLookup:  100_000,
			TurnPhaseKVAllocation:  100_000,
			TurnPhaseGPUKernel:     500_000,
			TurnPhaseTokenSampling: 100_000,
		}
		// Sum = 1,000,000 ns. TotalWall = 1,002,000 ns (delta = 2,000 ns = 0.20%, <= 0.5%)
		const totalWallNs = 1_002_000

		report, err := NewTurnTraceReport("within-tol", totalWallNs, phases, 0)
		if err != nil {
			t.Fatalf("expected valid report within tolerance, got: %v", err)
		}
		if !report.SumValid {
			t.Errorf("expected SumValid = true for 0.2%% diff")
		}
		if report.SumDeltaNs != 2_000 {
			t.Errorf("expected SumDeltaNs = 2000, got %d", report.SumDeltaNs)
		}
		if err := report.AssertSumInvariant(); err != nil {
			t.Errorf("AssertSumInvariant should pass for 0.2%% drift: %v", err)
		}
	})

	t.Run("boundary_tolerance_0.5_percent_passes", func(t *testing.T) {
		phases := map[TurnPhase]int64{
			TurnPhaseHostDispatch:  200_000,
			TurnPhasePrefixLookup:  100_000,
			TurnPhaseKVAllocation:  100_000,
			TurnPhaseGPUKernel:     500_000,
			TurnPhaseTokenSampling: 100_000,
		}
		// Sum = 1,000,000 ns. TotalWall = 1,005,000 ns (delta = 5,000 ns = 0.4975% <= 0.5%)
		const totalWallNs = 1_005_000

		report, err := NewTurnTraceReport("boundary-tol", totalWallNs, phases, 0)
		if err != nil {
			t.Fatalf("expected valid report at boundary tolerance, got: %v", err)
		}
		if !report.SumValid {
			t.Errorf("expected SumValid = true")
		}
	})

	t.Run("exceeding_tolerance_0.8_percent_fails", func(t *testing.T) {
		phases := map[TurnPhase]int64{
			TurnPhaseHostDispatch:  200_000,
			TurnPhasePrefixLookup:  100_000,
			TurnPhaseKVAllocation:  100_000,
			TurnPhaseGPUKernel:     500_000,
			TurnPhaseTokenSampling: 100_000,
		}
		// Sum = 1,000,000 ns. TotalWall = 1,008,000 ns (delta = 8,000 ns = 0.793% > 0.5%)
		const totalWallNs = 1_008_000

		_, err := NewTurnTraceReport("exceeding-tol", totalWallNs, phases, 0)
		if err == nil {
			t.Fatalf("expected sum invariant error for 0.8%% discrepancy, got nil")
		}
		if !strings.Contains(err.Error(), "sum invariant violated") {
			t.Errorf("expected 'sum invariant violated' error message, got: %v", err)
		}
	})

	t.Run("auto_computed_sum_when_wall_zero", func(t *testing.T) {
		tracer := NewTurnLatencyTracer()
		tracer.RecordPhaseNs(TurnPhaseHostDispatch, 100_000)
		tracer.RecordPhaseNs(TurnPhasePrefixLookup, 50_000)
		tracer.RecordPhaseNs(TurnPhaseKVAllocation, 50_000)
		tracer.RecordPhaseNs(TurnPhaseGPUKernel, 700_000)
		tracer.RecordPhaseNs(TurnPhaseTokenSampling, 100_000)

		report, err := tracer.Report()
		if err != nil {
			t.Fatalf("report failed: %v", err)
		}
		if report.TotalWallNs != 1_000_000 {
			t.Errorf("expected total wall = 1,000,000, got %d", report.TotalWallNs)
		}
		if !report.SumValid {
			t.Errorf("expected SumValid = true")
		}
	})
}

// TestTurnTrace_LowOverheadNanosecondTimers verifies that tracer measurement overhead
// is < 1% of total turn latency.
func TestTurnTrace_LowOverheadNanosecondTimers(t *testing.T) {
	t.Run("simulated_turn_overhead_under_1_percent", func(t *testing.T) {
		tracer := NewTurnLatencyTracer("simulated-turn")

		// Simulate phases with sleep of ~200 µs each (total ~1 ms)
		tracer.StartPhase(TurnPhaseHostDispatch)
		time.Sleep(200 * time.Microsecond)
		tracer.EndPhase()

		tracer.StartPhase(TurnPhasePrefixLookup)
		time.Sleep(100 * time.Microsecond)
		tracer.EndPhase()

		tracer.StartPhase(TurnPhaseKVAllocation)
		time.Sleep(100 * time.Microsecond)
		tracer.EndPhase()

		tracer.StartPhase(TurnPhaseGPUKernel)
		time.Sleep(500 * time.Microsecond)
		tracer.EndPhase()

		tracer.StartPhase(TurnPhaseTokenSampling)
		time.Sleep(100 * time.Microsecond)
		tracer.EndPhase()

		report, err := tracer.Report()
		if err != nil {
			t.Fatalf("report failed: %v", err)
		}

		if err := report.AssertLowOverhead(); err != nil {
			t.Fatalf("tracer overhead check failed: %v (overhead: %d ns, total: %d ns, overhead pct: %.4f%%)",
				err, report.OverheadNs, report.TotalWallNs, report.OverheadPct)
		}

		if !report.OverheadValid {
			t.Errorf("expected OverheadValid = true, got false (OverheadPct: %.4f%%)", report.OverheadPct)
		}

		if report.OverheadPct >= 1.0 {
			t.Errorf("overhead %.4f%% exceeds 1.0%% limit", report.OverheadPct)
		}
	})

	t.Run("nanosecond_timer_raw_overhead_benchmark", func(t *testing.T) {
		tracer := NewTurnLatencyTracer("bench")
		const iterations = 5000

		start := time.Now()
		for i := 0; i < iterations; i++ {
			stop := tracer.StartPhase(TurnPhaseHostDispatch)
			stop()
		}
		totalElapsed := time.Since(start)

		avgNsPerCall := float64(totalElapsed.Nanoseconds()) / float64(iterations)
		t.Logf("Average StartPhase+EndPhase duration: %.2f ns (total %v for %d iterations)",
			avgNsPerCall, totalElapsed, iterations)

		// Verification that nanosecond timer overhead is very small (< 10 µs per call, typically < 300 ns)
		if avgNsPerCall > 10_000.0 {
			t.Errorf("average timer call %.2f ns is too high (> 10 µs)", avgNsPerCall)
		}
	})

	t.Run("overhead_exceeding_limit_fails_assertion", func(t *testing.T) {
		phases := map[TurnPhase]int64{
			TurnPhaseHostDispatch:  200_000,
			TurnPhasePrefixLookup:  100_000,
			TurnPhaseKVAllocation:  100_000,
			TurnPhaseGPUKernel:     500_000,
			TurnPhaseTokenSampling: 100_000,
		}
		const totalWallNs = 1_000_000
		// Overhead of 20,000 ns on 1,000,000 ns turn is 2.0% (> 1.0% limit)
		const overheadNs = 20_000

		_, err := NewTurnTraceReport("high-overhead", totalWallNs, phases, overheadNs)
		if err == nil {
			t.Fatalf("expected overhead check error, got nil")
		}
		if !strings.Contains(err.Error(), "tracer measurement overhead") {
			t.Errorf("expected overhead error message, got: %v", err)
		}
	})
}

// TestTurnTrace_JSONSerializationAndFormatting verifies JSON marshaling, formatting,
// schema conformance, and roundtrip decoding.
func TestTurnTrace_JSONSerializationAndFormatting(t *testing.T) {
	phases := map[TurnPhase]int64{
		TurnPhaseHostDispatch:  120_000,
		TurnPhasePrefixLookup:  45_000,
		TurnPhaseKVAllocation:  35_000,
		TurnPhaseGPUKernel:     750_000,
		TurnPhaseTokenSampling: 50_000,
	}
	const totalWallNs = 1_000_000
	const overheadNs = 400

	report, err := NewTurnTraceReport("turn-serialization", totalWallNs, phases, overheadNs)
	if err != nil {
		t.Fatalf("failed to create report: %v", err)
	}

	data, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON() failed: %v", err)
	}

	jsonStr := string(data)

	// 1. Verify schema string presence
	if !strings.Contains(jsonStr, `"schema": "fak.trace.turn_latency/v1"`) {
		t.Errorf("missing or incorrect schema in JSON:\n%s", jsonStr)
	}

	// 2. Verify all 5 phase buckets are present with snake_case keys
	requiredKeys := []string{
		"host_dispatch_ns",
		"prefix_lookup_ns",
		"kv_allocation_ns",
		"gpu_kernel_ns",
		"token_sampling_ns",
		"host_dispatch_us",
		"prefix_lookup_us",
		"kv_allocation_us",
		"gpu_kernel_us",
		"token_sampling_us",
		"host_dispatch_pct",
		"prefix_lookup_pct",
		"kv_allocation_pct",
		"gpu_kernel_pct",
		"token_sampling_pct",
		"total_wall_ns",
		"total_wall_us",
		"overhead_ns",
		"overhead_pct",
		"sum_valid",
		"overhead_valid",
		"phases",
	}
	for _, key := range requiredKeys {
		if !strings.Contains(jsonStr, `"`+key+`"`) {
			t.Errorf("expected key %q in JSON output", key)
		}
	}

	// 3. Test roundtrip decode via DecodeTurnTraceReport
	decoded, err := DecodeTurnTraceReport(data)
	if err != nil {
		t.Fatalf("DecodeTurnTraceReport failed: %v", err)
	}

	if decoded.Schema != report.Schema {
		t.Errorf("decoded Schema = %q, want %q", decoded.Schema, report.Schema)
	}
	if decoded.TurnID != report.TurnID {
		t.Errorf("decoded TurnID = %q, want %q", decoded.TurnID, report.TurnID)
	}
	if decoded.HostDispatchNs != report.HostDispatchNs {
		t.Errorf("decoded HostDispatchNs = %d, want %d", decoded.HostDispatchNs, report.HostDispatchNs)
	}
	if decoded.PrefixLookupNs != report.PrefixLookupNs {
		t.Errorf("decoded PrefixLookupNs = %d, want %d", decoded.PrefixLookupNs, report.PrefixLookupNs)
	}
	if decoded.KVAllocationNs != report.KVAllocationNs {
		t.Errorf("decoded KVAllocationNs = %d, want %d", decoded.KVAllocationNs, report.KVAllocationNs)
	}
	if decoded.GPUKernelNs != report.GPUKernelNs {
		t.Errorf("decoded GPUKernelNs = %d, want %d", decoded.GPUKernelNs, report.GPUKernelNs)
	}
	if decoded.TokenSamplingNs != report.TokenSamplingNs {
		t.Errorf("decoded TokenSamplingNs = %d, want %d", decoded.TokenSamplingNs, report.TokenSamplingNs)
	}
	if decoded.TotalWallNs != report.TotalWallNs {
		t.Errorf("decoded TotalWallNs = %d, want %d", decoded.TotalWallNs, report.TotalWallNs)
	}

	// 4. Test PascalCase JSON compatibility
	pascalJSON := `{
		"schema": "fak.trace.turn_latency/v1",
		"turn_id": "pascal-turn",
		"TotalWallNs": 1000000,
		"HostDispatchNs": 150000,
		"PrefixLookupNs": 50000,
		"KVAllocationNs": 50000,
		"GPUKernelNs": 700000,
		"TokenSamplingNs": 50000,
		"OverheadNs": 200
	}`
	var fromPascal TurnTraceReport
	if err := json.Unmarshal([]byte(pascalJSON), &fromPascal); err != nil {
		t.Fatalf("unmarshal PascalCase JSON failed: %v", err)
	}
	if fromPascal.HostDispatchNs != 150_000 || fromPascal.GPUKernelNs != 700_000 {
		t.Errorf("failed to map PascalCase fields properly: %+v", fromPascal)
	}
	if err := fromPascal.Validate(); err != nil {
		t.Errorf("PascalCase report failed validation: %v", err)
	}
}

// TestTurnTrace_PhaseNormalization verifies canonical mapping from various string labels.
func TestTurnTrace_PhaseNormalization(t *testing.T) {
	tests := []struct {
		input string
		want  TurnPhase
	}{
		{"host_dispatch", TurnPhaseHostDispatch},
		{"HostDispatch", TurnPhaseHostDispatch},
		{"HostDispatchNs", TurnPhaseHostDispatch},
		{"hostdispatch", TurnPhaseHostDispatch},
		{"prefix_lookup", TurnPhasePrefixLookup},
		{"PrefixLookup", TurnPhasePrefixLookup},
		{"PrefixLookupNs", TurnPhasePrefixLookup},
		{"prefix_tree_lookup", TurnPhasePrefixLookup},
		{"PrefixTreeLookup", TurnPhasePrefixLookup},
		{"kv_allocation", TurnPhaseKVAllocation},
		{"KVAllocation", TurnPhaseKVAllocation},
		{"KVAllocationNs", TurnPhaseKVAllocation},
		{"gpu_kernel", TurnPhaseGPUKernel},
		{"GPUKernel", TurnPhaseGPUKernel},
		{"GPUKernelNs", TurnPhaseGPUKernel},
		{"kernel", TurnPhaseGPUKernel},
		{"token_sampling", TurnPhaseTokenSampling},
		{"TokenSampling", TurnPhaseTokenSampling},
		{"TokenSamplingNs", TurnPhaseTokenSampling},
		{"sampling", TurnPhaseTokenSampling},
	}

	for _, tc := range tests {
		got, err := NormalizeTurnPhase(tc.input)
		if err != nil {
			t.Errorf("NormalizeTurnPhase(%q) returned error: %v", tc.input, err)
		}
		if got != tc.want {
			t.Errorf("NormalizeTurnPhase(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// TestTurnTrace_TimePhaseAndTimeCallback tests the TimePhase and StartPhase callback patterns.
func TestTurnTrace_TimePhaseAndTimeCallback(t *testing.T) {
	tracer := NewTurnLatencyTracer("patterns")

	// Pattern 1: closure callback
	stop := tracer.StartPhase(TurnPhaseHostDispatch)
	time.Sleep(50 * time.Microsecond)
	stop()

	// Pattern 2: TimePhase helper
	tracer.TimePhase(TurnPhasePrefixLookup, func() {
		time.Sleep(30 * time.Microsecond)
	})

	// Pattern 3: explicit EndPhase
	tracer.Start(TurnPhaseKVAllocation)
	time.Sleep(20 * time.Microsecond)
	tracer.EndPhase()

	tracer.RecordPhase(TurnPhaseGPUKernel, 200*time.Microsecond)
	tracer.RecordPhase(TurnPhaseTokenSampling, 50*time.Microsecond)

	report, err := tracer.Report()
	if err != nil {
		t.Fatalf("report failed: %v", err)
	}

	if report.HostDispatchNs <= 0 {
		t.Errorf("expected positive HostDispatchNs")
	}
	if report.PrefixLookupNs <= 0 {
		t.Errorf("expected positive PrefixLookupNs")
	}
	if report.KVAllocationNs <= 0 {
		t.Errorf("expected positive KVAllocationNs")
	}
	if report.GPUKernelNs <= 0 {
		t.Errorf("expected positive GPUKernelNs")
	}
	if report.TokenSamplingNs <= 0 {
		t.Errorf("expected positive TokenSamplingNs")
	}
}

// TestTurnTrace_ConcurrentRecording proves thread safety under concurrent goroutines.
func TestTurnTrace_ConcurrentRecording(t *testing.T) {
	tracer := NewTurnLatencyTracer("concurrent")
	var wg sync.WaitGroup

	for i := 0; i < 50; i++ {
		wg.Add(5)
		go func() {
			defer wg.Done()
			tracer.RecordPhase(TurnPhaseHostDispatch, 100*time.Nanosecond)
		}()
		go func() {
			defer wg.Done()
			tracer.RecordPhase(TurnPhasePrefixLookup, 50*time.Nanosecond)
		}()
		go func() {
			defer wg.Done()
			tracer.RecordPhase(TurnPhaseKVAllocation, 30*time.Nanosecond)
		}()
		go func() {
			defer wg.Done()
			tracer.RecordPhase(TurnPhaseGPUKernel, 500*time.Nanosecond)
		}()
		go func() {
			defer wg.Done()
			tracer.RecordPhase(TurnPhaseTokenSampling, 20*time.Nanosecond)
		}()
	}

	wg.Wait()

	report, err := tracer.Report()
	if err != nil {
		t.Fatalf("report failed: %v", err)
	}

	if report.HostDispatchNs != 50*100 {
		t.Errorf("HostDispatchNs = %d, want %d", report.HostDispatchNs, 50*100)
	}
	if report.GPUKernelNs != 50*500 {
		t.Errorf("GPUKernelNs = %d, want %d", report.GPUKernelNs, 50*500)
	}
}
