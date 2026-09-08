package amdgpu

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestNewStrixCandidateRegistry_CanonicalBaselines(t *testing.T) {
	reg := NewStrixCandidateRegistry()
	if reg == nil {
		t.Fatal("expected non-nil registry")
	}

	tests := []struct {
		id                  string
		expectedDimension   string
		expectedFeature     string
		expectedBaseArm     string
		expectedBaseLatency int64
		expectedCandArm     string
		expectedCandLatency int64
	}{
		{
			id:                  CandidateIDTargetQ4KGEMV,
			expectedDimension:   "target",
			expectedFeature:     "cpu_vs_vulkan_gpu",
			expectedBaseArm:     "cpu_q4_reference",
			expectedBaseLatency: 75561,
			expectedCandArm:     "vulkan_gpu_q4k",
			expectedCandLatency: 451,
		},
		{
			id:                  CandidateIDTopologyNormMM,
			expectedDimension:   "topology",
			expectedFeature:     "fused_vs_discrete_norm_matmul",
			expectedBaseArm:     "discrete_rmsnorm_then_matmul",
			expectedBaseLatency: 28275,
			expectedCandArm:     "fused_rmsnorm_matmul",
			expectedCandLatency: 17400,
		},
		{
			id:                  CandidateIDQuantQ4KvsF32,
			expectedDimension:   "quantization",
			expectedFeature:     "quant_q4k_vs_q8_vs_f32",
			expectedBaseArm:     "f32_dense_weights",
			expectedBaseLatency: 1820,
			expectedCandArm:     "q4k_super_blocks",
			expectedCandLatency: 428,
		},
		{
			id:                  CandidateIDResidencyDevLoc,
			expectedDimension:   "residency",
			expectedFeature:     "device_local_vs_host_visible",
			expectedBaseArm:     "host_visible_streaming",
			expectedBaseLatency: 1420,
			expectedCandArm:     "device_local_pool",
			expectedCandLatency: 428,
		},
		{
			id:                  CandidateIDLayoutF16Contig,
			expectedDimension:   "layout",
			expectedFeature:     "strided_vs_contiguized_f16_kv",
			expectedBaseArm:     "strided_f16_kv_camping",
			expectedBaseLatency: 44869,
			expectedCandArm:     "contiguized_f16_kv_scratch",
			expectedCandLatency: 16680,
		},
		{
			id:                  CandidateIDPrefillSequence,
			expectedDimension:   "prefill",
			expectedFeature:     "prefill_sequence_vs_serial",
			expectedBaseArm:     "baseline_serial_prefill",
			expectedBaseLatency: 18580000,
			expectedCandArm:     "vulkan_sequence_prefill",
			expectedCandLatency: 1140000,
		},
		{
			id:                  CandidateIDDecodeResident,
			expectedDimension:   "decode",
			expectedFeature:     "decode_resident_vs_host_fallback",
			expectedBaseArm:     "host_fallback_decode",
			expectedBaseLatency: 2695800,
			expectedCandArm:     "vulkan_resident_decode",
			expectedCandLatency: 59500,
		},
	}

	for _, tc := range tests {
		t.Run(tc.id, func(t *testing.T) {
			b, ok := reg.GetBaseline(tc.id)
			if !ok || b == nil {
				t.Fatalf("expected baseline %q to be found", tc.id)
			}
			if b.Dimension != tc.expectedDimension {
				t.Errorf("got dimension %q, want %q", b.Dimension, tc.expectedDimension)
			}
			if b.Feature != tc.expectedFeature {
				t.Errorf("got feature %q, want %q", b.Feature, tc.expectedFeature)
			}
			if b.BaselineArm.Name != tc.expectedBaseArm {
				t.Errorf("got baseline arm %q, want %q", b.BaselineArm.Name, tc.expectedBaseArm)
			}
			if b.BaselineArm.LatencyUS != tc.expectedBaseLatency {
				t.Errorf("got baseline latency %d, want %d", b.BaselineArm.LatencyUS, tc.expectedBaseLatency)
			}
			if b.PinnedCandidate.Name != tc.expectedCandArm {
				t.Errorf("got candidate arm %q, want %q", b.PinnedCandidate.Name, tc.expectedCandArm)
			}
			if b.PinnedCandidate.LatencyUS != tc.expectedCandLatency {
				t.Errorf("got candidate latency %d, want %d", b.PinnedCandidate.LatencyUS, tc.expectedCandLatency)
			}
		})
	}

	// Verify alias lookups work
	aliasTests := []struct {
		alias      string
		expectedID string
	}{
		{"cpu_vs_vulkan_gpu", CandidateIDTargetQ4KGEMV},
		{"q4k_gemv", CandidateIDTargetQ4KGEMV},
		{"vulkan_gpu_q4k", CandidateIDTargetQ4KGEMV},
		{"fused_vs_discrete_norm_matmul", CandidateIDTopologyNormMM},
		{"norm_matmul", CandidateIDTopologyNormMM},
		{"quant_q4k_vs_q8_vs_f32", CandidateIDQuantQ4KvsF32},
		{"device_local_vs_host_visible", CandidateIDResidencyDevLoc},
		{"strided_vs_contiguized_f16_kv", CandidateIDLayoutF16Contig},
		{"prefill_sequence_vs_serial", CandidateIDPrefillSequence},
		{"sequence_prefill", CandidateIDPrefillSequence},
		{"decode_resident_vs_host_fallback", CandidateIDDecodeResident},
		{"resident_decode", CandidateIDDecodeResident},
	}

	for _, at := range aliasTests {
		t.Run("alias_"+at.alias, func(t *testing.T) {
			b, ok := reg.GetBaseline(at.alias)
			if !ok || b == nil {
				t.Fatalf("expected alias %q to resolve to baseline", at.alias)
			}
			if b.CandidateID != at.expectedID {
				t.Errorf("alias %q resolved to %q, want %q", at.alias, b.CandidateID, at.expectedID)
			}
		})
	}

	// Unknown ID returns false
	if _, ok := reg.GetBaseline("unknown_candidate"); ok {
		t.Error("expected unknown candidate to return false")
	}
}

func TestScoreboard_InitialCanonicalState(t *testing.T) {
	reg := NewStrixCandidateRegistry()
	sb := reg.Scoreboard()

	if len(sb) != 7 {
		t.Fatalf("expected 7 canonical items in scoreboard, got %d", len(sb))
	}

	expectedOrder := []string{
		CandidateIDTargetQ4KGEMV,
		CandidateIDTopologyNormMM,
		CandidateIDQuantQ4KvsF32,
		CandidateIDResidencyDevLoc,
		CandidateIDLayoutF16Contig,
		CandidateIDPrefillSequence,
		CandidateIDDecodeResident,
	}

	for i, expID := range expectedOrder {
		if sb[i].CandidateID != expID {
			t.Errorf("scoreboard[%d].CandidateID = %q, want %q", i, sb[i].CandidateID, expID)
		}
		if sb[i].Verdict != VerdictPromoted {
			t.Errorf("scoreboard[%d] %s verdict = %q, want %q", i, expID, sb[i].Verdict, VerdictPromoted)
		}
		if sb[i].Speedup <= 1.0 {
			t.Errorf("scoreboard[%d] %s speedup = %.2f, expected > 1.0", i, expID, sb[i].Speedup)
		}
		if sb[i].CosineParity < DefaultMinParity {
			t.Errorf("scoreboard[%d] %s parity = %.6f, expected >= %.6f", i, expID, sb[i].CosineParity, DefaultMinParity)
		}
	}

	// Test specific metrics for target.q4k_gemv
	q4k := sb[0]
	if q4k.CandidateLatencyUS != 451 || q4k.BaselineLatencyUS != 75561 {
		t.Errorf("unexpected latencies for q4k_gemv: base=%d cand=%d", q4k.BaselineLatencyUS, q4k.CandidateLatencyUS)
	}
	if q4k.LatencyDeltaUS != 451-75561 {
		t.Errorf("unexpected latency delta: got %d, want %d", q4k.LatencyDeltaUS, 451-75561)
	}
	expectedSpeedup := float64(75561) / 451.0
	if math.Abs(q4k.Speedup-expectedSpeedup) > 0.01 {
		t.Errorf("unexpected speedup for q4k_gemv: got %.2f, want %.2f", q4k.Speedup, expectedSpeedup)
	}

	// Test compression ratio for quant.q4k_vs_f32
	quant := sb[2]
	expectedCompression := float64(356515840) / 50135040.0
	if math.Abs(quant.CompressionRatio-expectedCompression) > 0.01 {
		t.Errorf("unexpected compression ratio: got %.2f, want %.2f", quant.CompressionRatio, expectedCompression)
	}
}

func TestEvaluateCandidate_Promoted(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 75561,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 400, // Faster than previous 451µs
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictPromoted {
		t.Errorf("got verdict %q, want %q (reason: %s)", comp.Verdict, VerdictPromoted, comp.Reason)
	}
	if comp.Speedup < 180.0 {
		t.Errorf("got speedup %.2f, want >= 180.0", comp.Speedup)
	}
	if comp.LatencyDeltaUS != 400-75561 {
		t.Errorf("got latency delta %d, want %d", comp.LatencyDeltaUS, 400-75561)
	}

	// Verify scoreboard updated
	updated, ok := reg.GetComparison(CandidateIDTargetQ4KGEMV)
	if !ok || updated == nil {
		t.Fatal("expected comparison in scoreboard")
	}
	if updated.CandidateLatencyUS != 400 {
		t.Errorf("scoreboard candidate latency = %d, want 400", updated.CandidateLatencyUS)
	}
}

func TestEvaluateCandidate_Neutral_WithinNoiseBand(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// Candidate speedup is 1.02 (within the neutral 5% noise band)
	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 1000,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 980, // speedup = 1000/980 = 1.0204x (< 1.05 threshold)
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictNeutral {
		t.Errorf("got verdict %q, want %q (reason: %s)", comp.Verdict, VerdictNeutral, comp.Reason)
	}
	if !strings.Contains(comp.Reason, "within the noise band") {
		t.Errorf("unexpected reason: %s", comp.Reason)
	}
}

func TestEvaluateCandidate_Neutral_HighNoise(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// High speedup (1.50x), good parity, but noise is 8% (> 5% max tolerance)
	result := StrixAblationResult{
		Dimension: "topology",
		Feature:   CandidateIDTopologyNormMM,
		BaselineArm: StrixArmResult{
			Name:      "discrete_rmsnorm_then_matmul",
			LatencyUS: 1500,
		},
		CandidateArm: StrixArmResult{
			Name:      "fused_rmsnorm_matmul",
			LatencyUS: 1000,
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidateWithOptions(result, WithNoise(0.08))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictNeutral {
		t.Errorf("got verdict %q, want %q with high noise (reason: %s)", comp.Verdict, VerdictNeutral, comp.Reason)
	}
	if !strings.Contains(comp.Reason, "noise 8.0% exceeds") {
		t.Errorf("unexpected reason: %s", comp.Reason)
	}
}

func TestEvaluateCandidate_Regressed_Slower(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// Candidate takes longer than baseline (speedup 0.80x < 0.95 floor)
	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 1000,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 1250, // speedup = 0.80x
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictRegressed {
		t.Errorf("got verdict %q, want %q (reason: %s)", comp.Verdict, VerdictRegressed, comp.Reason)
	}
	if !strings.Contains(comp.Reason, "slower than baseline") {
		t.Errorf("unexpected reason: %s", comp.Reason)
	}
}

func TestEvaluateCandidate_Regressed_ParityViolated(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// Great speedup (10x), but cosine parity is 0.999500 (< 0.999900)
	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 10000,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 1000,
		},
		CosineParity: 0.999500, // Below 0.999900
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictRegressed {
		t.Errorf("got verdict %q, want %q (reason: %s)", comp.Verdict, VerdictRegressed, comp.Reason)
	}
	if !strings.Contains(comp.Reason, "numerical parity violated") {
		t.Errorf("unexpected reason: %s", comp.Reason)
	}
}

func TestEvaluateCandidate_Regressed_NaNParity(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 10000,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 1000,
		},
		CosineParity: math.NaN(),
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictRegressed {
		t.Errorf("got verdict %q, want %q for NaN parity", comp.Verdict, VerdictRegressed)
	}
}

func TestEvaluateCandidate_Regressed_NonPositiveLatency(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 0,
		},
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != VerdictRegressed {
		t.Errorf("got verdict %q, want %q for zero latency", comp.Verdict, VerdictRegressed)
	}
}

func TestEvaluateCandidate_ThroughputAndCompressionCalculations(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	result := StrixAblationResult{
		Dimension: "quantization",
		Feature:   CandidateIDQuantQ4KvsF32,
		BaselineArm: StrixArmResult{
			Name:           "f32_dense_weights",
			LatencyUS:      1820,
			ThroughputTokS: 50.0,
			AllocatedBytes: 100000,
		},
		CandidateArm: StrixArmResult{
			Name:           "q4k_super_blocks",
			LatencyUS:      428,
			ThroughputTokS: 150.0,
			AllocatedBytes: 25000,
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Throughput
	if math.Abs(comp.ThroughputDelta-100.0) > 0.001 {
		t.Errorf("got throughput delta %.2f, want 100.0", comp.ThroughputDelta)
	}
	if math.Abs(comp.LiftRatio-3.0) > 0.001 {
		t.Errorf("got lift ratio %.2f, want 3.0", comp.LiftRatio)
	}

	// Memory
	if comp.AllocatedBytesDelta != -75000 {
		t.Errorf("got alloc delta %d, want -75000", comp.AllocatedBytesDelta)
	}
	if math.Abs(comp.CompressionRatio-4.0) > 0.001 {
		t.Errorf("got compression ratio %.2f, want 4.0", comp.CompressionRatio)
	}
}

func TestEvaluateCandidate_UnknownCandidate(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	result := StrixAblationResult{
		Dimension: "unknown_dim",
		Feature:   "non_existent_feature",
		CandidateArm: StrixArmResult{
			Name:      "non_existent_arm",
			LatencyUS: 100,
		},
	}

	_, err := reg.EvaluateCandidate(result)
	if err == nil {
		t.Error("expected error for unknown candidate, got nil")
	}
}

func TestEvaluateReceipt(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	receipt := validStrixReceipt(t)
	receipt.Ablations = []StrixAblationResult{
		{
			Dimension: "target",
			Feature:   "cpu_vs_vulkan_gpu",
			BaselineArm: StrixArmResult{
				Name:      "cpu_q4_reference",
				LatencyUS: 75561,
				Samples:   1,
			},
			CandidateArm: StrixArmResult{
				Name:      "vulkan_gpu_q4k",
				LatencyUS: 450,
				Samples:   1,
			},
			Speedup: 167.9, LiftRatio: 167.9, CosineParity: 0.999999, Verdict: "VERIFIED_LIFT", Evidence: validStrixExecutionEvidence(),
		},
		{
			Dimension: "residency",
			Feature:   "device_local_vs_host_visible",
			BaselineArm: StrixArmResult{
				Name:      "host_visible_streaming",
				LatencyUS: 1420,
				Samples:   1,
			},
			CandidateArm: StrixArmResult{
				Name:      "device_local_pool",
				LatencyUS: 420,
				Samples:   1,
			},
			Speedup: 3.38, LiftRatio: 3.38, CosineParity: 1.0, Verdict: "VERIFIED_LIFT", Evidence: validStrixExecutionEvidence(),
		},
	}
	receipt.SelectedAblations = 2
	receipt.ExecutedAblations = 2
	receipt.Provenance.ExecutionManifestSHA256 = executionManifestDigest(receipt)

	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	receipt.Digest = digest

	comparisons, err := reg.EvaluateReceipt(receipt)
	if err != nil {
		t.Fatalf("unexpected error evaluating receipt: %v", err)
	}

	if len(comparisons) != 2 {
		t.Fatalf("expected 2 comparisons, got %d", len(comparisons))
	}
	for _, c := range comparisons {
		if c.Verdict != VerdictPromoted {
			t.Errorf("receipt comparison for %s verdict = %q, want PROMOTED", c.CandidateID, c.Verdict)
		}
	}
}

func TestRegisterBaseline_Custom(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	custom := StrixCandidateBaseline{
		CandidateID: "custom.gemm_simd",
		Dimension:   "compute",
		Feature:     "simd_wave32_gemm",
		Description: "Wave32 SIMD GEMM custom candidate",
		BaselineArm: StrixArmResult{
			Name:           "scalar_reference",
			LatencyUS:      10000,
			AllocatedBytes: 20000,
		},
		PinnedCandidate: StrixArmResult{
			Name:           "wave32_kernel",
			LatencyUS:      2000,
			AllocatedBytes: 20000,
		},
		SpeedupThreshold: 1.20,
		MinParity:        0.999990,
		NoiseBand:        0.05,
	}

	if err := reg.RegisterBaseline(custom); err != nil {
		t.Fatalf("failed to register custom baseline: %v", err)
	}

	b, ok := reg.GetBaseline("custom.gemm_simd")
	if !ok || b == nil {
		t.Fatal("expected custom baseline to be retrievable by ID")
	}
	if b.ReferenceSpeedup() != 5.0 {
		t.Errorf("got reference speedup %.2f, want 5.0", b.ReferenceSpeedup())
	}
	if b.ReferenceCompressionRatio() != 1.0 {
		t.Errorf("got compression ratio %.2f, want 1.0", b.ReferenceCompressionRatio())
	}

	// Evaluate candidate against custom baseline
	res := StrixAblationResult{
		Dimension: "compute",
		Feature:   "simd_wave32_gemm",
		CandidateArm: StrixArmResult{
			Name:      "wave32_kernel",
			LatencyUS: 1800,
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(res)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if comp.Verdict != VerdictPromoted {
		t.Errorf("got verdict %q, want PROMOTED", comp.Verdict)
	}

	// Test validation error on empty CandidateID
	if err := reg.RegisterBaseline(StrixCandidateBaseline{}); err == nil {
		t.Error("expected error when registering baseline with empty CandidateID")
	}
}

func TestFormatScoreboard(t *testing.T) {
	reg := NewStrixCandidateRegistry()
	formatted := reg.FormatScoreboard()

	if !strings.Contains(formatted, "| Candidate ID |") {
		t.Error("expected scoreboard header")
	}
	if !strings.Contains(formatted, CandidateIDTargetQ4KGEMV) {
		t.Errorf("expected table to contain %s", CandidateIDTargetQ4KGEMV)
	}
	if !strings.Contains(formatted, "PROMOTED") {
		t.Error("expected table to contain PROMOTED")
	}
}

func TestFormatLatencyUS(t *testing.T) {
	cases := []struct {
		us       int64
		expected string
	}{
		{451, "451µs"},
		{17400, "17.40ms"},
		{1500000, "1.50s"},
	}

	for _, c := range cases {
		got := formatLatencyUS(c.us)
		if got != c.expected {
			t.Errorf("formatLatencyUS(%d) = %q, want %q", c.us, got, c.expected)
		}
	}
}

func TestConcurrency_SafeRegistry(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	var wg sync.WaitGroup
	workers := 16
	iterations := 50

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				// Read scoreboard
				sb := reg.Scoreboard()
				if len(sb) == 0 {
					t.Errorf("worker %d: empty scoreboard", workerID)
				}

				// Read baseline
				_, _ = reg.GetBaseline(CandidateIDTargetQ4KGEMV)

				// Evaluate candidate
				res := StrixAblationResult{
					Dimension: "target",
					Feature:   CandidateIDTargetQ4KGEMV,
					CandidateArm: StrixArmResult{
						Name:      "vulkan_gpu_q4k",
						LatencyUS: int64(400 + (j % 10)),
					},
					CosineParity: 0.999999,
				}
				_, _ = reg.EvaluateCandidate(res)
			}
		}(i)
	}

	wg.Wait()
}

func TestStrixValidationBenchmarkArtifact(t *testing.T) {
	artifacts := []string{
		"../../docs/benchmarks/strix-halo-validation-11940.json",
		"../../docs/benchmarks/strix-halo-validation-latest.json",
	}

	for _, artifactPath := range artifacts {
		data, err := os.ReadFile(artifactPath)
		if err != nil {
			t.Skipf("benchmark artifact not found at %s: %v", artifactPath, err)
		}

		// 1. Genuine benchmark artifact must validate and evaluate cleanly
		receipt, err := ValidateBenchmarkArtifact(data)
		if err != nil {
			t.Fatalf("ValidateBenchmarkArtifact failed for %s: %v", artifactPath, err)
		}
		if receipt.Digest == "" {
			t.Fatalf("expected non-empty digest for %s", artifactPath)
		}
		if !receipt.Verified {
			t.Fatalf("expected verified=true for %s", artifactPath)
		}

		reg := NewStrixCandidateRegistry()
		if _, err := reg.EvaluateReceipt(receipt); err == nil || !strings.Contains(err.Error(), "historical v1") {
			t.Fatalf("historical artifact %s incorrectly earned current credit: %v", artifactPath, err)
		}

		// 2. Direct digest corruption must cause failure
		corruptedReceipt := *receipt
		corruptedReceipt.Digest = "sha256:corrupted_digest_deadbeef_0123456789abcdef"
		if err := corruptedReceipt.Validate(); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected digest mismatch error from Validate(), got: %v", err)
		}
		if _, err := reg.EvaluateReceipt(&corruptedReceipt); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected digest mismatch error from EvaluateReceipt(), got: %v", err)
		}

		corruptedData, err := json.Marshal(corruptedReceipt)
		if err != nil {
			t.Fatalf("marshal corrupted receipt failed: %v", err)
		}
		if _, err := ValidateBenchmarkArtifact(corruptedData); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected digest mismatch error from ValidateBenchmarkArtifact(), got: %v", err)
		}

		// 3. Fabricated ablation evidence with unchanged digest must fail closed
		fabricatedReceipt := *receipt
		fabricatedReceipt.Ablations = append([]StrixAblationResult(nil), receipt.Ablations...)
		if len(fabricatedReceipt.Ablations) > 0 {
			// Fabricate an impossibly low candidate latency (1µs) to fake promotion
			fabricatedReceipt.Ablations[0].CandidateArm.LatencyUS = 1
			if err := fabricatedReceipt.Validate(); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
				t.Errorf("expected fabricated ablation to trigger digest mismatch in Validate(), got: %v", err)
			}
			if _, err := reg.EvaluateReceipt(&fabricatedReceipt); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
				t.Errorf("expected fabricated ablation to trigger digest mismatch in EvaluateReceipt(), got: %v", err)
			}

			fabricatedData, err := json.Marshal(fabricatedReceipt)
			if err != nil {
				t.Fatalf("marshal fabricated receipt failed: %v", err)
			}
			if _, err := ValidateBenchmarkArtifact(fabricatedData); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
				t.Errorf("expected fabricated ablation to trigger digest mismatch in ValidateBenchmarkArtifact(), got: %v", err)
			}
		}

		// 4. Missing digest in benchmark artifact must fail closed
		noDigestReceipt := *receipt
		noDigestReceipt.Digest = ""
		noDigestData, err := json.Marshal(noDigestReceipt)
		if err != nil {
			t.Fatalf("marshal no-digest receipt failed: %v", err)
		}
		if _, err := ValidateBenchmarkArtifact(noDigestData); err == nil || !strings.Contains(err.Error(), "missing required digest") {
			t.Errorf("expected missing required digest error from ValidateBenchmarkArtifact(), got: %v", err)
		}
	}

	// 5. Synthetic benchmark artifact verification with deterministic failure matrix
	t.Run("synthetic_artifact_digest_mismatch", func(t *testing.T) {
		receipt := validStrixReceipt(t)
		digest, err := receipt.ComputeDigest()
		if err != nil {
			t.Fatalf("ComputeDigest failed: %v", err)
		}
		receipt.Digest = digest

		// Valid synthetic receipt passes
		if err := receipt.Validate(); err != nil {
			t.Fatalf("synthetic receipt should validate: %v", err)
		}
		reg := NewStrixCandidateRegistry()
		if _, err := reg.EvaluateReceipt(receipt); err != nil {
			t.Fatalf("EvaluateReceipt failed on valid synthetic receipt: %v", err)
		}

		// Tampered digest fails closed
		tampered := *receipt
		tampered.Digest = "sha256:badf00d_mismatched_digest"
		if err := tampered.Validate(); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected digest mismatch error in Validate(), got: %v", err)
		}
		if _, err := reg.EvaluateReceipt(&tampered); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected digest mismatch error in EvaluateReceipt(), got: %v", err)
		}

		// Fabricated ablation with original digest fails closed
		fabricated := *receipt
		fabricated.Ablations = []StrixAblationResult{
			{
				Dimension: "target",
				Feature:   "cpu_vs_vulkan_gpu",
				BaselineArm: StrixArmResult{
					Name:      "cpu_q4_reference",
					LatencyUS: 75561,
				},
				CandidateArm: StrixArmResult{
					Name:      "vulkan_gpu_q4k",
					LatencyUS: 10, // Fabricated
				},
				Speedup:      7556.1,
				CosineParity: 0.999999,
				Verdict:      "VERIFIED_LIFT",
			},
		}
		if err := fabricated.Validate(); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected fabricated ablation to fail Validate() with digest mismatch, got: %v", err)
		}
		if _, err := reg.EvaluateReceipt(&fabricated); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Errorf("expected fabricated ablation to fail EvaluateReceipt() with digest mismatch, got: %v", err)
		}

		// Verified receipt missing digest fails closed in EvaluateReceipt
		unsealedVerified := *receipt
		unsealedVerified.Digest = ""
		unsealedVerified.Verified = true
		if _, err := reg.EvaluateReceipt(&unsealedVerified); err == nil || !strings.Contains(err.Error(), "missing required digest") {
			t.Errorf("expected unsealed verified receipt to fail EvaluateReceipt, got: %v", err)
		}
	})
}
