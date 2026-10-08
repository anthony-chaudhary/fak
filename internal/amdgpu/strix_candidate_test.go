package amdgpu

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
)

var canonicalCandidateOrder = []string{
	CandidateIDTargetQ4KGEMV,
	CandidateIDTopologyNormMM,
	CandidateIDQuantQ4KvsF32,
	CandidateIDResidencyDevLoc,
	CandidateIDLayoutF16Contig,
	CandidateIDPrefillSequence,
	CandidateIDDecodeResident,
}

// TestStrixCandidateRegistry_StartsUncredited is the ticket witness: a freshly
// constructed registry must contain no receipt-free PROMOTED rows; historical
// constants remain reference-only.
func TestStrixCandidateRegistry_StartsUncredited(t *testing.T) {
	reg := NewStrixCandidateRegistry()
	sb := reg.Scoreboard()
	if len(sb) != len(canonicalCandidateOrder) {
		t.Fatalf("expected %d reference rows (one per canonical candidate), got %d", len(canonicalCandidateOrder), len(sb))
	}
	for _, c := range sb {
		if c.Verdict == VerdictPromoted {
			t.Errorf("fresh registry row %s is PROMOTED without trusted evidence: %s", c.CandidateID, c.Reason)
		}
		if c.Verdict != StrixCandidateVerdict("UNVERIFIED") {
			t.Errorf("fresh registry row %s verdict = %q, want %q", c.CandidateID, c.Verdict, StrixCandidateVerdict("UNVERIFIED"))
		}
	}
}

// TestStrixCandidateRegistry_RejectsCallerBaselinePromotion is the ticket
// witness: a caller-supplied inflated baseline cannot replace the pinned
// denominator, and a receipt-free raw evaluation cannot award promotion.
func TestStrixCandidateRegistry_RejectsCallerBaselinePromotion(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// The caller tries to inflate the baseline to 1_000_000µs with a 1µs
	// candidate to mint an enormous speedup. The pinned baseline (451µs
	// candidate / 75561µs baseline) must govern the recompute instead.
	comp, err := reg.EvaluateCandidate(StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "attacker_inflated_baseline",
			LatencyUS: 1000000,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 1,
		},
		CosineParity: 0.999999,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if comp.BaselineLatencyUS != 75561 {
		t.Errorf("caller baseline replaced the pinned denominator: got %d, want 75561", comp.BaselineLatencyUS)
	}
	if comp.Verdict == VerdictPromoted {
		t.Errorf("receipt-free raw evaluation awarded promotion: %s", comp.Reason)
	}
	if comp.Verdict != StrixCandidateVerdict("UNVERIFIED") {
		t.Errorf("got verdict %q, want %q", comp.Verdict, StrixCandidateVerdict("UNVERIFIED"))
	}
	if _, ok := reg.GetComparison(CandidateIDTargetQ4KGEMV); ok {
		if promoted, _ := reg.GetComparison(CandidateIDTargetQ4KGEMV); promoted.Verdict == VerdictPromoted {
			t.Error("scoreboard row became PROMOTED from a receipt-free evaluation")
		}
	}
}

// TestStrixCandidateRegistry_PromotesTrustedReceipt is the ticket witness: only
// an authority-backed receipt, recomputed against the pinned baseline, may
// transition a row to PROMOTED.
func TestStrixCandidateRegistry_PromotesTrustedReceipt(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	receipt := validStrixReceipt(t)
	receipt.Ablations = []StrixAblationResult{
		{
			Dimension: "target",
			Feature:   "cpu_vs_vulkan_gpu",
			BaselineArm: StrixArmResult{
				Name:      "attacker_inflated_baseline",
				LatencyUS: 999999,
				Samples:   1,
			},
			CandidateArm: StrixArmResult{
				Name:      "vulkan_gpu_q4k",
				LatencyUS: 400,
				Samples:   1,
			},
			Speedup: 2500, LiftRatio: 2500, CosineParity: 0.999999, Verdict: "VERIFIED_LIFT", Evidence: validStrixExecutionEvidence(),
		},
	}
	receipt.SelectedAblations = 1
	receipt.ExecutedAblations = 1
	receipt.Provenance.ExecutionManifestSHA256 = executionManifestDigest(receipt)
	authorizeStrixReceiptForTest(t, receipt)
	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil || !receipt.authenticatedPass() {
		t.Fatalf("trusted receipt must validate and carry authority: %v", err)
	}

	comparisons, err := reg.EvaluateReceipt(receipt)
	if err != nil || len(comparisons) != 1 {
		t.Fatalf("trusted receipt evaluation failed: comparisons=%d err=%v", len(comparisons), err)
	}
	got := comparisons[0]
	if got.Verdict != VerdictPromoted {
		t.Fatalf("trusted receipt did not promote: verdict=%q reason=%s", got.Verdict, got.Reason)
	}
	// The recomputed speedup must use the pinned baseline, not the caller's.
	if got.BaselineLatencyUS != 75561 {
		t.Errorf("trusted promotion used caller baseline: got %d, want 75561", got.BaselineLatencyUS)
	}
	if after, ok := reg.GetComparison(CandidateIDTargetQ4KGEMV); !ok || after.Verdict != VerdictPromoted {
		t.Fatalf("scoreboard not promoted from trusted receipt: %+v", after)
	}
}

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

	if len(sb) != len(canonicalCandidateOrder) {
		t.Fatalf("expected %d canonical items in scoreboard, got %d", len(canonicalCandidateOrder), len(sb))
	}

	expectedOrder := canonicalCandidateOrder

	for i, expID := range expectedOrder {
		if sb[i].CandidateID != expID {
			t.Errorf("scoreboard[%d].CandidateID = %q, want %q", i, sb[i].CandidateID, expID)
		}
		// A freshly constructed registry carries no trusted evidence, so every
		// historical row is reference-only: never PROMOTED.
		if sb[i].Verdict != StrixCandidateVerdict("UNVERIFIED") {
			t.Errorf("scoreboard[%d] %s verdict = %q, want %q", i, expID, sb[i].Verdict, StrixCandidateVerdict("UNVERIFIED"))
		}
		if sb[i].Speedup <= 1.0 {
			t.Errorf("scoreboard[%d] %s reference speedup = %.2f, expected > 1.0", i, expID, sb[i].Speedup)
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

func TestEvaluateCandidate_RawPathUnverified(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// A receipt-free result whose metrics would otherwise pass thresholds is
	// reference-only: the raw path can never award promotion.
	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 75561,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 400, // Faster than the pinned 451µs candidate reference
		},
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if comp.Verdict != StrixCandidateVerdict("UNVERIFIED") {
		t.Errorf("got verdict %q, want %q (reason: %s)", comp.Verdict, StrixCandidateVerdict("UNVERIFIED"), comp.Reason)
	}
	if comp.Speedup < 180.0 {
		t.Errorf("got speedup %.2f, want >= 180.0", comp.Speedup)
	}
	if comp.LatencyDeltaUS != 400-75561 {
		t.Errorf("got latency delta %d, want %d", comp.LatencyDeltaUS, 400-75561)
	}

	// Verify scoreboard updated (as UNVERIFIED, never PROMOTED)
	updated, ok := reg.GetComparison(CandidateIDTargetQ4KGEMV)
	if !ok || updated == nil {
		t.Fatal("expected comparison in scoreboard")
	}
	if updated.CandidateLatencyUS != 400 {
		t.Errorf("scoreboard candidate latency = %d, want 400", updated.CandidateLatencyUS)
	}
	if updated.Verdict != StrixCandidateVerdict("UNVERIFIED") {
		t.Errorf("scoreboard verdict = %q, want %q", updated.Verdict, StrixCandidateVerdict("UNVERIFIED"))
	}
}

func TestEvaluateCandidate_Neutral_WithinNoiseBand(t *testing.T) {
	reg := NewStrixCandidateRegistry()

	// Speedup is computed against the pinned baseline (75561µs). A candidate of
	// 74080µs yields 1.0200x, inside the neutral [0.95, 1.05) noise band.
	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 74080,
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

	// 1.50x against the pinned topology baseline, with noise 8% (> 5% limit).
	result := StrixAblationResult{
		Dimension: "topology",
		Feature:   CandidateIDTopologyNormMM,
		CandidateArm: StrixArmResult{
			Name:      "fused_rmsnorm_matmul",
			LatencyUS: 18850, // 28275/18850 = 1.50x
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

	// 0.80x against the pinned baseline (75561µs): candidate 94451µs.
	result := StrixAblationResult{
		Dimension: "target",
		Feature:   CandidateIDTargetQ4KGEMV,
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 94451, // 75561/94451 = 0.80x
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

	// Metrics are computed against the registry-owned pinned baseline. For
	// quant.q4k_vs_f32 the pinned baseline allocates 356515840 bytes.
	result := StrixAblationResult{
		Dimension: "quantization",
		Feature:   CandidateIDQuantQ4KvsF32,
		CandidateArm: StrixArmResult{
			Name:           "q4k_super_blocks",
			LatencyUS:      428,
			ThroughputTokS: 150.0,
			AllocatedBytes: 25000,
		},
		LiftRatio:    3.0,
		CosineParity: 0.999999,
	}

	comp, err := reg.EvaluateCandidate(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Compression ratio uses the pinned baseline bytes (356515840) over the
	// candidate bytes (25000).
	wantCompression := float64(356515840) / 25000.0
	if math.Abs(comp.CompressionRatio-wantCompression) > 0.01 {
		t.Errorf("got compression ratio %.2f, want %.2f", comp.CompressionRatio, wantCompression)
	}
	if comp.AllocatedBytesDelta != 25000-356515840 {
		t.Errorf("got alloc delta %d, want %d", comp.AllocatedBytesDelta, 25000-356515840)
	}

	// The pinned baseline has no throughput reference, so lift falls back to the
	// caller-provided ratio and the candidate throughput is preserved.
	if math.Abs(comp.CandidateThroughputTokS-150.0) > 0.001 {
		t.Errorf("got candidate throughput %.2f, want 150.0", comp.CandidateThroughputTokS)
	}
	if math.Abs(comp.LiftRatio-3.0) > 0.001 {
		t.Errorf("got lift ratio %.2f, want 3.0", comp.LiftRatio)
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
	authorizeStrixReceiptForTest(t, receipt)

	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil || !receipt.authenticatedPass() {
		t.Fatalf("ablation receipt must be structurally valid and authenticated: %v", err)
	}
	if receipt.CreditEligible() {
		t.Fatal("ablation-bearing receipt must not be physical-credit eligible")
	}
	// Authority-backed ablation receipts are the trusted promotion path even
	// though they are outside the narrower physical-credit envelope.
	comparisons, err := reg.EvaluateReceipt(receipt)
	if err != nil || len(comparisons) != 2 {
		t.Fatalf("authority-backed ablation receipt was not evaluated: comparisons=%d err=%v", len(comparisons), err)
	}
	for _, c := range comparisons {
		if c.Verdict != VerdictPromoted {
			t.Errorf("comparison %s verdict = %q, want %q (reason: %s)", c.CandidateID, c.Verdict, VerdictPromoted, c.Reason)
		}
	}
	if after, ok := reg.GetComparison(CandidateIDTargetQ4KGEMV); !ok || after.Verdict != VerdictPromoted {
		t.Fatalf("trusted receipt did not promote target.q4k_gemv: %+v", after)
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
	if comp.Verdict != StrixCandidateVerdict("UNVERIFIED") {
		t.Errorf("got verdict %q, want UNVERIFIED for a receipt-free raw evaluation", comp.Verdict)
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
	if !strings.Contains(formatted, "UNVERIFIED") {
		t.Error("expected table to contain UNVERIFIED reference rows")
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
		// A serialized artifact retains digest integrity, not private verifier authority.
		data, err := json.Marshal(validStrixReceipt(t))
		if err != nil {
			t.Fatalf("marshal synthetic receipt: %v", err)
		}
		var decoded StrixValidationReceipt
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("unmarshal synthetic receipt: %v", err)
		}
		receipt := &decoded
		digest, err := receipt.ComputeDigest()
		if err != nil {
			t.Fatalf("ComputeDigest failed: %v", err)
		}
		receipt.Digest = digest

		// Valid synthetic receipt passes
		if err := receipt.Validate(); err != nil {
			t.Fatalf("synthetic receipt should validate: %v", err)
		}
		if receipt.authority.validFor(receipt) || receipt.authenticatedPass() {
			t.Fatal("serialized synthetic receipt retained verifier authority")
		}
		if receipt.CreditEligible() {
			t.Fatal("serialized synthetic receipt earned physical credit")
		}
		reg := NewStrixCandidateRegistry()
		before := reg.Scoreboard()
		comparisons, err := reg.EvaluateReceipt(receipt)
		if err == nil || len(comparisons) != 0 {
			t.Fatalf("synthetic receipt must be rejected without comparisons: comparisons=%d err=%v", len(comparisons), err)
		}
		if after := reg.Scoreboard(); !reflect.DeepEqual(after, before) {
			t.Fatalf("synthetic receipt mutated scoreboard: before=%v after=%v", before, after)
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
