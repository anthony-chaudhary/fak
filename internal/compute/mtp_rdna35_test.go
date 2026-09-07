package compute

import (
	"math"
	"testing"
)

// TestMTPK4CausalVerificationTreeMask verifies that the 2D causal verification
// tree mask generator satisfies Mask[i, j] = 1 for j <= i, and 0 for j > i.
func TestMTPK4CausalVerificationTreeMask(t *testing.T) {
	mask := MTPK4CausalVerificationTreeMask()

	// Verify exact 4x4 matrix values
	expected := [4][4]float32{
		{1.0, 0.0, 0.0, 0.0},
		{1.0, 1.0, 0.0, 0.0},
		{1.0, 1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0, 1.0},
	}

	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if mask[i][j] != expected[i][j] {
				t.Errorf("mask[%d][%d] = %g, want %g", i, j, mask[i][j], expected[i][j])
			}
			if j <= i {
				if mask[i][j] != 1.0 {
					t.Errorf("causal invariant violated: mask[%d][%d] must be 1.0 for j <= i", i, j)
				}
				if !IsCausalVerificationMaskAllowed(i, j) {
					t.Errorf("IsCausalVerificationMaskAllowed(%d, %d) must be true", i, j)
				}
			} else {
				if mask[i][j] != 0.0 {
					t.Errorf("future mask violated: mask[%d][%d] must be 0.0 for j > i", i, j)
				}
				if IsCausalVerificationMaskAllowed(i, j) {
					t.Errorf("IsCausalVerificationMaskAllowed(%d, %d) must be false", i, j)
				}
			}
		}
	}

	// Verify general K causal tree mask
	genMask := CausalVerificationTreeMask(4)
	if len(genMask) != 4 {
		t.Fatalf("len(genMask) = %d, want 4", len(genMask))
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if genMask[i][j] != expected[i][j] {
				t.Errorf("genMask[%d][%d] = %g, want %g", i, j, genMask[i][j], expected[i][j])
			}
		}
	}

	// Verify additive attention bias mask
	bias := CausalAttentionBiasMask(4)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if j <= i {
				if bias[i][j] != 0.0 {
					t.Errorf("bias[%d][%d] = %g, want 0.0 (allowed)", i, j, bias[i][j])
				}
			} else {
				if bias[i][j] > -1e8 {
					t.Errorf("bias[%d][%d] = %g, want negative infinity (masked)", i, j, bias[i][j])
				}
			}
		}
	}
}

// TestMTPSinglePassWeightReuseAcross40CUs witnesses single-pass weight reuse,
// 40 CU execution, and quadrupled arithmetic intensity from LPDDR5X DRAM.
func TestMTPSinglePassWeightReuseAcross40CUs(t *testing.T) {
	outDim := 128
	inDim := 64
	weights := make([]float32, outDim*inDim)
	for i := range weights {
		weights[i] = float32(i%17) * 0.05
	}

	drafts := make([][]float32, 4)
	for k := 0; k < 4; k++ {
		drafts[k] = make([]float32, inDim)
		for d := 0; d < inDim; d++ {
			drafts[k][d] = float32((k+1)*(d+1)%13) * 0.1
		}
	}

	treeMask := MTPK4CausalVerificationTreeMask()
	outputs, audit, err := MTPK4MicroBatchVerify(weights, outDim, inDim, drafts, treeMask)
	if err != nil {
		t.Fatalf("MTPK4MicroBatchVerify failed: %v", err)
	}

	if len(outputs) != 4 {
		t.Fatalf("len(outputs) = %d, want 4", len(outputs))
	}
	for k := 0; k < 4; k++ {
		if len(outputs[k]) != outDim {
			t.Fatalf("outputs[%d] length = %d, want %d", k, len(outputs[k]), outDim)
		}
	}

	// Verify audit invariants on gfx1151
	if audit.ComputeUnitsEngaged <= 0 || audit.ComputeUnitsEngaged > StrixHaloComputeUnits {
		t.Errorf("ComputeUnitsEngaged = %d, want <= %d", audit.ComputeUnitsEngaged, StrixHaloComputeUnits)
	}
	if audit.WavefrontSize != 32 {
		t.Errorf("WavefrontSize = %d, want 32 (Wave32)", audit.WavefrontSize)
	}
	if audit.DraftDepthK != 4 {
		t.Errorf("DraftDepthK = %d, want 4", audit.DraftDepthK)
	}
	if !audit.CausalTreeMaskApplied {
		t.Errorf("CausalTreeMaskApplied must be true")
	}

	// Verify weight reuse ratio: sequential reads 4x weights, single pass reads 1x weights
	// Expect WeightReuseRatio close to ~4.0x
	if audit.WeightReuseRatio < 3.5 || audit.WeightReuseRatio > 4.1 {
		t.Errorf("WeightReuseRatio = %g, want ~4.0x", audit.WeightReuseRatio)
	}

	// Verify arithmetic intensity: ~2.0 FLOP/byte for 4 draft proposals
	if audit.ArithmeticIntensity < 1.5 {
		t.Errorf("ArithmeticIntensity = %g FLOP/byte, want >= 1.5 FLOP/byte", audit.ArithmeticIntensity)
	}

	// Verify correctness against direct computation
	for k := 0; k < 4; k++ {
		for r := 0; r < outDim; r++ {
			var expected float32
			wBase := r * inDim
			for d := 0; d < inDim; d++ {
				expected += weights[wBase+d] * drafts[k][d]
			}
			diff := math.Abs(float64(outputs[k][r] - expected))
			if diff > 1e-4 {
				t.Fatalf("token %d row %d mismatch: got %g, want %g (diff %g)", k, r, outputs[k][r], expected, diff)
			}
		}
	}
}

// TestMTPAcceptanceScoringAndRollback verifies sequential prefix acceptance,
// rollback of rejected speculative branches, and speedup evaluation in the 75-85% band.
func TestMTPAcceptanceScoringAndRollback(t *testing.T) {
	// Case 1: All 4 tokens accepted (100% acceptance, 0 rollbacks)
	draftAll := []int{101, 102, 103, 104}
	targetAll := []int{101, 102, 103, 104, 105} // including 5th bonus token
	resAll := EvaluateDraftAcceptance(draftAll, targetAll)
	if resAll.AcceptedCount != 4 {
		t.Errorf("resAll.AcceptedCount = %d, want 4", resAll.AcceptedCount)
	}
	if resAll.RollbackCount != 0 {
		t.Errorf("resAll.RollbackCount = %d, want 0", resAll.RollbackCount)
	}
	if resAll.RejectedAt != -1 {
		t.Errorf("resAll.RejectedAt = %d, want -1", resAll.RejectedAt)
	}
	if len(resAll.NextTokens) != 5 || resAll.NextTokens[4] != 105 {
		t.Errorf("resAll.NextTokens = %+v, want 5 tokens ending with 105", resAll.NextTokens)
	}

	// Case 2: First 3 tokens accepted, 4th rejected (75% acceptance, 1 rollback)
	draft3 := []int{101, 102, 103, 999}
	target3 := []int{101, 102, 103, 204}
	res3 := EvaluateDraftAcceptance(draft3, target3)
	if res3.AcceptedCount != 3 {
		t.Errorf("res3.AcceptedCount = %d, want 3", res3.AcceptedCount)
	}
	if res3.RollbackCount != 1 {
		t.Errorf("res3.RollbackCount = %d, want 1", res3.RollbackCount)
	}
	if res3.RejectedAt != 3 {
		t.Errorf("res3.RejectedAt = %d, want 3", res3.RejectedAt)
	}
	if res3.AcceptanceRate != 0.75 {
		t.Errorf("res3.AcceptanceRate = %g, want 0.75 (75%%)", res3.AcceptanceRate)
	}
	// NextTokens should contain [101, 102, 103, 204] (3 accepted + 1 replacement)
	if len(res3.NextTokens) != 4 || res3.NextTokens[3] != 204 {
		t.Errorf("res3.NextTokens = %+v, want [101 102 103 204]", res3.NextTokens)
	}

	// Case 3: First token rejected (0% acceptance, 4 rollbacks)
	draft0 := []int{999, 102, 103, 104}
	target0 := []int{201, 102, 103, 104}
	res0 := EvaluateDraftAcceptance(draft0, target0)
	if res0.AcceptedCount != 0 {
		t.Errorf("res0.AcceptedCount = %d, want 0", res0.AcceptedCount)
	}
	if res0.RollbackCount != 4 {
		t.Errorf("res0.RollbackCount = %d, want 4", res0.RollbackCount)
	}
	if res0.RejectedAt != 0 {
		t.Errorf("res0.RejectedAt = %d, want 0", res0.RejectedAt)
	}
	if len(res0.NextTokens) != 1 || res0.NextTokens[0] != 201 {
		t.Errorf("res0.NextTokens = %+v, want [201]", res0.NextTokens)
	}

	// Test speedup calculation across the 75-85% acceptance band
	speedup75 := CalculateExpectedSpeedup(0.75, 4)
	speedup80 := CalculateExpectedSpeedup(0.80, 4)
	speedup85 := CalculateExpectedSpeedup(0.85, 4)

	// Theoretical:
	// alpha=0.75: 1 + 0.75 + 0.5625 + 0.421875 + 0.31640625 = 3.05
	// alpha=0.80: 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.36
	// alpha=0.85: 1 + 0.85 + 0.7225 + 0.614125 + 0.52200625 = 3.71
	if speedup75 < 2.9 || speedup75 > 3.2 {
		t.Errorf("speedup75 = %g, want ~3.05", speedup75)
	}
	if speedup80 < 3.2 || speedup80 > 3.5 {
		t.Errorf("speedup80 = %g, want ~3.36", speedup80)
	}
	if speedup85 < 3.5 || speedup85 > 3.9 {
		t.Errorf("speedup85 = %g, want ~3.71", speedup85)
	}
}

// TestMTPDynamicThrottling tests adaptive throttling of K between 4 and 1.
func TestMTPDynamicThrottling(t *testing.T) {
	gov := NewMTPAdaptiveGovernor()
	if gov.CurrentK() != 4 {
		t.Fatalf("initial K = %d, want 4", gov.CurrentK())
	}

	// 1. Simulate high acceptance (80%): K remains 4
	for i := 0; i < 20; i++ {
		gov.RecordStep(4, 3, 1) // 75% per step
	}
	if gov.CurrentK() != 4 {
		t.Errorf("after high acceptance, K = %d, want 4", gov.CurrentK())
	}

	// 2. Simulate degraded acceptance (25% < 50% threshold): K throttles down to 1
	for i := 0; i < 20; i++ {
		gov.RecordStep(4, 1, 3) // 25% acceptance
	}
	if gov.CurrentK() != 1 {
		t.Errorf("after low acceptance, K = %d, want 1 (throttled)", gov.CurrentK())
	}

	// 3. Simulate recovery (80% >= 75% threshold): K recovers back to 4
	for i := 0; i < 20; i++ {
		gov.RecordStep(4, 4, 0) // 100% acceptance
	}
	if gov.CurrentK() != 4 {
		t.Errorf("after recovery, K = %d, want 4 (restored)", gov.CurrentK())
	}
}
