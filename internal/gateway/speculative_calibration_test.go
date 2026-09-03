package gateway

import (
	"errors"
	"testing"
)

func TestSpeculativeCalibrationWitness(t *testing.T) {
	// First witness requirements (#9923):
	// 1. Run one frozen target-logit fixture through real (greedy/stochastic) and synthetic rates.
	// 2. Real arms report QualityClaimAllowed == true and pass certification.
	// 3. Synthetic arm reports IsSyntheticCalibration == true and QualityClaimAllowed == false.
	// 4. Prove quality claims strictly reject / fail-closed on the synthetic arm.

	// Frozen target logit fixture: K=4 steps, vocabSize=5
	targetLogits := [][]float32{
		{0.1, 5.0, 0.2, -1.0, 0.5}, // argmax = 1
		{1.0, 0.2, 8.0, 0.5, 0.1},  // argmax = 2
		{0.0, 0.1, 0.2, 6.0, 0.3},  // argmax = 3
		{10.0, 1.0, 0.5, 0.2, 0.1}, // argmax = 0
	}

	// Draft tokens match argmax for first 2 steps, then diverge at step 2
	draftTokens := []int{1, 2, 0, 0}
	draftProbs := []float32{0.9, 0.9, 0.9, 0.9}

	// 1. Evaluate Greedy mode (real arm)
	greedyRec, err := EvaluateSpeculativeAcceptance(targetLogits, draftTokens, draftProbs, SpecAcceptGreedy, 0)
	if err != nil {
		t.Fatalf("greedy evaluation failed: %v", err)
	}
	if greedyRec.AcceptedTokens != 2 {
		t.Fatalf("expected 2 accepted tokens for greedy, got %d", greedyRec.AcceptedTokens)
	}
	if greedyRec.IsSyntheticCalibration {
		t.Fatal("greedy arm should not be synthetic calibration")
	}
	if !greedyRec.QualityClaimAllowed {
		t.Fatal("greedy arm should be allowed for quality claims")
	}
	if err := CertifyQualityClaim(greedyRec); err != nil {
		t.Fatalf("certify greedy quality failed: %v", err)
	}

	// 2. Evaluate Synthetic mode (calibrated throughput benchmarking arm)
	syntheticRate := 0.75
	synthRec, err := EvaluateSpeculativeAcceptance(targetLogits, draftTokens, draftProbs, SpecAcceptSynthetic, syntheticRate)
	if err != nil {
		t.Fatalf("synthetic evaluation failed: %v", err)
	}
	if !synthRec.IsSyntheticCalibration {
		t.Fatal("expected IsSyntheticCalibration = true for synthetic arm")
	}
	if synthRec.QualityClaimAllowed {
		t.Fatal("expected QualityClaimAllowed = false for synthetic arm")
	}
	if synthRec.AcceptanceRate != syntheticRate {
		t.Fatalf("expected rate %v, got %v", syntheticRate, synthRec.AcceptanceRate)
	}

	// 3. Prove quality claim certification strictly refuses synthetic calibration
	err = CertifyQualityClaim(synthRec)
	if !errors.Is(err, ErrSyntheticQualityClaimProhibited) {
		t.Fatalf("expected ErrSyntheticQualityClaimProhibited, got %v", err)
	}
}
