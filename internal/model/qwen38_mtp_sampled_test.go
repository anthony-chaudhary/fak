package model

import (
	"errors"
	"math"
	"testing"
)

func makeTestDist(seed int64, n int) []float32 {
	rng := NewQwen38PRNG(uint64(seed), 1)
	dist := make([]float32, n)
	var sum float64
	for i := 0; i < n; i++ {
		val := float64(rng.NextFloat32())
		dist[i] = float32(val)
		sum += val
	}
	invSum := 1.0 / sum
	for i := range dist {
		dist[i] = float32(float64(dist[i]) * invSum)
	}
	return dist
}

// TestQwen38MTP_Sampled_ResidualDistributionSumToOne verifies that the residual distribution
// sums to 1.0 within numerical precision across varied distribution shapes.
func TestQwen38MTP_Sampled_ResidualDistributionSumToOne(t *testing.T) {
	v, err := NewQwen38SampledSpeculativeVerifier(Qwen38SamplerConfig{
		Temperature: 0.8,
		TopP:        0.95,
		Seed:        12345,
	})
	if err != nil {
		t.Fatalf("unexpected init error: %v", err)
	}

	testCases := []struct {
		name    string
		pTarget []float32
		pDraft  []float32
	}{
		{
			name:    "skewed vs uniform",
			pTarget: []float32{0.7, 0.2, 0.05, 0.05},
			pDraft:  []float32{0.25, 0.25, 0.25, 0.25},
		},
		{
			name:    "disjoint support",
			pTarget: []float32{0.5, 0.5, 0.0, 0.0},
			pDraft:  []float32{0.0, 0.0, 0.5, 0.5},
		},
		{
			name:    "partial overlap",
			pTarget: []float32{0.1, 0.4, 0.3, 0.2},
			pDraft:  []float32{0.3, 0.2, 0.4, 0.1},
		},
		{
			name:    "identical distributions",
			pTarget: []float32{0.25, 0.25, 0.25, 0.25},
			pDraft:  []float32{0.25, 0.25, 0.25, 0.25},
		},
		{
			name:    "high dimension 128",
			pTarget: makeTestDist(42, 128),
			pDraft:  makeTestDist(99, 128),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			residual := v.ResidualDistribution(tc.pTarget, tc.pDraft)
			if len(residual) != len(tc.pTarget) {
				t.Fatalf("residual len = %d, want %d", len(residual), len(tc.pTarget))
			}
			var sum float64
			for i, p := range residual {
				if p < 0 {
					t.Fatalf("negative probability at index %d: %g", i, p)
				}
				sum += float64(p)
			}
			if math.Abs(sum-1.0) > 1e-6 {
				t.Fatalf("residual distribution sum = %g, want 1.0 (difference %g exceeds 1e-6)", sum, math.Abs(sum-1.0))
			}
		})
	}
}

// TestQwen38MTP_Sampled_GreedyRecovery verifies that temperature=0 recovers exact greedy verification.
func TestQwen38MTP_Sampled_GreedyRecovery(t *testing.T) {
	// Temperature 0
	vGreedy, err := NewQwen38SampledSpeculativeVerifier(Qwen38SamplerConfig{
		Temperature: 0.0,
		Seed:        42,
	})
	if err != nil {
		t.Fatalf("unexpected init error: %v", err)
	}

	pTarget := []float32{0.1, 0.2, 0.6, 0.1} // argmax is index 2
	pDraft := []float32{0.6, 0.2, 0.1, 0.1}  // draft argmax is index 0

	// Test 1: Draft token matches target argmax (token 2) -> must accept deterministically
	resAccept := vGreedy.VerifyToken(2, pTarget, pDraft, -1)
	if !resAccept.Accepted {
		t.Fatalf("greedy verification rejected matching argmax token 2")
	}
	if resAccept.Alpha != 1.0 {
		t.Fatalf("greedy accept alpha = %g, want 1.0", resAccept.Alpha)
	}

	// Test 2: Draft token mismatches target argmax (token 0 != 2) -> must reject and replace with argmax (token 2)
	resReject := vGreedy.VerifyToken(0, pTarget, pDraft, -1)
	if resReject.Accepted {
		t.Fatalf("greedy verification accepted mismatched token 0")
	}
	if resReject.ReplacementToken != 2 {
		t.Fatalf("greedy replacement token = %d, want 2 (target argmax)", resReject.ReplacementToken)
	}
	if resReject.Alpha != 0.0 {
		t.Fatalf("greedy reject alpha = %g, want 0.0", resReject.Alpha)
	}

	// Test 3: Verify draft sequence with logits under greedy mode
	targetLogits := [][]float32{
		{1.0, 5.0, 2.0}, // argmax = 1
		{8.0, 2.0, 1.0}, // argmax = 0
		{0.5, 1.0, 9.0}, // argmax = 2
	}
	draftLogits := [][]float32{
		{1.0, 4.0, 2.0}, // draft argmax = 1
		{1.0, 2.0, 8.0}, // draft argmax = 2 (mismatch!)
		{0.5, 1.0, 9.0}, // draft argmax = 2
	}
	draftTokens := []int{1, 2, 2}

	receipt, err := vGreedy.VerifyDraftSequenceLogits(draftTokens, targetLogits, draftLogits)
	if err != nil {
		t.Fatalf("failed to verify draft sequence logits: %v", err)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}
	if receipt.SamplingMode != "greedy" {
		t.Fatalf("receipt sampling mode = %q, want greedy", receipt.SamplingMode)
	}
	if len(receipt.AcceptedTokens) != 1 || receipt.AcceptedTokens[0] != 1 {
		t.Fatalf("accepted tokens = %v, want [1]", receipt.AcceptedTokens)
	}
	if receipt.RejectedAt != 1 {
		t.Fatalf("rejected at = %d, want 1", receipt.RejectedAt)
	}
	if receipt.ReplacementToken != 0 {
		t.Fatalf("replacement token = %d, want 0 (target argmax at step 1)", receipt.ReplacementToken)
	}
}

// TestQwen38MTP_Sampled_RNGTransactionRollback verifies PRNG checkpointing, transaction rollback,
// and deterministic replay.
func TestQwen38MTP_Sampled_RNGTransactionRollback(t *testing.T) {
	v, err := NewQwen38SampledSpeculativeVerifier(Qwen38SamplerConfig{
		Temperature: 0.7,
		TopP:        0.9,
		Seed:        987654321,
		Seq:         42,
	})
	if err != nil {
		t.Fatalf("unexpected init error: %v", err)
	}

	// Checkpoint initial state
	initialState := v.CheckpointRNG()

	// Draw 5 numbers
	var run1 []float32
	for i := 0; i < 5; i++ {
		run1 = append(run1, v.PRNG().NextFloat32())
	}

	// Roll back to initial state
	v.RollbackRNG(initialState)

	// Draw 5 numbers again -> must match run1 bit-for-bit
	for i := 0; i < 5; i++ {
		val := v.PRNG().NextFloat32()
		if val != run1[i] {
			t.Fatalf("deterministic replay mismatch at index %d: got %g, want %g", i, val, run1[i])
		}
	}

	// Test explicit RNG transaction abort
	tx := v.BeginTx()
	_ = v.PRNG().NextFloat32()
	_ = v.PRNG().NextFloat32()
	tx.Rollback()

	// After rollback, the next drawn value must match the value drawn right after the tx checkpoint
	v.RollbackRNG(initialState)
	wantVal := v.PRNG().NextFloat32()

	v.RollbackRNG(initialState)
	tx2 := v.BeginTx()
	gotVal := v.PRNG().NextFloat32()
	if gotVal != wantVal {
		t.Fatalf("transaction value %g != wantVal %g", gotVal, wantVal)
	}
	tx2.Rollback()

	// Check that state was rolled back to initialState
	currState := v.CheckpointRNG()
	if currState.State != initialState.State || currState.Steps != initialState.Steps {
		t.Fatalf("state after tx.Rollback() not restored to checkpoint: got %+v, want %+v", currState, initialState)
	}
}

// TestQwen38MTP_Sampled_DowngradeUnsupportedSamplers asserts typed downgrade on unsupported sampler parameters.
func TestQwen38MTP_Sampled_DowngradeUnsupportedSamplers(t *testing.T) {
	cases := []struct {
		name   string
		config Qwen38SamplerConfig
	}{
		{
			name:   "top_k unsupported",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.9, TopK: 50},
		},
		{
			name:   "min_p unsupported",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.9, MinP: 0.05},
		},
		{
			name:   "repetition penalty unsupported",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.9, RepetitionPenalty: 1.2},
		},
		{
			name:   "presence penalty unsupported",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.9, PresencePenalty: 0.5},
		},
		{
			name:   "frequency penalty unsupported",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.9, FrequencyPenalty: 0.5},
		},
		{
			name:   "negative temperature",
			config: Qwen38SamplerConfig{Temperature: -0.5, TopP: 0.9},
		},
		{
			name:   "top_p zero",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.0},
		},
		{
			name:   "top_p greater than 1",
			config: Qwen38SamplerConfig{Temperature: 0.7, TopP: 1.5},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewQwen38SampledSpeculativeVerifier(tc.config)
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
			if !errors.Is(err, ErrSamplingModeUnsupported) {
				t.Fatalf("error %v does not unwrap to ErrSamplingModeUnsupported", err)
			}
			var downgradeErr *Qwen38SamplingDowngradeError
			if !errors.As(err, &downgradeErr) {
				t.Fatalf("error %v is not *Qwen38SamplingDowngradeError", err)
			}
			if downgradeErr.Reason != Qwen38MTPSamplingUnsupported {
				t.Fatalf("downgrade reason = %q, want %q", downgradeErr.Reason, Qwen38MTPSamplingUnsupported)
			}
		})
	}
}

// TestQwen38MTP_Sampled_QualityEnvelopeAgainstTargetOnly validates that speculative rejection sampling
// exactly preserves the target distribution mathematically and empirically within sampling bounds.
func TestQwen38MTP_Sampled_QualityEnvelopeAgainstTargetOnly(t *testing.T) {
	v, err := NewQwen38SampledSpeculativeVerifier(Qwen38SamplerConfig{
		Temperature: 0.8,
		TopP:        1.0,
		Seed:        20260907,
	})
	if err != nil {
		t.Fatalf("unexpected init error: %v", err)
	}

	pTarget := []float32{0.45, 0.30, 0.15, 0.10}
	pDraft := []float32{0.20, 0.50, 0.20, 0.10}

	// 1. Verify exact mathematical distribution identity (Leviathan-Chen Theorem 1)
	effective := v.EffectiveDistribution(pTarget, pDraft)
	for i := range effective {
		diff := math.Abs(float64(effective[i] - pTarget[i]))
		if diff > 1e-6 {
			t.Fatalf("theoretical distribution divergence at token %d: effective=%g, target=%g, diff=%g",
				i, effective[i], pTarget[i], diff)
		}
	}

	// 2. Monte Carlo verification: sample 20,000 steps through speculative verification
	// and assert output frequencies closely approximate target distribution pTarget.
	numSamples := 20000
	counts := make([]int, len(pTarget))

	for s := 0; s < numSamples; s++ {
		// Draft token sampled from pDraft
		uDraft := v.PRNG().NextFloat32()
		draftToken := v.SampleFromDistribution(pDraft, uDraft)

		// Speculative verification step
		ver := v.VerifyToken(draftToken, pTarget, pDraft, -1)
		var outputToken int
		if ver.Accepted {
			outputToken = ver.DraftToken
		} else {
			outputToken = ver.ReplacementToken
		}
		counts[outputToken]++
	}

	// Assert empirical frequencies match target probabilities within sampling margin of error (tol = 0.02)
	for i, targetProb := range pTarget {
		empiricalProb := float64(counts[i]) / float64(numSamples)
		diff := math.Abs(empiricalProb - float64(targetProb))
		if diff > 0.02 {
			t.Errorf("empirical probability divergence at token %d: empirical=%g, target=%g, diff=%g > 0.02",
				i, empiricalProb, targetProb, diff)
		}
	}
}

// TestQwen38MTP_Sampled_VerificationReceipt validates that the emitted receipt contains
// the required schema, engine, distribution metrics, and passes Validate().
func TestQwen38MTP_Sampled_VerificationReceipt(t *testing.T) {
	v, err := NewQwen38SampledSpeculativeVerifier(Qwen38SamplerConfig{
		Temperature: 0.7,
		TopP:        0.9,
		Seed:        1337,
	})
	if err != nil {
		t.Fatalf("unexpected init error: %v", err)
	}

	draftTokens := []int{0, 1, 2}
	pTargets := [][]float32{
		{0.7, 0.2, 0.1},
		{0.1, 0.8, 0.1},
		{0.1, 0.1, 0.8},
		{0.5, 0.5, 0.0}, // bonus distribution
	}
	pDrafts := [][]float32{
		{0.6, 0.3, 0.1},
		{0.2, 0.7, 0.1},
		{0.1, 0.2, 0.7},
	}

	receipt := v.VerifyDraftSequence(draftTokens, pTargets, pDrafts)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if receipt.Engine != Qwen38EngineMTP {
		t.Fatalf("receipt Engine = %q, want %q", receipt.Engine, Qwen38EngineMTP)
	}
	if receipt.SchemaVersion != Qwen38SampledVerificationReceiptSchema {
		t.Fatalf("receipt SchemaVersion = %q, want %q", receipt.SchemaVersion, Qwen38SampledVerificationReceiptSchema)
	}
	if receipt.Metrics.TotalVariationDistance <= 0 {
		t.Fatalf("expected positive TV distance, got %g", receipt.Metrics.TotalVariationDistance)
	}
	if receipt.Metrics.ExpectedAcceptanceRate <= 0 || receipt.Metrics.ExpectedAcceptanceRate > 1.0 {
		t.Fatalf("expected valid acceptance rate in (0, 1], got %g", receipt.Metrics.ExpectedAcceptanceRate)
	}
}
