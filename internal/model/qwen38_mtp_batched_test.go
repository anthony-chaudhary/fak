package model

import (
	"testing"
	"time"
)

func TestQwen38MTP_Batched_SingleOperation(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3

	ref := m.NewSession()
	ref.captureTargetHidden = true
	t.Cleanup(ref.Close)
	logits := ref.Prefill(prompt)

	draft := make([]int, depth)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = ref.Step(draft[i])
	}

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)

	verifier, err := NewQwen38BatchedVerifier(target)
	if err != nil {
		t.Fatalf("failed to create verifier: %v", err)
	}

	receipt, err := verifier.VerifyBlock(draft, boundary)
	if err != nil {
		t.Fatalf("batched verify failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if receipt.Engine != Qwen38EngineMTP {
		t.Fatalf("receipt engine=%q, want %q", receipt.Engine, Qwen38EngineMTP)
	}
	if !receipt.OneOperation {
		t.Fatalf("receipt.OneOperation=%v, want true", receipt.OneOperation)
	}
	if receipt.TargetVerificationOperations != 1 {
		t.Fatalf("receipt.TargetVerificationOperations=%d, want 1", receipt.TargetVerificationOperations)
	}
	if receipt.TargetDecodeSteps != 0 {
		t.Fatalf("receipt.TargetDecodeSteps=%d, want 0", receipt.TargetDecodeSteps)
	}
	if receipt.AcceptedCount != depth {
		t.Fatalf("accepted count=%d, want %d", receipt.AcceptedCount, depth)
	}
	if receipt.RejectedCount != 0 {
		t.Fatalf("rejected count=%d, want 0", receipt.RejectedCount)
	}
}

func TestQwen38MTP_Batched_FullAccept(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 4

	ref := m.NewSession()
	ref.captureTargetHidden = true
	t.Cleanup(ref.Close)
	logits := ref.Prefill(prompt)

	draft := make([]int, depth)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = ref.Step(draft[i])
	}

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)

	receipt, err := VerifyMTPBlockBatched(target, draft, boundary)
	if err != nil {
		t.Fatalf("VerifyMTPBlockBatched failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("invalid receipt: %v", err)
	}

	if receipt.AcceptedCount != depth || receipt.RejectedCount != 0 {
		t.Fatalf("accepted=%d rejected=%d, want %d/0", receipt.AcceptedCount, receipt.RejectedCount, depth)
	}
	if len(receipt.AcceptedTokens) != depth {
		t.Fatalf("accepted tokens len=%d, want %d", len(receipt.AcceptedTokens), depth)
	}
	for i, tok := range draft {
		if receipt.AcceptedTokens[i] != tok {
			t.Fatalf("token %d: got %d, want %d", i, receipt.AcceptedTokens[i], tok)
		}
	}

	if target.Cache.Len() != len(prompt)+depth {
		t.Fatalf("target cache length=%d, want %d", target.Cache.Len(), len(prompt)+depth)
	}
	if receipt.BonusToken < 0 {
		t.Fatalf("bonus token=%d, want >= 0 on full accept", receipt.BonusToken)
	}
}

func TestQwen38MTP_Batched_PartialAccept(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3

	ref := m.NewSession()
	ref.captureTargetHidden = true
	t.Cleanup(ref.Close)
	logits := ref.Prefill(prompt)

	draft := make([]int, depth)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = ref.Step(draft[i])
	}

	// Corrupt the 3rd draft token (index 2) so only 2 match
	originalToken2 := draft[2]
	draft[2] = (draft[2] + 7) % m.Cfg.VocabSize
	if draft[2] == originalToken2 {
		draft[2] = (originalToken2 + 13) % m.Cfg.VocabSize
	}

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)

	receipt, err := VerifyMTPBlockBatched(target, draft, boundary)
	if err != nil {
		t.Fatalf("VerifyMTPBlockBatched failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("invalid receipt: %v", err)
	}

	if receipt.AcceptedCount != 2 {
		t.Fatalf("accepted count=%d, want 2", receipt.AcceptedCount)
	}
	if receipt.RejectedCount != 1 {
		t.Fatalf("rejected count=%d, want 1", receipt.RejectedCount)
	}
	if len(receipt.AcceptedTokens) != 2 {
		t.Fatalf("len(acceptedTokens)=%d, want 2", len(receipt.AcceptedTokens))
	}
	if receipt.BonusToken != -1 {
		t.Fatalf("bonus token=%d, want -1 on partial accept", receipt.BonusToken)
	}

	// Check target cache length: exactly prompt + 2 accepted tokens
	if target.Cache.Len() != len(prompt)+2 {
		t.Fatalf("target cache length=%d, want %d", target.Cache.Len(), len(prompt)+2)
	}

	// Compare target state with independent reference that stepped exactly the 2 accepted tokens
	expectedRef := m.NewSession()
	expectedRef.captureTargetHidden = true
	t.Cleanup(expectedRef.Close)
	expectedRef.Prefill(prompt)
	expectedRef.Step(draft[0])
	expectedRef.Step(draft[1])

	assertQwen35MTPTargetStateEqual(t, target, expectedRef)
}

func TestQwen38MTP_Batched_TotalRejection_ExactRollback(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)

	expected0 := argmaxF32(boundary)
	draft := make([]int, depth)
	// Corrupt first draft token so divergence is immediate at index 0
	draft[0] = (expected0 + 11) % m.Cfg.VocabSize
	if draft[0] == expected0 {
		draft[0] = (expected0 + 23) % m.Cfg.VocabSize
	}
	draft[1] = 42
	draft[2] = 43

	wantPrefix := m.NewSession()
	wantPrefix.captureTargetHidden = true
	t.Cleanup(wantPrefix.Close)
	wantPrefix.Prefill(prompt)

	receipt, err := VerifyMTPBlockBatched(target, draft, boundary)
	if err != nil {
		t.Fatalf("VerifyMTPBlockBatched failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("invalid receipt: %v", err)
	}

	if receipt.AcceptedCount != 0 {
		t.Fatalf("accepted count=%d, want 0", receipt.AcceptedCount)
	}
	if receipt.RejectedCount != depth {
		t.Fatalf("rejected count=%d, want %d", receipt.RejectedCount, depth)
	}
	if len(receipt.AcceptedTokens) != 0 {
		t.Fatalf("accepted tokens=%v, want empty", receipt.AcceptedTokens)
	}

	// Verify exact rollback with zero state leaks
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("target cache length=%d, want %d", target.Cache.Len(), len(prompt))
	}
	assertQwen35MTPTargetStateEqual(t, target, wantPrefix)
}

func TestQwen38MTP_Batched_LatencyAccounting(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	draft := []int{2, 3}

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)

	draftTime := 2500 * time.Nanosecond
	receipt, err := VerifyMTPBlockBatched(target, draft, boundary, draftTime)
	if err != nil {
		t.Fatalf("VerifyMTPBlockBatched failed: %v", err)
	}

	if receipt.DraftTimeNS != draftTime {
		t.Fatalf("draft time=%v, want %v", receipt.DraftTimeNS, draftTime)
	}
	if receipt.VerifyTimeNS <= 0 {
		t.Fatalf("verify time=%v, want > 0", receipt.VerifyTimeNS)
	}
	if receipt.TotalWallTimeNS < receipt.DraftTimeNS+receipt.VerifyTimeNS {
		t.Fatalf("total wall time=%v, want >= draft+verify (%v)", receipt.TotalWallTimeNS, receipt.DraftTimeNS+receipt.VerifyTimeNS)
	}

	if !receipt.Accounting.Setup.Measured {
		t.Fatal("setup cost unmeasured")
	}
	if !receipt.Accounting.Drafting.Measured {
		t.Fatal("drafting cost unmeasured")
	}
	if !receipt.Accounting.TargetVerification.Measured {
		t.Fatal("target verification cost unmeasured")
	}
	if !receipt.Accounting.Rejection.Measured {
		t.Fatal("rejection cost unmeasured")
	}
	if !receipt.Accounting.MemoryMeasured {
		t.Fatal("memory cost unmeasured")
	}
}

func TestQwen38MTP_Batched_DowngradeHandling(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3

	ref := m.NewSession()
	ref.captureTargetHidden = true
	t.Cleanup(ref.Close)
	logits := ref.Prefill(prompt)

	draft := make([]int, depth)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = ref.Step(draft[i])
	}

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)

	// F16 unsupported on one-operation verify -> triggers typed downgrade
	target.F16 = true

	receipt, err := VerifyMTPBlockBatched(target, draft, boundary)
	if err != nil {
		t.Fatalf("VerifyMTPBlockBatched failed on downgrade: %v", err)
	}

	if receipt.OneOperation {
		t.Fatal("receipt.OneOperation=true on downgrade, want false")
	}
	if receipt.FallbackEngine != Qwen38EngineTargetDecode {
		t.Fatalf("fallback engine=%q, want %q", receipt.FallbackEngine, Qwen38EngineTargetDecode)
	}
	if receipt.TargetVerificationOperations != 0 {
		t.Fatalf("target_verification_operations=%d, want 0", receipt.TargetVerificationOperations)
	}
	if receipt.TargetDecodeSteps != depth {
		t.Fatalf("target_decode_steps=%d, want %d", receipt.TargetDecodeSteps, depth)
	}
	if receipt.DowngradeReason == "" {
		t.Fatal("downgrade reason empty on downgrade")
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	// Verify that despite downgrade, acceptance was correctly calculated
	if receipt.AcceptedCount != depth {
		t.Fatalf("accepted count=%d, want %d", receipt.AcceptedCount, depth)
	}
}
