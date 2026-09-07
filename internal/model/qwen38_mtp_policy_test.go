package model

import (
	"testing"
)

func TestQwen38MTP_Policy_DeterministicSelection(t *testing.T) {
	policy := NewQwen38CompositeDraftPolicy()

	// 1. All eligible -> MTP selected deterministically as primary, others rejected under mutual exclusion.
	inputAllEligible := Qwen38DraftPolicyInput{
		OperatorEnabled: true,
		Prompt: PromptCharacteristics{
			TokenCount:     128,
			Repetitive:     true,
			HasLookupMatch: true,
			HasNGramMatch:  true,
		},
		Cache: CacheCharacteristics{
			PrefixCacheReady: true,
			CacheHitRate:     0.95,
			ResidentTokens:   128,
		},
		Candidates: map[DraftSource]DraftCandidateState{
			DraftSourceMTP: {
				Source:   DraftSourceMTP,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
				Performance: DraftSourcePerformance{
					ProposedTokens: 50,
					AcceptedTokens: 40,
					AcceptanceRate: 0.80,
				},
			},
			DraftSourcePromptLookup: {
				Source:   DraftSourcePromptLookup,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
				Performance: DraftSourcePerformance{
					ProposedTokens: 30,
					AcceptedTokens: 25,
					AcceptanceRate: 0.83,
				},
			},
			DraftSourceNGram: {
				Source:   DraftSourceNGram,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
				Performance: DraftSourcePerformance{
					ProposedTokens: 20,
					AcceptedTokens: 15,
					AcceptanceRate: 0.75,
				},
			},
		},
	}

	receipt1 := policy.SelectDraftSource(inputAllEligible)
	if err := receipt1.Validate(); err != nil {
		t.Fatalf("receipt1.Validate() error = %v", err)
	}
	if receipt1.ChosenSource != DraftSourceMTP {
		t.Fatalf("receipt1.ChosenSource = %q, want %q", receipt1.ChosenSource, DraftSourceMTP)
	}
	if receipt1.Engine != Qwen38EngineMTP {
		t.Fatalf("receipt1.Engine = %q, want %q", receipt1.Engine, Qwen38EngineMTP)
	}
	if len(receipt1.Rejected) != 2 {
		t.Fatalf("receipt1 len(Rejected) = %d, want 2", len(receipt1.Rejected))
	}
	for _, rej := range receipt1.Rejected {
		if rej.Reason != DraftRejectionMutualExclusion {
			t.Errorf("rejected source %s reason = %q, want %q", rej.Source, rej.Reason, DraftRejectionMutualExclusion)
		}
	}

	// Prove determinism: running identical input returns identical output
	receipt2 := policy.SelectDraftSource(inputAllEligible)
	if receipt1.ChosenSource != receipt2.ChosenSource || receipt1.Engine != receipt2.Engine || len(receipt1.Rejected) != len(receipt2.Rejected) {
		t.Fatalf("non-deterministic selection across consecutive invocations")
	}

	// 2. MTP unhealthy -> fallback to PromptLookup
	inputMTPUnhealthy := inputAllEligible
	mtpCandidate := inputMTPUnhealthy.Candidates[DraftSourceMTP]
	mtpCandidate.Health = DraftSourceHealth{Healthy: false, Detail: "memory pressure or device fault"}
	inputMTPUnhealthy.Candidates[DraftSourceMTP] = mtpCandidate

	receiptPL := policy.SelectDraftSource(inputMTPUnhealthy)
	if err := receiptPL.Validate(); err != nil {
		t.Fatalf("receiptPL.Validate() error = %v", err)
	}
	if receiptPL.ChosenSource != DraftSourcePromptLookup {
		t.Fatalf("receiptPL.ChosenSource = %q, want %q", receiptPL.ChosenSource, DraftSourcePromptLookup)
	}
	if receiptPL.Engine != Qwen38EngineTargetDecode {
		t.Fatalf("receiptPL.Engine = %q, want %q", receiptPL.Engine, Qwen38EngineTargetDecode)
	}

	// 3. MTP and PromptLookup ineligible -> fallback to NGram
	inputNGramOnly := inputMTPUnhealthy
	plCandidate := inputNGramOnly.Candidates[DraftSourcePromptLookup]
	plCandidate.Eligible = false
	inputNGramOnly.Candidates[DraftSourcePromptLookup] = plCandidate

	receiptNG := policy.SelectDraftSource(inputNGramOnly)
	if err := receiptNG.Validate(); err != nil {
		t.Fatalf("receiptNG.Validate() error = %v", err)
	}
	if receiptNG.ChosenSource != DraftSourceNGram {
		t.Fatalf("receiptNG.ChosenSource = %q, want %q", receiptNG.ChosenSource, DraftSourceNGram)
	}
	if receiptNG.Engine != Qwen38EngineTargetDecode {
		t.Fatalf("receiptNG.Engine = %q, want %q", receiptNG.Engine, Qwen38EngineTargetDecode)
	}

	// 4. All ineligible -> fallback to Target-only (DraftSourceNone)
	inputNone := inputNGramOnly
	ngCandidate := inputNone.Candidates[DraftSourceNGram]
	ngCandidate.Eligible = false
	inputNone.Candidates[DraftSourceNGram] = ngCandidate

	receiptNone := policy.SelectDraftSource(inputNone)
	if err := receiptNone.Validate(); err != nil {
		t.Fatalf("receiptNone.Validate() error = %v", err)
	}
	if receiptNone.ChosenSource != DraftSourceNone {
		t.Fatalf("receiptNone.ChosenSource = %q, want %q", receiptNone.ChosenSource, DraftSourceNone)
	}
	if receiptNone.Engine != Qwen38EngineTargetDecode {
		t.Fatalf("receiptNone.Engine = %q, want %q", receiptNone.Engine, Qwen38EngineTargetDecode)
	}
	if receiptNone.DowngradeReason == "" {
		t.Fatalf("receiptNone expected non-empty DowngradeReason")
	}
}

func TestQwen38MTP_Policy_OperatorDisabledDowngrade(t *testing.T) {
	policy := NewQwen38CompositeDraftPolicy()

	inputDisabled := Qwen38DraftPolicyInput{
		OperatorEnabled: false,
		Candidates: map[DraftSource]DraftCandidateState{
			DraftSourceMTP: {Eligible: true, Health: DraftSourceHealth{Healthy: true}},
		},
	}

	receipt := policy.SelectDraftSource(inputDisabled)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate() error = %v", err)
	}
	if receipt.ChosenSource != DraftSourceNone {
		t.Fatalf("receipt.ChosenSource = %q, want %q", receipt.ChosenSource, DraftSourceNone)
	}
	if receipt.Engine != Qwen38EngineTargetDecode {
		t.Fatalf("receipt.Engine = %q, want %q", receipt.Engine, Qwen38EngineTargetDecode)
	}
	if receipt.DowngradeReason != Qwen38MTPDisabledByPolicy {
		t.Fatalf("receipt.DowngradeReason = %q, want %q", receipt.DowngradeReason, Qwen38MTPDisabledByPolicy)
	}
	if len(receipt.Rejected) != 3 {
		t.Fatalf("receipt len(Rejected) = %d, want 3", len(receipt.Rejected))
	}
	for _, rej := range receipt.Rejected {
		if rej.Reason != DraftRejectionPolicyDisabled {
			t.Errorf("rejected source %s reason = %q, want %q", rej.Source, rej.Reason, DraftRejectionPolicyDisabled)
		}
	}
}

func TestQwen38MTP_Policy_LowAcceptanceDowngrade(t *testing.T) {
	policy := NewQwen38CompositeDraftPolicy()
	policy.MinAcceptanceRate = 0.50

	input := Qwen38DraftPolicyInput{
		OperatorEnabled: true,
		Candidates: map[DraftSource]DraftCandidateState{
			DraftSourceMTP: {
				Source:   DraftSourceMTP,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
				Performance: DraftSourcePerformance{
					ProposedTokens: 100,
					AcceptedTokens: 20,
					AcceptanceRate: 0.20, // Low acceptance rate < 0.50
				},
			},
		},
	}

	receipt := policy.SelectDraftSource(input)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate() error = %v", err)
	}
	if receipt.ChosenSource != DraftSourceNone {
		t.Fatalf("receipt.ChosenSource = %q, want %q", receipt.ChosenSource, DraftSourceNone)
	}
	if len(receipt.Rejected) != 3 {
		t.Fatalf("len(receipt.Rejected) = %d, want 3", len(receipt.Rejected))
	}

	var mtpRej *RejectedDraftSource
	for i := range receipt.Rejected {
		if receipt.Rejected[i].Source == DraftSourceMTP {
			mtpRej = &receipt.Rejected[i]
			break
		}
	}
	if mtpRej == nil {
		t.Fatal("DraftSourceMTP missing from rejected list")
	}
	if mtpRej.Reason != DraftRejectionLowAcceptance {
		t.Fatalf("mtpRej.Reason = %q, want %q", mtpRej.Reason, DraftRejectionLowAcceptance)
	}
}

func TestQwen38MTP_Policy_PromptAndCacheConstraints(t *testing.T) {
	policy := NewQwen38CompositeDraftPolicy()

	// Short prompt, cold cache, no repetitions -> PromptLookup and NGram rejected under prompt/cache invariants
	input := Qwen38DraftPolicyInput{
		OperatorEnabled: true,
		Prompt: PromptCharacteristics{
			TokenCount:     8,
			Repetitive:     false,
			HasLookupMatch: false,
			HasNGramMatch:  false,
		},
		Cache: CacheCharacteristics{
			PrefixCacheReady: false,
			ResidentTokens:   0,
		},
		Candidates: map[DraftSource]DraftCandidateState{
			DraftSourcePromptLookup: {
				Source:   DraftSourcePromptLookup,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
			},
			DraftSourceNGram: {
				Source:   DraftSourceNGram,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
			},
		},
	}

	receipt := policy.SelectDraftSource(input)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate() error = %v", err)
	}
	if receipt.ChosenSource != DraftSourceNone {
		t.Fatalf("receipt.ChosenSource = %q, want %q", receipt.ChosenSource, DraftSourceNone)
	}

	rejections := make(map[DraftSource]DraftRejectionReason)
	for _, rej := range receipt.Rejected {
		rejections[rej.Source] = rej.Reason
	}
	if rejections[DraftSourcePromptLookup] != DraftRejectionPromptIneligible {
		t.Errorf("PromptLookup reason = %q, want %q", rejections[DraftSourcePromptLookup], DraftRejectionPromptIneligible)
	}
	if rejections[DraftSourceNGram] != DraftRejectionPromptIneligible {
		t.Errorf("NGram reason = %q, want %q", rejections[DraftSourceNGram], DraftRejectionPromptIneligible)
	}
}

func TestQwen38MTP_Policy_StackingRules(t *testing.T) {
	policy := NewQwen38CompositeDraftPolicy()

	// Case 1: AllowStacked requested on input, but ExplicitStackAuth is false -> rejects stacking
	inputNoAuth := Qwen38DraftPolicyInput{
		OperatorEnabled:   true,
		AllowStacked:      true,
		ExplicitStackAuth: false,
		Prompt: PromptCharacteristics{
			TokenCount:     128,
			Repetitive:     true,
			HasLookupMatch: true,
			HasNGramMatch:  true,
		},
		Cache: CacheCharacteristics{
			PrefixCacheReady: true,
		},
		Candidates: map[DraftSource]DraftCandidateState{
			DraftSourcePromptLookup: {
				Source:   DraftSourcePromptLookup,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
			},
			DraftSourceNGram: {
				Source:   DraftSourceNGram,
				Eligible: true,
				Health:   DraftSourceHealth{Healthy: true},
			},
		},
	}

	receipt1 := policy.SelectDraftSource(inputNoAuth)
	if err := receipt1.Validate(); err != nil {
		t.Fatalf("receipt1.Validate() error = %v", err)
	}
	if len(receipt1.StackedSources) != 0 {
		t.Fatalf("expected 0 stacked sources without explicit auth, got %v", receipt1.StackedSources)
	}
	if receipt1.ChosenSource != DraftSourcePromptLookup {
		t.Fatalf("receipt1.ChosenSource = %q, want %q", receipt1.ChosenSource, DraftSourcePromptLookup)
	}

	// Case 2: AllowStacked requested with ExplicitStackAuth = true -> stacks PromptLookup and NGram
	inputAuth := inputNoAuth
	inputAuth.ExplicitStackAuth = true

	receipt2 := policy.SelectDraftSource(inputAuth)
	if err := receipt2.Validate(); err != nil {
		t.Fatalf("receipt2.Validate() error = %v", err)
	}
	if len(receipt2.StackedSources) != 2 {
		t.Fatalf("expected 2 stacked sources with explicit auth, got %v", receipt2.StackedSources)
	}
	if receipt2.StackedSources[0] != DraftSourcePromptLookup || receipt2.StackedSources[1] != DraftSourceNGram {
		t.Fatalf("unexpected stacked sources: %v", receipt2.StackedSources)
	}
}
