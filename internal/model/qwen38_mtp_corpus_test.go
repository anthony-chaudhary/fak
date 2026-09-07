package model

import (
	"math"
	"testing"
)

func TestQwen38MTP_QualityCorpusDomains(t *testing.T) {
	corpus := NewDefaultQwen38QualityCorpus()
	if len(corpus.Prompts) == 0 {
		t.Fatal("expected non-empty corpus prompts")
	}

	domainCounts := make(map[Qwen38CorpusDomain]int)
	for _, p := range corpus.Prompts {
		if p.ID == "" || p.Title == "" || p.SubCategory == "" {
			t.Fatalf("prompt %+v has empty required metadata", p)
		}
		if len(p.Tokens) == 0 {
			t.Fatalf("prompt %q has empty tokens", p.ID)
		}
		// Invariant: tokens must stay within vocabulary [0, 97) for synthetic test models
		for i, tok := range p.Tokens {
			if tok < 0 || tok >= 97 {
				t.Fatalf("prompt %q token %d at index %d exceeds vocabulary [0, 97)", p.ID, tok, i)
			}
		}
		domainCounts[p.Domain]++
	}

	// Invariant: all 4 domains must have representative prompts
	requiredDomains := []Qwen38CorpusDomain{
		CorpusDomainCoding,
		CorpusDomainAgentToolCall,
		CorpusDomainLongContext,
		CorpusDomainOrdinaryGeneration,
	}

	for _, req := range requiredDomains {
		if domainCounts[req] == 0 {
			t.Fatalf("missing required domain %q in corpus", req)
		}
	}

	// Invariant: LongContext domain prompt must have prefix > 4096 tokens
	foundLong := false
	for _, p := range corpus.Prompts {
		if p.Domain == CorpusDomainLongContext {
			foundLong = true
			if len(p.Tokens) <= 4096 {
				t.Fatalf("long context prompt %q has length %d <= 4096", p.ID, len(p.Tokens))
			}
			if p.NeedlePos <= 0 || p.NeedlePos >= len(p.Tokens) {
				t.Fatalf("invalid needle position %d in long context prompt", p.NeedlePos)
			}
			if p.Tokens[p.NeedlePos] != p.ExpectedNeedle {
				t.Fatalf("token at needle pos %d is %d, want %d", p.NeedlePos, p.Tokens[p.NeedlePos], p.ExpectedNeedle)
			}
		}
	}
	if !foundLong {
		t.Fatal("long context domain prompt not found")
	}
}

func TestQwen38MTP_QualityGreedyBitExactParity(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	corpus := NewDefaultQwen38QualityCorpus()

	cfg := DefaultQwen38CorpusEvaluationConfig("greedy")
	cfg.MaxGenTokens = 4

	receipt, err := corpus.Evaluate(m, cfg)
	if err != nil {
		t.Fatalf("greedy corpus evaluation failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	// Invariant 1: Greedy output equality = 100% bit-exact match
	if !receipt.BitExactMatch {
		t.Fatal("expected 100% bit-exact match across all domains")
	}
	if receipt.TokenMatchRate != 1.0 {
		t.Fatalf("token match rate = %g, want 1.0", receipt.TokenMatchRate)
	}

	// Invariant 2: All 4 domains individually achieve bit-exact parity
	for _, dm := range receipt.DomainMetrics {
		if !dm.BitExactMatch || dm.TokenMatchRate != 1.0 {
			t.Fatalf("domain %q failed bit-exact parity: exact=%v, rate=%g", dm.Domain, dm.BitExactMatch, dm.TokenMatchRate)
		}
		if dm.PromptCount <= 0 || dm.TokensEvaluated <= 0 {
			t.Fatalf("domain %q has empty prompt/token counts", dm.Domain)
		}
	}

	if !receipt.QualityEnvelopePassed {
		t.Fatal("quality envelope failed in greedy mode")
	}
}

func TestQwen38MTP_QualitySampledEnvelope(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	corpus := NewDefaultQwen38QualityCorpus()

	cfg := DefaultQwen38CorpusEvaluationConfig("sampled")
	cfg.MaxGenTokens = 4
	cfg.MaxTVDThreshold = 0.05
	cfg.MaxPPLDelta = 0.05
	cfg.MinAcceptRate = 0.30

	receipt, err := corpus.Evaluate(m, cfg)
	if err != nil {
		t.Fatalf("sampled corpus evaluation failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	// Invariant 1: Total Variation Distance <= threshold
	if receipt.TotalVariationDistance > cfg.MaxTVDThreshold {
		t.Fatalf("total variation distance %g exceeds threshold %g", receipt.TotalVariationDistance, cfg.MaxTVDThreshold)
	}

	// Invariant 2: Perplexity delta <= threshold
	if receipt.PerplexityDelta > cfg.MaxPPLDelta {
		t.Fatalf("perplexity delta %g exceeds threshold %g", receipt.PerplexityDelta, cfg.MaxPPLDelta)
	}

	// Invariant 3: Acceptance rate within expected bounds
	if receipt.AcceptanceRate < cfg.MinAcceptRate {
		t.Fatalf("acceptance rate %g below minimum %g", receipt.AcceptanceRate, cfg.MinAcceptRate)
	}

	if !receipt.QualityEnvelopePassed {
		t.Fatal("sampled quality envelope check failed")
	}
}

func TestQwen38MTP_QualityReceiptValidation(t *testing.T) {
	allDomains := []Qwen38CorpusDomain{
		CorpusDomainCoding,
		CorpusDomainAgentToolCall,
		CorpusDomainLongContext,
		CorpusDomainOrdinaryGeneration,
	}

	validReceipt := Qwen38MTPQualityReceipt{
		SchemaVersion:          Qwen38MTPQualityReceiptSchema,
		Engine:                 Qwen38EngineMTP,
		EvaluationMode:         "greedy",
		DomainsCovered:         allDomains,
		TotalPrompts:           8,
		TotalTokensEvaluated:   32,
		BitExactMatch:          true,
		TokenMatchRate:         1.0,
		TotalVariationDistance: 0.0,
		PerplexityDelta:        0.0,
		AcceptanceRate:         1.0,
		QualityEnvelopePassed:  true,
	}

	if err := validReceipt.Validate(); err != nil {
		t.Fatalf("valid receipt failed validation: %v", err)
	}

	// 1. Invalid schema version
	r1 := validReceipt
	r1.SchemaVersion = "bad/schema/v1"
	if err := r1.Validate(); err == nil {
		t.Fatal("expected error for bad schema version")
	}

	// 2. Non-native engine
	r2 := validReceipt
	r2.Engine = Qwen38MTPEngine("llama.cpp")
	if err := r2.Validate(); err == nil {
		t.Fatal("expected error for non-native engine")
	}

	// 3. Missing domain
	r3 := validReceipt
	r3.DomainsCovered = []Qwen38CorpusDomain{CorpusDomainCoding, CorpusDomainAgentToolCall}
	if err := r3.Validate(); err == nil {
		t.Fatal("expected error for missing domains")
	}

	// 4. Greedy mismatch
	r4 := validReceipt
	r4.BitExactMatch = false
	r4.TokenMatchRate = 0.95
	if err := r4.Validate(); err == nil {
		t.Fatal("expected error for greedy mismatch")
	}

	// 5. Sampled mode invalid TVD or NaN PPL
	r5 := validReceipt
	r5.EvaluationMode = "sampled"
	r5.PerplexityDelta = math.NaN()
	if err := r5.Validate(); err == nil {
		t.Fatal("expected error for NaN perplexity delta")
	}

	// 6. Quality envelope failed flag
	r6 := validReceipt
	r6.QualityEnvelopePassed = false
	if err := r6.Validate(); err == nil {
		t.Fatal("expected error when quality envelope passed is false")
	}
}
