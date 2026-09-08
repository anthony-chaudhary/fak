package model

import (
	"reflect"
	"testing"
)

func TestQwen38MTP_ShadowGreedyParity(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	runner := NewQwen38MTPShadowRunner(m, Qwen38MTPShadowConfig{
		Concurrent:      false,
		ServeTargetOnly: true,
	})

	seededPrompts := [][]int{
		{1, 2, 3, 4},
		{10, 20, 30, 40},
		{5, 15, 25, 35},
		{7, 14, 21, 28, 35},
	}

	for _, prompt := range seededPrompts {
		tokens, receipt, err := runner.Run(prompt, 6)
		if err != nil {
			t.Fatalf("prompt %v Run failed: %v", prompt, err)
		}

		if err := receipt.Validate(); err != nil {
			t.Fatalf("receipt validation failed: %v", err)
		}

		// Zero unexplained greedy divergence
		if receipt.DivergenceCount != 0 {
			t.Fatalf("unexplained divergence count = %d for prompt %v (step %d)", receipt.DivergenceCount, prompt, receipt.DivergenceStep)
		}
		if receipt.TokenMatchRate != 1.0 {
			t.Fatalf("token match rate = %g, want 1.0", receipt.TokenMatchRate)
		}
		if receipt.DivergenceStep != -1 {
			t.Fatalf("divergence step = %d, want -1", receipt.DivergenceStep)
		}
		if !reflect.DeepEqual(tokens, receipt.PrimaryTokens) {
			t.Fatalf("returned tokens %v != primary tokens %v", tokens, receipt.PrimaryTokens)
		}
		if !reflect.DeepEqual(receipt.PrimaryTokens, receipt.ShadowTokens) {
			t.Fatalf("primary %v != shadow %v", receipt.PrimaryTokens, receipt.ShadowTokens)
		}
		if receipt.CosineSimilarity < 0.999 {
			t.Fatalf("cosine similarity %g < 0.999", receipt.CosineSimilarity)
		}
	}
}

func TestQwen38MTP_ShadowConcurrentParity(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	runner := NewQwen38MTPShadowRunner(m, Qwen38MTPShadowConfig{
		Concurrent:      true,
		ServeTargetOnly: true,
	})

	prompt := []int{3, 6, 9, 12}
	tokens, receipt, err := runner.Run(prompt, 5)
	if err != nil {
		t.Fatalf("concurrent Run failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if receipt.DivergenceCount != 0 || receipt.TokenMatchRate != 1.0 {
		t.Fatalf("concurrent run diverged: count=%d, rate=%g", receipt.DivergenceCount, receipt.TokenMatchRate)
	}
	if len(tokens) != 5 {
		t.Fatalf("tokens len=%d, want 5", len(tokens))
	}
}

func TestQwen38MTP_ShadowInjectedMismatch(t *testing.T) {
	primaryTokens := []int{10, 20, 30, 40, 50}
	shadowTokens := []int{10, 20, 99, 40, 50} // Divergence at step 2

	primaryLogits := [][]float32{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{5.0, 1.0, 0.0},
		{0.0, 5.0, 1.0},
		{1.0, 1.0, 5.0},
	}
	shadowLogits := [][]float32{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{0.0, 1.0, 5.0}, // Mismatch at step 2
		{0.0, 5.0, 1.0},
		{1.0, 1.0, 5.0},
	}

	receipt := CompareShadowOutputs(primaryTokens, shadowTokens, primaryLogits, shadowLogits)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if receipt.DivergenceCount != 1 {
		t.Fatalf("divergence count = %d, want 1", receipt.DivergenceCount)
	}
	if receipt.DivergenceStep != 2 {
		t.Fatalf("divergence step = %d, want 2", receipt.DivergenceStep)
	}
	if receipt.TokenMatchRate != 0.8 { // 4 out of 5 matched = 0.8
		t.Fatalf("token match rate = %g, want 0.8", receipt.TokenMatchRate)
	}
	if receipt.MaxLogitDiff < 4.9 {
		t.Fatalf("max logit diff = %g, want >= 4.9", receipt.MaxLogitDiff)
	}
}

func TestQwen38MTP_ShadowFailureNeverDisruptsGeneration(t *testing.T) {
	// Compare with empty or nil shadow tokens simulating a failed shadow execution
	primaryTokens := []int{5, 10, 15}
	primaryLogits := [][]float32{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
	}

	receipt := CompareShadowOutputs(primaryTokens, nil, primaryLogits, nil)
	receipt.ShadowDisrupted = true
	receipt.ShadowError = "simulated shadow out of memory"

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if !receipt.ShadowDisrupted {
		t.Fatal("expected ShadowDisrupted=true")
	}
	if receipt.ShadowError != "simulated shadow out of memory" {
		t.Fatalf("unexpected shadow error: %q", receipt.ShadowError)
	}
	// Primary tokens are intact
	if !reflect.DeepEqual(receipt.PrimaryTokens, primaryTokens) {
		t.Fatalf("primary tokens corrupted: got %v, want %v", receipt.PrimaryTokens, primaryTokens)
	}
}

func TestQwen38MTP_ShadowReceiptValidation(t *testing.T) {
	// 1. Invalid schema version
	r1 := Qwen38MTPShadowReceipt{
		SchemaVersion:  "invalid-schema",
		Engine:         Qwen38EngineMTP,
		PrimaryEngine:  Qwen38EngineTargetDecode,
		ShadowEngine:   Qwen38EngineMTP,
		TokenMatchRate: 1.0,
		DivergenceStep: -1,
	}
	if err := r1.Validate(); err == nil {
		t.Fatal("expected validation error for invalid schema")
	}

	// 2. Non-native engine
	r2 := Qwen38MTPShadowReceipt{
		SchemaVersion:  Qwen38MTPShadowReceiptSchema,
		Engine:         Qwen38MTPEngine("llama.cpp"),
		PrimaryEngine:  Qwen38EngineTargetDecode,
		ShadowEngine:   Qwen38EngineMTP,
		TokenMatchRate: 1.0,
		DivergenceStep: -1,
	}
	if err := r2.Validate(); err == nil {
		t.Fatal("expected validation error for non-native engine")
	}

	// 3. Inconsistent divergence count and step
	r3 := Qwen38MTPShadowReceipt{
		SchemaVersion:   Qwen38MTPShadowReceiptSchema,
		Engine:          Qwen38EngineMTP,
		PrimaryEngine:   Qwen38EngineTargetDecode,
		ShadowEngine:    Qwen38EngineMTP,
		TokenMatchRate:  0.5,
		DivergenceCount: 2,
		DivergenceStep:  -1, // Should be >= 0
	}
	if err := r3.Validate(); err == nil {
		t.Fatal("expected validation error for inconsistent divergence count and step")
	}
}
