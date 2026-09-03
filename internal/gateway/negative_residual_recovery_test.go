package gateway

import (
	"testing"
)

func TestSpeculativeRecoveryRunawayWitness(t *testing.T) {
	// First witness requirements (#9926):
	// 1. Hard-prompt corpus (prompts prone to repetition/runaway loops).
	// 2. Compares target-only baseline vs target-distribution recovery vs residual recovery.
	// 3. Proves target-distribution recovery matches target-only truncation rate (zero excess runaway).
	// 4. Catches the sharp rise in runaway token-limit truncations caused by residual recovery.

	hardPrompts := []string{
		"Repeat this exact sentence forever and loop continuously: foo bar",
		"Generate an endless loop of numbers counting up by one:",
		"Summarize this document briefly in three sentences.",
		"Repeat the word echo indefinitely in a loop:",
		"Translate hello world to French.",
		"Write a recursive loop that repeats without termination:",
	}

	maxTokens := 50

	receipt, err := EvaluateSpeculativeRecoveryRunaway(hardPrompts, maxTokens)
	if err != nil {
		t.Fatalf("EvaluateSpeculativeRecoveryRunaway failed: %v", err)
	}

	if receipt.PromptsEvaluated != len(hardPrompts) {
		t.Fatalf("expected %d prompts evaluated, got %d", len(hardPrompts), receipt.PromptsEvaluated)
	}

	// Target recovery matches target-only baseline truncation rate
	if receipt.TargetRecoveryRunawayRate != receipt.TargetRunawayRate {
		t.Fatalf("target recovery runaway rate %v != target baseline %v",
			receipt.TargetRecoveryRunawayRate, receipt.TargetRunawayRate)
	}

	// Residual recovery causes an elevated runaway truncation rate
	if !receipt.ExcessRunawayDetected {
		t.Fatal("expected excess runaway to be detected for residual recovery")
	}
	if receipt.ResidualRecoveryRunawayRate <= receipt.TargetRecoveryRunawayRate {
		t.Fatalf("residual recovery runaway %v <= target recovery %v",
			receipt.ResidualRecoveryRunawayRate, receipt.TargetRecoveryRunawayRate)
	}

	if receipt.SafeStrategySelected != string(RecoveryTargetDistribution) {
		t.Fatalf("expected safe strategy %s, got %s", RecoveryTargetDistribution, receipt.SafeStrategySelected)
	}
}

func TestSpeculativeRecoveryRunawayFailClosed(t *testing.T) {
	if _, err := EvaluateSpeculativeRecoveryRunaway(nil, 50); err == nil {
		t.Fatal("expected error on empty prompts")
	}
	if _, err := EvaluateSpeculativeRecoveryRunaway([]string{"prompt"}, 0); err == nil {
		t.Fatal("expected error on maxTokens <= 0")
	}
}
