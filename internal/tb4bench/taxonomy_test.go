package tb4bench

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestFailureTaxonomyExhaustive(t *testing.T) {
	reasons := AllValidReasons()
	if len(reasons) != 9 {
		t.Fatalf("expected 9 valid closed reasons, got %d", len(reasons))
	}

	for _, r := range reasons {
		if !IsValidReason(r) {
			t.Errorf("expected reason %s to be valid", r)
		}
		if err := ValidateReason(r); err != nil {
			t.Errorf("expected ValidateReason(%s) to pass, got %v", r, err)
		}
	}

	if IsValidReason(ReasonUnclassified) {
		t.Errorf("UNCLASSIFIED must not be in valid reasons map")
	}
	if err := ValidateReason(ReasonUnclassified); err == nil {
		t.Errorf("expected error for UNCLASSIFIED reason, got nil")
	}
	if err := ValidateReason("SOME_RANDOM_REASON"); err == nil {
		t.Errorf("expected error for random unknown reason, got nil")
	}
}

func TestOracleContractAndResult(t *testing.T) {
	script := "echo pass\nexit 0\n"
	h := sha256.Sum256([]byte(script))
	scriptHash := "sha256:" + hex.EncodeToString(h[:])

	contract := OracleContract{
		Script:         script,
		ScriptHash:     scriptHash,
		TimeoutSeconds: 30,
	}
	if err := contract.Validate(); err != nil {
		t.Fatalf("expected valid contract: %v", err)
	}

	// Tamper script
	contractBad := contract
	contractBad.Script = "echo fail\n"
	if err := contractBad.Validate(); err == nil {
		t.Errorf("expected error when script doesn't match hash, got nil")
	}

	// Valid passing result
	passingResult := OracleResult{
		TaskID:     "task-1",
		ExitCode:   0,
		Stdout:     "pass",
		DurationMs: 120,
		Passed:     true,
	}
	if err := passingResult.Validate(); err != nil {
		t.Errorf("expected passing result to validate: %v", err)
	}

	// Inconsistent passing result with non-zero exit
	badPass := passingResult
	badPass.ExitCode = 1
	if err := badPass.Validate(); err == nil {
		t.Errorf("expected error for passed=true with exit code 1, got nil")
	}

	// Valid failing result
	failingResult := OracleResult{
		TaskID:        "task-2",
		ExitCode:      1,
		Stderr:        "assertion failed",
		DurationMs:    150,
		Passed:        false,
		FailureReason: ReasonTestFailed,
	}
	if err := failingResult.Validate(); err != nil {
		t.Errorf("expected failing result to validate: %v", err)
	}

	// Failing result with missing reason
	badFail := failingResult
	badFail.FailureReason = ""
	if err := badFail.Validate(); err == nil {
		t.Errorf("expected error for passed=false with empty failure reason, got nil")
	}
}

func TestComputeArmMetrics(t *testing.T) {
	results := []OracleResult{
		{TaskID: "task-1", Passed: true, ExitCode: 0, DurationMs: 1000},
		{TaskID: "task-2", Passed: false, ExitCode: 1, DurationMs: 2000, FailureReason: ReasonTestFailed},
	}
	telemetry := TelemetryTierMetrics{
		TotalPromptTokens:     1000,
		TotalCompletionTokens: 500,
		VDSOHits:              12,
		CompactedTokens:       300,
	}

	metrics, err := ComputeArmMetrics("fak_inkernel", results, telemetry)
	if err != nil {
		t.Fatalf("failed to compute arm metrics: %v", err)
	}

	if metrics.Official.TotalTasks != 2 {
		t.Errorf("expected 2 total tasks, got %d", metrics.Official.TotalTasks)
	}
	if metrics.Official.SolvedTasks != 1 {
		t.Errorf("expected 1 solved task, got %d", metrics.Official.SolvedTasks)
	}
	if metrics.Official.SolveRate != 0.5 {
		t.Errorf("expected 0.5 solve rate, got %f", metrics.Official.SolveRate)
	}
	if metrics.Official.MeanTaskDurationSeconds != 1.5 {
		t.Errorf("expected 1.5s mean duration, got %f", metrics.Official.MeanTaskDurationSeconds)
	}
	if metrics.Telemetry.TokenEfficiency != 1500.0 {
		t.Errorf("expected 1500 token efficiency (1500 tokens / 1 solved), got %f", metrics.Telemetry.TokenEfficiency)
	}
}
