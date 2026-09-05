package loopgate

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find dos.toml in any parent directory")
		}
		dir = parent
	}
}

func TestIsNarrationTestClaim(t *testing.T) {
	tests := []struct {
		phrase string
		want   bool
	}{
		// Positive cases
		{"all tests pass", true},
		{"all tests passed", true},
		{"all 25 tests pass", true},
		{"all 25 tests passed", true},
		{"all 1 test passed", true},
		{"tests passed", true},
		{"test passed", true},
		{"test suite pass", true},
		{"test suite passed", true},
		{"verified green", true},
		{"unit tests pass", true},
		{"unit tests passed", true},
		{"unit test passed", true},
		{"all checks pass", true},
		{"all checks passed", true},
		{"feat: added auth gate, all 10 tests pass", true},
		{"fix: resolved race condition; verified green", true},

		// Negative cases
		{"updated README links", false},
		{"refactored database schema", false},
		{"fix: handle nil pointer in loopgate", false},
		{"added test cases for validation", false},
		{"writing test suite", false},
		{"investigate test failures", false},
		{"test runner refactoring", false},
		{"all checks implemented", false},
		{"unit test coverage report", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.phrase, func(t *testing.T) {
			got := IsNarrationTestClaim(tt.phrase)
			if got != tt.want {
				t.Errorf("IsNarrationTestClaim(%q) = %v, want %v", tt.phrase, got, tt.want)
			}
		})
	}
}

func TestNarrationGateRejectsUnwitnessedClaim(t *testing.T) {
	witnessCalled := false
	mockWitness := func(_ context.Context, _ Request) (WitnessResult, error) {
		witnessCalled = true
		return WitnessResult{Outcome: OutcomeWitnessed, Reason: "OK"}, nil
	}

	// 1. Via AdjudicateTurnWithReceipts with empty/nil receipts
	turn := Turn{
		ClaimedDone: true,
		Claim:       "all 25 tests pass",
		HeadRef:     "HEAD",
	}

	dec := AdjudicateTurnWithReceipts(context.Background(), turn, nil, mockWitness)
	if dec.Verdict != VerdictRefused {
		t.Fatalf("verdict = %s, want %s", dec.Verdict, VerdictRefused)
	}
	if dec.Reason != ReasonUnwitnessedNarrationClaim {
		t.Fatalf("reason = %q, want %s", dec.Reason, ReasonUnwitnessedNarrationClaim)
	}
	if witnessCalled {
		t.Fatal("external witness must not be called when narration gate refuses")
	}

	// 2. Via Adjudicate with empty receipts slice on Turn
	witnessCalled = false
	turnWithEmptyReceipts := Turn{
		ClaimedDone: true,
		Claim:       "all 25 tests pass",
		HeadRef:     "HEAD",
		Receipts:    []ReceiptRecord{},
	}
	dec2 := Adjudicate(context.Background(), turnWithEmptyReceipts, mockWitness)
	if dec2.Verdict != VerdictRefused {
		t.Fatalf("verdict = %s, want %s", dec2.Verdict, VerdictRefused)
	}
	if dec2.Reason != ReasonUnwitnessedNarrationClaim {
		t.Fatalf("reason = %q, want %s", dec2.Reason, ReasonUnwitnessedNarrationClaim)
	}
	if witnessCalled {
		t.Fatal("external witness must not be called when narration gate refuses in Adjudicate")
	}
}

func TestNarrationGateAllowsWitnessedClaim(t *testing.T) {
	witnessCalled := false
	mockWitness := func(_ context.Context, req Request) (WitnessResult, error) {
		witnessCalled = true
		return WitnessResult{
			Outcome: OutcomeWitnessed,
			Reason:  "OK",
			Detail:  "diff-witnessed",
			Rung:    "diff-witnessed",
		}, nil
	}

	turn := Turn{
		ClaimedDone: true,
		Claim:       "all tests pass",
		HeadRef:     "HEAD",
	}
	receipts := []ReceiptRecord{
		{
			Tool:     "go test",
			ExitCode: 0,
			Verdict:  "ALLOW",
		},
	}

	dec := AdjudicateTurnWithReceipts(context.Background(), turn, receipts, mockWitness)
	if dec.Verdict != VerdictWitnessed {
		t.Fatalf("verdict = %s, want %s (reason=%q, summary=%q)", dec.Verdict, VerdictWitnessed, dec.Reason, dec.Summary)
	}
	if !witnessCalled {
		t.Fatal("external witness should be called when execution receipt is present")
	}

	// Also verify that failing execution receipt (exit code 1) is rejected
	witnessCalled = false
	failingReceipts := []ReceiptRecord{
		{
			Tool:     "go test",
			ExitCode: 1,
			Verdict:  "ALLOW",
		},
	}
	decFailing := AdjudicateTurnWithReceipts(context.Background(), turn, failingReceipts, mockWitness)
	if decFailing.Verdict != VerdictRefused || decFailing.Reason != ReasonUnwitnessedNarrationClaim {
		t.Fatalf("failing receipt verdict = %s, reason = %s, want REFUSED/%s", decFailing.Verdict, decFailing.Reason, ReasonUnwitnessedNarrationClaim)
	}
	if witnessCalled {
		t.Fatal("external witness must not be called when receipt exit code is non-zero")
	}
}

func TestNarrationGateAllowsUnrelatedDoneClaim(t *testing.T) {
	witnessCalled := false
	mockWitness := func(_ context.Context, req Request) (WitnessResult, error) {
		witnessCalled = true
		return WitnessResult{
			Outcome: OutcomeWitnessed,
			Reason:  "OK",
			Detail:  "diff-witnessed",
			Rung:    "diff-witnessed",
		}, nil
	}

	turn := Turn{
		ClaimedDone: true,
		Claim:       "updated README links",
		HeadRef:     "HEAD",
	}

	dec := AdjudicateTurnWithReceipts(context.Background(), turn, nil, mockWitness)
	if dec.Verdict != VerdictWitnessed {
		t.Fatalf("verdict = %s, want %s (reason=%q, summary=%q)", dec.Verdict, VerdictWitnessed, dec.Reason, dec.Summary)
	}
	if !witnessCalled {
		t.Fatal("external witness should be called for unrelated done claims")
	}
}

func TestNarrationDosReasonRegistered(t *testing.T) {
	root := findRepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	content := string(raw)

	header := "[reasons.UNWITNESSED_NARRATION_CLAIM]"
	if !strings.Contains(content, header) {
		t.Fatalf("dos.toml does not declare %s", header)
	}
}
