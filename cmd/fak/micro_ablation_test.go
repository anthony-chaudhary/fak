package main

import (
	"context"
	"testing"
)

func TestRunMicroRetryAblationWitnessesGroundedContribution(t *testing.T) {
	r, err := runMicroRetryAblation(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !retryAblationPassed(r) {
		t.Fatalf("retry ablation did not isolate contribution: %+v", r)
	}
}

func TestRunMicroVerifierAblationCatchesUnsupportedClaim(t *testing.T) {
	r, err := runMicroVerifierAblation(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !verifierAblationPassed(r) {
		t.Fatalf("verifier ablation did not isolate contribution: %+v", r)
	}
}

func TestRunMicroHistoryAblationRetainsDurablePointerWithinCap(t *testing.T) {
	r := runMicroHistoryAblation()
	if !historyAblationPassed(r) {
		t.Fatalf("context ablation did not isolate contribution: %+v", r)
	}
}

func TestRunMicroModeAblationSameTask(t *testing.T) {
	r, err := runMicroModeAblation()
	if err != nil {
		t.Fatal(err)
	}
	if !modeAblationPassed(r) {
		t.Fatalf("bad mode receipt: %+v", r)
	}
}
