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
