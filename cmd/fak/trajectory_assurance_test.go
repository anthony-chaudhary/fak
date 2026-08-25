package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectoryassurance"
)

func TestRunTrajectoryAssuranceWritesShadowReceipt(t *testing.T) {
	passed := true
	units := int64(5)
	evidence := trajectoryassurance.Evidence{Source: "metric", Provenance: "run", Authority: "verifier", Freshness: "same_run", Reason: "verified"}
	input := trajectoryassurance.Input{
		DeterministicFloor:  []trajectoryassurance.DeterministicCheck{{Name: "test", Passed: &passed, Evidence: evidence}},
		ObjectiveProgress:   trajectoryassurance.Observation{State: trajectoryassurance.Pass, Evidence: evidence},
		Efficiency:          trajectoryassurance.EfficiencyInput{Outcome: &passed, ConstraintsSatisfied: &passed, ParentUnits: &units, AccountingComplete: true, Evidence: evidence},
		DelegationIntegrity: trajectoryassurance.Observation{State: trajectoryassurance.Pass, Evidence: evidence},
		SemanticReview:      trajectoryassurance.Observation{State: trajectoryassurance.Pass, Evidence: evidence},
	}
	payload, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runTrajectoryAssurance(bytes.NewReader(payload), &stdout, &stderr, nil); code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	var receipt trajectoryassurance.Receipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if !receipt.Shadow || receipt.State != trajectoryassurance.Pass {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestRunTrajectoryAssuranceRejectsRawPayloadFields(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTrajectoryAssurance(strings.NewReader(`{"raw_prompt":"secret"}`), &stdout, &stderr, nil)
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "unknown field") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}
