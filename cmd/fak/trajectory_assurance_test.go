package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
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

func TestRunTrajectoryAssuranceUltracodeStatusMode(t *testing.T) {
	path := filepath.Join(t.TempDir(), "status.json")
	payload := `{"schema":"fak.ultracode_status.v0","session_id":"s","run_id":"r","requested_profile":"on","resolved_profile":"on","state":"complete","outcome":{"verdict":"unverified","effect_readback":"not_observed","independent_witness":"not_observed","reconciliation":"not_observed","reason":"pending"},"activation":{"schema":"fak.ultracode_activation.v1","total":0,"active":0,"inactive":0,"degraded":0,"unknown":0,"verified":0,"ratio":0,"children":[]},"budget":{"schema":"fak.ultracode_budget_receipt.v1","declared_tokens":1,"wall_budget_ms":1,"started_at":"2099-01-01T00:00:00Z","deadline_at":"2099-01-01T00:00:01Z","authority":"incomplete","covered_children":0,"total_children":0,"consumed_tokens":0,"remaining_tokens":1,"consumed_wall_ms":0,"remaining_wall_ms":1,"token_overrun":false,"wall_overrun":false,"overrun":false,"complete":false,"admitted":false,"children":[]},"budget_phase":"provisional","workers":[]}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runTrajectoryAssurance(strings.NewReader("ignored"), &stdout, &stderr, []string{"--ultracode-status", path}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got trajectoryassurance.Receipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.SessionID != "s" || got.RunID != "r" || got.TrajectoryID != "r" || got.Layers[3].ReasonToken != trajectoryassurance.ReasonUltracodeSchemaUnsupported {
		t.Fatalf("receipt=%+v", got)
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
