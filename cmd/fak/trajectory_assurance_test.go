package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

func TestRunTrajectoryAssuranceBuildsFromDeclaredReceipts(t *testing.T) {
	dir := t.TempDir()
	now := time.Now().UTC()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	curve := write("curve.json", fmt.Sprintf(`{"schema":"fak-trajctl-curve/1","objectives":[{"objective_id":"issue-8828","signal":"HEALTHY","methods":[{"points":[{"unix_millis":%d,"run_id":"run-1"}]}]}]}`, now.UnixMilli()))
	audit := write("audit.jsonl", `{"schema":"fak-trajectory-audit/1","kind":"session","source":"codex","session_id":"run-1","usage_records":1}`+"\n")
	dojo := write("dojo.json", `{"schema":"fak-dojo-rsi/1","kept":true,"reason":"gain","witness":{"ok":true,"outcome":true,"constraints_satisfied":true,"parent_units":10,"child_units":[4,5],"accounting_complete":true}}`)
	effects := write("effects.json", fmt.Sprintf(`{"schema":"fak.orchestration_effect_receipt.v1","run_id":"run-1","child_id":"child","state":"VERIFIED","reconciliation":"RECONCILED","observed_at":%q,"witness":{"authority_id":"observer","author_child_id":"other"}}`, now.Format(time.RFC3339Nano)))
	var stdout, stderr bytes.Buffer
	args := []string{"--trajctl-curve", curve, "--trajectory-audit", audit, "--dojo-receipt", dojo, "--effect-receipts", effects, "--trajectory-id", "run-1"}
	if code := runTrajectoryAssurance(strings.NewReader("ignored"), &stdout, &stderr, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if strings.Contains(stdout.String(), "prompt") || !strings.Contains(stdout.String(), `"objective_id":"issue-8828"`) || !strings.Contains(stdout.String(), `"total_units":19`) {
		t.Fatalf("receipt=%s", stdout.String())
	}
}
