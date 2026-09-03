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

func gymCorpusV2Path(t *testing.T) string {
	t.Helper()
	candidates := []string{
		filepath.Join("..", "..", "internal", "trajectoryassurance", "testdata", "gym-corpus.v2.json"),
		filepath.Join("internal", "trajectoryassurance", "testdata", "gym-corpus.v2.json"),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Fatalf("gym-corpus.v2.json not found in %v", candidates)
	return ""
}

func TestTrajectoryAssuranceGym(t *testing.T) {
	corpusPath := gymCorpusV2Path(t)
	var stdout, stderr bytes.Buffer
	code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym", "--corpus", corpusPath})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Gym Report: PROMOTE") {
		t.Fatalf("expected stdout to contain 'Gym Report: PROMOTE', got:\n%s", out)
	}
	if !strings.Contains(out, "Verdict:            PROMOTE") {
		t.Fatalf("expected stdout to contain 'Verdict:            PROMOTE', got:\n%s", out)
	}
}

func TestTrajectoryAssuranceGym_JSON(t *testing.T) {
	corpusPath := gymCorpusV2Path(t)
	var stdout, stderr bytes.Buffer
	code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym", "--corpus", corpusPath, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	var report trajectoryassurance.GymReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON report: %v\noutput:\n%s", err, stdout.String())
	}
	if report.Schema != trajectoryassurance.GymReportSchema {
		t.Fatalf("expected schema %q, got %q", trajectoryassurance.GymReportSchema, report.Schema)
	}
	if report.Promotion.Verdict != "PROMOTE" {
		t.Fatalf("expected PROMOTE verdict, got %q (reasons: %v)", report.Promotion.Verdict, report.Promotion.Reasons)
	}
	if report.Overall.Cases != 192 {
		t.Fatalf("expected 192 cases, got %d", report.Overall.Cases)
	}
}

func TestTrajectoryAssuranceGym_Thresholds(t *testing.T) {
	corpusPath := gymCorpusV2Path(t)
	threshPath := filepath.Join(t.TempDir(), "thresholds.json")
	// Set an impossibly high min_utility_ci95_low threshold (0.99) to force NO_PROMOTION
	customThresh := `{"proposed": true, "min_utility_ci95_low": 0.99, "min_security_ci95_low": 0.85, "max_false_hold": 0.05, "max_intervention_regret": 0.12}`
	if err := os.WriteFile(threshPath, []byte(customThresh), 0644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym", "--corpus", corpusPath, "--thresholds", threshPath, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	var report trajectoryassurance.GymReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON report: %v", err)
	}
	if report.Promotion.Verdict != "NO_PROMOTION" {
		t.Fatalf("expected NO_PROMOTION with strict threshold, got %q", report.Promotion.Verdict)
	}
	if report.Promotion.Threshold.MinUtilityCI95Low != 0.99 {
		t.Fatalf("expected Threshold.MinUtilityCI95Low = 0.99, got %f", report.Promotion.Threshold.MinUtilityCI95Low)
	}
}

func TestTrajectoryAssuranceGym_ReportFile(t *testing.T) {
	corpusPath := gymCorpusV2Path(t)
	reportOut := filepath.Join(t.TempDir(), "report.json")
	var stdout, stderr bytes.Buffer
	code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym", "--corpus", corpusPath, "--report", reportOut})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	// Verify human summary on stdout
	if !strings.Contains(stdout.String(), "Gym Report: PROMOTE") {
		t.Fatalf("expected summary on stdout, got: %s", stdout.String())
	}
	// Verify report file was written with JSON
	data, err := os.ReadFile(reportOut)
	if err != nil {
		t.Fatalf("failed to read report file: %v", err)
	}
	var fileReport trajectoryassurance.GymReport
	if err := json.Unmarshal(data, &fileReport); err != nil {
		t.Fatalf("failed to unmarshal report file JSON: %v", err)
	}
	if fileReport.Promotion.Verdict != "PROMOTE" {
		t.Fatalf("expected file report to have PROMOTE verdict, got %q", fileReport.Promotion.Verdict)
	}
}

func TestTrajectoryAssuranceGym_UsageAndErrors(t *testing.T) {
	corpusPath := gymCorpusV2Path(t)

	// Missing --corpus flag
	{
		var stdout, stderr bytes.Buffer
		code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym"})
		if code != 2 {
			t.Fatalf("expected exit 2 on missing corpus, got %d", code)
		}
		if !strings.Contains(stderr.String(), "usage: fak trajectory assurance gym") {
			t.Fatalf("expected usage message in stderr, got: %s", stderr.String())
		}
	}

	// Non-existent corpus file
	{
		var stdout, stderr bytes.Buffer
		code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym", "--corpus", "nonexistent-corpus.json"})
		if code != 1 {
			t.Fatalf("expected exit 1 on non-existent corpus, got %d", code)
		}
		if !strings.Contains(stderr.String(), "load corpus") {
			t.Fatalf("expected 'load corpus' in stderr, got: %s", stderr.String())
		}
	}

	// Non-existent thresholds file
	{
		var stdout, stderr bytes.Buffer
		code := runTrajectoryAssurance(nil, &stdout, &stderr, []string{"gym", "--corpus", corpusPath, "--thresholds", "nonexistent-thresholds.json"})
		if code != 1 {
			t.Fatalf("expected exit 1 on non-existent thresholds, got %d", code)
		}
		if !strings.Contains(stderr.String(), "read thresholds") {
			t.Fatalf("expected 'read thresholds' in stderr, got: %s", stderr.String())
		}
	}

	// Dispatch through runTrajectory entrypoint
	{
		var stdout, stderr bytes.Buffer
		code := runTrajectory(&stdout, &stderr, []string{"assurance", "gym", "--corpus", corpusPath, "--json"})
		if code != 0 {
			t.Fatalf("runTrajectory assurance gym failed: exit = %d, stderr = %s", code, stderr.String())
		}
		var report trajectoryassurance.GymReport
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to unmarshal JSON report from runTrajectory: %v", err)
		}
		if report.Promotion.Verdict != "PROMOTE" {
			t.Fatalf("expected PROMOTE, got %s", report.Promotion.Verdict)
		}
	}
}
