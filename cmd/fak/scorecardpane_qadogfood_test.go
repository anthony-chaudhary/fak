package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// qaDogfoodSpineBody is a QA-dogfood-spine issue body in the real template shape
// (#1982's own shape): a spine marker, a Root-point change, a Done condition, and a
// Witness command.
const qaDogfoodSpineBody = "<!-- fak-qa-dogfood-spine:QD-022 -->\n\n" +
	"## Current state\nSome origin lacks a control.\n\n" +
	"## Root-point change\nAdd the at-origin control.\n\n" +
	"## Done condition\nThe control refuses the failure at origin.\n\n" +
	"## Witness\ngo test ./internal/x -run Y\n"

// TestQADogfoodPanelBuildsFromIssueBodies proves buildQADogfoodIssues reads state,
// staleness, and the closure-witness + root-point sections out of real issue bodies,
// and that the fold reports the four figures the done condition names.
func TestQADogfoodPanelBuildsFromIssueBodies(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	horizon := 14 * 24 * time.Hour
	rows := []qaDogfoodIssueRow{
		// Open, fresh, fully-fielded origin control.
		{Number: 1, State: "OPEN", Body: qaDogfoodSpineBody, UpdatedAt: now.Add(-2 * time.Hour).Format(time.RFC3339)},
		// Open, stale (untouched 40d), still fully-fielded.
		{Number: 2, State: "OPEN", Body: qaDogfoodSpineBody, UpdatedAt: now.Add(-40 * 24 * time.Hour).Format(time.RFC3339)},
		// Open, bare body — no witness, no root-point sections.
		{Number: 3, State: "open", Body: "just a sentence, no sections", UpdatedAt: now.Add(-1 * time.Hour).Format(time.RFC3339)},
		// Closed but fully-fielded: counts for witness + root-point, not open/stale.
		{Number: 4, State: "CLOSED", Body: qaDogfoodSpineBody, UpdatedAt: now.Add(-90 * 24 * time.Hour).Format(time.RFC3339)},
	}

	issues := buildQADogfoodIssues(rows, now, horizon)
	if got := issues[0].RootPointChange; got != "Add the at-origin control." {
		t.Errorf("issue 1 RootPointChange = %q, want the section text", got)
	}
	if got := issues[0].ClosureWitness; got != "go test ./internal/x -run Y" {
		t.Errorf("issue 1 ClosureWitness = %q, want the witness command", got)
	}
	if issues[2].HasRootPointFields() || issues[2].HasClosureWitness() {
		t.Error("the bare issue should have neither root-point fields nor a closure witness")
	}

	p := scorecardpane.FoldQADogfoodPanel(issues)
	if p.Total != 4 {
		t.Errorf("Total = %d, want 4", p.Total)
	}
	if p.OpenCount != 3 {
		t.Errorf("OpenCount = %d, want 3", p.OpenCount)
	}
	if p.StaleCount != 1 {
		t.Errorf("StaleCount = %d, want 1", p.StaleCount)
	}
	if p.ClosureWitnessCount != 3 {
		t.Errorf("ClosureWitnessCount = %d, want 3", p.ClosureWitnessCount)
	}
	if p.RootPointCount != 3 {
		t.Errorf("RootPointCount = %d, want 3", p.RootPointCount)
	}
	if p.RootPointPercent != 75 {
		t.Errorf("RootPointPercent = %v, want 75", p.RootPointPercent)
	}
}

// TestQADogfoodPanelCommandJSONFromFixture drives the whole verb from a cached JSON
// fixture (the offline --existing-json path) and asserts the emitted panel payload.
func TestQADogfoodPanelCommandJSONFromFixture(t *testing.T) {
	rows := []qaDogfoodIssueRow{
		{Number: 1, State: "OPEN", Body: qaDogfoodSpineBody, UpdatedAt: time.Now().Add(-1 * time.Hour).Format(time.RFC3339)},
		{Number: 2, State: "CLOSED", Body: qaDogfoodSpineBody, UpdatedAt: time.Now().Add(-100 * 24 * time.Hour).Format(time.RFC3339)},
	}
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	fixture := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(fixture, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runScorecardQADogfood(&stdout, &stderr, []string{"--json", "--existing-json", fixture})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, stderr.String())
	}
	var panel scorecardpane.QADogfoodPanel
	if err := json.Unmarshal(stdout.Bytes(), &panel); err != nil {
		t.Fatalf("decode panel: %v (stdout=%q)", err, stdout.String())
	}
	if panel.Schema != scorecardpane.QADogfoodPanelSchema {
		t.Errorf("schema = %q, want %q", panel.Schema, scorecardpane.QADogfoodPanelSchema)
	}
	if panel.Total != 2 || panel.OpenCount != 1 || panel.ClosureWitnessCount != 2 || panel.RootPointCount != 2 {
		t.Errorf("panel = %+v, want total=2 open=1 closure=2 rootpoint=2", panel)
	}
	if panel.RootPointPercent != 100 {
		t.Errorf("RootPointPercent = %v, want 100", panel.RootPointPercent)
	}
}

// TestQADogfoodPanelCommandPlainRender proves the default (non-JSON) render emits the
// one-line control-pane card.
func TestQADogfoodPanelCommandPlainRender(t *testing.T) {
	rows := []qaDogfoodIssueRow{{Number: 1, State: "OPEN", Body: qaDogfoodSpineBody, UpdatedAt: time.Now().Format(time.RFC3339)}}
	raw, _ := json.Marshal(rows)
	fixture := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(fixture, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runScorecardQADogfood(&stdout, &stderr, []string{"--existing-json", fixture})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "qa-dogfood issue health") {
		t.Errorf("render = %q, want the control-pane card line", stdout.String())
	}
}
