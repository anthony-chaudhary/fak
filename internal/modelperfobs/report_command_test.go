package modelperfobs

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestServiceGoalReportCommand(t *testing.T) {
	dir := t.TempDir()
	input, goal := filepath.Join(dir, "observations.jsonl"), filepath.Join(dir, "goal.json")
	rows := []Observation{serviceObservation("one", 3, 5, 5)}
	data, _ := json.Marshal(rows[0])
	if err := os.WriteFile(input, append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	g := serviceGoalFixture()
	data, _ = json.Marshal(g)
	if err := os.WriteFile(goal, data, 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := RunReport([]string{"--input", input, "--goal", goal, "--format", "json"}, &out); err != nil {
		t.Fatal(err)
	}
	var report ServiceGoalReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil || report.MissingRequests != 5 ||
		report.MeasuredSLOGoodput.RequestsPerSecond != .1 || report.QualityQualifiedGoodput != nil {
		t.Fatalf("report: %s, %v", out.String(), err)
	}
	out.Reset()
	if err := RunReport([]string{"--input", input, "--format", "json"}, &out); err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(Summarize(rows))
	if strings.TrimSpace(out.String()) != string(want) {
		t.Fatalf("legacy JSON changed: %s, want %s", out.String(), want)
	}
	out.Reset()
	if err := RunReport([]string{"--input", input, "--goal", goal}, &out); err != nil ||
		!strings.Contains(out.String(), "quality-qualified goodput: **unknown**") {
		t.Fatalf("markdown lost unknown: %s, %v", out.String(), err)
	}
	for _, raw := range []string{string(data) + "{}", strings.Replace(string(data), `"id":`, `"typo_id":`, 1)} {
		if _, err := ReadServiceGoal(strings.NewReader(raw)); err == nil {
			t.Fatalf("accepted malformed goal %s", raw)
		}
	}
	out.Reset()
	if err := RunReport([]string{"--help"}, &out); err != nil || !strings.Contains(out.String(), "-goal") {
		t.Fatalf("report help: %s, %v", out.String(), err)
	}
}
