package qwen38quantrun

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildAMDScoreboardFileWritesTypedReport(t *testing.T) {
	dir := t.TempDir()
	inputPath, reportPath := filepath.Join(dir, "input.json"), filepath.Join(dir, "report.json")
	raw, err := json.Marshal(validAMDScoreboardInput())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(inputPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	report, err := BuildAMDScoreboardFile(inputPath, reportPath)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Comparable {
		t.Fatalf("report=%+v", report)
	}
	written, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AMDScoreboardReport
	if err := json.Unmarshal(written, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAMDScoreboardReport(decoded); err != nil {
		t.Fatal(err)
	}
}

func TestBuildAMDScoreboardFileRejectsUnknownFields(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "input.json")
	if err := os.WriteFile(inputPath, []byte(`{"schema":"fak.qwen38.amd-scoreboard-input.v1","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildAMDScoreboardFile(inputPath, filepath.Join(dir, "report.json")); err == nil {
		t.Fatal("expected strict decode failure")
	}
}
