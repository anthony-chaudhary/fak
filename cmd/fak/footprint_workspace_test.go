package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestFootprintWorkspaceVerb(t *testing.T) {
	var out bytes.Buffer
	code := runMCPFootprint(&out, io.Discard, []string{"--workspace"})
	if code != 0 {
		t.Fatalf("runMCPFootprint --workspace exit code = %d, want 0; output:\n%s", code, out.String())
	}
	output := out.String()
	if !bytes.Contains(out.Bytes(), []byte("workspace-footprint: PASS")) {
		t.Fatalf("expected PASS in output, got:\n%s", output)
	}
	if !bytes.Contains(out.Bytes(), []byte("context conservation")) {
		t.Fatalf("expected 'context conservation' in output, got:\n%s", output)
	}
}

func TestFootprintWorkspaceJSON(t *testing.T) {
	var out bytes.Buffer
	code := runMCPFootprint(&out, io.Discard, []string{"--workspace", "--json"})
	if code != 0 {
		t.Fatalf("runMCPFootprint --workspace --json exit code = %d, want 0; output:\n%s", code, out.String())
	}

	var report workspaceFootprintReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v; raw:\n%s", err, out.String())
	}

	if report.Schema != "fak-workspace-footprint/1" {
		t.Fatalf("report.Schema = %q, want fak-workspace-footprint/1", report.Schema)
	}
	if report.Verdict != "PASS" {
		t.Fatalf("report.Verdict = %q, want PASS", report.Verdict)
	}
	if report.ConservationRatio < 2.0 {
		t.Fatalf("report.ConservationRatio = %.2f, want >= 2.0", report.ConservationRatio)
	}
	if len(report.Components) < 4 {
		t.Fatalf("len(report.Components) = %d, want at least 4", len(report.Components))
	}
	if report.TotalSavedTok <= 0 {
		t.Fatalf("report.TotalSavedTok = %d, want > 0", report.TotalSavedTok)
	}
}

func TestEvaluateWorkspaceFootprintMock(t *testing.T) {
	tmp := t.TempDir()

	// Write bloated opencode.json
	ocContent := `{
		"instructions": ["CONTRIBUTING.md", "AGENTS.md"],
		"agent": {
			"build": { "variant": "high" },
			"general": { "variant": "high" }
		}
	}`
	if err := os.WriteFile(filepath.Join(tmp, "opencode.json"), []byte(ocContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "CONTRIBUTING.md"), []byte("contributor guidelines"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "AGENTS.md"), []byte("agent instructions"), 0644); err != nil {
		t.Fatal(err)
	}

	report := evaluateWorkspaceFootprint(tmp)
	if len(report.Findings) < 2 {
		t.Fatalf("expected at least 2 findings for bloated workspace, got %d: %v", len(report.Findings), report.Findings)
	}
}
