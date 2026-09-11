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
	// Exit 0 = PASS; exit 3 = the report rendered but conservation targets are
	// unmet (the operator-pinned max-effort posture). Both are valid renders of
	// the verb; exit 3 is asserted in Tandem with the FAIL verdict below.
	if code != 0 && code != 3 {
		t.Fatalf("runMCPFootprint --workspace exit code = %d, want 0 or 3; output:\n%s", code, out.String())
	}
	output := out.String()
	if code == 0 && !bytes.Contains(out.Bytes(), []byte("workspace-footprint: PASS")) {
		t.Fatalf("expected PASS in output, got:\n%s", output)
	}
	if code == 3 && !bytes.Contains(out.Bytes(), []byte("workspace-footprint: FAIL")) {
		t.Fatalf("expected FAIL in output, got:\n%s", output)
	}
	if !bytes.Contains(out.Bytes(), []byte("context conservation")) {
		t.Fatalf("expected 'context conservation' in output, got:\n%s", output)
	}
}

func TestFootprintWorkspaceJSON(t *testing.T) {
	var out bytes.Buffer
	code := runMCPFootprint(&out, io.Discard, []string{"--workspace", "--json"})
	if code != 0 && code != 3 {
		t.Fatalf("runMCPFootprint --workspace --json exit code = %d, want 0 (PASS) or 3 (operator-approved elevated effort); output:\n%s", code, out.String())
	}

	var report workspaceFootprintReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v; raw:\n%s", err, out.String())
	}

	if report.Schema != "fak-workspace-footprint/1" {
		t.Fatalf("report.Schema = %q, want fak-workspace-footprint/1", report.Schema)
	}
	// The repo now deliberately pins max reasoning effort on every OpenCode agent
	// (operator directive), so the live report legitimately FAILs the 2.0x
	// conservation target on the Subagent Thinking component. Assert the axes that
	// must still hold, and that the bloat signal fires only when a routine agent
	// carries a NON-default variant (max is not silently stale-passed).
	if report.Verdict != "FAIL" {
		t.Fatalf("report.Verdict = %q, want FAIL (operator-pinned max effort on routine agents)", report.Verdict)
	}
	if report.ConservationRatio >= 2.0 {
		t.Fatalf("report.ConservationRatio = %.2f, want < 2.0 while max effort is pinned (the guard must not silently stale-pass)", report.ConservationRatio)
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
