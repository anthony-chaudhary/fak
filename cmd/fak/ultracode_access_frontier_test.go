package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestUltracodeBenchAccessFrontierJSON(t *testing.T) {
	fixture := ultracodebench.AccessModeFrontierFixture()
	var out, errOut bytes.Buffer
	code := runUltracodeBench(&out, &errOut, []string{"--scenario", "access-frontier", "--widths", "1,2,4,8", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report ultracodebench.AccessModeFrontierReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.ClaimScope != "same-task agentic context work only" {
		t.Fatalf("claim scope = %q", report.ClaimScope)
	}
	joined := strings.Join(report.ExcludedBaselines, "|")
	for _, excluded := range []string{"single-request throughput", "traditional batching throughput", "provider billed tokens"} {
		if !strings.Contains(joined, excluded) {
			t.Fatalf("missing exclusion %q in %q", excluded, joined)
		}
	}
	if len(report.Artifacts) != len(fixture.Artifacts) {
		t.Fatalf("matrix shape = %+v", report.Artifacts)
	}
	for i, artifact := range report.Artifacts {
		if artifact.EvidenceKind != fixture.Artifacts[i].EvidenceKind || artifact.SourceArtifact != fixture.Artifacts[i].SourceArtifact || len(artifact.Cells) != len(fixture.Artifacts[i].Cells) {
			t.Fatalf("artifact[%d] not bound to fixture: got=%+v want=%+v", i, artifact, fixture.Artifacts[i])
		}
	}
	if report.Artifacts[0].Cells[4].Verdict != "GAIN" || report.Artifacts[1].Cells[0].Verdict != "ABSTAIN" {
		t.Fatalf("offline/live verdicts = %+v", report.Artifacts)
	}
}

func TestUltracodeBenchAccessFrontierRejectsBadWidth(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runUltracodeBench(&out, &errOut, []string{"--scenario", "access-frontier", "--widths", "2,nope"}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestUltracodeBenchAccessFrontierObservedInput(t *testing.T) {
	f := ultracodebench.AccessFrontierFixture()
	f.EvidenceKind = "observed_run"
	f.SourceArtifact = "run://small-model/session-1"
	b, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "run.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runUltracodeBench(&out, &errOut, []string{"--scenario", "access-frontier", "--scenario-input", path, "--widths", "1", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report ultracodebench.AccessModeFrontierReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.EvidenceKind != "observed_run" || report.SourceArtifact != f.SourceArtifact {
		t.Fatalf("report = %+v", report)
	}
}
