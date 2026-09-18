package devcmd

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// TestRunIssueContractSectionsJSON asserts `issue contract-sections --json`
// exits 0 and emits valid JSON carrying the declared schema plus problem-frame
// and required-section information.
func TestRunIssueContractSectionsJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIssueContractSections(&stdout, &stderr, []string{"--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}

	var decoded issuepolicy.RequiredSectionsContract
	if err := json.Unmarshal(stdout.Bytes(), &decoded); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, stdout.String())
	}
	if decoded.Schema != issuepolicy.RequiredSectionsSchema {
		t.Errorf("schema = %q, want %q", decoded.Schema, issuepolicy.RequiredSectionsSchema)
	}
	if decoded.Schema != "fak.issue-required-sections/1" {
		t.Errorf("schema literal = %q", decoded.Schema)
	}
	if len(decoded.Sections) == 0 {
		t.Error("no required sections emitted")
	}
	if decoded.ProblemFrame.Section != "Value" {
		t.Errorf("problem frame section = %q, want Value", decoded.ProblemFrame.Section)
	}
	if decoded.ProblemFrame.CentralityLine != "Centrality:" {
		t.Errorf("centrality line = %q", decoded.ProblemFrame.CentralityLine)
	}
	if len(decoded.ProblemFrame.CheckLabels) != 4 {
		t.Errorf("check labels = %v, want P1-P4", decoded.ProblemFrame.CheckLabels)
	}

	// The exported Go accessor must return the same manifest the CLI encodes.
	viaGo := IssueRequiredSections()
	if viaGo.Schema != decoded.Schema || len(viaGo.Sections) != len(decoded.Sections) {
		t.Errorf("Go accessor diverges from CLI JSON: %+v vs %+v", viaGo.Schema, decoded.Schema)
	}
}

// TestRunIssueContractSectionsText asserts the default human-readable output
// exits 0 and names the problem frame.
func TestRunIssueContractSectionsText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIssueContractSections(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(issuepolicy.RequiredSectionsSchema)) {
		t.Errorf("text output missing schema:\n%s", stdout.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("problem frame")) {
		t.Errorf("text output missing problem frame:\n%s", stdout.String())
	}
}

// TestRunIssueContractSectionsBadFlag asserts an unknown flag exits 2.
func TestRunIssueContractSectionsBadFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIssueContractSections(&stdout, &stderr, []string{"--nope"}); code != 2 {
		t.Fatalf("exit=%d, want 2; stderr=%s", code, stderr.String())
	}
}

// TestRunIssueContractSectionsRejectsArgs asserts a positional argument exits 2.
func TestRunIssueContractSectionsRejectsArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIssueContractSections(&stdout, &stderr, []string{"extra"}); code != 2 {
		t.Fatalf("exit=%d, want 2; stderr=%s", code, stderr.String())
	}
}
