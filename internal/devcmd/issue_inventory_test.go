package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func writeInventoryFixture(t *testing.T, dir, name string, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunIssueInventoryGateAndStableJSON(t *testing.T) {
	dir := t.TempDir()
	issues := writeInventoryFixture(t, dir, "issues.json", []map[string]any{{"number": 1, "title": "panic in x"}})
	fail := issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: "aaa", ModuleVersion: "internal/x@r1+gaaa", Witness: "observed"}
	obs := writeInventoryFixture(t, dir, "obs.json", issueInventoryInput{Schema: issuepolicy.ErrorInventorySchema, GeneratedAt: "2026-09-01T00:00:00Z", Observations: []issuepolicy.ErrorObservation{{Issue: 1, Observed: fail, Current: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: "bbb", ModuleVersion: "internal/x@r2+gbbb", Witness: "current"}}}})
	var out, errb bytes.Buffer
	code := RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", obs, "--json", "--require-actionable", "1"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"schema": "fak.issue-error-inventory/1"`) || !strings.Contains(out.String(), `"disposition": "ACTIONABLE"`) {
		t.Fatalf("output=%s", out.String())
	}
	first := out.String()
	out.Reset()
	errb.Reset()
	if code = RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", obs, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if out.String() != first {
		t.Fatalf("fixture output changed\nfirst=%s\nsecond=%s", first, out.String())
	}
}

func TestRunIssueInventoryRefusesFixedAndEvidenceDeficient(t *testing.T) {
	dir := t.TempDir()
	issues := writeInventoryFixture(t, dir, "issues.json", []map[string]any{{"number": 1, "title": "fixed"}, {"number": 2, "title": "unknown"}})
	obs := issueInventoryInput{GeneratedAt: time.Unix(1, 0).UTC().Format(time.RFC3339), Observations: []issuepolicy.ErrorObservation{
		{Issue: 1, Observed: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Witness: "old"}, Fix: issuepolicy.ErrorEvidence{Commit: "fix", Witness: "fix"}, Current: issuepolicy.ErrorEvidence{Status: "pass", Witness: "trunk"}},
		{Issue: 2},
	}}
	path := writeInventoryFixture(t, dir, "obs.json", obs)
	for _, tc := range []struct{ issue, want int }{{1, 3}, {2, 4}, {99, 2}} {
		var out, errb bytes.Buffer
		if got := RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", path, "--require-actionable", string(rune('0' + tc.issue))}); got != tc.want {
			t.Fatalf("issue=%d code=%d want=%d stderr=%s", tc.issue, got, tc.want, errb.String())
		}
	}
}

func TestRunIssueInventoryTreatsIssueBodyAsInertData(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "executed")
	issues := writeInventoryFixture(t, dir, "issues.json", []map[string]any{{"number": 1, "title": "report", "body": "touch " + marker}})
	obs := writeInventoryFixture(t, dir, "obs.json", issueInventoryInput{GeneratedAt: "2026-09-01T00:00:00Z", Observations: []issuepolicy.ErrorObservation{{Issue: 1}}})
	var out, errb bytes.Buffer
	RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", obs, "--json"})
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("issue body was executed: %v", err)
	}
}
