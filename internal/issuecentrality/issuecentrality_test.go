package issuecentrality

import (
	"strings"
	"testing"
	"time"
)

func TestAuditUsesCanonicalProblemFrame(t *testing.T) {
	frame := "\n- P1: advanced - measured context effect\n- P2: preserved - no efficiency regression\n- P3: N/A - no adaptation surface\n- P4: advanced - live operator path\n"
	issues := []Issue{
		{Number: 1, Body: "- Centrality: Core" + frame},
		{Number: 2, Body: "- Centrality: Enabling(named cache outcome)" + frame},
		{Number: 3, Body: "Centrality: Stewardship(release obligation)"},
		{Number: 4, Body: "- Centrality: Peripheral"},
		{Number: 5, Body: "legacy issue"},
	}
	r := Audit(issues, "fixture", "testdata", time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC), nil)
	if r.Total != 5 || r.Classified != 4 || r.Counts.Core != 1 || r.Counts.Enabling != 1 || r.Counts.Stewardship != 1 || r.Counts.Peripheral != 1 || r.Counts.Unknown != 1 {
		t.Fatalf("unexpected report: %#v", r)
	}
	if r.Counts.ProblemFrame != 2 {
		t.Fatalf("complete frames = %d, want 2", r.Counts.ProblemFrame)
	}
	if r.CoveragePct != 80 {
		t.Fatalf("coverage = %v, want 80", r.CoveragePct)
	}
	if got := r.Text(); !strings.Contains(got, "CENTRALITY COVERAGE 80.0% (4/5)") || !strings.Contains(got, "Unknown 1") {
		t.Fatalf("text output:\n%s", got)
	}
}

func TestAuditReportsCanonicalInvalidFrameAsIncomplete(t *testing.T) {
	body := "Centrality: Enabling\nP1: N/A\nP2: advanced - measured\nP3: preserved - unchanged\nP4: advanced - operated"
	r := Audit([]Issue{{Body: body}}, "fixture", "test", time.Time{}, []string{"partial collection"})
	if r.Classified != 1 || r.Counts.Enabling != 1 {
		t.Fatalf("canonical class not reported: %#v", r)
	}
	if r.Counts.ProblemFrame != 0 {
		t.Fatalf("invalid canonical frame accepted: %#v", r)
	}
	if len(r.Errors) != 1 {
		t.Fatalf("errors = %#v", r.Errors)
	}
}
