package projectcompletion

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// Operator status never renders a bare "complete" for a non-production leaf
// (#4640): every closed bucket carries a maturity-qualified label, and an
// undeclared standard reads as undeclared, never as complete.
func TestMaturityLabelNeverBareComplete(t *testing.T) {
	for _, standard := range []string{"production", "integrated", "staging", "development", "demo", "prototype", "experiment", "research"} {
		got := MaturityLabel(standard)
		if got != standard+"-complete" {
			t.Fatalf("MaturityLabel(%q) = %q, want maturity-qualified label", standard, got)
		}
	}
	if got := MaturityLabel(""); got != "closed (maturity undeclared)" {
		t.Fatalf("MaturityLabel(empty) = %q, want undeclared label, never complete", got)
	}
	if got := MaturityLabel("  "); strings.Contains(got, "complete") {
		t.Fatalf("undeclared standard must not render any complete claim, got %q", got)
	}
}

// The #4640 witness: a captured status render where a closed toy bring-up is
// visibly demo-complete while the parent remains partially production-complete.
func TestRenderTextToyBringupStaysBelowProductionComplete(t *testing.T) {
	report := Summarize([]Issue{
		{Number: 1, Title: "toy model bring-up", State: "closed", ProjectWork: work("demo", 5)},
		{Number: 2, Title: "production serving path", State: "closed", ProjectWork: work("production", 5)},
		{Number: 3, Title: "production hardening", State: "open", ProjectWork: work("production", 10)},
	})
	got := RenderText(report)
	if !strings.Contains(got, "production complete: 5.00/20.00 points (25.0%)") {
		t.Fatalf("parent must stay partially production-complete:\n%s", got)
	}
	if !strings.Contains(got, "closed demo-complete") {
		t.Fatalf("closed toy bring-up must render demo-complete:\n%s", got)
	}
	// No closed bucket line may claim a bare, maturity-free "complete".
	if regexp.MustCompile(`(?m)^closed\s+complete`).MatchString(got) {
		t.Fatalf("status render shows a bare complete claim:\n%s", got)
	}

	// JSON consumers keep explicit stable fields: the machine standard is
	// unchanged and the render label is additive.
	b, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	for _, field := range []string{`"standard":"demo"`, `"label":"demo-complete"`, `"production_complete_points"`, `"baseline_points"`} {
		if !strings.Contains(string(b), field) {
			t.Fatalf("report JSON missing stable field %s:\n%s", field, b)
		}
	}
}

// A legacy close that never declared its project-work metadata stays an honest
// unknown in the render — visible, and never counted or labeled complete.
func TestRenderTextKeepsUndeclaredWorkVisible(t *testing.T) {
	report := Summarize([]Issue{
		{Number: 1, Title: "legacy close", State: "closed", ProjectWork: issuepolicy.ProjectWorkReadout{Status: issuepolicy.ProjectWorkUndeclared}},
		{Number: 2, Title: "production serving path", State: "open", ProjectWork: work("production", 20)},
	})
	got := RenderText(report)
	if !strings.Contains(got, "unknown #1 legacy close: undeclared") {
		t.Fatalf("legacy close must stay visible as unknown:\n%s", got)
	}
	if !strings.Contains(got, "[incomplete]") {
		t.Fatalf("confidence must degrade with undeclared work:\n%s", got)
	}
}
