package dispatchtick

import (
	"encoding/json"
	"strings"
	"testing"
)

// Dispatch packets must state the issue's estimated work and its contribution
// against the parent production baseline (#4640): a worker reading the packet
// for a 5-point demo leaf must see that the parent scope is 34 points and that
// closing the leaf satisfies "demo", not production completion of the parent.
func TestIssuePromptPacketStatesProjectWorkAndParentContribution(t *testing.T) {
	in := sampleIssuePrompt()
	in.Body = strings.Join([]string{
		"## Parent context",
		"Parent: #4636. Depends on #4637.",
		"## Work estimate",
		"Estimate: 5 points (medium).",
		"## Overall completion contribution",
		"Contribution: 5/34 points (14.7%).",
		"## Completion standard",
		"demo",
	}, "\n")
	rec := BuildIssuePrompt(in)
	for _, want := range []string{
		"- Work estimate: Estimate: 5 points (medium).",
		"- Parent contribution: Contribution: 5/34 points (14.7%).",
		"- Completion standard: demo",
	} {
		if !strings.Contains(rec.Prompt, want) {
			t.Fatalf("dispatch packet missing %q:\n%s", want, rec.Prompt)
		}
	}
	if rec.WorkEstimate != "Estimate: 5 points (medium)." {
		t.Fatalf("record work estimate = %q, want the declared estimate", rec.WorkEstimate)
	}
	if rec.ParentContribution != "Contribution: 5/34 points (14.7%)." {
		t.Fatalf("record parent contribution = %q, want the declared contribution", rec.ParentContribution)
	}
	if rec.CompletionStandard != "demo" {
		t.Fatalf("record completion standard = %q, want demo", rec.CompletionStandard)
	}
	// JSON consumers get explicit stable fields, not prompt-text re-parsing.
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal packet: %v", err)
	}
	for _, field := range []string{`"work_estimate"`, `"parent_contribution"`, `"completion_standard"`} {
		if !strings.Contains(string(b), field) {
			t.Fatalf("packet JSON missing stable field %s:\n%s", field, b)
		}
	}
}

// A legacy issue with no project-work sections keeps the packet unchanged: the
// fields stay empty (visible omission), never a fabricated estimate.
func TestIssuePromptPacketKeepsUndeclaredProjectWorkEmpty(t *testing.T) {
	rec := BuildIssuePrompt(sampleIssuePrompt())
	if rec.WorkEstimate != "" || rec.ParentContribution != "" || rec.CompletionStandard != "" {
		t.Fatalf("legacy issue must keep project-work fields empty: %+v", rec)
	}
	if strings.Contains(rec.Prompt, "- Work estimate:") || strings.Contains(rec.Prompt, "- Parent contribution:") {
		t.Fatalf("legacy packet must not render fabricated project-work rows:\n%s", rec.Prompt)
	}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal packet: %v", err)
	}
	for _, field := range []string{`"work_estimate"`, `"parent_contribution"`, `"completion_standard"`} {
		if strings.Contains(string(b), field) {
			t.Fatalf("undeclared project work must stay omitted from packet JSON (%s):\n%s", field, b)
		}
	}
}
