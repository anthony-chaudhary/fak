package issuecentrality

import (
	"strings"
	"testing"
)

func TestPreviewMigrationRequiresExplicitEvidenceBackedSelection(t *testing.T) {
	issues := []Issue{{Number: 9, Title: "legacy", Body: "## Outcome\nKeep the kernel path working."}}
	selection := Selection{
		Number: 9, Centrality: "Enabling (managed-context outcome)", Evidence: "Outcome names the managed-context kernel path.",
		P1: "advanced - records the context link once", P2: "preserved - no runtime cost", P3: "N/A - no adaptation surface", P4: "advanced - makes the issue dispatchable",
	}
	plan, err := PreviewMigration(issues, []Selection{selection})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != "preview" || len(plan.Patches) != 1 || plan.Patches[0].Number != 9 {
		t.Fatalf("plan = %+v", plan)
	}
	body := plan.Patches[0].NewBody
	for _, want := range []string{"## Problem frame", "Centrality: Enabling (managed-context outcome)", "P1 Context: advanced", "Migration evidence: Outcome names"} {
		if !strings.Contains(body, want) {
			t.Fatalf("patch missing %q:\n%s", want, body)
		}
	}
	if issues[0].Body != "## Outcome\nKeep the kernel path working." {
		t.Fatalf("preview mutated input: %q", issues[0].Body)
	}
}

func TestPreviewMigrationRefusesInferenceAndBlanketRewrite(t *testing.T) {
	issues := []Issue{{Number: 1, Title: "Core-looking title", Body: "legacy"}, {Number: 2, Body: "legacy"}}
	valid := Selection{Number: 1, Centrality: "Core", Evidence: "linked witness", P1: "preserved - bounded", P2: "preserved - bounded", P3: "N/A - no adaptation", P4: "advanced - operated"}
	cases := []struct {
		name string
		sel  []Selection
		want string
	}{
		{"missing explicit issue", []Selection{{Number: 3, Centrality: "Core", Evidence: "x", P1: "preserved - x", P2: "preserved - x", P3: "preserved - x", P4: "preserved - x"}}, "not in the audited input"},
		{"missing evidence", []Selection{{Number: 1, Centrality: "Core", P1: "preserved - x", P2: "preserved - x", P3: "preserved - x", P4: "preserved - x"}}, "requires evidence"},
		{"duplicate selection", []Selection{valid, valid}, "duplicate selection"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PreviewMigration(issues, tc.sel)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}
