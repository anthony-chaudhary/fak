package issuecentrality

import (
	"reflect"
	"testing"
)

func TestBuildSampleIncludesMandatorySetsAndDeterministicStrata(t *testing.T) {
	milestone := &Milestone{Title: "m1"}
	issues := []PortfolioIssue{
		{Number: 9, Title: "P0 observed", Labels: []Label{{Name: "priority/P0"}, {Name: "observability"}}, Milestone: milestone},
		{Number: 2, Title: "now context", Labels: []Label{{Name: "gen/now"}, {Name: "managed-context"}}, Milestone: milestone},
		{Number: 7, Title: "both", Labels: []Label{{Name: "priority/P0"}, {Name: "gen/now"}, {Name: "security"}}},
		{Number: 6, Title: "second context", Labels: []Label{{Name: "managed-context"}}},
		{Number: 4, Title: "first context", Labels: []Label{{Name: "managed-context"}}},
		{Number: 5, Title: "unknown family"},
	}
	got := BuildSample(issues, SampleOptions{PerFamily: 1})
	if got.Total != 6 || got.P0Total != 2 || got.GenNowTotal != 2 || got.Milestoneless != 4 {
		t.Fatalf("counts = %+v", got)
	}
	var numbers []int
	for _, row := range got.Rows {
		numbers = append(numbers, row.Number)
	}
	if want := []int{2, 4, 5, 7, 9}; !reflect.DeepEqual(numbers, want) {
		t.Fatalf("sample numbers = %v, want %v", numbers, want)
	}
	if got.Rows[3].Number != 7 || !reflect.DeepEqual(got.Rows[3].Reasons, []string{"gen/now", "priority/P0"}) {
		t.Fatalf("deduped mandatory row = %+v", got.Rows[3])
	}
	if got.Rows[1].Family != "managed-context" || got.Rows[2].Family != FamilyUnknown {
		t.Fatalf("strata = %+v", got.Rows)
	}
	if got.Rows[0].Decision != "unknown-with-missing-evidence" || got.Rows[0].Centrality != "unclassified" || len(got.Rows[0].EvidenceGaps) == 0 {
		t.Fatalf("legacy decision = %+v", got.Rows[0])
	}
}

func TestBuildSampleDoesNotInferFamilyFromTitle(t *testing.T) {
	got := BuildSample([]PortfolioIssue{{Number: 1, Title: "security core GPU runtime"}}, SampleOptions{})
	if len(got.Rows) != 1 || got.Rows[0].Family != FamilyUnknown {
		t.Fatalf("title inferred family: %+v", got.Rows)
	}
}

func TestBuildSampleCarriesCanonicalRetainAndReframeDecisions(t *testing.T) {
	valid := "Centrality: Enabling (managed context)\nP1 Context: advanced - captures once\nP2 Net value: preserved - no cost\nP3 Adaptation: N/A - no adaptive surface\nP4 Operations: advanced - real path"
	invalid := "Centrality: Stewardship\nP1 Context: N/A\nP2 Net value: preserved - bounded\nP3 Adaptation: preserved - bounded\nP4 Operations: advanced - real path"
	got := BuildSample([]PortfolioIssue{
		{Number: 1, Title: "valid", Body: valid, Labels: []Label{{Name: "gen/now"}}},
		{Number: 2, Title: "invalid", Body: invalid, Labels: []Label{{Name: "priority/P0"}}},
	}, SampleOptions{})
	if got.Rows[0].Decision != "retain" || got.Rows[0].NamedOutcome != "managed context" || len(got.Rows[0].EvidenceGaps) != 0 {
		t.Fatalf("valid row = %+v", got.Rows[0])
	}
	if got.Rows[1].Decision != "reframe" || len(got.Rows[1].EvidenceGaps) == 0 || len(got.Rows[1].RepairActions) == 0 {
		t.Fatalf("invalid row = %+v", got.Rows[1])
	}
}
