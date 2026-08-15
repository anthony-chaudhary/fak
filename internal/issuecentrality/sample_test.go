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

func TestBuildSampleChoosesMostRecentlyUpdatedFamilyRow(t *testing.T) {
	issues := []PortfolioIssue{
		{Number: 9, Title: "older", UpdatedAt: "2026-08-01T00:00:00Z", Labels: []Label{{Name: "observability"}}},
		{Number: 7, Title: "newer high id", UpdatedAt: "2026-08-10T00:00:00Z", Labels: []Label{{Name: "observability"}}},
		{Number: 5, Title: "newer tie low id", UpdatedAt: "2026-08-10T00:00:00Z", Labels: []Label{{Name: "observability"}}},
		{Number: 1, Title: "missing timestamp", Labels: []Label{{Name: "observability"}}},
	}
	got := BuildSample(issues, SampleOptions{PerFamily: 1})
	if len(got.Rows) != 1 || got.Rows[0].Number != 5 {
		t.Fatalf("active stratum = %+v, want newest timestamp then lowest issue number", got.Rows)
	}
	for i, j := 0, len(issues)-1; i < j; i, j = i+1, j-1 {
		issues[i], issues[j] = issues[j], issues[i]
	}
	again := BuildSample(issues, SampleOptions{PerFamily: 1})
	if !reflect.DeepEqual(got.Rows, again.Rows) {
		t.Fatalf("input order changed sample:\nfirst=%+v\nsecond=%+v", got.Rows, again.Rows)
	}
}

func TestBuildSampleMandatorySetsIgnoreActivityOrdering(t *testing.T) {
	got := BuildSample([]PortfolioIssue{
		{Number: 10, UpdatedAt: "2020-01-01T00:00:00Z", Labels: []Label{{Name: "priority/P0"}, {Name: "security"}}},
		{Number: 11, UpdatedAt: "2020-01-01T00:00:00Z", Labels: []Label{{Name: "gen/now"}, {Name: "security"}}},
		{Number: 12, UpdatedAt: "2026-08-15T00:00:00Z", Labels: []Label{{Name: "security"}}},
	}, SampleOptions{})
	if len(got.Rows) != 3 || got.P0Total != 1 || got.GenNowTotal != 1 {
		t.Fatalf("activity displaced mandatory rows: %+v", got)
	}
}

func TestBuildSampleMissingTimestampsFallBackToIssueNumber(t *testing.T) {
	got := BuildSample([]PortfolioIssue{
		{Number: 8, Labels: []Label{{Name: "security"}}},
		{Number: 3, UpdatedAt: "not-a-time", Labels: []Label{{Name: "security"}}},
	}, SampleOptions{})
	if len(got.Rows) != 1 || got.Rows[0].Number != 3 {
		t.Fatalf("missing timestamp fallback = %+v", got.Rows)
	}
}
