package issuepolicy

import (
	"strings"
	"testing"
)

func TestAppendProjectWorkDefaultsProduction(t *testing.T) {
	body, err := AppendProjectWorkDefaults("## Parent context\n#4638\n", ProjectWorkAuthoring{EstimatePoints: 3, ParentBaseline: 8, TargetEnvelope: "- paths: >= 1 command", WitnessedEnvelope: "- paths: 1 command"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"## Work estimate", "Estimate: 3 points", "Contribution: 3/8 points", "## Completion standard\nproduction", "## Target operating envelope", "## Witnessed operating envelope"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body=%q missing %q", body, want)
		}
	}
}
func TestAppendProjectWorkDefaultsPreservesExplicitDemoAndSections(t *testing.T) {
	original := "## Work estimate\nEstimate: 1 point.\n\n## Overall completion contribution\nContribution: 1/8 points.\n\n## Completion standard\ndemo\n"
	body, err := AppendProjectWorkDefaults(original, ProjectWorkAuthoring{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(body, "## Work estimate") != 1 || !strings.Contains(body, "\ndemo\n") {
		t.Fatalf("body=%q", body)
	}
}
func TestAppendProjectWorkDefaultsRefusesUnknownNumbers(t *testing.T) {
	if _, err := AppendProjectWorkDefaults("## Parent context\n#1", ProjectWorkAuthoring{}); err == nil || !strings.Contains(err.Error(), "estimate-points") {
		t.Fatalf("err=%v", err)
	}
}
func TestAppendProjectWorkDefaultsRefusesContributionAboveBaseline(t *testing.T) {
	if _, err := AppendProjectWorkDefaults("body", ProjectWorkAuthoring{EstimatePoints: 3, ContributionPoints: 9, ParentBaseline: 8}); err == nil || !strings.Contains(err.Error(), "<= parent baseline") {
		t.Fatalf("err=%v", err)
	}
}
