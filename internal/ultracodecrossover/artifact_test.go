package ultracodecrossover

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestIssue8674ArtifactReplay(t *testing.T) {
	campaignBytes, err := os.ReadFile("issue8674-campaign.json")
	if err != nil {
		t.Fatal(err)
	}
	var campaign ComplexityCampaign
	if err := json.Unmarshal(campaignBytes, &campaign); err != nil {
		t.Fatal(err)
	}
	got, err := EvaluateComplexityCampaign(campaign)
	if err != nil {
		t.Fatal(err)
	}
	reportBytes, err := os.ReadFile("issue8674-report.json")
	if err != nil {
		t.Fatal(err)
	}
	var want ComplexityReport
	if err := json.Unmarshal(reportBytes, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("replay drifted\ngot:  %+v\nwant: %+v", got, want)
	}
	if got.LastEqualOutcomeRung != 2 || got.FirstFailureRung != 3 || got.StoppedAfterRung != 4 {
		t.Fatalf("unexpected observed crossover: %+v", got)
	}
	if len(got.Rungs) != 4 || got.Rungs[2].Verdict != "ABSTAIN" || got.Rungs[3].Verdict != "ABSTAIN" {
		t.Fatalf("quality failures were averaged away: %+v", got.Rungs)
	}
	if got.Rungs[0].ScopeAvoidedTokens <= 0 || got.Rungs[1].ScopeAvoidedTokens <= 0 {
		t.Fatalf("equal outcomes did not retain scoped attribution: %+v", got.Rungs)
	}
}
