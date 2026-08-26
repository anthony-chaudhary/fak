package ultracodenegcontrol

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestReplayArtifact(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "campaign.json"))
	if err != nil {
		t.Fatal(err)
	}
	var campaign Campaign
	if err := json.Unmarshal(data, &campaign); err != nil {
		t.Fatal(err)
	}
	report, err := Evaluate(campaign)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]Verdict{
		"shuffled-role":            Contradictory,
		"duplicated-context":       NoGain,
		"omitted-required-context": Abstain,
		"random-truncation":        Abstain,
	}
	for _, result := range report.Results {
		if result.Verdict != want[result.Name] {
			t.Errorf("%s verdict = %s, want %s", result.Name, result.Verdict, want[result.Name])
		}
		if result.CreditedSavings != 0 {
			t.Errorf("%s credited harmful/placebo savings: %d", result.Name, result.CreditedSavings)
		}
		if len(result.SourceReceipts) != 2 || result.SourceReceipts[0] == "" || result.SourceReceipts[1] == "" {
			t.Errorf("%s missing replay receipts", result.Name)
		}
	}
}

func TestUnexpectedPassIsPublishedAsContradiction(t *testing.T) {
	campaign := fixtureCampaign(t)
	for i := range campaign.Controls {
		if campaign.Controls[i].Name == "random-truncation" {
			campaign.Controls[i].Observed.AcceptedOutcome = campaign.Controls[i].Baseline.AcceptedOutcome
		}
	}
	report, err := Evaluate(campaign)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range report.Results {
		if result.Name == "random-truncation" {
			if result.Verdict != Contradictory || !result.PublishContradiction || result.CreditedSavings != 0 {
				t.Fatalf("unexpected pass hidden or credited: %+v", result)
			}
			return
		}
	}
	t.Fatal("random-truncation result missing")
}

func TestMissingTelemetryAbstainsWithoutSavings(t *testing.T) {
	campaign := fixtureCampaign(t)
	campaign.Controls[0].Observed.AuthoritativeUsage = false
	report, err := Evaluate(campaign)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range report.Results {
		if result.Name == campaign.Controls[0].Name && (result.Verdict != Abstain || result.CreditedSavings != 0) {
			t.Fatalf("missing telemetry result = %+v", result)
		}
	}
}

func fixtureCampaign(t *testing.T) Campaign {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "campaign.json"))
	if err != nil {
		t.Fatal(err)
	}
	var campaign Campaign
	if err := json.Unmarshal(data, &campaign); err != nil {
		t.Fatal(err)
	}
	return campaign
}
