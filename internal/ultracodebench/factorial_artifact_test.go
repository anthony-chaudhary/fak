package ultracodebench

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestIssue8648FactorialArtifact(t *testing.T) {
	const source = "testdata/issue8648-factorial-campaign.json"
	raw, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	var campaign FactorialCampaign
	if err := json.Unmarshal(raw, &campaign); err != nil {
		t.Fatal(err)
	}
	if got, want := FactorialWidths(campaign), []int{1, 2, 4, 8}; !reflect.DeepEqual(got, want) {
		t.Fatalf("widths = %v, want %v", got, want)
	}
	for _, cell := range campaign.Cells {
		for _, rep := range cell.Replicates {
			if !rep.Accepted || rep.OutcomeDigest != campaign.OutcomeDigest {
				t.Fatalf("width %d treatment %s has unequal outcome", cell.Width, cell.Treatment)
			}
			if cell.Cache == "warm" && rep.CachedTokens <= 0 {
				t.Fatalf("width %d treatment %s lacks positive warm-cache telemetry", cell.Width, cell.Treatment)
			}
		}
	}
	got, err := EvaluateFactorialCampaign(campaign, []int{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	reportRaw, err := os.ReadFile("testdata/issue8648-factorial-report.json")
	if err != nil {
		t.Fatal(err)
	}
	var want FactorialReport
	if err := json.Unmarshal(reportRaw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		t.Fatalf("replayed report differs from captured evaluator report:\n%s", gotJSON)
	}
	if got.PromotionEvidence == "" || got.DemotionEvidence == "" || got.InvalidatingAssumption == "" || got.ReplayCommand == "" {
		t.Fatal("report lacks generation evidence or replay command")
	}
}
