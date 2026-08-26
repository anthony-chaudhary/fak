package ultracodebench

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestIssue8677ToolResultShapeArtifact(t *testing.T) {
	raw, err := os.ReadFile("testdata/issue8677-tool-result-shape-campaign.json")
	if err != nil {
		t.Fatal(err)
	}
	var campaign ToolResultShapeCampaign
	if err := json.Unmarshal(raw, &campaign); err != nil {
		t.Fatal(err)
	}
	got, err := EvaluateToolResultShape(campaign)
	if err != nil {
		t.Fatal(err)
	}
	reportRaw, err := os.ReadFile("testdata/issue8677-tool-result-shape-report.json")
	if err != nil {
		t.Fatal(err)
	}
	var want ToolResultShapeReport
	if err := json.Unmarshal(reportRaw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		t.Fatalf("replayed report differs from captured report:\n%s", gotJSON)
	}
	if got.Verdict != "GAIN" || got.Crossovers["irrelevant"] != "small" || got.Crossovers["relevant"] != "large" {
		t.Fatalf("crossover = %+v", got)
	}
	for _, cell := range got.Cells {
		if cell.Omitted.Role != 0 || cell.Omitted.Repository != 0 || cell.Omitted.History != 0 {
			t.Fatalf("non-tool-result context changed: %+v", cell)
		}
	}
	if got.PromotionEvidence == "" || got.DemotionEvidence == "" || got.InvalidatingAssumption == "" || got.ReplayCommand == "" {
		t.Fatal("report lacks generation evidence or replay command")
	}
}
