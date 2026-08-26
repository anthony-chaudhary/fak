package ultracodenegcontrol

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCommittedReportMatchesReplay(t *testing.T) {
	campaign := fixtureCampaign(t)
	got, err := Evaluate(campaign)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join("testdata", "report.json"))
	if err != nil {
		t.Fatal(err)
	}
	var artifact struct {
		Schema                 string   `json:"schema"`
		Results                []Result `json:"results"`
		Promotion              string   `json:"promotion_evidence"`
		Demotion               string   `json:"demotion_or_retirement_evidence"`
		InvalidatingAssumption string   `json:"invalidating_assumption"`
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatal(err)
	}
	if artifact.Schema != got.Schema || !reflect.DeepEqual(artifact.Results, got.Results) {
		t.Fatalf("committed report drifted from replay\ngot:  %+v\nwant: %+v", got.Results, artifact.Results)
	}
	if artifact.Promotion == "" || artifact.Demotion == "" || artifact.InvalidatingAssumption == "" {
		t.Fatal("report must state promotion, retirement, and invalidation evidence")
	}
}
