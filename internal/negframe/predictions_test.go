package negframe

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type negationPredictionRow struct {
	ID          string   `json:"id"`
	Claim       string   `json:"claim"`
	Variables   []string `json:"variables"`
	Prediction  string   `json:"prediction"`
	Falsifier   string   `json:"falsifier"`
	Measurement string   `json:"measurement"`
	Rungs       []string `json:"rungs"`
}

func TestNegationPredictions(t *testing.T) {
	data, err := os.ReadFile("testdata/negation_cost_predictions.json")
	if err != nil {
		t.Fatal(err)
	}
	var rows []negationPredictionRow
	if err := json.Unmarshal(data, &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows) < 4 {
		t.Fatalf("prediction rows=%d, want at least 4", len(rows))
	}
	seen := make(map[string]bool, len(rows))
	validRung := map[string]bool{"L0": true, "L1": true, "L2": true, "L3": true, "L4": true}
	for _, row := range rows {
		if row.ID == "" || seen[row.ID] {
			t.Fatalf("missing or duplicate prediction id %q", row.ID)
		}
		seen[row.ID] = true
		if strings.TrimSpace(row.Claim) == "" || len(row.Variables) == 0 || strings.TrimSpace(row.Prediction) == "" || strings.TrimSpace(row.Falsifier) == "" || strings.TrimSpace(row.Measurement) == "" || len(row.Rungs) == 0 {
			t.Fatalf("incomplete prediction %+v", row)
		}
		for _, rung := range row.Rungs {
			if !validRung[rung] {
				t.Fatalf("%s has unknown rung %q", row.ID, rung)
			}
		}
		t.Logf("%s | predicts: %s | falsifier: %s | measurement: %s | rungs=%s", row.ID, row.Prediction, row.Falsifier, row.Measurement, strings.Join(row.Rungs, ","))
	}
}
