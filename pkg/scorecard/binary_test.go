package scorecard

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestProjectBinaryPreservesOrderFindingsAndAxisShares(t *testing.T) {
	rows := []BinaryResult{
		{Key: "pass", Axis: "a", Weight: 1, Passed: true, Detail: "green"},
		{Key: "hard", Axis: "a", Weight: 3, Hard: true, Detail: "broken"},
		{Key: "soft", Axis: "b", Weight: 2, Detail: "thin"},
	}
	kpis, weights := ProjectBinary(rows, map[string]float64{"a": 0.4, "b": 0.6}, map[string][]string{
		"hard": {"hard: additional magnitude"},
	})

	if got, want := []string{kpis[0].Key, kpis[1].Key, kpis[2].Key}, []string{"pass", "hard", "soft"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("KPI order = %v, want %v", got, want)
	}
	if got, want := kpis[1].Defects, []string{"hard: broken", "hard: additional magnitude"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("hard defects = %v, want %v", got, want)
	}
	if got, want := kpis[2].Soft, []string{"soft: thin"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("soft findings = %v, want %v", got, want)
	}
	if math.Abs(weights["pass"]-0.1) > 1e-12 || math.Abs(weights["hard"]-0.3) > 1e-12 || math.Abs(weights["soft"]-0.6) > 1e-12 {
		t.Fatalf("weights = %#v, want axis-normalized 0.1/0.3/0.6", weights)
	}
}

func TestProjectBinaryScaledPreservesRawAxisFormula(t *testing.T) {
	rows := []BinaryResult{
		{Key: "light", Axis: "maturity", Weight: 1, Passed: true},
		{Key: "heavy", Axis: "maturity", Weight: 3, Passed: true},
	}
	_, weights := ProjectBinaryScaled(rows, map[string]float64{"maturity": 0.4}, nil)
	if math.Abs(weights["light"]-0.4) > 1e-12 || math.Abs(weights["heavy"]-1.2) > 1e-12 {
		t.Fatalf("weights = %#v, want raw axis-scaled 0.4/1.2", weights)
	}
}

func TestBinaryAxisScoreAndPayloads(t *testing.T) {
	rows := []BinaryResult{
		{Key: "pass", Axis: "usage", Weight: 1, Passed: true, Detail: "green"},
		{Key: "hard", Axis: "usage", Weight: 3, Hard: true, Detail: "broken"},
	}
	if got := BinaryAxisScore(rows); got != 25 {
		t.Fatalf("BinaryAxisScore = %d, want 25", got)
	}
	payloads := BinaryPayloads(rows)
	if payloads[0].Score != 100 || payloads[0].Value != 1 || payloads[0].Defects != nil || payloads[0].Soft != nil {
		t.Fatalf("pass payload = %#v", payloads[0])
	}
	if got, want := payloads[1].Defects, []string{"hard: broken"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("hard payload defects = %v, want %v", got, want)
	}
	if got := BinaryAxisScore(nil); got != 0 {
		t.Fatalf("empty BinaryAxisScore = %d, want 0", got)
	}
}

func TestBinaryResultAndPayloadJSONContract(t *testing.T) {
	row := BinaryResult{Key: "pass", Weight: 1, Axis: "usage", Passed: true, Detail: "green"}
	gotRow, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"key":"pass","label":"","hard":false,"weight":1,"axis":"usage","passed":true,"detail":"green"}`; string(gotRow) != want {
		t.Fatalf("BinaryResult JSON = %s, want %s", gotRow, want)
	}

	gotPayload, err := json.Marshal(BinaryPayloads([]BinaryResult{row}))
	if err != nil {
		t.Fatal(err)
	}
	if want := `[{"kpi":"pass","group":"usage","score":100,"value":1,"detail":"green","defects":null,"soft":null}]`; string(gotPayload) != want {
		t.Fatalf("BinaryPayload JSON = %s, want %s", gotPayload, want)
	}
}
