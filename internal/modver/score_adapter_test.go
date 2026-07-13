package modver

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestScorecardFileScoresMeanByModule(t *testing.T) {
	fixture := []byte(`{
	  "schema": "fak-scorecard-files/1",
	  "files": [
	    {"path":"internal/gateway/http.go", "score":80},
	    {"path":"internal/gateway/cache.go", "score":60},
	    {"path":"cmd/fak/main.go", "score":90},
	    {"path":"cmd/fak/serve.go", "score":70},
	    {"path":"tools/code_slop_scorecard.py", "score":40}
	  ]
	}`)
	got, err := ScorecardFileScores(fixture)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{
		"internal/gateway":          70,
		"cmd/fak":                   80,
		"tools/code_slop_scorecard": 40,
	}
	if len(got) != len(want) {
		t.Fatalf("scores = %#v, want %#v", got, want)
	}
	for module, score := range want {
		if got[module] != score {
			t.Errorf("%s score = %v, want %v", module, got[module], score)
		}
	}

	encoded, err := MarshalModuleScores(got)
	if err != nil {
		t.Fatal(err)
	}
	// The adapter output must feed the existing --scores decoder unchanged.
	joined, err := LoadScores(encoded)
	if err != nil {
		t.Fatalf("LoadScores(adapter output): %v\n%s", err, encoded)
	}
	if joined["internal/gateway"].Score != 70 {
		t.Fatalf("joined gateway score = %v, want 70", joined["internal/gateway"].Score)
	}
	var flat map[string]float64
	if err := json.Unmarshal(encoded, &flat); err != nil {
		t.Fatalf("adapter output is not flat module-number JSON: %v", err)
	}
}

func TestScorecardFileScoresRejectsDuplicatePath(t *testing.T) {
	_, err := ScorecardFileScores([]byte(`{"files":[
		{"path":"internal/gateway/a.go","score":1},
		{"path":"./internal/gateway/a.go","score":2}
	]}`))
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("error = %v, want duplicate-path refusal", err)
	}
}

func TestScorecardFileScoresNeedsRows(t *testing.T) {
	if _, err := ScorecardFileScores([]byte(`{"files":[]}`)); err == nil {
		t.Fatal("empty scorecard accepted")
	}
}
