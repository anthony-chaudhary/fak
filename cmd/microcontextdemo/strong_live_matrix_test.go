package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTopKExamplesExcludesTuneSelf(t *testing.T) {
	r := semanticRecord{ID: "a", Title: "alpha cache"}
	xs := []semanticRecord{r, {ID: "b", Title: "alpha"}, {ID: "c", Title: "other"}}
	g := map[string]semanticConsensus{"a": {ID: "a"}, "b": {ID: "b"}, "c": {ID: "c"}}
	_, tr := topKExamples(r, xs, g, 2)
	for _, id := range tr.SelectedIDs {
		if id == "a" {
			t.Fatal("tune retrieval leaked query answer")
		}
	}
}
func TestVerifyStrongLiveMatrix(t *testing.T) {
	if err := verifyStrongLiveMatrix(filepath.Join("..", "..", "experiments", "microcontext", "s8k-strong-live-matrix-2026-08-10.json")); err != nil {
		t.Fatal(err)
	}
}
func TestStrongVerifierRejectsMissingTrace(t *testing.T) {
	r := strongLiveReport{Schema: strongLiveMatrixSchema, Trials: 2, Workers: 8, RetrievalK: 3}
	for _, n := range []string{"retrieval-rerank", "long-context", "chunk-map-reduce", "micro-context"} {
		r.Pipelines = append(r.Pipelines, strongPipeline{Pipeline: n, Calibration: abstentionCalibration{Candidates: []float64{0, .5, .8, 1}, TuneRecords: 16}, TrialResults: make([]liveTrial, 2), Aggregate: liveAggregate{PromptTokens: 1}, Grade: semanticGrade{Records: 16}})
	}
	b, _ := json.Marshal(r)
	p := filepath.Join(t.TempDir(), "bad.json")
	_ = os.WriteFile(p, b, 0644)
	if verifyStrongLiveMatrix(p) == nil {
		t.Fatal("accepted missing retrieval trace")
	}
}
