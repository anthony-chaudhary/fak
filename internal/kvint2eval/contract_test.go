package kvint2eval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestProjectionDispatchsAndUnknownRefuses(t *testing.T) {
	cases := []struct {
		name   string
		want   Disposition
		reason DecisionCode
	}{
		{"modeled-delegate.json", Dispatch, ProjectionNeedsRun},
		{"unsupported.json", Refuse, MethodRefuse},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", tc.name))
			if err != nil {
				t.Fatal(err)
			}
			var req Request
			if err = json.Unmarshal(raw, &req); err != nil {
				t.Fatal(err)
			}
			got := Evaluate(req)
			if got.Outcome != tc.want || got.Reason != tc.reason {
				t.Fatalf("got %s/%s", got.Outcome, got.Reason)
			}
			if got.Metrics != nil {
				t.Fatal("non-supported result leaked metrics")
			}
		})
	}
}

func TestOutputAwareINT2KVRotationWitness(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "l4-observed.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req Request
	if err = json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	got := Evaluate(req)
	if got.Outcome != Permit || got.Reason != ReasonOK {
		t.Fatalf("got %s/%s: %s", got.Outcome, got.Reason, got.Detail)
	}
	if got.Metrics == nil {
		t.Fatal("supported result omitted full measurements")
	}
	if got.Metrics.OutputNMSEAfterRotation >= got.Metrics.BaselineOutputNMSE {
		t.Fatalf("bounded output-aware candidate did not improve output NMSE: %+v", *got.Metrics)
	}
	if got.Metrics.CandidateCount != 9 || got.Metrics.DecodeTrials != 10 || len(got.Metrics.Candidates) != 9 || got.Metrics.SelectedRotation == 0 {
		t.Fatalf("candidate search or repeated decode evidence missing: %+v", *got.Metrics)
	}
	var baseline, selected *CandidateMetric
	for i := range got.Metrics.Candidates {
		c := &got.Metrics.Candidates[i]
		if c.Rotation == 0 {
			baseline = c
		}
		if c.Rotation == got.Metrics.SelectedRotation {
			selected = c
		}
	}
	if baseline == nil || selected == nil || baseline.ClipRatio == 0 || baseline.OutputNMSEStddev == 0 || selected.DecodeMicrosecondsStddev == 0 {
		t.Fatalf("tuned baseline, candidate, or variance evidence missing: %+v", *got.Metrics)
	}
	if got.Metrics.CandidateTaskAccuracy < got.Metrics.BaselineTaskAccuracy {
		t.Fatalf("bounded task witness regressed: %+v", *got.Metrics)
	}
	if got.Artifact.ID == "" || got.RecipeArtifact.Version != "v2" || got.Model.ID == "" || got.Runtime.Name != "cuda" {
		t.Fatal("provenance pins missing")
	}
}

func TestTamperedWitnessRefuses(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "l4-observed.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req Request
	if err = json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	req.Metrics.CacheBytes++
	got := Evaluate(req)
	if got.Outcome != Refuse || got.Reason != MetricsInvalid {
		t.Fatalf("got %s/%s", got.Outcome, got.Reason)
	}
}
