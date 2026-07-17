package model

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func loadNegationGeometry(t *testing.T) []NegationPair {
	t.Helper()
	data, err := os.ReadFile("testdata/negation_geometry_pairs.json")
	if err != nil {
		t.Fatal(err)
	}
	var pairs []NegationPair
	if err := json.Unmarshal(data, &pairs); err != nil {
		t.Fatal(err)
	}
	return pairs
}
func TestNegationDirectionConsistency(t *testing.T) {
	const cosineFloor = .98
	const singularFloor = .99
	r, err := MeasureNegationGeometry(loadNegationGeometry(t), cosineFloor, singularFloor)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "shared" {
		t.Fatalf("verdict=%s report=%+v", r.Verdict, r)
	}
	if r.MeanCosine < cosineFloor || r.TopSingularValueFraction < singularFloor {
		t.Fatalf("threshold failure: %+v", r)
	}
	t.Logf("mean_cosine=%.6f top_singular_value_fraction=%.6f verdict=%s thresholds=(%.2f,%.2f)", r.MeanCosine, r.TopSingularValueFraction, r.Verdict, cosineFloor, singularFloor)
}
func TestNegationLinearMap(t *testing.T) {
	r, err := MeasureNegationGeometry(loadNegationGeometry(t), .98, .99)
	if err != nil {
		t.Fatal(err)
	}
	if r.SharedMapHeldOutRMSE >= r.PerConceptBaselineRMSE {
		t.Fatalf("shared map did not beat per-concept baseline: %+v", r)
	}
	if r.SharedMapHeldOutRMSE > 0.08 {
		t.Fatalf("held-out reconstruction too high: %+v", r)
	}
	t.Logf("shared_map_rmse=%.6f per_concept_baseline_rmse=%.6f mean_displacement=%v", r.SharedMapHeldOutRMSE, r.PerConceptBaselineRMSE, r.MeanDisplacement)
}
func TestNegationDirectionEntangledVerdict(t *testing.T) {
	pairs := []NegationPair{{Concept: "a", Split: "train", Affirmed: []float64{0, 0}, Negated: []float64{1, 0}}, {Concept: "b", Split: "train", Affirmed: []float64{0, 0}, Negated: []float64{0, 1}}, {Concept: "c", Split: "test", Affirmed: []float64{0, 0}, Negated: []float64{1, 0}}}
	r, err := MeasureNegationGeometry(pairs, .9, .9)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "entangled" {
		t.Fatalf("orthogonal directions verdict=%s", r.Verdict)
	}
}
func TestNegationGeometryWitness(t *testing.T) {
	r, err := MeasureNegationGeometry(loadNegationGeometry(t), .98, .99)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/negation_geometry_witness.json")
	if err != nil {
		t.Fatal(err)
	}
	var want NegationGeometryReport
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	if math.Abs(r.MeanCosine-want.MeanCosine) > 1e-12 || math.Abs(r.TopSingularValueFraction-want.TopSingularValueFraction) > 1e-12 || math.Abs(r.SharedMapHeldOutRMSE-want.SharedMapHeldOutRMSE) > 1e-12 || r.Verdict != want.Verdict {
		t.Fatalf("witness drift got=%+v want=%+v", r, want)
	}
}
