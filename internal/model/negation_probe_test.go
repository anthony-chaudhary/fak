package model

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func loadNegationExamples(t *testing.T) []NegationActivationExample {
	t.Helper()
	data, err := os.ReadFile("testdata/negation_probe_examples.json")
	if err != nil {
		t.Fatal(err)
	}
	var ex []NegationActivationExample
	if err := json.Unmarshal(data, &ex); err != nil {
		t.Fatal(err)
	}
	return ex
}
func TestNegationProbe(t *testing.T) {
	const accuracyFloor = .90
	p, err := TrainNegationProbe(loadNegationExamples(t))
	if err != nil {
		t.Fatal(err)
	}
	if p.Layer != 2 {
		t.Fatalf("selected layer=%d want=2", p.Layer)
	}
	if p.HeldOutAccuracy < accuracyFloor {
		t.Fatalf("held-out accuracy=%.3f floor=%.2f", p.HeldOutAccuracy, accuracyFloor)
	}
	t.Logf("selected_layer=%d held_out_accuracy=%.3f floor=%.2f threshold=%.2f weights=%v", p.Layer, p.HeldOutAccuracy, accuracyFloor, p.Threshold, p.Weights)
}
func TestNegationProbeDetect(t *testing.T) {
	p, err := TrainNegationProbe(loadNegationExamples(t))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name       string
		activation []float64
		want       bool
	}{{"negated", []float64{2.1, .1}, true}, {"affirmative", []float64{-2.1, .1}, false}, {"neutral", []float64{0, 0}, false}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fires, score := p.Detect(tc.activation)
			if fires != tc.want {
				t.Fatalf("Detect=%v score=%.4f want=%v", fires, score, tc.want)
			}
			t.Logf("fires=%v score=%.4f", fires, score)
		})
	}
}
func TestNegationProbeArtifactRoundTrip(t *testing.T) {
	p, err := TrainNegationProbe(loadNegationExamples(t))
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/negation_probe_artifact.json")
	if err != nil {
		t.Fatal(err)
	}
	var wantP NegationProbeArtifact
	if err := json.Unmarshal(want, &wantP); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p, wantP) {
		t.Fatalf("artifact drift got=%+v want=%+v", p, wantP)
	}
	loaded, err := LoadNegationProbe(got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded, p) {
		t.Fatalf("roundtrip got=%+v want=%+v", loaded, p)
	}
}
